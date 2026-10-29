from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from posture_coach.engine import EXERCISES, FrameAnalyzer, PoseEstimator, pose_model_available, train_phase_model

st.set_page_config(page_title="LSTM Posture Coach", page_icon="🏋️", layout="wide")


@st.cache_resource(show_spinner="Preparing the synthetic-data LSTM…")
def load_phase_model():
    return train_phase_model()


@st.cache_resource(show_spinner=False)
def load_pose_estimator():
    return PoseEstimator()


def draw_pose(frame: np.ndarray, landmarks) -> np.ndarray:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    links = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
             (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32))
    for start, end in links:
        cv2.line(frame, points[start], points[end], (80, 230, 170), 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 4, (40, 210, 255), -1, cv2.LINE_AA)
    return frame


def overlay_status(frame: np.ndarray, result, exercise: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 120), (22, 28, 40), -1)
    if result is None:
        lines = [f"{exercise} | Waiting for a visible pose", "Reposition to show the required joints"]
    else:
        lines = [f"{exercise} | Reps: {result.count} | LSTM: {result.phase}",
                 f"{result.metric_name}: {result.metric_degrees:.0f} deg | tracked side: {result.side}",
                 result.feedback]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (18, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.61 if i == 0 else 0.52, (245, 248, 255), 2, cv2.LINE_AA)
    return out


class LiveSession:
    """Thread-safe state shared by the WebRTC frame callback and Streamlit UI."""
    SIDE_VIEW_EXERCISES = ("Push-up", "Bicep curl", "Front raise")

    def __init__(self, estimator, model, exercise):
        self.estimator = estimator
        self.analyzer = FrameAnalyzer(model, exercise)
        self.exercise = exercise
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_result = None
        self.pose_missing = False
        self.target_reps = None
        self.last_motion_at = time.monotonic()
        self.last_metric = None
        self.motivation = ""

    def set_exercise(self, exercise):
        with self.lock:
            if exercise != self.exercise:
                self.exercise = exercise
                self.analyzer.reset(exercise)
                self.last_metric = None
                self.last_motion_at = time.monotonic()
                self.motivation = ""

    def set_target(self, target_reps):
        with self.lock:
            self.target_reps = target_reps

    def process_frame(self, frame):
        import av

        image = frame.to_ndarray(format="bgr24")
        detected = self.estimator.process(image)
        with self.lock:
            if detected is None:
                result = None
                rendered = overlay_status(image, None, self.exercise)
                self.motivation = ""
            else:
                features, side_scores, trunk_angles, landmarks = detected
                if self.exercise in self.SIDE_VIEW_EXERCISES:
                    side = max(side_scores, key=side_scores.get)
                    confidence = side_scores[side]
                    trunk_alignment = trunk_angles[side]
                else:
                    side = "left"
                    confidence = min(side_scores.values())
                    trunk_alignment = float(np.mean(list(trunk_angles.values())))
                already_done = self.target_reps is not None and self.analyzer.counter.count >= self.target_reps
                result = self.analyzer.last_result if already_done else self.analyzer.process_features(
                    features, confidence, side, trunk_alignment
                )
                if result is not None and confidence >= 0.55:
                    if self.last_metric is None or abs(result.metric_degrees - self.last_metric) >= 5.0:
                        self.last_motion_at = time.monotonic()
                        self.last_metric = result.metric_degrees
                    paused_for = time.monotonic() - self.last_motion_at
                    self.motivation = "" if paused_for < 8.0 else "Keep going—you’ve got this. One controlled rep at a time!"
                else:
                    self.motivation = ""
                rendered = overlay_status(draw_pose(image, landmarks), result, self.exercise)
            self.latest_result = result
            self.latest_image = rendered
            self.pose_missing = detected is None
        return av.VideoFrame.from_ndarray(rendered, format="bgr24")

    def snapshot(self):
        with self.lock:
            return self.latest_image, self.latest_result, self.pose_missing, self.motivation


VIEW_GUIDANCE = {
    "Push-up": "Side-on. Keep head, hips, ankles, and one elbow visible.",
    "Squat": "Face the camera. Keep both hips, knees, and ankles visible.",
    "Bicep curl": "Side-on. Show one full arm and your torso; the clearer arm is tracked.",
    "Front raise": "Side-on. Show one full arm and torso; the clearer arm is tracked.",
    "High knees": "Face the camera. Keep both knees and ankles visible; each completed knee lift counts once.",
    "Lateral raise": "Face the camera. Keep both arms and shoulders visible; both arms must reach the target range.",
}

LEVEL_BASE_REPS = {
    "Beginner": {"Push-up": 8, "Squat": 10, "Bicep curl": 10, "Front raise": 8,
                 "High knees": 20, "Lateral raise": 8},
    "Intermediate": {"Push-up": 12, "Squat": 15, "Bicep curl": 12, "Front raise": 10,
                     "High knees": 30, "Lateral raise": 12},
    "Advanced": {"Push-up": 15, "Squat": 20, "Bicep curl": 15, "Front raise": 12,
                 "High knees": 40, "Lateral raise": 15},
}
SIDE_VIEW_EXERCISES = LiveSession.SIDE_VIEW_EXERCISES


def calculate_rep_target(level: str, exercise: str, height_cm: float, weight_kg: float):
    """Return a transparent demo target using level baseline plus a small BMI adjustment."""
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    if bmi < 18.5:
        adjustment = -1
    elif bmi < 25.0:
        adjustment = 0
    elif bmi < 30.0:
        adjustment = -1
    else:
        adjustment = -2
    baseline = LEVEL_BASE_REPS[level][exercise]
    return max(5, baseline + adjustment), baseline, adjustment, bmi


def show_result(result):
    a, b, c = st.columns(3)
    a.metric(result.metric_name, f"{result.metric_degrees:.0f}°")
    b.metric("Pose confidence", f"{result.pose_confidence:.0%}", f"tracked {result.side}")
    c.metric("LSTM phase", result.phase, f"{result.phase_confidence:.0%} confidence")
    st.info(result.feedback)


st.title("🏋️ LSTM Posture Coach")
st.caption("Choose exercises and let the coach calculate each target from workout level, height, and weight.")
st.warning("Prototype note: rep personalization is a transparent demo heuristic, not a validated fitness prescription. The LSTM uses synthetic training sequences.")
model = load_phase_model()

for key, default in (("workout_plan", None), ("workout_profile", None), ("workout_active", False),
                     ("workout_complete", False), ("workout_started", False),
                     ("workout_index", 0), ("completed_exercises", [])):
    if key not in st.session_state:
        st.session_state[key] = default

mode = st.radio("Mode", ("Workout coach", "Analyze a video"), horizontal=True)

if mode == "Workout coach":
    st.subheader("Choose your workout")
    level = st.selectbox("Workout level", tuple(LEVEL_BASE_REPS), key="profile_level")
    col1, col2 = st.columns(2)
    height_cm = col1.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0, key="profile_height")
    weight_kg = col2.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=65.0, step=1.0, key="profile_weight")
    selected_exercises = st.multiselect(
        "Choose one or more exercises",
        EXERCISES,
        default=["Squat"],
        key="selected_workout_exercises",
    )
    st.caption("The coach assigns each rep target; you choose only the exercise(s) and workout level.")
    submitted = st.button("Calculate targets and create workout", type="primary")
    if submitted:
        if not selected_exercises:
            st.error("Select at least one exercise for this workout.")
        else:
            st.session_state.workout_profile = {"level": level, "height_cm": height_cm, "weight_kg": weight_kg}
            st.session_state.workout_plan = []
            for exercise in selected_exercises:
                target, baseline, adjustment, bmi = calculate_rep_target(level, exercise, height_cm, weight_kg)
                st.session_state.workout_plan.append({
                    "exercise": exercise,
                    "target": target,
                    "baseline": baseline,
                    "adjustment": adjustment,
                    "done": False,
                })
            st.session_state.workout_index = 0
            st.session_state.workout_active = False
            st.session_state.workout_complete = False
            st.session_state.workout_started = False
            st.session_state.completed_exercises = []
            st.session_state.pop("live_session", None)

    plan = st.session_state.workout_plan
    if plan:
        profile = st.session_state.workout_profile
        bmi = profile["weight_kg"] / ((profile["height_cm"] / 100.0) ** 2)
        st.caption(f"{profile['level']} · {profile['height_cm']:.0f} cm · {profile['weight_kg']:.1f} kg · BMI input {bmi:.1f}. The BMI-based adjustment is a demo-only heuristic, not a measure of fitness.")
        st.dataframe([{"Exercise": item["exercise"], "Level baseline": item["baseline"],
                       "Height/weight adjustment": item["adjustment"], "Coach target (reps)": item["target"],
                       "Status": "Complete" if item["done"] else "Planned"} for item in plan],
                     hide_index=True, use_container_width=True)

    if plan and st.session_state.workout_complete:
        completed_names = ", ".join(item["exercise"] for item in plan)
        st.success("Selected workout completed! You reached every rep goal in this workout.")
        st.info(f"Completed: {completed_names}. Great work finishing the workout you chose!")
        if st.button("Create a new workout"):
            st.session_state.workout_plan = None
            st.session_state.workout_profile = None
            st.session_state.workout_complete = False
            st.session_state.completed_exercises = []
            st.session_state.pop("live_session", None)
            st.rerun()
    elif plan and not st.session_state.workout_active:
        if st.button("Start workout", type="primary"):
            st.session_state.workout_active = True
            if not st.session_state.workout_started:
                st.session_state.workout_index = 0
                st.session_state.completed_exercises = []
                st.session_state.pop("live_session", None)
                st.session_state.workout_started = True
            st.rerun()

    if st.session_state.workout_active:
        if not pose_model_available():
            st.error("Pose model is missing. Run `python scripts/download_pose_model.py` from this project folder.")
        else:
            try:
                from streamlit_webrtc import webrtc_streamer
                from streamlit_autorefresh import st_autorefresh

                index = st.session_state.workout_index
                item = st.session_state.workout_plan[index]
                exercise, target = item["exercise"], item["target"]
                st.subheader(f"Selected workout · exercise {index + 1} of {len(st.session_state.workout_plan)}: {exercise}")
                st.info(VIEW_GUIDANCE[exercise] + " Angles use MediaPipe 3D world landmarks and temporal smoothing.")
                estimator = load_pose_estimator()
                if "live_session" not in st.session_state:
                    st.session_state.live_session = LiveSession(estimator, model, exercise)
                live = st.session_state.live_session
                live.set_exercise(exercise)
                live.set_target(target)
                st_autorefresh(interval=900, key="coach-refresh")
                webrtc_streamer(key="posture-coach-camera", video_frame_callback=live.process_frame,
                                media_stream_constraints={"video": True, "audio": False}, async_processing=True)
                image, result, pose_missing, motivation = live.snapshot()
                video_col, stats_col = st.columns([1.2, 1])
                with video_col:
                    if image is not None:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Live coach view", use_container_width=True)
                with stats_col:
                    current = result.count if result else 0
                    st.metric("Repetitions", f"{min(current, target)} / {target}")
                    st.progress(min(current / target, 1.0))
                    if result:
                        show_result(result)
                    elif pose_missing:
                        st.warning("Step back and make the required joints visible to start tracking.")
                    if motivation:
                        st.success(motivation)
                if st.button("Pause workout"):
                    st.session_state.workout_active = False
                    st.rerun()
                if result and result.count >= target:
                    st.session_state.workout_plan[index]["done"] = True
                    st.session_state.completed_exercises.append(exercise)
                    if index + 1 >= len(st.session_state.workout_plan):
                        st.session_state.workout_active = False
                        st.session_state.workout_complete = True
                    else:
                        st.session_state.workout_index += 1
                    st.rerun()
            except Exception as exc:
                st.error(f"Live workout setup is unavailable: {exc}")
                st.info("Check browser camera permission, or switch to Analyze a video.")

else:
    exercise = st.selectbox("Exercise to analyze", EXERCISES, key="video-exercise")
    st.info(VIEW_GUIDANCE[exercise])
    uploaded = st.file_uploader("Choose a video", type=("mp4", "mov", "avi", "m4v"))
    if uploaded:
        st.video(uploaded.getvalue())
        if st.button("Analyze uploaded video", type="primary"):
            if not pose_model_available():
                st.error("Pose model is missing. Run `python scripts/download_pose_model.py` first.")
            else:
                try:
                    estimator = load_pose_estimator()
                    analyzer = FrameAnalyzer(model, exercise)
                    suffix = Path(uploaded.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp.flush()
                        capture = cv2.VideoCapture(tmp.name)
                        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        stride = max(1, total // 180)
                        progress = st.progress(0, text="Analyzing pose frames…")
                        frame_no, results, last_annotated = 0, [], None
                        while True:
                            ok, frame = capture.read()
                            if not ok:
                                break
                            if frame_no % stride == 0:
                                detected = estimator.process(frame)
                                if detected is not None:
                                    features, side_scores, trunk_angles, landmarks = detected
                                    if exercise in SIDE_VIEW_EXERCISES:
                                        side = max(side_scores, key=side_scores.get)
                                        confidence = side_scores[side]
                                        trunk_alignment = trunk_angles[side]
                                    else:
                                        side = "left"
                                        confidence = min(side_scores.values())
                                        trunk_alignment = float(np.mean(list(trunk_angles.values())))
                                    result = analyzer.process_features(features, confidence, side, trunk_alignment)
                                    results.append(result)
                                    last_annotated = overlay_status(draw_pose(frame, landmarks), result, exercise)
                            frame_no += 1
                            if total > 0 and frame_no % max(1, total // 100) == 0:
                                progress.progress(min(frame_no / total, 1.0), text=f"Analyzing frame {frame_no}/{total}")
                        capture.release()
                        progress.empty()
                    if not results:
                        st.error("No clear body pose was detected. Try a brighter, wider shot.")
                    else:
                        left, right = st.columns(2)
                        with left:
                            st.image(cv2.cvtColor(last_annotated, cv2.COLOR_BGR2RGB), caption="Last detected pose", use_container_width=True)
                        with right:
                            show_result(results[-1])
                            st.caption(f"Processed {len(results)} pose frames.")
                        st.line_chart({results[-1].metric_name: [r.metric_degrees for r in results]})
                except Exception as exc:
                    st.error(f"Could not analyze this clip: {exc}")

with st.expander("How the coach works"):
    st.markdown("""
- Workout level selects the rep target for each movement; height and weight are recorded as profile details.
- The clearest body side is selected for side-view movements. Bilateral movements require both sides to be visible.
- Smoothed angle state machines require a start range, held movement range, and return before counting.
- The workout advances when each target is reached and displays a gentle prompt after eight seconds without detected movement.
- The LSTM phase estimate is trained on synthetic sequences and is not a real-world accuracy measure.
""")
