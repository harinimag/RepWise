"""Pose extraction, synthetic LSTM training, repetition counting, and coaching."""

from __future__ import annotations

import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

EXERCISES = (
    "Push-up", "Squat", "Bicep curl", "Front raise", "High knees",
    "Lateral raise",
)
WINDOW = 24
ANGLE_NAMES = (
    "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip", "left_knee", "right_knee",
)
FEATURE_SIZE = len(ANGLE_NAMES) + len(EXERCISES)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the interior angle ABC in degrees."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def extract_pose_features(landmarks, world_landmarks=None):
    """Return 3D joint angles, per-side visibility, and per-side trunk alignment."""
    source = world_landmarks if world_landmarks and len(world_landmarks) == len(landmarks) else landmarks
    dimensions = 3 if source is world_landmarks else 2
    points = np.asarray([[p.x, p.y, p.z] if dimensions == 3 else [p.x, p.y] for p in source], dtype=np.float32)
    visibility = []
    for point in landmarks:
        score = getattr(point, "visibility", None)
        if score is None:
            score = getattr(point, "presence", None)
        visibility.append(float(score) if score is not None else 1.0)
    visibility = np.asarray(visibility, dtype=np.float32)
    triplets = ((11, 13, 15), (12, 14, 16), (13, 11, 23), (14, 12, 24),
                (11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28))
    angles = np.asarray([angle_degrees(points[a], points[b], points[c]) for a, b, c in triplets], dtype=np.float32)
    left_indices = np.asarray([11, 13, 15, 23, 25, 27])
    right_indices = np.asarray([12, 14, 16, 24, 26, 28])
    side_confidence = {
        "left": float(np.mean(visibility[left_indices])),
        "right": float(np.mean(visibility[right_indices])),
    }
    trunk_alignment = {
        "left": angle_degrees(points[11], points[23], points[27]),
        "right": angle_degrees(points[12], points[24], points[28]),
    }
    if not np.all(np.isfinite(angles)):
        side_confidence = {side: min(score, 0.2) for side, score in side_confidence.items()}
        angles = np.nan_to_num(angles, nan=90.0, posinf=90.0, neginf=90.0)
    return np.clip(angles, 0, 180) / 180.0, side_confidence, trunk_alignment


def add_exercise_code(angles: np.ndarray, exercise: str) -> np.ndarray:
    one_hot = np.zeros(len(EXERCISES), dtype=np.float32)
    one_hot[EXERCISES.index(exercise)] = 1.0
    return np.concatenate((np.asarray(angles, dtype=np.float32), one_hot))


def synthetic_angles(exercise: str, progress: float) -> np.ndarray:
    """A deliberately simple angle-cycle generator used only to bootstrap the demo model."""
    phase = 2.0 * math.pi * progress
    movement = (1.0 - math.cos(phase)) / 2.0
    # Values are normalized degrees, in the same feature order as extract_pose_features.
    presets = {
        "Push-up": ([165, 165, 48, 48, 170, 170, 170, 170], [-88, -88, -14, -14, -8, -8, 0, 0]),
        "Squat": ([165, 165, 35, 35, 165, 165, 168, 168], [0, 0, 8, 8, -78, -78, -82, -82]),
        "Bicep curl": ([160, 160, 24, 24, 165, 165, 170, 170], [-116, -116, 4, 4, 0, 0, 0, 0]),
        "Front raise": ([165, 165, 24, 24, 165, 165, 170, 170], [0, 0, 78, 78, 0, 0, 0, 0]),
        "High knees": ([165, 165, 20, 20, 165, 165, 165, 165], [0, 0, 0, 0, 0, 0, -82, -82]),
        "Lateral raise": ([165, 165, 20, 20, 165, 165, 170, 170], [0, 0, 78, 78, 0, 0, 0, 0]),
    }
    baseline, delta = presets[exercise]
    values = np.asarray(baseline, dtype=np.float32) + movement * np.asarray(delta, dtype=np.float32)
    if exercise == "High knees":
        # Alternate legs to make the generated trace visibly temporal.
        alternating = (1.0 - math.cos(phase + math.pi)) / 2.0
        values[6] = 165 - 82 * movement
        values[7] = 165 - 82 * alternating
    return np.clip(values, 8, 178) / 180.0


def create_synthetic_training_data(seed: int = 41, sequences_per_exercise: int = 48):
    """Return generated sequences and phase labels. These are not human recordings."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    length = 32
    for exercise in EXERCISES:
        for _ in range(sequences_per_exercise):
            start = float(rng.uniform(0, 1))
            speed = float(rng.uniform(0.82, 1.18))
            frames = []
            labels = []
            for i in range(length):
                progress = (start + (i / length) * speed) % 1.0
                angles = synthetic_angles(exercise, progress)
                angles = np.clip(angles + rng.normal(0, 0.025, size=angles.shape), 0, 1)
                frames.append(add_exercise_code(angles, exercise))
                labels.append(int((1.0 - math.cos(2 * math.pi * progress)) / 2.0 > 0.5))
            xs.append(frames)
            ys.append(labels[-1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def train_phase_model():
    """Train a small LSTM on generated data; returned weights are demo-only."""
    import torch
    from torch import nn

    torch.manual_seed(41)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    class PhaseLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(FEATURE_SIZE, 24, batch_first=True)
            self.dropout = nn.Dropout(0.15)
            self.head = nn.Linear(24, 2)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.head(self.dropout(sequence[:, -1, :]))

    x, y = create_synthetic_training_data()
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    model = PhaseLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    batch_size = 32
    for _ in range(7):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_tensor[idx]), y_tensor[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def pose_model_available() -> bool:
    return POSE_MODEL_PATH.exists()


def download_pose_model() -> Path:
    """Download MediaPipe's small official Pose Landmarker model bundle."""
    POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


class PoseEstimator:
    """Thin wrapper around MediaPipe Tasks Pose Landmarker (image mode)."""
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, bgr_image: np.ndarray):
        rgb = np.ascontiguousarray(bgr_image[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]
        world_pose = getattr(result, "pose_world_landmarks", None)
        world_landmarks = world_pose[0] if world_pose else None
        angles, side_confidence, trunk_alignment = extract_pose_features(landmarks, world_landmarks)
        return angles, side_confidence, trunk_alignment, landmarks

    def close(self):
        self._landmarker.close()


@dataclass
class FrameResult:
    count: int
    phase: str
    phase_confidence: float
    feedback: str
    pose_confidence: float
    side: str
    metric_name: str
    metric_degrees: float
    angles_degrees: np.ndarray


class RepCounter:
    """Count only full, sustained ROM cycles with a clear start and return position."""
    SIDE_INDEX = {"left": (0, 2, 4, 6), "right": (1, 3, 5, 7)}

    def __init__(self, exercise: str):
        self.exercise = exercise
        self.count = 0
        self.frames = 0
        self.armed = False
        self.in_rep = False
        self.start_streak = 0
        self.active_streak = 0
        self.return_streak = 0
        self.active_started_at = 0
        self.last_rep_at = -10_000
        self.leg_state = [self._new_leg_state(), self._new_leg_state()]

    @staticmethod
    def _new_leg_state():
        return {"armed": False, "active": False, "start": 0, "lift": 0, "returned": 0, "lift_at": 0}

    def metric_name(self):
        return {
            "Push-up": "Selected elbow flexion",
            "Squat": "Mean knee flexion",
            "Bicep curl": "Selected elbow flexion",
            "Front raise": "Selected shoulder raise",
            "High knees": "Most-raised knee flexion",
            "Lateral raise": "Mean shoulder raise",
        }[self.exercise]

    @staticmethod
    def _advance_cycle(state, at_start, at_active, frame, min_gap):
        """Advance one limb's start → active → returned state, with 3-frame dwell."""
        if not state["armed"]:
            state["start"] = state["start"] + 1 if at_start else 0
            if state["start"] >= 3:
                state["armed"] = True
                state["start"] = 0
            return False
        if not state["active"]:
            state["lift"] = state["lift"] + 1 if at_active else 0
            if state["lift"] >= 3:
                state["active"] = True
                state["lift_at"] = frame
                state["lift"] = 0
            return False
        if at_start and frame - state["lift_at"] >= 6:
            state["returned"] += 1
            if state["returned"] >= 3 and frame - state["lift_at"] >= min_gap:
                state.update(RepCounter._new_leg_state())
                return True
        else:
            state["returned"] = 0
        return False

    def update(self, angles_degrees: np.ndarray, side: str, allow_count: bool = True) -> float:
        self.frames += 1
        left_elbow, right_elbow, left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee = angles_degrees
        left_i = self.SIDE_INDEX["left"]
        right_i = self.SIDE_INDEX["right"]
        if self.exercise == "Push-up":
            elbow = angles_degrees[left_i[0] if side == "left" else right_i[0]]
            metric = float(elbow)
            at_start, at_active = metric >= 155.0, metric <= 95.0
        elif self.exercise == "Squat":
            metric = float((left_knee + right_knee) / 2)
            hip = float((left_hip + right_hip) / 2)
            at_start, at_active = metric >= 150.0, metric <= 105.0 and hip <= 125.0
        elif self.exercise == "Bicep curl":
            elbow_i, shoulder_i = (left_i[0], left_i[1]) if side == "left" else (right_i[0], right_i[1])
            metric = float(angles_degrees[elbow_i])
            upper_arm_angle = float(angles_degrees[shoulder_i])
            at_start, at_active = metric >= 150.0, metric <= 65.0 and upper_arm_angle <= 45.0
        elif self.exercise == "Front raise":
            shoulder_i = left_i[1] if side == "left" else right_i[1]
            elbow_i = left_i[0] if side == "left" else right_i[0]
            metric = float(angles_degrees[shoulder_i])
            straight_arm = float(angles_degrees[elbow_i]) >= 145.0
            at_start, at_active = metric <= 25.0, metric >= 75.0 and metric <= 125.0 and straight_arm
        elif self.exercise == "Lateral raise":
            metric = float((left_shoulder + right_shoulder) / 2)
            straight_arms = left_elbow >= 145.0 and right_elbow >= 145.0
            at_start = left_shoulder <= 25.0 and right_shoulder <= 25.0
            at_active = left_shoulder >= 70.0 and right_shoulder >= 70.0 and straight_arms
        else:
            metric = float(min(left_knee, right_knee))
            if allow_count:
                for index, knee in enumerate((left_knee, right_knee)):
                    completed = self._advance_cycle(
                        self.leg_state[index], at_start=knee >= 150.0,
                        at_active=knee <= 100.0, frame=self.frames, min_gap=8,
                    )
                    if completed:
                        self.count += 1
            return metric

        if allow_count:
            if not self.armed:
                self.start_streak = self.start_streak + 1 if at_start else 0
                if self.start_streak >= 3:
                    self.armed = True
                    self.start_streak = 0
            elif not self.in_rep:
                self.active_streak = self.active_streak + 1 if at_active else 0
                if self.active_streak >= 3:
                    self.in_rep = True
                    self.active_started_at = self.frames
                    self.active_streak = 0
            elif at_start and self.frames - self.active_started_at >= 6:
                self.return_streak += 1
                if self.return_streak >= 3 and self.frames - self.last_rep_at >= 8:
                    self.count += 1
                    self.last_rep_at = self.frames
                    self.armed = False
                    self.in_rep = False
                    self.start_streak = 0
                    self.return_streak = 0
            else:
                self.return_streak = 0
        return metric


def feedback_for(exercise: str, angles: np.ndarray, pose_confidence: float, side: str,
                 trunk_alignment: float) -> str:
    if pose_confidence < 0.55:
        return "Pose confidence is low. Reposition so the required joints are visible before counting."
    a = angles
    elbow_i, shoulder_i, hip_i, knee_i = ((0, 2, 4, 6) if side == "left" else (1, 3, 5, 7))
    if exercise == "Push-up":
        elbow = float(a[elbow_i])
        if trunk_alignment < 155:
            return "Keep your head, hips, and ankles closer to one straight line."
        if elbow > 145:
            return "Start with elbows extended; lower until the selected elbow reaches about 90°."
        if elbow < 70:
            return "Your elbow is very deeply bent; use a controlled, comfortable range."
        return "Good plank line. Keep your elbow tracking back as you press."
    if exercise == "Squat":
        knee = float((a[6] + a[7]) / 2)
        hip = float((a[4] + a[5]) / 2)
        symmetry = abs(float(a[6] - a[7]))
        if symmetry > 20:
            return "Shift your weight so both knees bend more evenly."
        if knee > 140:
            return "Begin the squat by sending your hips back and bending both knees."
        if knee < 65 or hip < 55:
            return "You are at a very deep range; keep the motion controlled and comfortable."
        if knee > 110 or hip > 125:
            return "Continue lowering until your hips and knees are clearly bent."
        return "Good depth. Keep your knees tracking in line with your feet."
    if exercise == "Bicep curl":
        elbow = float(a[elbow_i])
        upper_arm = float(a[shoulder_i])
        if upper_arm > 45:
            return "Keep the upper arm beside your torso; avoid swinging the elbow forward."
        if elbow > 125:
            return "Curl until the selected elbow bends to about 60°, then lower fully."
        if elbow < 30:
            return "Avoid forcing the curl tighter; pause and lower slowly."
        return "Keep your upper arm still and control the return to a straight elbow."
    if exercise == "Front raise":
        shoulder = float(a[shoulder_i])
        elbow = float(a[elbow_i])
        if elbow < 140:
            return "Straighten the working arm slightly; keep a soft elbow, not a curl."
        if shoulder < 60:
            return "Raise the arm forward until it is near shoulder height."
        if shoulder > 125:
            return "Stop at shoulder height; avoid continuing overhead."
        return "Good raise. Keep your torso still and lower the arm fully."
    if exercise == "Lateral raise":
        shoulder = float((a[2] + a[3]) / 2)
        symmetry = abs(float(a[2] - a[3]))
        elbow = min(float(a[0]), float(a[1]))
        if symmetry > 22:
            return "Raise both arms evenly; keep the shoulders level."
        if elbow < 140:
            return "Keep a soft bend, but do not turn the movement into a curl."
        if shoulder < 55:
            return "Lift both arms out to the sides toward shoulder height."
        if shoulder > 115:
            return "Stop around shoulder height; avoid shrugging overhead."
        return "Good lateral raise. Lower both arms slowly to your sides."
    knee = float(min(a[6], a[7]))
    if abs(float(a[6] - a[7])) < 12:
        return "Alternate one knee at a time; return each foot before lifting the other."
    if knee > 125:
        return "Lift one knee until it is clearly bent, then return it under control."
    return "Land softly and keep your torso upright."


class FrameAnalyzer:
    def __init__(self, model, exercise: str):
        self.model = model
        self.exercise = exercise
        self.counter = RepCounter(exercise)
        self.window: list[np.ndarray] = []
        self.last_result: Optional[FrameResult] = None

    def reset(self, exercise: str):
        self.exercise = exercise
        self.counter = RepCounter(exercise)
        self.window = []
        self.smoothed_angles = None
        self.last_result = None

    def process_features(self, normalized_angles: np.ndarray, pose_confidence: float,
                         side: str = "left", trunk_alignment: float = 180.0) -> FrameResult:
        raw_angles = np.asarray(normalized_angles, dtype=np.float32) * 180.0
        if not hasattr(self, "smoothed_angles") or self.smoothed_angles is None:
            self.smoothed_angles = raw_angles.copy()
        else:
            self.smoothed_angles = 0.35 * raw_angles + 0.65 * self.smoothed_angles
        angles = self.smoothed_angles.copy()
        phase_confidence = 0.0
        phase = "UNCERTAIN"
        if pose_confidence >= 0.38:
            sample = add_exercise_code(normalized_angles, self.exercise)
            self.window.append(sample)
            self.window = self.window[-WINDOW:]
            padded = [self.window[0]] * (WINDOW - len(self.window)) + self.window
            import torch
            with torch.no_grad():
                probabilities = torch.softmax(self.model(torch.from_numpy(np.asarray([padded], dtype=np.float32)))[0], dim=0).numpy()
            phase_index = int(np.argmax(probabilities))
            phase_confidence = float(probabilities[phase_index])
            phase = ("MOVING" if phase_index else "READY") if phase_confidence >= 0.58 else "UNCERTAIN"
        metric = self.counter.update(angles, side=side, allow_count=pose_confidence >= 0.55)
        feedback = feedback_for(self.exercise, angles, pose_confidence, side, trunk_alignment)
        result = FrameResult(self.counter.count, phase, phase_confidence, feedback, pose_confidence,
                             side, self.counter.metric_name(), metric, angles)
        self.last_result = result
        return result

