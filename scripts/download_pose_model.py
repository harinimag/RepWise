import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posture_coach.engine import download_pose_model


if __name__ == "__main__":
    print(f"Downloading pose model to: {download_pose_model()}")

