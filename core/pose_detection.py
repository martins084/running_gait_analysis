"""
pose_detection.py — MediaPipe PoseLandmarker wrapper for pose extraction.

Uses the new MediaPipe Tasks API (>= 0.10.x) with PoseLandmarker
instead of the deprecated mp.solutions.pose.

Provides a PoseDetector class that:
  - Detects 33 body landmarks in individual frames
  - Processes entire videos and saves landmark sequences to JSON
  - Returns normalized (0–1) x, y, z coordinates plus visibility
"""

import cv2
import json
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ── MediaPipe Tasks API aliases ─────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Default model path (relative to project root)
_DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "pretrained" / "pose_landmarker_full.task"
)


class PoseDetector:
    """
    Wraps MediaPipe PoseLandmarker (Tasks API) to detect 33 body
    landmarks from RGB frames or full video files.
    """

    # Human-readable names for the 33 MediaPipe BlazePose landmarks
    LANDMARK_NAMES = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ):
        """
        Args:
            model_path: Path to the .task model file.
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence:  Minimum confidence for frame-to-frame tracking.
            num_poses: Max number of people to detect (usually 1 for gait analysis).
        """
        self._model_path = model_path
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._num_poses = num_poses

    # -------------------------------------------------------------------------
    # Internal — create a landmarker for a given running mode
    # -------------------------------------------------------------------------

    def _create_landmarker(self, mode: RunningMode) -> PoseLandmarker:
        """Build a PoseLandmarker configured for IMAGE or VIDEO mode."""
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=mode,
            num_poses=self._num_poses,
            min_pose_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        return PoseLandmarker.create_from_options(options)

    # -------------------------------------------------------------------------
    # Single-frame detection
    # -------------------------------------------------------------------------

    def detect_pose(self, frame: np.ndarray):
        """
        Detect pose landmarks in a single BGR frame (IMAGE mode).

        Args:
            frame: OpenCV BGR image (H x W x 3).

        Returns:
            landmarks: np.ndarray of shape (33, 4) — [x, y, z, visibility]
                       or None if no person detected.
            result: The raw PoseLandmarkerResult object.
        """
        # Convert BGR → RGB and wrap in MediaPipe Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        landmarker = self._create_landmarker(RunningMode.IMAGE)
        result = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None, None

        # Take the first detected person
        person = result.pose_landmarks[0]
        landmarks = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in person]
        )

        return landmarks, result

    # -------------------------------------------------------------------------
    # Full-video processing
    # -------------------------------------------------------------------------

    def process_video(self, video_path: str, output_json: str) -> list:
        """
        Run pose detection on every frame of a video (VIDEO mode)
        and save results to JSON.

        The output JSON contains:
          - video: source path
          - fps: source FPS
          - frame_count: total frames
          - detected_frames: how many had a pose
          - landmark_names: list of 33 joint names
          - poses: list of {frame, timestamp, landmarks} dicts

        Args:
            video_path:  Path to input video.
            output_json: Where to write the JSON results.

        Returns:
            List of per-frame pose dicts (same data written to JSON).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use VIDEO mode — requires monotonically increasing timestamps
        landmarker = self._create_landmarker(RunningMode.VIDEO)

        poses = []
        frame_idx = 0
        detected_count = 0

        with tqdm(total=total_frames, desc="Detecting poses") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to MediaPipe Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Timestamp in milliseconds (must be monotonic)
                timestamp_ms = int(frame_idx * 1000.0 / fps) if fps else frame_idx

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                landmarks = None
                if result.pose_landmarks:
                    person = result.pose_landmarks[0]
                    landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in person]
                    detected_count += 1

                poses.append({
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps if fps else 0,
                    "landmarks": landmarks,
                })

                frame_idx += 1
                pbar.update(1)

        cap.release()

        # Write JSON output
        output_data = {
            "video": video_path,
            "fps": fps,
            "frame_count": total_frames,
            "detected_frames": detected_count,
            "landmark_names": self.LANDMARK_NAMES,
            "poses": poses,
        }

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output_data, f)

        return poses
