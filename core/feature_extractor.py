"""
feature_extractor.py — Biomechanical feature computation from pose data.

Takes raw landmark sequences (from pose_detection.py) and computes:
  - Joint angles (hip, knee, ankle)
  - Stride / temporal metrics (stride length, cadence, ground contact time)
  - Left–right symmetry index
  - Vertical oscillation

Temporal smoothing (Savitzky–Golay filter) is applied to landmark
trajectories before any feature calculation — this is critical for
reducing MediaPipe jitter.
"""

import json
import numpy as np
from scipy import signal
from typing import Optional


class FeatureExtractor:
    """
    Computes biomechanical running features from a sequence of
    MediaPipe BlazePose landmarks.
    """

    # ── MediaPipe joint indices ──────────────────────────────────────────
    JOINTS = {
        "nose": 0,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
        "left_heel": 29,
        "right_heel": 30,
        "left_foot_index": 31,
        "right_foot_index": 32,
    }

    def __init__(self, fps: int = 30, smooth_window: int = 7):
        """
        Args:
            fps: Frames per second of the source video.
            smooth_window: Window size for the Savitzky–Golay filter.
                           Must be odd and >= 3. Larger = smoother.
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.smooth_window = smooth_window

    # =====================================================================
    # Public API — high-level
    # =====================================================================

    def extract_all_features(self, poses_json: str) -> dict:
        """
        One-shot: load a pose JSON file and compute every feature.

        Returns a dict with keys:
          stride_metrics, joint_angles, symmetry, vertical_oscillation
        """
        with open(poses_json) as f:
            data = json.load(f)

        fps = data.get("fps", self.fps)
        raw_landmarks = [
            np.array(p["landmarks"]) if p["landmarks"] is not None else None
            for p in data["poses"]
        ]

        # Apply temporal smoothing across the whole sequence
        smoothed = self._smooth_sequence(raw_landmarks)

        # Compute each feature group
        stride = self.compute_stride_metrics(smoothed, fps)
        angles = [
            self.compute_joint_angles(lm) for lm in smoothed if lm is not None
        ]
        symmetry = [
            self.compute_symmetry(lm) for lm in smoothed if lm is not None
        ]
        vert_osc = self._compute_vertical_oscillation(smoothed)

        return {
            "stride_metrics": stride,
            "joint_angles": angles,
            "symmetry": symmetry,
            "vertical_oscillation_px": vert_osc,
        }

    # =====================================================================
    # Joint angles
    # =====================================================================

    def compute_joint_angles(self, landmarks: np.ndarray) -> dict:
        """
        Compute hip and knee angles for both sides from a single frame.

        Returns dict: left_hip, right_hip, left_knee, right_knee (degrees).
        """
        j = self.JOINTS

        # Left side
        left_hip_angle = self._angle_at(
            landmarks[j["left_shoulder"]],
            landmarks[j["left_hip"]],
            landmarks[j["left_knee"]],
        )
        left_knee_angle = self._angle_at(
            landmarks[j["left_hip"]],
            landmarks[j["left_knee"]],
            landmarks[j["left_ankle"]],
        )

        # Right side
        right_hip_angle = self._angle_at(
            landmarks[j["right_shoulder"]],
            landmarks[j["right_hip"]],
            landmarks[j["right_knee"]],
        )
        right_knee_angle = self._angle_at(
            landmarks[j["right_hip"]],
            landmarks[j["right_knee"]],
            landmarks[j["right_ankle"]],
        )

        return {
            "left_hip": round(left_hip_angle, 2),
            "right_hip": round(right_hip_angle, 2),
            "left_knee": round(left_knee_angle, 2),
            "right_knee": round(right_knee_angle, 2),
        }

    # =====================================================================
    # Stride / temporal metrics
    # =====================================================================

    def compute_stride_metrics(self, landmarks_seq: list, fps: int) -> dict:
        """
        Compute stride length (in pixel-space), stride time, and cadence.

        Uses the left ankle's vertical position to find ground-contact
        frames (local maxima of y, since y increases downward in image
        coordinates).
        """
        # Collect left-ankle y-positions (ground contact = high y)
        ankle_y = []
        ankle_x = []
        for lm in landmarks_seq:
            if lm is not None:
                ankle_y.append(lm[self.JOINTS["left_ankle"]][1])
                ankle_x.append(lm[self.JOINTS["left_ankle"]][0])
            else:
                ankle_y.append(np.nan)
                ankle_x.append(np.nan)

        ankle_y = np.array(ankle_y)
        ankle_x = np.array(ankle_x)

        # Interpolate NaNs so peak-finding works on a continuous signal
        valid = ~np.isnan(ankle_y)
        if valid.sum() < 10:
            return {}

        ankle_y_clean = np.interp(
            np.arange(len(ankle_y)),
            np.where(valid)[0],
            ankle_y[valid],
        )
        ankle_x_clean = np.interp(
            np.arange(len(ankle_x)),
            np.where(valid)[0],
            ankle_x[valid],
        )

        # Find ground-contact peaks (high y = foot low in image)
        # Minimum distance between peaks is roughly half a stride (~0.3 s)
        min_dist = max(1, int(fps * 0.3))
        peaks, _ = signal.find_peaks(ankle_y_clean, distance=min_dist)

        if len(peaks) < 2:
            return {}

        # Stride length: horizontal distance between consecutive contacts
        stride_lengths = [
            abs(ankle_x_clean[peaks[i + 1]] - ankle_x_clean[peaks[i]])
            for i in range(len(peaks) - 1)
        ]

        # Stride time: time between consecutive contacts
        stride_times = [
            (peaks[i + 1] - peaks[i]) / fps for i in range(len(peaks) - 1)
        ]

        mean_stride_time = float(np.mean(stride_times)) if stride_times else 0

        return {
            "stride_length_px": round(float(np.mean(stride_lengths)), 4),
            "stride_time_sec": round(mean_stride_time, 4),
            "cadence_steps_per_min": round(60.0 / mean_stride_time, 1) if mean_stride_time > 0 else 0,
            "num_strides_detected": len(peaks),
        }

    # =====================================================================
    # Symmetry
    # =====================================================================

    def compute_symmetry(self, landmarks: np.ndarray) -> float:
        """
        Compute a left–right symmetry index (0 = asymmetric, 1 = perfect).

        Compares the distance of left vs right knee and ankle from the
        midline (average of left and right hip x).
        """
        j = self.JOINTS
        midline_x = (landmarks[j["left_hip"]][0] + landmarks[j["right_hip"]][0]) / 2.0

        left_dev = (
            abs(landmarks[j["left_knee"]][0] - midline_x)
            + abs(landmarks[j["left_ankle"]][0] - midline_x)
        )
        right_dev = (
            abs(landmarks[j["right_knee"]][0] - midline_x)
            + abs(landmarks[j["right_ankle"]][0] - midline_x)
        )

        total = left_dev + right_dev
        if total < 1e-8:
            return 1.0

        symmetry = 1.0 - abs(left_dev - right_dev) / total
        return round(float(symmetry), 4)

    # =====================================================================
    # Vertical oscillation
    # =====================================================================

    def _compute_vertical_oscillation(self, landmarks_seq: list) -> Optional[float]:
        """
        Vertical oscillation = range of the midpoint between the two hips.

        This approximates center-of-mass bounce during running.
        """
        j = self.JOINTS
        hip_y = []

        for lm in landmarks_seq:
            if lm is not None:
                mid_y = (lm[j["left_hip"]][1] + lm[j["right_hip"]][1]) / 2.0
                hip_y.append(mid_y)

        if len(hip_y) < 10:
            return None

        hip_y = np.array(hip_y)

        # Peak-to-trough range (use interquartile to ignore outliers)
        q95 = np.percentile(hip_y, 95)
        q05 = np.percentile(hip_y, 5)

        return round(float(q95 - q05), 4)

    # =====================================================================
    # Internals — math helpers
    # =====================================================================

    @staticmethod
    def _angle_at(p1: np.ndarray, vertex: np.ndarray, p3: np.ndarray) -> float:
        """
        Angle (in degrees) at `vertex` formed by vectors vertex→p1
        and vertex→p3.  Uses only x and y coordinates.
        """
        v1 = p1[:2] - vertex[:2]
        v2 = p3[:2] - vertex[:2]

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    def _smooth_sequence(self, landmarks_seq: list) -> list:
        """
        Apply Savitzky–Golay filter along the time axis to every
        coordinate of every joint.  Frames where landmarks are None
        are left as None.
        """
        n = len(landmarks_seq)
        if n < self.smooth_window:
            return landmarks_seq

        # Stack valid frames into a 3-D array: (time, 33, cols)
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if len(valid_indices) < self.smooth_window:
            return landmarks_seq

        stacked = np.array([landmarks_seq[i] for i in valid_indices])  # (T, 33, C)
        num_joints, num_coords = stacked.shape[1], stacked.shape[2]

        # Smooth each joint / coordinate independently along time
        for j in range(num_joints):
            for c in range(num_coords):
                stacked[:, j, c] = signal.savgol_filter(
                    stacked[:, j, c],
                    window_length=self.smooth_window,
                    polyorder=2,
                )

        # Write back into the original list
        result = list(landmarks_seq)  # shallow copy
        for idx_out, idx_in in enumerate(valid_indices):
            result[idx_in] = stacked[idx_out]

        return result
