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
        Stride length, stride period, and cadence from ankle motion.

        Ground contact is approximated by **local maxima of ankle y**
        (image coords: y grows downward, so high y = foot low / near ground).

        **Cadence (steps/min)** — One *stride* = left foot strike to next left
        foot strike; that interval contains **two steps** (L then R). So:

            cadence = 120 / stride_time_sec

        (Using 60/stride_time was half the true step rate — common bug.)

        We also merge L+R ankle peaks to estimate cadence independently and
        report **stride_length** normalized by hip width (scale-free across zoom).
        """
        j = self.JOINTS

        def _series(idx: int) -> tuple[np.ndarray, np.ndarray]:
            ys, xs = [], []
            for lm in landmarks_seq:
                if lm is not None:
                    ys.append(lm[idx][1])
                    xs.append(lm[idx][0])
                else:
                    ys.append(np.nan)
                    xs.append(np.nan)
            return np.array(ys), np.array(xs)

        def _interp(y_arr: np.ndarray) -> np.ndarray | None:
            valid = ~np.isnan(y_arr)
            if valid.sum() < 10:
                return None
            return np.interp(
                np.arange(len(y_arr)),
                np.where(valid)[0],
                y_arr[valid],
            )

        left_y, left_x = _series(j["left_ankle"])
        right_y, right_x = _series(j["right_ankle"])

        left_y_clean = _interp(left_y)
        right_y_clean = _interp(right_y)
        if left_y_clean is None:
            return {}

        valid = ~np.isnan(left_y)
        left_x_clean = np.interp(
            np.arange(len(left_x)),
            np.where(valid)[0],
            left_x[valid],
        )

        # --- Same-foot (left) peaks: stride period = L → next L ---
        min_dist = max(1, int(fps * 0.28))
        peaks_left, _ = signal.find_peaks(left_y_clean, distance=min_dist)

        out: dict = {}

        if len(peaks_left) >= 2:
            stride_lengths = [
                abs(left_x_clean[peaks_left[i + 1]] - left_x_clean[peaks_left[i]])
                for i in range(len(peaks_left) - 1)
            ]
            stride_times = np.diff(peaks_left) / fps
            mean_stride_time = float(np.mean(stride_times))
            out["stride_length_px"] = round(float(np.mean(stride_lengths)), 4)
            out["stride_time_sec"] = round(mean_stride_time, 4)
            # Two steps per full stride (L–R–L)
            out["cadence_steps_per_min"] = (
                round(120.0 / mean_stride_time, 1) if mean_stride_time > 0 else 0.0
            )
            out["num_same_foot_contacts_left"] = int(len(peaks_left))

        # Hip width (median) for normalized stride — robust to single-frame noise
        hip_w = []
        for lm in landmarks_seq:
            if lm is not None:
                hip_w.append(abs(lm[j["left_hip"]][0] - lm[j["right_hip"]][0]))
        if hip_w:
            w = float(np.median(hip_w))
            if w > 1e-8 and "stride_length_px" in out:
                out["stride_length_over_hip_width"] = round(out["stride_length_px"] / w, 4)

        # --- Merged L+R foot-strike times (dedupe double-hits) ---
        peaks_right = np.array([], dtype=int)
        if right_y_clean is not None:
            pr, _ = signal.find_peaks(right_y_clean, distance=min_dist)
            peaks_right = pr

        events = sorted(
            [(int(i), "L") for i in peaks_left] + [(int(i), "R") for i in peaks_right]
        )
        merged_frames = self._merge_close_events(events, min_sep_frames=max(1, int(fps * 0.12)))

        if len(merged_frames) >= 2:
            dt = np.diff(merged_frames) / fps
            mean_step = float(np.mean(dt))
            out["cadence_steps_per_min_merged"] = (
                round(60.0 / mean_step, 1) if mean_step > 0 else 0.0
            )
            out["num_foot_strikes_merged"] = int(len(merged_frames))

        # Legacy-compatible key: total left peaks (informative for debugging)
        if peaks_left.size:
            out["num_strides_detected"] = int(len(peaks_left))

        return out if out else {}

    @staticmethod
    def _merge_close_events(
        events: list[tuple[int, str]],
        min_sep_frames: int,
    ) -> list[int]:
        """
        Sort by frame index; drop events closer than min_sep_frames to the
        previous kept event (avoids duplicate L+R peaks same instant).
        """
        if not events:
            return []
        events = sorted(events, key=lambda e: e[0])
        kept = [events[0][0]]
        for frame_idx, _side in events[1:]:
            if frame_idx - kept[-1] >= min_sep_frames:
                kept.append(frame_idx)
        return kept

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
