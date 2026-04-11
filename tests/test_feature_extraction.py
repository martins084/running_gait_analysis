"""
test_feature_extraction.py — Unit tests for biomechanical feature calculations.

Validates angle math, stride detection, and symmetry against known inputs.
"""

import json

import numpy as np
import pytest

from core.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor(fps=30)


# ── Joint angle tests ───────────────────────────────────────────────────

class TestJointAngles:
    """Verify the angle-at-vertex calculation with known geometry."""

    def test_right_angle(self, extractor):
        """Three points forming a perfect 90-degree angle."""
        p1 = np.array([0.0, 0.0, 0.0, 1.0])
        vertex = np.array([1.0, 0.0, 0.0, 1.0])
        p3 = np.array([1.0, 1.0, 0.0, 1.0])

        angle = extractor._angle_at(p1, vertex, p3)
        assert abs(angle - 90.0) < 1.0, f"Expected ~90°, got {angle:.1f}°"

    def test_straight_line(self, extractor):
        """Three collinear points → 180 degrees."""
        p1 = np.array([0.0, 0.0, 0.0, 1.0])
        vertex = np.array([1.0, 0.0, 0.0, 1.0])
        p3 = np.array([2.0, 0.0, 0.0, 1.0])

        angle = extractor._angle_at(p1, vertex, p3)
        assert abs(angle - 180.0) < 1.0, f"Expected ~180°, got {angle:.1f}°"

    def test_acute_angle(self, extractor):
        """45-degree angle from known geometry."""
        p1 = np.array([0.0, 0.0, 0.0, 1.0])
        vertex = np.array([0.0, 1.0, 0.0, 1.0])
        p3 = np.array([1.0, 0.0, 0.0, 1.0])

        angle = extractor._angle_at(p1, vertex, p3)
        assert abs(angle - 45.0) < 2.0, f"Expected ~45°, got {angle:.1f}°"


# ── Symmetry tests ──────────────────────────────────────────────────────

class TestSymmetry:
    """Verify the symmetry index with controlled landmarks."""

    def _make_landmarks(self, **overrides) -> np.ndarray:
        """Build a minimal 33x4 landmark array with sensible defaults."""
        lm = np.zeros((33, 4))
        # Default hips centered at x=0.5
        lm[23] = [0.45, 0.5, 0.0, 1.0]  # left hip
        lm[24] = [0.55, 0.5, 0.0, 1.0]  # right hip
        # Default knees + ankles roughly symmetric
        lm[25] = [0.40, 0.7, 0.0, 1.0]  # left knee
        lm[26] = [0.60, 0.7, 0.0, 1.0]  # right knee
        lm[27] = [0.38, 0.9, 0.0, 1.0]  # left ankle
        lm[28] = [0.62, 0.9, 0.0, 1.0]  # right ankle

        for key, val in overrides.items():
            idx = FeatureExtractor.JOINTS[key]
            lm[idx] = val
        return lm

    def test_perfect_symmetry(self, extractor):
        """Mirror-image left/right should give symmetry ≈ 1.0."""
        lm = self._make_landmarks()
        sym = extractor.compute_symmetry(lm)
        assert sym > 0.90, f"Expected high symmetry, got {sym:.3f}"

    def test_asymmetric(self, extractor):
        """Shift left knee far out → symmetry drops."""
        lm = self._make_landmarks(left_knee=np.array([0.1, 0.7, 0.0, 1.0]))
        sym = extractor.compute_symmetry(lm)
        assert sym < 0.80, f"Expected low symmetry, got {sym:.3f}"


# ── Stride metric tests ────────────────────────────────────────────────

class TestStrideMetrics:
    """Verify stride detection on a synthetic oscillating signal."""

    def test_synthetic_stride(self, extractor):
        """
        Create a sine-wave ankle trajectory and check that the detected
        cadence is in the right ballpark.
        """
        fps = 30
        duration = 4  # seconds — enough for several strides
        n_frames = fps * duration

        landmarks_seq = []
        for i in range(n_frames):
            lm = np.zeros((33, 4))
            phase = i / fps

            # Left ankle oscillates vertically (simulating ground contact)
            lm[27] = [
                0.5 + 0.05 * np.sin(2 * np.pi * 1.5 * phase),  # x: slight sway
                0.7 + 0.05 * np.sin(2 * np.pi * 1.5 * phase),  # y: stride bounce
                0.0,
                1.0,
            ]
            landmarks_seq.append(lm)

        metrics = extractor.compute_stride_metrics(landmarks_seq, fps)
        assert metrics, "No stride metrics returned"
        assert metrics["num_strides_detected"] >= 2, "Too few strides detected"
        # Cadence = 120 / stride_time (L→L interval); should be in a sane range
        assert "cadence_steps_per_min" in metrics
        assert metrics["cadence_steps_per_min"] > 50


# ── Temporal smoothing tests ────────────────────────────────────────────

class TestTemporalSmoothing:
    """Validate Savitzky–Golay sequence smoothing behavior."""

    def _make_frame(self, ankle_x: float, ankle_y: float) -> np.ndarray:
        """
        Create one 33x4 landmark frame with deterministic values.
        Only ankle coordinates vary between frames for easy assertions.
        """
        lm = np.zeros((33, 4))
        lm[:, 3] = 1.0  # full visibility by default
        lm[27, 0] = ankle_x
        lm[27, 1] = ankle_y
        return lm

    def test_short_sequence_is_returned_unchanged(self):
        """
        If frame count is below smooth_window, smoothing should be skipped.
        This avoids introducing artifacts on very short clips.
        """
        extractor = FeatureExtractor(fps=30, smooth_window=7)
        seq = [self._make_frame(0.5 + i * 0.01, 0.7) for i in range(5)]

        out = extractor._smooth_sequence(seq)

        assert len(out) == len(seq)
        for i in range(len(seq)):
            assert np.allclose(out[i], seq[i]), "Short sequence should remain unchanged"

    def test_none_frames_are_preserved_in_place(self):
        """
        Frames with missing detections must stay None after smoothing.
        Only valid frames should be smoothed and written back.
        """
        extractor = FeatureExtractor(fps=30, smooth_window=5)
        seq = [
            self._make_frame(0.45, 0.70),
            None,
            self._make_frame(0.50, 0.72),
            self._make_frame(0.55, 0.74),
            None,
            self._make_frame(0.60, 0.76),
            self._make_frame(0.65, 0.78),
        ]

        out = extractor._smooth_sequence(seq)

        assert out[1] is None
        assert out[4] is None
        assert isinstance(out[0], np.ndarray)
        assert isinstance(out[2], np.ndarray)

    def test_smoothing_reduces_high_frequency_noise(self):
        """
        A noisy ankle trajectory should become closer to the clean reference
        after smoothing (lower mean squared error).
        """
        extractor = FeatureExtractor(fps=30, smooth_window=7)
        n = 25
        t = np.arange(n)
        clean = 0.5 + 0.04 * np.sin(2 * np.pi * t / 12.0)
        # Alternating jitter term (Nyquist-frequency-like) to emulate frame jitter.
        noise = 0.015 * np.where((t % 2) == 0, 1.0, -1.0)
        noisy = clean + noise

        seq = [self._make_frame(float(noisy[i]), 0.7) for i in range(n)]
        out = extractor._smooth_sequence(seq)
        smoothed = np.array([frame[27, 0] for frame in out])

        mse_noisy = float(np.mean((noisy - clean) ** 2))
        mse_smooth = float(np.mean((smoothed - clean) ** 2))

        assert mse_smooth < mse_noisy, "Smoothing should reduce trajectory noise"


# ── Full extract alignment (video frame index) ───────────────────────────


def _minimal_landmarks_list() -> list:
    """33×4 BlazePose-like list for tests (enough for angles / COM)."""
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    lm[0] = [0.5, 0.1, 0.0, 1.0]
    lm[7] = lm[8] = [0.5, 0.12, 0.0, 1.0]
    lm[11] = [0.45, 0.35, 0.0, 1.0]
    lm[12] = [0.55, 0.35, 0.0, 1.0]
    lm[23] = [0.45, 0.5, 0.0, 1.0]
    lm[24] = [0.55, 0.5, 0.0, 1.0]
    lm[25] = [0.40, 0.7, 0.0, 1.0]
    lm[26] = [0.60, 0.7, 0.0, 1.0]
    lm[27] = [0.38, 0.9, 0.0, 1.0]
    lm[28] = [0.62, 0.9, 0.0, 1.0]
    lm[29] = [0.37, 0.92, 0.0, 1.0]
    lm[30] = [0.63, 0.92, 0.0, 1.0]
    lm[31] = [0.36, 0.95, 0.0, 1.0]
    lm[32] = [0.64, 0.95, 0.0, 1.0]
    lm[13] = [0.42, 0.45, 0.0, 1.0]
    lm[14] = [0.58, 0.45, 0.0, 1.0]
    lm[15] = [0.40, 0.55, 0.0, 1.0]
    lm[16] = [0.60, 0.55, 0.0, 1.0]
    lm[19] = [0.39, 0.58, 0.0, 1.0]
    lm[20] = [0.61, 0.58, 0.0, 1.0]
    return lm.tolist()


def test_extract_all_joint_angles_aligned_to_video_frames(tmp_path):
    """Each pose JSON frame must map to one joint_angles / symmetry slot (null if no pose)."""
    extractor = FeatureExtractor(fps=30, smooth_window=7)
    poses = [
        {"frame": 0, "timestamp": 0.0, "landmarks": _minimal_landmarks_list()},
        {"frame": 1, "timestamp": 1 / 30, "landmarks": None},
        {"frame": 2, "timestamp": 2 / 30, "landmarks": _minimal_landmarks_list()},
    ]
    p = tmp_path / "poses.json"
    p.write_text(json.dumps({"fps": 30, "poses": poses}), encoding="utf-8")

    out = extractor.extract_all_features(str(p))

    assert out["video_frame_count"] == 3
    assert out["frame_count"] == 3
    assert len(out["joint_angles"]) == 3
    assert len(out["symmetry"]) == 3
    assert out["joint_angles"][1] is None
    assert out["symmetry"][1] is None
    assert isinstance(out["joint_angles"][0], dict)
    assert isinstance(out["joint_angles"][2], dict)
