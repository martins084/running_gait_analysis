"""
test_feature_extraction.py — Unit tests for biomechanical feature calculations.

Validates angle math, stride detection, and symmetry against known inputs.
"""

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
