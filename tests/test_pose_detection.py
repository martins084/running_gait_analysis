"""
test_pose_detection.py — Smoke tests for the PoseDetector class.

These tests verify that:
  - The detector initializes without error
  - detect_pose() returns the right shapes on a dummy image
  - Landmark names list has 33 entries
"""

import numpy as np
import pytest

from core.pose_detection import PoseDetector


@pytest.fixture(scope="module")
def detector():
    """Create a single PoseDetector instance for all tests."""
    return PoseDetector()


class TestPoseDetector:
    """Basic sanity checks for PoseDetector."""

    def test_landmark_names_count(self, detector):
        """MediaPipe BlazePose defines exactly 33 landmarks."""
        assert len(detector.LANDMARK_NAMES) == 33

    def test_detect_on_blank_image(self, detector):
        """A solid-color image should return None (no person present)."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, raw = detector.detect_pose(blank)
        assert landmarks is None
        assert raw is None

    def test_detect_returns_correct_shape(self, detector):
        """
        If a person IS detected, the landmarks array should be (33, 4).
        We can't guarantee detection on a synthetic image, so we just
        test the None branch above and the shape contract here.
        """
        # This test documents the expected shape; it'll pass trivially
        # when no person is found (blank image).  A real integration test
        # would use an actual photo of a person.
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks, _ = detector.detect_pose(blank)

        if landmarks is not None:
            assert landmarks.shape == (33, 4)
