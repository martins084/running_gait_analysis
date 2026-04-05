"""Unit tests for pose overlay drawing (utils/visualization.py)."""

from __future__ import annotations

import numpy as np

from utils import visualization as vis


def _landmarks_all_visible() -> np.ndarray:
    """33x4 grid of points in a simple standing pose (normalized coords)."""
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    # Head / hands — used by segment-weighted COM (nose, ears, index fingers).
    lm[0] = [0.5, 0.14, 0.0, 1.0]
    lm[7] = [0.38, 0.16, 0.0, 1.0]
    lm[8] = [0.62, 0.16, 0.0, 1.0]
    lm[19] = [0.34, 0.54, 0.0, 1.0]
    lm[20] = [0.66, 0.54, 0.0, 1.0]
    lm[11] = [0.4, 0.25, 0.0, 1.0]
    lm[12] = [0.6, 0.25, 0.0, 1.0]
    lm[13] = [0.38, 0.4, 0.0, 1.0]
    lm[14] = [0.62, 0.4, 0.0, 1.0]
    lm[15] = [0.36, 0.52, 0.0, 1.0]
    lm[16] = [0.64, 0.52, 0.0, 1.0]
    lm[23] = [0.45, 0.55, 0.0, 1.0]
    lm[24] = [0.55, 0.55, 0.0, 1.0]
    lm[25] = [0.44, 0.72, 0.0, 1.0]
    lm[26] = [0.56, 0.72, 0.0, 1.0]
    lm[27] = [0.43, 0.88, 0.0, 1.0]
    lm[28] = [0.57, 0.88, 0.0, 1.0]
    lm[29] = [0.42, 0.92, 0.0, 1.0]
    lm[30] = [0.58, 0.92, 0.0, 1.0]
    lm[31] = [0.44, 0.95, 0.0, 1.0]
    lm[32] = [0.56, 0.95, 0.0, 1.0]
    return lm


def test_draw_pose_adds_green_and_bone_pixels():
    """Skeleton uses green joints (high visibility) and green-tinted bones."""
    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    before = frame.copy()
    vis.draw_pose(frame, _landmarks_all_visible())
    diff = cv2.absdiff(frame, before)
    assert diff.sum() > 1000, "Expected visible drawing on frame"

    # BGR: bone color (0,255,100) → strong G channel
    g = frame[:, :, 1]
    assert g.max() > 80, "Expected green channel activity from bones/joints"


def test_draw_pose_low_visibility_uses_red_joint():
    import cv2

    lm = _landmarks_all_visible()
    lm[16, 3] = 0.2  # right wrist low visibility → red in BGR
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vis.draw_pose(frame, lm)
    # Red joints use high B channel in BGR
    assert frame[:, :, 2].max() > 100


def test_skeleton_connections_cover_expected_edges():
    """Every connection pair references valid landmark indices 0..32."""
    for i, j in vis.SKELETON_CONNECTIONS:
        assert 0 <= i < 33 and 0 <= j < 33


def test_visibility_strictly_above_half_is_green_joint():
    """
    draw_pose uses visibility > 0.5 → green (BGR high G); <= 0.5 → red (BGR high R).
    Isolate landmark 0 (nose) at frame center; others off-screen with low vis.
    """
    lm = np.zeros((33, 4))
    lm[:, 0] = 0.01
    lm[:, 1] = 0.01
    lm[:, 3] = 0.1
    lm[0] = [0.5, 0.5, 0.0, 0.51]  # just above threshold
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vis.draw_pose(frame, lm)
    # Normalized (0.5, 0.5) → pixel (320, 240) on 640×480
    y, x = 240, 320
    b, g, r = frame[y, x]
    assert g > r + 50, "Expected green-dominant joint for visibility > 0.5"


def test_visibility_at_or_below_half_is_red_joint():
    """At 0.5 exactly, code uses `> 0.5` so 0.5 maps to low-visibility (red)."""
    lm = np.zeros((33, 4))
    lm[:, 0] = 0.01
    lm[:, 1] = 0.01
    lm[:, 3] = 0.1
    lm[0] = [0.5, 0.5, 0.0, 0.5]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    vis.draw_pose(frame, lm)
    y, x = 240, 320
    b, g, r = frame[y, x]
    assert r >= g, "Expected red joint for visibility <= 0.5"


def test_draw_pose_accepts_list_of_lists_for_landmarks():
    """Landmarks may be list-of-lists (from JSON); drawing should not raise."""
    lm = _landmarks_all_visible()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    vis.draw_pose(frame, lm.tolist())
    assert frame.sum() > 0


def test_compute_segment_weighted_com_symmetric_pose_near_midline():
    """Left–right symmetric standing pose → approximate COM x ≈ image midline."""
    lm = _landmarks_all_visible()
    com = vis.compute_segment_weighted_com_xy(lm)
    assert com is not None
    assert abs(com[0] - 0.5) < 0.12


def test_draw_pose_returns_com_coordinates():
    lm = _landmarks_all_visible()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vis.draw_pose(frame, lm)
    assert out is not None
    assert len(out) == 2
