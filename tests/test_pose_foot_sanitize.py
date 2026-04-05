"""Sanitizer pulls implausible ankles (e.g. background flags) back toward the knee."""

import numpy as np

from utils.pose_foot_sanitize import sanitize_pose_sequence

# BlazePose indices (match utils.pose_foot_sanitize)
L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
R_HIP, R_KNEE, R_ANKLE = 24, 26, 28


def _synthetic_pose_with_flag_foot():
    """Left ankle parked far left; legs otherwise coherent."""
    lm = [[0.0, 0.0, 0.0, 0.85] for _ in range(33)]
    lm[L_HIP] = [0.48, 0.48, 0.0, 0.9]
    lm[L_KNEE] = [0.50, 0.58, 0.0, 0.9]
    lm[L_ANKLE] = [0.02, 0.58, 0.0, 0.9]  # ~0.48 m away — "flag"
    lm[R_HIP] = [0.55, 0.48, 0.0, 0.9]
    lm[R_KNEE] = [0.55, 0.58, 0.0, 0.9]
    lm[R_ANKLE] = [0.55, 0.68, 0.0, 0.9]
    return lm


def test_extreme_lateral_ankle_is_clamped_toward_knee():
    poses = [
        {
            "frame": i,
            "timestamp": i / 30.0,
            "landmarks": _synthetic_pose_with_flag_foot(),
        }
        for i in range(8)
    ]
    knee_before = np.array(poses[0]["landmarks"][L_KNEE][:2])
    ankle_before = np.array(poses[0]["landmarks"][L_ANKLE][:2])
    assert np.linalg.norm(ankle_before - knee_before) > 0.35

    sanitize_pose_sequence(poses)

    knee_after = np.array(poses[0]["landmarks"][L_KNEE][:2])
    ankle_after = np.array(poses[0]["landmarks"][L_ANKLE][:2])
    shank_len = float(np.linalg.norm(ankle_after - knee_after))
    # Absolute cap in sanitizer is max 0.22 normalized units for shank
    assert shank_len <= 0.23


def test_ankle_identity_continuity_avoids_left_right_swap():
    # Frame 0 establishes stable tracks.
    base = [[0.0, 0.0, 0.0, 0.9] for _ in range(33)]
    base[L_ANKLE] = [0.40, 0.70, 0.0, 0.95]
    base[R_ANKLE] = [0.60, 0.70, 0.0, 0.95]
    base[L_KNEE] = [0.43, 0.58, 0.0, 0.95]
    base[R_KNEE] = [0.57, 0.58, 0.0, 0.95]
    base[L_HIP] = [0.45, 0.48, 0.0, 0.95]
    base[R_HIP] = [0.55, 0.48, 0.0, 0.95]

    # Frame 1 mimics an ID swap from the detector.
    swapped = [pt[:] for pt in base]
    swapped[L_ANKLE] = [0.61, 0.70, 0.0, 0.95]
    swapped[R_ANKLE] = [0.39, 0.70, 0.0, 0.95]

    poses = [
        {"frame": 0, "timestamp": 0.0, "landmarks": base},
        {"frame": 1, "timestamp": 1.0 / 30.0, "landmarks": swapped},
    ]

    sanitize_pose_sequence(poses)

    left_after = np.array(poses[1]["landmarks"][L_ANKLE][:2])
    right_after = np.array(poses[1]["landmarks"][R_ANKLE][:2])

    # The continuity check should restore identity consistency with frame 0.
    assert left_after[0] < right_after[0]
