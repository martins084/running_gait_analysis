"""
pose_foot_sanitize.py — Post-process pose sequences to reduce foot mis-attachments.

2D pose models sometimes lock ankles/feet onto similarly colored background
objects (e.g. a red flag vs. a red shoe), especially with low resolution,
motion blur, and leg occlusion.

This module applies **kinematic plausibility** constraints:
  - Hip–knee–ankle segment lengths must stay within robust per-video limits.
  - Heel / foot-index points are clamped to stay near the corrected ankle.
  - A light temporal blend reduces single-frame spikes.

Call this on the full `poses` list **before** writing JSON / drawing overlays.
"""

from __future__ import annotations

import numpy as np

# MediaPipe BlazePose indices (must match pose_detection.LANDMARK_NAMES order)
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32


def sanitize_pose_sequence(poses: list[dict]) -> list[dict]:
    """
    Mutates each pose dict's `landmarks` list in place (if present) and
    returns the same `poses` list for chaining.
    """
    if not poses:
        return poses

    # Work with numpy arrays; convert back to nested lists for JSON
    arrays: list[np.ndarray | None] = []
    for p in poses:
        lm = p.get("landmarks")
        if lm is None:
            arrays.append(None)
        else:
            arrays.append(np.array(lm, dtype=float))

    med_thigh_L, med_shank_L = _robust_leg_medians(arrays, "L")
    med_thigh_R, med_shank_R = _robust_leg_medians(arrays, "R")

    prev_L = prev_R = None

    for i, arr in enumerate(arrays):
        if arr is None:
            continue

        arr = arr.copy()

        # --- Left leg ---
        arr = _fix_leg_chain(
            arr,
            L_HIP,
            L_KNEE,
            L_ANKLE,
            med_thigh_L,
            med_shank_L,
            prev_ankle_xy=prev_L,
        )
        arr = _clamp_foot_points(arr, L_ANKLE, L_HEEL, L_TOE)

        # --- Right leg ---
        arr = _fix_leg_chain(
            arr,
            R_HIP,
            R_KNEE,
            R_ANKLE,
            med_thigh_R,
            med_shank_R,
            prev_ankle_xy=prev_R,
        )
        arr = _clamp_foot_points(arr, R_ANKLE, R_HEEL, R_TOE)

        # Keep left/right ankle identity stable across occlusions:
        # if current ankles better match the opposite previous tracks, swap them.
        arr = _enforce_ankle_identity_continuity(
            arr,
            prev_left=prev_L,
            prev_right=prev_R,
        )

        prev_L = arr[L_ANKLE, :2].copy()
        prev_R = arr[R_ANKLE, :2].copy()

        arrays[i] = arr
        poses[i]["landmarks"] = arr.tolist()

    return poses


def _robust_leg_medians(
    arrays: list[np.ndarray | None],
    side: str,
) -> tuple[float, float]:
    """Median thigh and shank lengths (x,y) for side L/R from valid frames."""
    hip_i, knee_i, ankle_i = (
        (L_HIP, L_KNEE, L_ANKLE) if side == "L" else (R_HIP, R_KNEE, R_ANKLE)
    )
    thighs, shanks = [], []
    for arr in arrays:
        if arr is None:
            continue
        if arr[ankle_i, 3] < 0.35:  # visibility
            continue
        t = _dist2(arr[hip_i], arr[knee_i])
        s = _dist2(arr[knee_i], arr[ankle_i])
        if t > 1e-6 and s > 1e-6:
            thighs.append(t)
            shanks.append(s)
    if len(thighs) < 5:
        # Fallback: typical normalized lengths for upright runner in frame
        return 0.12, 0.12
    mt = float(np.median(thighs))
    ms = float(np.median(shanks))
    return mt, ms


def _fix_leg_chain(
    arr: np.ndarray,
    hip_i: int,
    knee_i: int,
    ankle_i: int,
    med_thigh: float,
    med_shank: float,
    prev_ankle_xy: np.ndarray | None,
) -> np.ndarray:
    """Clamp implausible ankle positions; blend with previous frame if needed."""
    hip = arr[hip_i]
    knee = arr[knee_i]
    ankle = arr[ankle_i]

    # Absolute caps (normalized image coords) — catches “foot on flag” spikes
    max_thigh = min(0.28, max(0.10, 2.4 * med_thigh))
    max_shank = min(0.22, max(0.08, 2.4 * med_shank))

    # 1) Thigh: hip → knee
    knee[:2] = _clamp_segment(hip[:2], knee[:2], max_thigh)
    arr[knee_i] = knee

    # 2) Shank: knee → ankle (pulls foot back from background mis-detects)
    knee = arr[knee_i]
    ankle[:2] = _clamp_segment(knee[:2], ankle[:2], max_shank)

    # 3) Temporal blend if ankle still teleports vs previous frame
    if prev_ankle_xy is not None:
        jump = float(np.linalg.norm(ankle[:2] - prev_ankle_xy))
        if jump > 0.10:
            ankle[:2] = 0.65 * ankle[:2] + 0.35 * prev_ankle_xy

    arr[ankle_i] = ankle
    return arr


def _clamp_foot_points(
    arr: np.ndarray,
    ankle_i: int,
    heel_i: int,
    toe_i: int,
) -> np.ndarray:
    """Keep heel/toe near ankle (they should not fly to background blobs)."""
    ankle = arr[ankle_i]
    max_foot = 0.10
    arr[heel_i, :2] = _clamp_segment(ankle[:2], arr[heel_i, :2], max_foot)
    arr[toe_i, :2] = _clamp_segment(ankle[:2], arr[toe_i, :2], max_foot)
    return arr


def _clamp_segment(p0: np.ndarray, p1: np.ndarray, max_len: float) -> np.ndarray:
    """Clamp p1 so ||p1 - p0|| <= max_len."""
    v = p1 - p0
    n = float(np.linalg.norm(v))
    if n <= max_len or n < 1e-8:
        return p1
    return p0 + v * (max_len / n)


def _enforce_ankle_identity_continuity(
    arr: np.ndarray,
    prev_left: np.ndarray | None,
    prev_right: np.ndarray | None,
) -> np.ndarray:
    """
    Prevent left/right ankle ID swaps during leg crossing.

    The rule compares two assignments:
      - keep as-is:    L->prev_left + R->prev_right
      - swap ankles:   L->prev_right + R->prev_left
    If the swapped assignment is substantially more motion-consistent,
    we swap ankle + heel + toe landmarks together.
    """
    if prev_left is None or prev_right is None:
        return arr

    left_vis = float(arr[L_ANKLE, 3])
    right_vis = float(arr[R_ANKLE, 3])
    if left_vis < 0.35 or right_vis < 0.35:
        return arr

    curr_left = arr[L_ANKLE, :2].copy()
    curr_right = arr[R_ANKLE, :2].copy()

    keep_cost = float(np.linalg.norm(curr_left - prev_left) + np.linalg.norm(curr_right - prev_right))
    swap_cost = float(np.linalg.norm(curr_left - prev_right) + np.linalg.norm(curr_right - prev_left))

    # Require a clear margin so normal stride crossing does not trigger swaps.
    if swap_cost + 0.03 < keep_cost:
        left_block = arr[[L_ANKLE, L_HEEL, L_TOE], :].copy()
        right_block = arr[[R_ANKLE, R_HEEL, R_TOE], :].copy()
        arr[[L_ANKLE, L_HEEL, L_TOE], :] = right_block
        arr[[R_ANKLE, R_HEEL, R_TOE], :] = left_block

    return arr


def _dist2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))
