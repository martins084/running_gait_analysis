"""
com_segmentation.py — Segment-weighted whole-body COM in 2D normalized image space.

Shared by ``feature_extractor`` (per-frame series for the API) and ``visualization``
(annotated video). Logic mirrors the biomechanics literature (mass fractions +
COM along segment chords); see module docstring in the thesis / API docs.
"""

from __future__ import annotations

import numpy as np

# Mass fractions of total body mass (Plagenhoef / Winter–style anthropometry).
_SEGMENT_MASS = {
    "head": 8.1,
    "trunk": 46.7,
    "l_upper_arm": 2.8,
    "r_upper_arm": 2.8,
    "l_forearm": 1.6,
    "r_forearm": 1.6,
    "l_hand": 0.6,
    "r_hand": 0.6,
    "l_thigh": 10.0,
    "r_thigh": 10.0,
    "l_shank": 4.65,
    "r_shank": 4.65,
    "l_foot": 1.43,
    "r_foot": 1.43,
}
_FRAC_TRUNK = 0.43
_FRAC_UPPER_ARM = 0.436
_FRAC_FOREARM = 0.430
_FRAC_HAND = 0.50
_FRAC_THIGH = 0.433
_FRAC_SHANK = 0.433
_FRAC_FOOT = 0.50
_VIS_FLOOR = 0.18


def _vis(lm: np.ndarray, i: int) -> float:
    row = lm[i]
    if len(row) > 3:
        return float(max(0.0, min(1.0, row[3])))
    return 1.0


def _seg_weight(lm: np.ndarray, *idx: int) -> float:
    v = min(_vis(lm, i) for i in idx)
    return max(0.0, v)


def _com_on_segment(lm: np.ndarray, a: int, b: int, frac: float) -> np.ndarray:
    pa = lm[a][:2].astype(np.float64)
    pb = lm[b][:2].astype(np.float64)
    return pa + frac * (pb - pa)


def compute_segment_weighted_com_xy(landmarks: list | np.ndarray) -> tuple[float, float] | None:
    """
    Approximate whole-body COM in **normalized image coordinates** (same as landmarks).

    Returns ``(x, y)`` in [0, 1] normalized space, or ``None`` if nothing reliable remains.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[0] < 33:
        return None

    ear_mid = (lm[7][:2] + lm[8][:2]) / 2.0
    head_com = (lm[0][:2] + ear_mid) / 2.0
    w_head = _seg_weight(lm, 0, 7, 8) * _SEGMENT_MASS["head"]

    hip_mid = (lm[23][:2] + lm[24][:2]) / 2.0
    shoulder_mid = (lm[11][:2] + lm[12][:2]) / 2.0
    trunk_vec = shoulder_mid - hip_mid
    trunk_com = hip_mid + _FRAC_TRUNK * trunk_vec
    w_trunk = _seg_weight(lm, 11, 12, 23, 24) * _SEGMENT_MASS["trunk"]

    segments: list[tuple[float, np.ndarray]] = [
        (w_head, head_com),
        (w_trunk, trunk_com),
        (_seg_weight(lm, 11, 13) * _SEGMENT_MASS["l_upper_arm"], _com_on_segment(lm, 11, 13, _FRAC_UPPER_ARM)),
        (_seg_weight(lm, 12, 14) * _SEGMENT_MASS["r_upper_arm"], _com_on_segment(lm, 12, 14, _FRAC_UPPER_ARM)),
        (_seg_weight(lm, 13, 15) * _SEGMENT_MASS["l_forearm"], _com_on_segment(lm, 13, 15, _FRAC_FOREARM)),
        (_seg_weight(lm, 14, 16) * _SEGMENT_MASS["r_forearm"], _com_on_segment(lm, 14, 16, _FRAC_FOREARM)),
        (_seg_weight(lm, 15, 19) * _SEGMENT_MASS["l_hand"], _com_on_segment(lm, 15, 19, _FRAC_HAND)),
        (_seg_weight(lm, 16, 20) * _SEGMENT_MASS["r_hand"], _com_on_segment(lm, 16, 20, _FRAC_HAND)),
        (_seg_weight(lm, 23, 25) * _SEGMENT_MASS["l_thigh"], _com_on_segment(lm, 23, 25, _FRAC_THIGH)),
        (_seg_weight(lm, 24, 26) * _SEGMENT_MASS["r_thigh"], _com_on_segment(lm, 24, 26, _FRAC_THIGH)),
        (_seg_weight(lm, 25, 27) * _SEGMENT_MASS["l_shank"], _com_on_segment(lm, 25, 27, _FRAC_SHANK)),
        (_seg_weight(lm, 26, 28) * _SEGMENT_MASS["r_shank"], _com_on_segment(lm, 26, 28, _FRAC_SHANK)),
        (_seg_weight(lm, 27, 31) * _SEGMENT_MASS["l_foot"], _com_on_segment(lm, 27, 31, _FRAC_FOOT)),
        (_seg_weight(lm, 28, 32) * _SEGMENT_MASS["r_foot"], _com_on_segment(lm, 28, 32, _FRAC_FOOT)),
    ]

    weighted: list[tuple[float, np.ndarray]] = []
    for mass, pt in segments:
        if mass < _VIS_FLOOR:
            continue
        weighted.append((mass, pt))

    if not weighted:
        return None

    total_w = sum(m for m, _ in weighted)
    if total_w < 1e-8:
        return None

    com = sum(m * p for m, p in weighted) / total_w
    x, y = float(com[0]), float(com[1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return (x, y)
