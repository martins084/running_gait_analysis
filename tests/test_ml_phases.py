"""Tests for geometry-based gait phase estimates (core/ml_phases.py)."""

from __future__ import annotations

import numpy as np
import pytest

from core.ml_phases import estimate_ml_phases_for_sequence


def _lm(left_y: float, right_y: float) -> np.ndarray:
    """Minimal 33×4 landmarks with ankle y at indices 27/28."""
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    lm[27] = [0.4, left_y, 0.0, 1.0]
    lm[28] = [0.6, right_y, 0.0, 1.0]
    lm[23] = [0.42, 0.5, 0.0, 1.0]
    lm[24] = [0.58, 0.5, 0.0, 1.0]
    return lm


def test_estimate_ml_phases_returns_labels_and_confidence():
    n = 40
    fps = 30.0
    seq = []
    for i in range(n):
        # Oscillate which foot is "lower" (higher y) in image space
        t = i / fps
        ly = 0.75 + 0.04 * np.sin(t * 8.0)
        ry = 0.75 + 0.04 * np.sin(t * 8.0 + np.pi / 2)
        seq.append(_lm(ly, ry))

    stride = {"cadence_steps_per_min_merged": 160.0}
    out = estimate_ml_phases_for_sequence(seq, fps, stride)
    assert out is not None
    assert out["model_version"] == "geometry-cadence-v1"
    assert len(out["phases_per_frame"]) == n
    assert len(out["confidence"]) == n
    assert all(p in ("stance", "swing", "push") for p in out["phases_per_frame"])
    assert all(0.0 <= c <= 1.0 for c in out["confidence"])


def test_too_short_sequence_returns_none():
    assert estimate_ml_phases_for_sequence([], 30.0, {}) is None
    assert estimate_ml_phases_for_sequence([None] * 4, 30.0, {}) is None
