"""
Per-frame gait phase labels for the results UI.

Uses a **geometry + cadence** estimate (ankle height difference, stride timing from
`stride_metrics`). This is not the PyTorch CNN-LSTM in `models/gait_classifier.py`;
when a trained checkpoint is wired in, replace or blend via the same `features.ml`
shape so the frontend stays stable.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

# MediaPipe BlazePose indices (match feature_extractor.JOINTS)
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28


def _interp_series(raw: np.ndarray) -> np.ndarray:
    """Fill NaNs with linear interpolation; edges use nearest valid."""
    n = len(raw)
    if n == 0:
        return raw
    valid = np.isfinite(raw)
    if valid.sum() == 0:
        return np.zeros(n)
    if valid.all():
        return raw.astype(float)
    idx = np.arange(n)
    return np.interp(idx, idx[valid], raw[valid])


def _phase_to_label(phase01: float) -> str:
    """Map one gait-cycle phase in [0,1) to stance / push / swing."""
    p = phase01 % 1.0
    if p < 0.38:
        return "stance"
    if p < 0.58:
        return "push"
    return "swing"


def estimate_ml_phases_for_sequence(
    smoothed_landmarks: list,
    fps: float,
    stride_metrics: dict,
) -> dict | None:
    """
    Build per-frame phase labels and a coarse confidence score.

    Returns:
        {
          "model_version": "geometry-cadence-v1",
          "phases_per_frame": ["stance"|"swing"|"push", ...],
          "confidence": [0..1, ...]  (same length; null pose -> low confidence)
        }
        or None if the sequence is too short / empty.
    """
    n = len(smoothed_landmarks)
    if n < 8:
        return None

    la = np.full(n, np.nan)
    ra = np.full(n, np.nan)
    vis_l = np.full(n, np.nan)
    vis_r = np.full(n, np.nan)

    for i, lm in enumerate(smoothed_landmarks):
        if lm is None:
            continue
        arr = np.asarray(lm)
        if arr.shape[0] <= max(_LEFT_ANKLE, _RIGHT_ANKLE):
            continue
        la[i] = float(arr[_LEFT_ANKLE][1])
        ra[i] = float(arr[_RIGHT_ANKLE][1])
        if arr.shape[1] > 3:
            vis_l[i] = float(arr[_LEFT_ANKLE][3])
            vis_r[i] = float(arr[_RIGHT_ANKLE][3])

    la_i = _interp_series(la)
    ra_i = _interp_series(ra)
    diff = la_i - ra_i
    # Light smoothing so phase boundaries are not single-frame noise.
    if len(diff) >= 5:
        k = 5
        kernel = np.ones(k) / k
        diff = np.convolve(diff, kernel, mode="same")

    # --- Gait cycle length in frames (prefer merged cadence, else stride_time) ---
    period_frames: float
    if stride_metrics.get("cadence_steps_per_min_merged"):
        cspm = float(stride_metrics["cadence_steps_per_min_merged"])
        if cspm > 1e-6:
            step_t = 60.0 / cspm
            period_frames = max(4.0, step_t * fps)
        else:
            period_frames = max(12.0, n / 5.5)
    elif stride_metrics.get("stride_time_sec"):
        st = float(stride_metrics["stride_time_sec"])
        # One stride (L→L) ≈ one full left–right–left cycle; diff oscillates ~once per step.
        period_frames = max(6.0, (st * fps) / 2.0)
    else:
        period_frames = max(12.0, n / 5.5)

    # Anchor phase so stance aligns near foot-strike proxies (peaks of max ankle y).
    # Use max(la,ra) as "lowest foot" depth; peaks ≈ contacts.
    depth = np.maximum(la_i, ra_i)
    phase_offset = 0.0
    if np.isfinite(depth).any():
        dist = max(1, int(fps * 0.25))
        peaks, _ = scipy_signal.find_peaks(depth, distance=dist)
        if len(peaks) >= 1:
            # Start stance segment shortly after a contact peak (empirical offset).
            phase_offset = (peaks[0] % period_frames) / period_frames * 0.15

    phases: list[str] = []
    confidence: list[float] = []

    d_std = float(np.std(diff)) + 1e-6
    for i in range(n):
        if smoothed_landmarks[i] is None:
            phases.append("swing")
            confidence.append(0.15)
            continue

        # Linear phase 0..1 rolling through the clip
        phi = ((i + phase_offset * period_frames) % period_frames) / period_frames
        phases.append(_phase_to_label(phi))

        # Confidence: stronger differential + visible feet -> higher
        foot_vis = 1.0
        if np.isfinite(vis_l[i]) and np.isfinite(vis_r[i]):
            foot_vis = float(np.clip((vis_l[i] + vis_r[i]) / 2.0, 0.0, 1.0))
        elif np.isfinite(vis_l[i]):
            foot_vis = float(np.clip(vis_l[i], 0.0, 1.0))
        elif np.isfinite(vis_r[i]):
            foot_vis = float(np.clip(vis_r[i], 0.0, 1.0))
        else:
            foot_vis = 0.55

        mag = min(1.0, abs(diff[i]) / (3.0 * d_std))
        conf = float(np.clip(0.25 + 0.45 * mag + 0.35 * foot_vis, 0.0, 1.0))
        confidence.append(round(conf, 3))

    return {
        "model_version": "geometry-cadence-v1",
        "phases_per_frame": phases,
        "confidence": confidence,
    }
