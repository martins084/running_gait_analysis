"""
RIC dataset loader for anomaly training and supervised classification.

Data assumptions:
- Manifest rows come from `scripts/build_ric_manifest.py`.
- Split assignment comes from `scripts/build_subject_splits.py` (session_split_v1.csv).
- JSON files are available locally under `json_root` using manifest `source_ref`.

Dataset classes
---------------
RICAnomalyDataset     Healthy-only (or mixed) for reconstruction-based anomaly detection.
RICClassifierDataset  Fully labelled for supervised binary injury classification.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Sequence specification
# ---------------------------------------------------------------------------

@dataclass
class SequenceSpec:
    seq_len: int = 180
    prefer_mode: str = "run"           # run | walk | auto
    normalize: str = "zscore"          # zscore | none
    train_random_window: bool = True
    # none: marker coordinates only. motion_stats: append per-timestep |v|,|a|,|j|.
    feature_engineering: str = "none"  # none | motion_stats

    # ---- Low-pass filter (applied to the full raw sequence before windowing) ----
    # Butterworth 4th-order zero-phase filter — standard for 120 Hz MoCap data.
    # Set lp_filter_hz > 0 to enable (e.g. 15.0 Hz).
    lp_filter_hz: float = 0.0
    lp_filter_source_hz: float = 120.0  # assumed recording Hz for filter design

    # ---- Augmentation (training split only, all disabled by default) ----
    aug_mirror: bool = False       # negate Z-axis of each [x,y,z] triplet (left-right flip)
    aug_mirror_prob: float = 0.5   # probability of applying mirror each sample
    aug_noise_sigma: float = 0.0   # Gaussian noise std in normalised space (0 = off)
    aug_time_warp: bool = False    # random time stretch ×0.9–1.1


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _motion_stats_extra_channels() -> int:
    """Number of features appended for `motion_stats` (velocity / accel / jerk magnitudes)."""
    return 3


def _append_motion_stats(x: np.ndarray) -> np.ndarray:
    """
    Append biomechanical proxies: L2-norm of finite-difference velocity, acceleration,
    and jerk of the full pose vector at each time step.
    """
    t, f = x.shape
    if t < 2:
        raise ValueError("Sequence too short for motion statistics.")
    z = np.zeros((1, f), dtype=np.float32)
    v = np.vstack([z, np.diff(x, axis=0)]).astype(np.float32, copy=False)
    a = np.vstack([z, np.diff(v, axis=0)]).astype(np.float32, copy=False)
    j = np.vstack([z, np.diff(a, axis=0)]).astype(np.float32, copy=False)
    vel  = np.linalg.norm(v, axis=1, keepdims=True)
    acc  = np.linalg.norm(a, axis=1, keepdims=True)
    jerk = np.linalg.norm(j, axis=1, keepdims=True)
    return np.concatenate([x, vel, acc, jerk], axis=1).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def _butterworth_lowpass(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter applied per feature axis.

    Standard preprocessing for 120 Hz MoCap data; removes high-frequency
    skin-artefact noise while preserving gait kinematics (< ~10 Hz).
    """
    from scipy.signal import butter, sosfiltfilt

    min_len = order * 3 + 1
    if x.shape[0] < min_len:
        return x
    nyq = fs_hz / 2.0
    if cutoff_hz >= nyq:
        return x
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def _mirror_features(x: np.ndarray) -> np.ndarray:
    """Approximate left-right flip: negate the Z coordinate of every [x,y,z] triplet.

    For standard biomechanics coordinate systems (X=anterior, Y=vertical, Z=mediolateral),
    this swaps left and right sides.  Applied probabilistically during training to
    double effective sample count and improve bilateral generalisation.
    """
    out = x.copy()
    out[:, 2::3] = -out[:, 2::3]
    return out


def _time_warp(x: np.ndarray, rng: random.Random, lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """Linearly interpolate the sequence to a randomly stretched/compressed length,
    then crop or pad back to the original length.

    Simulates running at slightly different speeds without changing the sequence
    tensor shape fed to the model.
    """
    factor = rng.uniform(lo, hi)
    t_orig = x.shape[0]
    t_new = max(2, int(round(t_orig * factor)))
    if t_new == t_orig:
        return x
    old_t = np.linspace(0.0, 1.0, t_orig)
    new_t = np.linspace(0.0, 1.0, t_new)
    out = np.zeros((t_new, x.shape[1]), dtype=np.float32)
    for f in range(x.shape[1]):
        out[:, f] = np.interp(new_t, old_t, x[:, f])
    if t_new >= t_orig:
        return out[:t_orig]
    pad = np.repeat(out[-1:, :], t_orig - t_new, axis=0)
    return np.concatenate([out, pad], axis=0)


def _add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    """Add zero-mean Gaussian noise in the (already normalised) feature space."""
    return (x + np.random.normal(0.0, sigma, x.shape).astype(np.float32))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(str(v))
    except (TypeError, ValueError):
        return default


def _to_float_array(values: list) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected marker array (T,3), got shape {arr.shape}")
    return arr


def compute_max_feature_dim(
    session_split_csv: Path,
    json_root: Path,
    seq: SequenceSpec,
    split_filter: str | set[str] | None = None,
) -> int:
    """
    Scan sessions in the split CSV and return the maximum marker feature dimension
    (num_markers * 3) after mode selection.  Used to set a fixed model input size.
    """
    session_split_csv = Path(session_split_csv)
    json_root = Path(json_root)
    rows: list[dict] = []
    allowed_splits: set[str] | None = None
    if split_filter is not None:
        allowed_splits = {split_filter} if isinstance(split_filter, str) else set(str(v) for v in split_filter)
    with open(session_split_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Session split CSV has no header row.")
        for r in reader:
            if allowed_splits is not None and str(r.get("split", "")) not in allowed_splits:
                continue
            rows.append(r)
    if not rows:
        raise ValueError("Session split CSV has no rows.")

    def _json_path(row: dict) -> Path:
        src = str(row["source_ref"]).replace("\\", "/")
        if "reformat_data/" in src:
            src = src.split("reformat_data/", 1)[1]
        return json_root / src

    def _select_mode(payload: dict) -> tuple[str, dict]:
        run  = payload.get("running")  or {}
        walk = payload.get("walking")  or {}
        if seq.prefer_mode == "run":
            return ("run", run) if run else ("walk", walk)
        if seq.prefer_mode == "walk":
            return ("walk", walk) if walk else ("run", run)
        run_len  = len(next(iter(run.values()))) if run else 0
        walk_len = len(next(iter(walk.values()))) if walk else 0
        return ("run", run) if run_len >= walk_len else ("walk", walk)

    max_d = 0
    for row in rows:
        p = _json_path(row)
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        try:
            _mode, block = _select_mode(payload)
        except (ValueError, StopIteration):
            continue
        if not block:
            continue
        d = len(block) * 3
        if d > max_d:
            max_d = d
    if max_d <= 0:
        raise RuntimeError("Could not infer max feature dimension (no valid JSON blocks found).")
    return max_d


# ---------------------------------------------------------------------------
# Base dataset mixin
# ---------------------------------------------------------------------------

class _RICBaseDataset(Dataset):
    """Shared loading, windowing, normalisation, and augmentation logic."""

    def __init__(
        self,
        session_split_csv: Path,
        json_root: Path,
        split: str,
        seq: SequenceSpec | None,
        seed: int,
        feature_dim: int | None,
        label_col: str,
    ) -> None:
        super().__init__()
        self.session_split_csv = Path(session_split_csv)
        self.json_root = Path(json_root)
        self.split = split
        self.seq = seq or SequenceSpec()
        self.label_col = str(label_col or "is_injured")
        self.rng = random.Random(seed)

        if not self.session_split_csv.is_file():
            raise FileNotFoundError(f"Session split CSV not found: {self.session_split_csv}")
        if not self.json_root.exists():
            raise FileNotFoundError(f"JSON root not found: {self.json_root}")

        fe = (self.seq.feature_engineering or "none").lower()
        if fe not in ("none", "motion_stats"):
            raise ValueError("SequenceSpec.feature_engineering must be 'none' or 'motion_stats'")
        self.extra_channels = _motion_stats_extra_channels() if fe == "motion_stats" else 0

        if feature_dim is None:
            self.marker_feature_dim = compute_max_feature_dim(
                self.session_split_csv, self.json_root, self.seq
            )
            self.feature_dim = int(self.marker_feature_dim + self.extra_channels)
        else:
            self.feature_dim = int(feature_dim)
            self.marker_feature_dim = int(self.feature_dim - self.extra_channels)
            if self.marker_feature_dim < 1:
                raise ValueError("feature_dim is smaller than the motion_stats channel budget.")

    # ---- JSON loading helpers ----

    def _json_path(self, row: dict) -> Path:
        src = str(row["source_ref"]).replace("\\", "/")
        if "reformat_data/" in src:
            src = src.split("reformat_data/", 1)[1]
        return self.json_root / src

    def _select_mode_block(self, payload: dict) -> tuple[str, dict]:
        run  = payload.get("running")  or {}
        walk = payload.get("walking")  or {}
        if self.seq.prefer_mode == "run":
            if run:  return "run",  run
            if walk: return "walk", walk
        elif self.seq.prefer_mode == "walk":
            if walk: return "walk", walk
            if run:  return "run",  run
        else:
            if run and walk:
                run_len  = len(next(iter(run.values())))
                walk_len = len(next(iter(walk.values())))
                return ("run", run) if run_len >= walk_len else ("walk", walk)
            if run:  return "run",  run
            if walk: return "walk", walk
        raise ValueError("JSON has no usable walking/running marker block.")

    def _block_to_matrix(self, block: dict) -> np.ndarray:
        marker_names = sorted(block.keys())
        if not marker_names:
            raise ValueError("Empty marker block.")
        matrices = [_to_float_array(block[name]) for name in marker_names]
        t = min(m.shape[0] for m in matrices)
        if t <= 1:
            raise ValueError("Sequence too short after marker alignment.")
        x = np.concatenate([m[:t] for m in matrices], axis=1)
        return x.astype(np.float32, copy=False)

    def _fit_feature_dim(self, x: np.ndarray) -> np.ndarray:
        d, t = x.shape[1], self.marker_feature_dim
        if d == t:
            return x
        if d < t:
            return np.concatenate([x, np.zeros((x.shape[0], t - d), dtype=np.float32)], axis=1)
        return x[:, :t]

    def _slice_window(self, x: np.ndarray) -> np.ndarray:
        t, seq_len = x.shape[0], self.seq.seq_len
        if t == seq_len:
            return x
        if t < seq_len:
            pad = np.repeat(x[-1:, :], seq_len - t, axis=0)
            return np.concatenate([x, pad], axis=0)
        if self.split == "train" and self.seq.train_random_window:
            start = self.rng.randint(0, t - seq_len)
        else:
            start = (t - seq_len) // 2
        return x[start: start + seq_len]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.seq.normalize == "none":
            return x
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        return (x - mu) / (sd + 1e-6)

    # ---- Full preprocessing pipeline ----

    def _load_sequence(self, row: dict) -> np.ndarray:
        p = self._json_path(row)
        if not p.is_file():
            raise FileNotFoundError(f"JSON file not found: {p}")
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        mode, block = self._select_mode_block(payload)
        row["_selected_mode"] = mode

        x = self._block_to_matrix(block)

        # 1. Butterworth low-pass filter (full sequence, before windowing).
        if self.seq.lp_filter_hz > 0:
            x = _butterworth_lowpass(x, self.seq.lp_filter_hz, self.seq.lp_filter_source_hz)

        # 2. Uniform feature width + temporal window.
        x = self._fit_feature_dim(x)
        x = self._slice_window(x)

        # 3. Augmentations (training split only).
        if self.split == "train":
            if self.seq.aug_mirror and self.rng.random() < self.seq.aug_mirror_prob:
                x = _mirror_features(x)
            if self.seq.aug_time_warp:
                x = _time_warp(x, self.rng)

        # 4. Motion statistics (before normalisation).
        if self.extra_channels:
            x = _append_motion_stats(x)
            if x.shape[1] != self.feature_dim:
                raise RuntimeError(
                    f"Feature width mismatch: expected {self.feature_dim} got {x.shape[1]}"
                )

        # 5. Normalise.
        x = self._normalize(x)

        # 6. Post-normalisation noise augmentation.
        if self.split == "train" and self.seq.aug_noise_sigma > 0:
            x = _add_gaussian_noise(x, self.seq.aug_noise_sigma)

        return x


# ---------------------------------------------------------------------------
# RICAnomalyDataset  (reconstruction-based anomaly detection)
# ---------------------------------------------------------------------------

class RICAnomalyDataset(_RICBaseDataset):
    """
    Sequence dataset for `GaitAnomalyDetector` (unsupervised reconstruction).

    By default (`include_injured=False`) only healthy sessions are returned for
    training — the anomaly detection paradigm.  Set `include_injured=True` for
    the validation/test splits so AUROC can be computed.

    Item schema:
      {
        "x":          Tensor[seq_len, feature_dim],
        "is_injured": Tensor[]  int64  (0|1),
        "subject_id": str,
        "session_id": str,
        "mode":       str,
      }
    """

    def __init__(
        self,
        session_split_csv: Path,
        json_root: Path,
        split: str,
        seq: SequenceSpec | None = None,
        include_injured: bool = True,
        seed: int = 42,
        feature_dim: int | None = None,
        label_col: str = "is_injured",
    ) -> None:
        super().__init__(session_split_csv, json_root, split, seq, seed, feature_dim, label_col)
        self.include_injured = include_injured
        self.rows = self._load_rows()
        if not self.rows:
            raise ValueError(
                f"No rows found for split={self.split!r} "
                f"with include_injured={self.include_injured}"
            )

    def _load_rows(self) -> list[dict]:
        rows: list[dict] = []
        with open(self.session_split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Session split CSV has no header row.")
            required = {"subject_id", "session_id", "source_ref", "split", self.label_col}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Session split CSV missing columns: {sorted(missing)}")
            for r in reader:
                if r.get("split") != self.split:
                    continue
                if not self.include_injured and _to_int(r.get(self.label_col), 0) == 1:
                    continue
                rows.append(r)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        x = self._load_sequence(row)
        return {
            "x": torch.from_numpy(x),
            "is_injured": torch.tensor(_to_int(row.get(self.label_col), 0), dtype=torch.int64),
            "subject_id": row.get("subject_id", ""),
            "session_id": row.get("session_id", ""),
            "mode": row.get("_selected_mode") or row.get("mode") or "",
        }


def collate_ric_anomaly(batch: list[dict]) -> dict:
    return {
        "x":          torch.stack([b["x"] for b in batch]),
        "is_injured": torch.stack([b["is_injured"] for b in batch]),
        "subject_id": [b["subject_id"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "mode":       [b["mode"] for b in batch],
    }


# ---------------------------------------------------------------------------
# RICClassifierDataset  (supervised binary classification)
# ---------------------------------------------------------------------------

class RICClassifierDataset(_RICBaseDataset):
    """
    Fully labelled dataset for supervised binary injury classification.

    Always includes both healthy and injured sessions.  Use this with
    `GaitInjuryClassifier` and `BCEWithLogitsLoss`.

    Item schema:
      {
        "x":          Tensor[seq_len, feature_dim],
        "label":      Tensor[]  float32  (0.0 | 1.0)  — ready for BCEWithLogitsLoss,
        "subject_id": str,
        "session_id": str,
        "mode":       str,
      }
    """

    def __init__(
        self,
        session_split_csv: Path,
        json_root: Path,
        split: str,
        seq: SequenceSpec | None = None,
        seed: int = 42,
        feature_dim: int | None = None,
        label_col: str = "is_injured",
    ) -> None:
        super().__init__(session_split_csv, json_root, split, seq, seed, feature_dim, label_col)
        self.rows = self._load_rows()
        if not self.rows:
            raise ValueError(f"No rows found for split={self.split!r} (RICClassifierDataset).")

    def _load_rows(self) -> list[dict]:
        rows: list[dict] = []
        with open(self.session_split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Session split CSV has no header row.")
            required = {"subject_id", "session_id", "source_ref", "split", self.label_col}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Session split CSV missing columns: {sorted(missing)}")
            for r in reader:
                if r.get("split") == self.split:
                    rows.append(r)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        x = self._load_sequence(row)
        label = float(_to_int(row.get(self.label_col), 0))
        return {
            "x":          torch.from_numpy(x),
            "label":      torch.tensor(label, dtype=torch.float32),
            "subject_id": row.get("subject_id", ""),
            "session_id": row.get("session_id", ""),
            "mode":       row.get("_selected_mode") or row.get("mode") or "",
        }

    def class_counts(self) -> tuple[int, int]:
        """Return (n_healthy, n_injured) counts for pos_weight computation."""
        labels = [_to_int(r.get(self.label_col), 0) for r in self.rows]
        n1 = sum(labels)
        return len(labels) - n1, n1


def collate_ric_classifier(batch: list[dict]) -> dict:
    return {
        "x":          torch.stack([b["x"] for b in batch]),
        "label":      torch.stack([b["label"] for b in batch]),
        "subject_id": [b["subject_id"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "mode":       [b["mode"] for b in batch],
    }
