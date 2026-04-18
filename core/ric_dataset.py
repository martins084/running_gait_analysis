"""
RIC dataset loader for anomaly training.

Data assumptions:
- Manifest rows come from `scripts/build_ric_manifest.py`.
- Split assignment comes from `scripts/build_subject_splits.py` (session_split_v1.csv).
- JSON files are available locally under `json_root` using manifest `source_ref`.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceSpec:
    seq_len: int = 180
    prefer_mode: str = "run"  # run|walk|auto
    normalize: str = "zscore"  # zscore|none
    train_random_window: bool = True


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
) -> int:
    """
    Scan every session in the split CSV (all splits) and return the maximum
    feature dimension (num_markers * 3) after mode selection.

    Sessions differ in how many marker tracks exist; the model needs one fixed
    input size, so we pad shorter sequences to this width.
    """
    session_split_csv = Path(session_split_csv)
    json_root = Path(json_root)
    rows: list[dict] = []
    with open(session_split_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Session split CSV has no header row.")
        for r in reader:
            rows.append(r)
    if not rows:
        raise ValueError("Session split CSV has no rows.")

    def _json_path_from_row(row: dict) -> Path:
        src = str(row["source_ref"]).replace("\\", "/")
        if "reformat_data/" in src:
            src = src.split("reformat_data/", 1)[1]
        return json_root / src

    def _select_mode_block(payload: dict) -> tuple[str, dict]:
        run = payload.get("running") or {}
        walk = payload.get("walking") or {}
        if seq.prefer_mode == "run":
            if run:
                return "run", run
            if walk:
                return "walk", walk
        elif seq.prefer_mode == "walk":
            if walk:
                return "walk", walk
            if run:
                return "run", run
        else:
            if run and walk:
                run_len = len(next(iter(run.values()))) if run else 0
                walk_len = len(next(iter(walk.values()))) if walk else 0
                return ("run", run) if run_len >= walk_len else ("walk", walk)
            if run:
                return "run", run
            if walk:
                return "walk", walk
        raise ValueError("JSON has no usable walking/running marker block.")

    max_d = 0
    for row in rows:
        p = _json_path_from_row(row)
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        _mode, block = _select_mode_block(payload)
        if not block:
            continue
        d = len(block) * 3
        if d > max_d:
            max_d = d
    if max_d <= 0:
        raise RuntimeError("Could not infer max feature dimension (no valid JSON blocks).")
    return max_d


class RICAnomalyDataset(Dataset):
    """
    Sequence dataset for `GaitAnomalyDetector`.

    Output item:
      {
        "x": Tensor[seq_len, feature_dim],
        "is_injured": Tensor[] int64 (0|1),
        "subject_id": str,
        "session_id": str,
        "mode": str
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
    ) -> None:
        super().__init__()
        self.session_split_csv = Path(session_split_csv)
        self.json_root = Path(json_root)
        self.split = split
        self.seq = seq or SequenceSpec()
        self.include_injured = include_injured
        self.rng = random.Random(seed)

        if not self.session_split_csv.is_file():
            raise FileNotFoundError(f"Session split CSV not found: {self.session_split_csv}")
        if not self.json_root.exists():
            raise FileNotFoundError(f"JSON root not found: {self.json_root}")

        self.rows = self._load_rows()
        if not self.rows:
            raise ValueError(f"No rows found for split={self.split} with include_injured={self.include_injured}")

        # Sessions differ in marker count; pad/truncate to one canonical width for batching.
        if feature_dim is None:
            self.feature_dim = compute_max_feature_dim(self.session_split_csv, self.json_root, self.seq)
        else:
            self.feature_dim = int(feature_dim)

    def _load_rows(self) -> list[dict]:
        rows: list[dict] = []
        with open(self.session_split_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Session split CSV has no header row.")
            required = {"subject_id", "session_id", "source_ref", "split", "is_injured"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"Session split CSV missing required columns: {sorted(missing)}")
            for r in reader:
                if r.get("split") != self.split:
                    continue
                injured = _to_int(r.get("is_injured"), 0)
                if not self.include_injured and injured == 1:
                    continue
                rows.append(r)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _json_path_from_row(self, row: dict) -> Path:
        """
        source_ref may be:
        - local-style: "<subject>/<file>.json"
        - s3 key-style: ".../reformat_data/<subject>/<file>.json"
        """
        src = str(row["source_ref"]).replace("\\", "/")
        if "reformat_data/" in src:
            src = src.split("reformat_data/", 1)[1]
        return self.json_root / src

    def _select_mode_block(self, payload: dict) -> tuple[str, dict]:
        run = payload.get("running") or {}
        walk = payload.get("walking") or {}

        if self.seq.prefer_mode == "run":
            if run:
                return "run", run
            if walk:
                return "walk", walk
        elif self.seq.prefer_mode == "walk":
            if walk:
                return "walk", walk
            if run:
                return "run", run
        else:  # auto
            if run and walk:
                # Prefer block with longer sequence length.
                run_len = len(next(iter(run.values()))) if run else 0
                walk_len = len(next(iter(walk.values()))) if walk else 0
                return ("run", run) if run_len >= walk_len else ("walk", walk)
            if run:
                return "run", run
            if walk:
                return "walk", walk

        raise ValueError("JSON has no usable walking/running marker block.")

    def _block_to_matrix(self, block: dict) -> np.ndarray:
        """
        Convert marker dictionary to matrix [T, markers*3].
        Marker order is sorted for deterministic feature layout.
        """
        marker_names = sorted(block.keys())
        if not marker_names:
            raise ValueError("Empty marker block.")

        matrices = [_to_float_array(block[name]) for name in marker_names]
        lengths = [m.shape[0] for m in matrices]
        t = min(lengths)
        if t <= 1:
            raise ValueError("Sequence too short after marker alignment.")

        clipped = [m[:t] for m in matrices]
        # Concatenate marker coordinates into feature axis.
        x = np.concatenate(clipped, axis=1)  # [T, M*3]
        return x.astype(np.float32, copy=False)

    def _fit_feature_dim(self, x: np.ndarray) -> np.ndarray:
        """Pad or truncate along feature axis to `self.feature_dim`."""
        d = x.shape[1]
        if d == self.feature_dim:
            return x
        if d < self.feature_dim:
            pad = np.zeros((x.shape[0], self.feature_dim - d), dtype=np.float32)
            return np.concatenate([x, pad], axis=1)
        return x[:, : self.feature_dim]

    def _slice_window(self, x: np.ndarray) -> np.ndarray:
        t = x.shape[0]
        seq_len = self.seq.seq_len
        if t == seq_len:
            return x
        if t < seq_len:
            pad = np.repeat(x[-1:, :], seq_len - t, axis=0)
            return np.concatenate([x, pad], axis=0)

        if self.split == "train" and self.seq.train_random_window:
            start = self.rng.randint(0, t - seq_len)
        else:
            start = (t - seq_len) // 2
        return x[start : start + seq_len]

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.seq.normalize == "none":
            return x
        # Per-feature z-score with epsilon stability.
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        x = (x - mu) / (sd + 1e-6)
        return x

    def _load_sequence(self, row: dict) -> np.ndarray:
        p = self._json_path_from_row(row)
        if not p.is_file():
            raise FileNotFoundError(f"JSON file not found for row: {p}")
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
        mode, block = self._select_mode_block(payload)
        x = self._block_to_matrix(block)
        x = self._fit_feature_dim(x)
        x = self._slice_window(x)
        x = self._normalize(x)
        row["_selected_mode"] = mode
        return x

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        x = self._load_sequence(row)
        return {
            "x": torch.from_numpy(x),
            "is_injured": torch.tensor(_to_int(row.get("is_injured"), 0), dtype=torch.int64),
            "subject_id": row.get("subject_id", ""),
            "session_id": row.get("session_id", ""),
            "mode": row.get("_selected_mode") or row.get("mode") or "",
        }


def collate_ric_anomaly(batch: list[dict]) -> dict:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["is_injured"] for b in batch], dim=0)
    return {
        "x": x,
        "is_injured": y,
        "subject_id": [b["subject_id"] for b in batch],
        "session_id": [b["session_id"] for b in batch],
        "mode": [b["mode"] for b in batch],
    }

