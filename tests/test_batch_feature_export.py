"""Tests for batch_feature_export (features from pose JSON only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_pose_frame(i: int, fps: float) -> dict:
    """Minimal 33x4 landmarks with oscillating left ankle (stride-like)."""
    lm = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
    phase = i / fps
    # Indices from FeatureExtractor / MediaPipe
    lm[11] = [0.4, 0.3, 0.0, 1.0]  # L shoulder
    lm[12] = [0.6, 0.3, 0.0, 1.0]  # R shoulder
    lm[23] = [0.45, 0.5, 0.0, 1.0]  # L hip
    lm[24] = [0.55, 0.5, 0.0, 1.0]  # R hip
    lm[25] = [0.42, 0.65, 0.0, 1.0]  # L knee
    lm[26] = [0.58, 0.65, 0.0, 1.0]  # R knee
    x_ank = 0.5 + 0.05 * np.sin(2 * np.pi * 1.5 * phase)
    y_ank = 0.7 + 0.05 * np.sin(2 * np.pi * 1.5 * phase)
    lm[27] = [float(x_ank), float(y_ank), 0.0, 1.0]  # L ankle
    lm[28] = [0.52, 0.72, 0.0, 1.0]  # R ankle
    return {"frame": i, "timestamp": i / fps, "landmarks": lm}


def _write_poses(path: Path, n_frames: int = 120, fps: float = 30.0) -> None:
    poses = [_make_pose_frame(i, fps) for i in range(n_frames)]
    payload = {
        "video": str(ROOT / "data/raw/sample_videos/synthetic_test.mp4"),
        "fps": fps,
        "frame_count": n_frames,
        "poses": poses,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_batch_export_creates_three_outputs(tmp_path, monkeypatch):
    """Export produces JSON, frames CSV, and summary row with expected shapes."""
    monkeypatch.chdir(ROOT)
    poses_dir = tmp_path / "poses"
    _write_poses(poses_dir / "abc123_poses.json")

    out_dir = tmp_path / "features"
    summary = tmp_path / "features_summary.csv"

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "batch_feature_export",
        ROOT / "scripts" / "batch_feature_export.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from core.feature_extractor import FeatureExtractor

    mod.export_one(
        poses_dir / "abc123_poses.json",
        out_dir,
        FeatureExtractor(fps=30),
        summary,
        force=True,
        force_summary=True,
    )

    jf = out_dir / "abc123_features.json"
    cf = out_dir / "abc123_features_frames.csv"
    assert jf.is_file() and cf.is_file()
    data = json.loads(jf.read_text(encoding="utf-8"))
    assert "features" in data and "stride_metrics" in data["features"]

    lines = cf.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 121  # header + 120 frames

    summ = summary.read_text(encoding="utf-8")
    assert "abc123" in summ
    assert "synthetic_test.mp4" in summ
