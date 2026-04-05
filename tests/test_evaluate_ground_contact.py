"""Tests for ground-contact evaluation script logic."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_eval_mod():
    path = ROOT / "scripts" / "evaluate_ground_contact.py"
    spec = importlib.util.spec_from_file_location("evaluate_ground_contact", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _poses_with_peaks(n: int = 90, fps: float = 30.0) -> dict:
    poses = []
    for i in range(n):
        lm = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
        lm[11] = [0.4, 0.3, 0.0, 1.0]
        lm[12] = [0.6, 0.3, 0.0, 1.0]
        lm[23] = [0.45, 0.5, 0.0, 1.0]
        lm[24] = [0.55, 0.5, 0.0, 1.0]
        lm[25] = [0.42, 0.65, 0.0, 1.0]
        lm[26] = [0.58, 0.65, 0.0, 1.0]
        phase = i / fps
        y = 0.7 + 0.08 * np.sin(2 * np.pi * 1.2 * phase)
        lm[27] = [0.5, float(y), 0.0, 1.0]
        lm[28] = [0.52, 0.71, 0.0, 1.0]
        poses.append({"frame": i, "timestamp": i / fps, "landmarks": lm})
    return {"fps": fps, "poses": poses}


def test_ankle_ground_score_series_runs():
    mod = _load_eval_mod()
    data = _poses_with_peaks()
    s = mod.ankle_ground_score_series(data["poses"])
    assert len(s) == len(data["poses"])
    assert np.isfinite(s).all()


def test_run_evaluation_writes_metrics(tmp_path):
    mod = _load_eval_mod()

    poses_path = tmp_path / "clip_poses.json"
    labels_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "out"

    data = _poses_with_peaks()
    poses_path.write_text(json.dumps(data), encoding="utf-8")

    # Label frames near expected peaks (heuristic labels)
    rows = []
    for i in range(0, 90, 15):
        rows.append({"frame_id": i, "is_contact": 1})
        rows.append({"frame_id": i + 1, "is_contact": 0})
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id", "is_contact"])
        w.writeheader()
        w.writerows(rows)

    res = mod.run_evaluation(labels_path, poses_path, out_dir)
    assert (out_dir / "ground_contact_metrics.csv").is_file()
    assert (out_dir / "ground_contact_summary.txt").is_file()
    assert "f1" in res or res.get("f1") is not None
    assert 0.0 <= res["sensitivity"] <= 1.0
    assert 0.0 <= res["specificity"] <= 1.0
