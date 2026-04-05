"""Tests for manual vs computed angle validation."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_val_mod():
    path = ROOT / "scripts" / "validate_features.py"
    spec = importlib.util.spec_from_file_location("validate_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_validate_features_known_offset(tmp_path):
    mod = _load_val_mod()

    feat_csv = tmp_path / "feat.csv"
    manual_csv = tmp_path / "man.csv"
    out_dir = tmp_path / "out"

    # Computed "left_knee" = 90 everywhere in features file
    with open(feat_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "video_name",
                "left_hip_deg",
                "left_knee_deg",
            ],
        )
        w.writeheader()
        for i in range(5):
            w.writerow(
                {
                    "frame_id": i,
                    "video_name": "test.mp4",
                    "left_hip_deg": 170,
                    "left_knee_deg": 90,
                }
            )

    # Manual knee = 88 → error 2°
    with open(manual_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["frame_id", "video_name", "manual_angle", "angle_name"],
        )
        w.writeheader()
        for i in range(5):
            w.writerow(
                {
                    "frame_id": i,
                    "video_name": "test.mp4",
                    "manual_angle": 88,
                    "angle_name": "left_knee",
                }
            )

    idx = mod.load_features_rows(feat_csv)
    with open(manual_csv, newline="", encoding="utf-8") as f:
        manual = list(csv.DictReader(f))

    rc = mod.run_validation(manual, idx, out_dir)
    assert rc == 0

    metrics = (out_dir / "angle_validation_metrics.csv").read_text(encoding="utf-8")
    assert "left_knee" in metrics
    assert "PASS" in metrics or "2.0" in metrics.replace(" ", "")
