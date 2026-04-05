"""Edge-case tests for batch export, ground-contact eval, and angle validation scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_labels_csv_skips_bad_rows(tmp_path):
    ev = _load("evaluate_ground_contact")
    p = tmp_path / "lab.csv"
    p.write_text("frame_id,is_contact\n0,1\noops,1\n2,0\n", encoding="utf-8")
    frames, labs = ev.load_labels_csv(p)
    assert 0 in frames and 2 in frames
    assert len(frames) == 2


def test_load_labels_csv_empty_after_bad_rows_raises(tmp_path):
    ev = _load("evaluate_ground_contact")
    p = tmp_path / "bad.csv"
    p.write_text("frame_id,is_contact\nxx,yy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No valid label rows"):
        ev.load_labels_csv(p)


def test_run_evaluation_rejects_empty_poses(tmp_path):
    ev = _load("evaluate_ground_contact")
    labels = tmp_path / "l.csv"
    labels.write_text("frame_id,is_contact\n0,0\n", encoding="utf-8")
    poses = tmp_path / "p.json"
    poses.write_text(json.dumps({"fps": 30, "poses": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="zero frames"):
        ev.run_evaluation(labels, poses, tmp_path / "out")


def test_validate_features_no_matches_returns_two(tmp_path):
    vf = _load("validate_features")
    man = tmp_path / "m.csv"
    man.write_text("frame_id,video_name,manual_angle,angle_name\n0,wrong.mp4,90,left_knee\n", encoding="utf-8")
    feat = tmp_path / "f.csv"
    feat.write_text(
        "frame_id,video_name,left_knee_deg\n0,right.mp4,90.0\n",
        encoding="utf-8",
    )
    manual = vf.load_manual(man)
    idx = vf.load_features_rows(feat)
    rc = vf.run_validation(manual, idx, tmp_path / "vo")
    assert rc == 2


def test_batch_export_rejects_invalid_poses_json(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    batch = _load("batch_feature_export")
    bad = tmp_path / "x_poses.json"
    bad.write_text('{"not_poses": []}', encoding="utf-8")
    from core.feature_extractor import FeatureExtractor

    with pytest.raises(ValueError, match="poses"):
        batch.export_one(
            bad,
            tmp_path / "out",
            FeatureExtractor(fps=30),
            tmp_path / "sum.csv",
            force=True,
            force_summary=True,
        )


def test_validate_features_rejects_headerless_csv(tmp_path):
    vf = _load("validate_features")
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        vf.load_manual(p)
