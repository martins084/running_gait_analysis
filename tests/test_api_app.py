"""API error-handling tests for analyze endpoint."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient

import api.app as app_module


class _OkDetector:
    def process_video(self, _video_path: str, poses_json: str) -> None:
        payload = {
            "fps": 30,
            "poses": [{"frame": 0, "timestamp": 0.0, "landmarks": [[0, 0, 0, 1]] * 33}],
        }
        Path(poses_json).write_text(json.dumps(payload), encoding="utf-8")


class _FailDetector:
    def process_video(self, _video_path: str, _poses_json: str) -> None:
        raise RuntimeError("bad/corrupt stream")


class _SlowDetector:
    def process_video(self, _video_path: str, _poses_json: str) -> None:
        time.sleep(0.2)


class _OkExtractor:
    def extract_all_features(self, _poses_json: str) -> dict:
        return {"stride_metrics": {"cadence_steps_per_min": 170.0}}


def _fake_annotator(_video_path: str, _poses_json: str, annotated_path: str) -> None:
    Path(annotated_path).write_bytes(b"fake-mp4")


def _client_with_tmpdirs(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "RESULTS_DIR", tmp_path / "results")
    app_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app_module.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return TestClient(app_module.app)


def test_analyze_rejects_unsupported_extension(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    response = client.post(
        "/analyze",
        files={"video": ("run.txt", b"not-a-video", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_analyze_rejects_empty_upload(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_maps_pose_failure_to_422(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _FailDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert response.status_code == 422
    assert "Pose analysis failed" in response.json()["detail"]


def test_analyze_timeout_returns_504(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _SlowDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)
    monkeypatch.setattr(app_module, "ANALYZE_TIMEOUT_SEC", 0.01)

    response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_analyze_persists_result_in_sqlite(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert response.status_code == 200
    payload = response.json()
    analysis_id = payload["id"]

    db_file = app_module._db_path()
    assert db_file.exists(), "SQLite DB file should be created"

    with sqlite3.connect(db_file) as conn:
        row = conn.execute(
            "SELECT id, status FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    assert row is not None
    assert row[0] == analysis_id
    assert row[1] == "completed"


def test_get_results_reads_from_sqlite_when_json_missing(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    create_response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert create_response.status_code == 200
    analysis_id = create_response.json()["id"]

    # Simulate legacy JSON file loss; DB should still serve the result.
    result_file = app_module.RESULTS_DIR / f"{analysis_id}_result.json"
    result_file.unlink(missing_ok=True)

    read_response = client.get(f"/results/{analysis_id}")
    assert read_response.status_code == 200
    assert read_response.json()["id"] == analysis_id


def test_health_endpoint_returns_ok(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_results_missing_id_returns_404(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    response = client.get("/results/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_download_missing_video_returns_404(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    response = client.get("/download/does-not-exist")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_poses_returns_stored_json(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    results_dir = tmp_path / "results"
    aid = "pose-test-id"
    payload = {"fps": 30, "poses": []}
    (results_dir / f"{aid}_poses.json").write_text(json.dumps(payload), encoding="utf-8")

    r = client.get(f"/poses/{aid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["fps"] == 30


def test_get_poses_missing_returns_404(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    r = client.get("/poses/does-not-exist")
    assert r.status_code == 404


def test_download_returns_mp4_for_completed_analysis(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)

    create_response = client.post(
        "/analyze",
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert create_response.status_code == 200
    analysis_id = create_response.json()["id"]

    download_response = client.get(f"/download/{analysis_id}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"].startswith("video/mp4")

