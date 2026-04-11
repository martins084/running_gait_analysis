"""Core API privacy/auth tests."""

from __future__ import annotations

import json
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


class _OkExtractor:
    def extract_all_features(self, _poses_json: str) -> dict:
        return {"stride_metrics": {"cadence_steps_per_min": 170.0}, "joint_angles": [], "symmetry": [], "vertical_oscillation_px": 0}


def _fake_annotator(_video_path: str, _poses_json: str, annotated_path: str) -> None:
    Path(annotated_path).write_bytes(b"fake-mp4")


def _client_with_tmpdirs(monkeypatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(app_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(app_module, "RESULTS_DIR", tmp_path / "results")
    app_module.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    app_module.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return TestClient(app_module.app)


def _auth_header(client: TestClient, email: str = "u@example.com") -> dict[str, str]:
    r = client.post("/auth/register", json={"email": email, "password": "password123"})
    assert r.status_code in (200, 409)
    if r.status_code == 409:
        r = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_analyze_requires_auth(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    response = client.post("/analyze", files={"video": ("run.mp4", b"fake", "video/mp4")})
    assert response.status_code == 401


def test_analyze_minimal_does_not_store_or_advertise_media_urls(monkeypatch, tmp_path):
    """Minimal profile deletes poses + annotated MP4; API must not point clients at /download or /poses."""
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)
    h = _auth_header(client, "minimal-profile@example.com")

    response = client.post(
        "/analyze",
        headers=h,
        data={"consent_accepted": "true", "storage_profile": "minimal"},
        files={"video": ("run.mp4", b"x", "video/mp4")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["annotated_video"] is None
    assert payload["poses"] is None

    aid = payload["id"]
    assert not (tmp_path / "results" / f"{aid}_annotated.mp4").exists()
    assert not (tmp_path / "results" / f"{aid}_poses.json").exists()

    again = client.get(f"/results/{aid}", headers=h)
    assert again.status_code == 200
    assert again.json()["annotated_video"] is None

    assert client.get(f"/download/{aid}", headers=h).status_code == 404
    assert client.get(f"/poses/{aid}", headers=h).status_code == 404


def test_analyze_and_get_results_user_scoped(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    monkeypatch.setattr(app_module, "_detector", _OkDetector())
    monkeypatch.setattr(app_module, "_extractor", _OkExtractor())
    monkeypatch.setattr(app_module, "create_annotated_video", _fake_annotator)
    h1 = _auth_header(client, "a@example.com")
    h2 = _auth_header(client, "b@example.com")

    response = client.post(
        "/analyze",
        headers=h1,
        data={"consent_accepted": "true", "storage_profile": "standard"},
        files={"video": ("run.mp4", b"fake-content", "video/mp4")},
    )
    assert response.status_code == 200
    analysis_id = response.json()["id"]

    own = client.get(f"/results/{analysis_id}", headers=h1)
    other = client.get(f"/results/{analysis_id}", headers=h2)
    assert own.status_code == 200
    assert other.status_code == 404


def test_health_endpoint_returns_ok(monkeypatch, tmp_path):
    client = _client_with_tmpdirs(monkeypatch, tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
