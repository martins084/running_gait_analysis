from __future__ import annotations

from fastapi.testclient import TestClient

import api.app as app_module


def _auth_header(client: TestClient, email: str) -> dict[str, str]:
    r = client.post("/auth/register", json={"email": email, "password": "password123"})
    if r.status_code == 409:
        r = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_dsar_export_and_status_roundtrip():
    client = TestClient(app_module.app)
    h = _auth_header(client, "dsar@example.com")
    r = client.post("/dsar/export", headers=h, json={"analysis_id": None, "reason": "test"})
    assert r.status_code == 200
    request_id = r.json()["request_id"]
    s = client.get(f"/dsar/{request_id}", headers=h)
    assert s.status_code == 200
    assert s.json()["status"] == "completed"


def test_dsar_delete_roundtrip():
    client = TestClient(app_module.app)
    h = _auth_header(client, "dsar-delete@example.com")
    r = client.post("/dsar/delete", headers=h, json={"analysis_id": None, "reason": "test"})
    assert r.status_code == 200
    assert "request_id" in r.json()
