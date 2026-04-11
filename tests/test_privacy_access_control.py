from __future__ import annotations

from fastapi.testclient import TestClient

import api.app as app_module


def _auth_header(client: TestClient, email: str) -> dict[str, str]:
    r = client.post("/auth/register", json={"email": email, "password": "password123"})
    if r.status_code == 409:
        r = client.post("/auth/login", json={"email": email, "password": "password123"})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_protected_endpoints_require_auth():
    client = TestClient(app_module.app)
    assert client.get("/results/abc").status_code == 401
    assert client.get("/poses/abc").status_code == 401
    assert client.get("/download/abc").status_code == 401


def test_delete_non_owned_analysis_returns_404():
    client = TestClient(app_module.app)
    h = _auth_header(client, "owner-check@example.com")
    r = client.request("DELETE", "/analyses/not-found", headers=h, json={"reason": "test"})
    assert r.status_code == 404
