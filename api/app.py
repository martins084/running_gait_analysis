"""
app.py - FastAPI server for Running Gait Analysis with GDPR controls.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.privacy_db import init_db, load_analysis, save_analysis
from api.retention import RetentionPolicy, purge_expired_analyses
from core.feature_extractor import FeatureExtractor
from core.pose_detection import PoseDetector
from utils.visualization import create_annotated_video

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="Running Gait Analysis API",
    version="0.2.0",
    description="Upload a running video and receive biomechanical analysis.",
)

_default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
_cors_raw = os.environ.get("CORS_ORIGINS", _default_origins).strip()
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "uploads"
RESULTS_DIR = _PROJECT_ROOT / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

_detector = PoseDetector()
_extractor = FeatureExtractor(fps=30)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ANALYZE_TIMEOUT_SEC = 300
AUTH_TOKEN_TTL_HOURS = int(os.getenv("AUTH_TOKEN_TTL_HOURS", "24"))
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-only-change-me")
PURGE_INTERVAL_SEC = int(os.getenv("RETENTION_PURGE_INTERVAL_SEC", "3600"))
RETENTION_POLICY = RetentionPolicy(
    keep_minimal_days=int(os.getenv("RETENTION_DAYS_MINIMAL", "14")),
    keep_standard_days=int(os.getenv("RETENTION_DAYS_STANDARD", "30")),
    keep_full_days=int(os.getenv("RETENTION_DAYS_FULL", "90")),
)
CONSENT_VERSION = os.getenv("CONSENT_VERSION", "v1")


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str


class DeleteRequest(BaseModel):
    reason: str = "user_requested"


class DsarRequestBody(BaseModel):
    analysis_id: str | None = None
    reason: str = "data_subject_request"


def _db_path() -> Path:
    return RESULTS_DIR / "analysis_results.db"


def _init_db() -> None:
    init_db(_db_path())


def _hash_password(password: str, *, salt: str | None = None) -> str:
    use_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), use_salt.encode("utf-8"), 120_000)
    return f"{use_salt}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        salt, stored = encoded.split("$", 1)
    except ValueError:
        return False
    got = _hash_password(password, salt=salt).split("$", 1)[1]
    return hmac.compare_digest(got, stored)


def _sign_token(payload_b64: str) -> str:
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")


def _create_token(user_id: str, email: str) -> str:
    exp = int((datetime.now(timezone.utc) + timedelta(hours=AUTH_TOKEN_TTL_HOURS)).timestamp())
    payload = {"uid": user_id, "email": email, "exp": exp}
    payload_raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_raw).decode("utf-8").rstrip("=")
    sig_b64 = _sign_token(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def _decode_token(token: str) -> dict:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        if not hmac.compare_digest(_sign_token(payload_b64), sig_b64):
            raise HTTPException(status_code=401, detail="Invalid token signature.")
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")))
        if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            raise HTTPException(status_code=401, detail="Token expired.")
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token format: {exc}")


async def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    token = authorization[7:].strip()
    payload = _decode_token(token)
    return {"id": payload["uid"], "email": payload["email"]}


def _get_user_by_email(email: str) -> tuple[str, str] | None:
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email.lower(),),
        ).fetchone()
    return row


def _insert_user(email: str, password: str) -> tuple[str, str]:
    _init_db()
    user_id = str(uuid.uuid4())
    password_hash = _hash_password(password)
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
            (user_id, email.lower(), password_hash),
        )
        conn.commit()
    return user_id, email.lower()


def _delete_analysis_owned_by_user(analysis_id: str, user_id: str, reason: str) -> bool:
    _init_db()
    _delete_files_for_analysis(analysis_id)
    with sqlite3.connect(_db_path()) as conn:
        deleted = conn.execute(
            "DELETE FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).rowcount
        if deleted:
            conn.execute(
                "INSERT INTO delete_audit (analysis_id, user_id, reason) VALUES (?, ?, ?)",
                (analysis_id, user_id, reason),
            )
        conn.commit()
    return deleted > 0


def _delete_files_for_analysis(analysis_id: str) -> None:
    for suffix in ("_poses.json", "_annotated.mp4", "_result.json"):
        (RESULTS_DIR / f"{analysis_id}{suffix}").unlink(missing_ok=True)


def _normalize_result_media_urls(analysis_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """
    `annotated_video` / `poses` must match files under RESULTS_DIR.
    Minimal storage profile deletes both after metrics are computed; older rows may still list URLs.
    """
    out = dict(result)
    annotated = RESULTS_DIR / f"{analysis_id}_annotated.mp4"
    poses_file = RESULTS_DIR / f"{analysis_id}_poses.json"
    out["annotated_video"] = f"/download/{analysis_id}" if annotated.exists() else None
    out["poses"] = f"/poses/{analysis_id}" if poses_file.exists() else None
    return out


def _load_owned_result(analysis_id: str, user_id: str) -> dict:
    db_result = load_analysis(_db_path(), analysis_id, user_id)
    if db_result is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return _normalize_result_media_urls(analysis_id, db_result)


def _save_dsar_request(user_id: str, req_type: str, status: str, payload: dict) -> str:
    _init_db()
    req_id = str(uuid.uuid4())
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO dsar_requests (id, user_id, type, status, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (req_id, user_id, req_type, status, json.dumps(payload)),
        )
        conn.commit()
    return req_id


def _update_dsar_status(req_id: str, status: str, payload: dict) -> None:
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            UPDATE dsar_requests
            SET status = ?, payload_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, json.dumps(payload), req_id),
        )
        conn.commit()


@app.on_event("startup")
async def on_startup() -> None:
    _init_db()
    app.state.retention_task = asyncio.create_task(_retention_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    t = getattr(app.state, "retention_task", None)
    if t:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t


async def _retention_loop() -> None:
    while True:
        await asyncio.sleep(PURGE_INTERVAL_SEC)
        await asyncio.to_thread(purge_expired_analyses, _db_path(), RESULTS_DIR, RETENTION_POLICY)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
async def register(payload: RegisterRequest):
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if _get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="Email already exists.")
    user_id, email = _insert_user(payload.email, payload.password)
    return {"token": _create_token(user_id, email), "user_id": user_id, "email": email}


@app.post("/auth/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    row = _get_user_by_email(payload.email)
    if not row or not _verify_password(payload.password, row[1]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user_id = row[0]
    return {"token": _create_token(user_id, payload.email.lower()), "user_id": user_id, "email": payload.email.lower()}


@app.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    storage_profile: Literal["minimal", "standard", "full"] = Form("minimal"),
    consent_accepted: bool = Form(False),
    consent_version: str = Form(CONSENT_VERSION),
    consent_timestamp: str = Form(""),
    consent_locale: str = Form("en"),
    current_user: dict = Depends(get_current_user),
):
    if not consent_accepted:
        raise HTTPException(status_code=400, detail="Consent is required before upload.")
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'. Use one of {ALLOWED_EXTENSIONS}.")

    analysis_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{analysis_id}_input{ext}"
    total_bytes = 0
    with open(video_path, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).")
            f.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    poses_json = RESULTS_DIR / f"{analysis_id}_poses.json"
    annotated_path = RESULTS_DIR / f"{analysis_id}_annotated.mp4"
    try:
        async def _run_pipeline():
            await asyncio.to_thread(_detector.process_video, str(video_path), str(poses_json))
            features_local = await asyncio.to_thread(_extractor.extract_all_features, str(poses_json))
            if storage_profile in {"standard", "full"}:
                await asyncio.to_thread(create_annotated_video, str(video_path), str(poses_json), str(annotated_path))
            return features_local

        try:
            features = await asyncio.wait_for(_run_pipeline(), timeout=ANALYZE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail=f"Analysis timed out after {ANALYZE_TIMEOUT_SEC} seconds.")
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Pose analysis failed: {exc}")

        if not poses_json.exists():
            raise HTTPException(status_code=422, detail="Pose detection failed: no pose output generated.")

        if storage_profile == "minimal":
            # Privacy-by-default: keep only derived metrics in minimal profile.
            poses_json.unlink(missing_ok=True)
            annotated_path.unlink(missing_ok=True)

        # Only advertise endpoints when files still exist (standard/full keep MP4 + poses JSON).
        if storage_profile == "minimal":
            media_video: str | None = None
            media_poses: str | None = None
        else:
            media_video = f"/download/{analysis_id}"
            media_poses = f"/poses/{analysis_id}"

        result = {
            "id": analysis_id,
            "status": "completed",
            "features": features,
            "annotated_video": media_video,
            "poses": media_poses,
            "storage_profile": storage_profile,
        }
        with open(RESULTS_DIR / f"{analysis_id}_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        save_analysis(
            _db_path(),
            analysis_id=analysis_id,
            user_id=current_user["id"],
            status="completed",
            result=result,
            storage_profile=storage_profile,
            consent_version=consent_version,
            consent_timestamp=consent_timestamp or datetime.now(timezone.utc).isoformat(),
            consent_locale=consent_locale,
        )
        return result
    finally:
        await video.close()
        video_path.unlink(missing_ok=True)


@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str, current_user: dict = Depends(get_current_user)):
    return _load_owned_result(analysis_id, current_user["id"])


@app.get("/download/{analysis_id}")
async def download_video(analysis_id: str, current_user: dict = Depends(get_current_user)):
    _load_owned_result(analysis_id, current_user["id"])
    video_file = RESULTS_DIR / f"{analysis_id}_annotated.mp4"
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found.")
    return FileResponse(
        str(video_file),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Disposition": 'inline; filename="annotated.mp4"'},
    )


@app.get("/poses/{analysis_id}")
async def get_poses_json(analysis_id: str, current_user: dict = Depends(get_current_user)):
    _load_owned_result(analysis_id, current_user["id"])
    poses_file = RESULTS_DIR / f"{analysis_id}_poses.json"
    if not poses_file.exists():
        raise HTTPException(status_code=404, detail="Pose data not found.")
    return FileResponse(str(poses_file), media_type="application/json", filename=f"{analysis_id}_poses.json")


@app.delete("/analyses/{analysis_id}")
async def delete_analysis(analysis_id: str, body: DeleteRequest, current_user: dict = Depends(get_current_user)):
    deleted = _delete_analysis_owned_by_user(analysis_id, current_user["id"], body.reason)
    if not deleted:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return {"status": "deleted", "id": analysis_id}


@app.post("/dsar/export")
async def dsar_export(body: DsarRequestBody, current_user: dict = Depends(get_current_user)):
    req_id = _save_dsar_request(current_user["id"], "export", "processing", body.model_dump())
    with sqlite3.connect(_db_path()) as conn:
        rows = conn.execute(
            "SELECT id, result_json, storage_profile, created_at FROM analyses WHERE user_id = ?",
            (current_user["id"],),
        ).fetchall()
    export_payload = {
        "user_id": current_user["id"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "analyses": [
            {"id": r[0], "storage_profile": r[2], "created_at": r[3], "result": json.loads(r[1])}
            for r in rows
            if body.analysis_id is None or body.analysis_id == r[0]
        ],
    }
    export_file = RESULTS_DIR / f"dsar_export_{req_id}.json"
    export_file.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
    _update_dsar_status(req_id, "completed", {"file": export_file.name, "analysis_id": body.analysis_id})
    return {"request_id": req_id, "status": "completed", "download": f"/dsar/export/{req_id}"}


@app.get("/dsar/export/{request_id}")
async def download_dsar_export(request_id: str, current_user: dict = Depends(get_current_user)):
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT payload_json FROM dsar_requests WHERE id = ? AND user_id = ? AND type = 'export'",
            (request_id, current_user["id"]),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="DSAR export request not found.")
    payload = json.loads(row[0])
    file_name = payload.get("file")
    if not file_name:
        raise HTTPException(status_code=404, detail="Export file missing.")
    path = RESULTS_DIR / file_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Export file missing.")
    return FileResponse(str(path), media_type="application/json", filename=file_name)


@app.post("/dsar/delete")
async def dsar_delete(body: DsarRequestBody, current_user: dict = Depends(get_current_user)):
    req_id = _save_dsar_request(current_user["id"], "delete", "processing", body.model_dump())
    deleted_ids: list[str] = []
    with sqlite3.connect(_db_path()) as conn:
        if body.analysis_id:
            ids = [body.analysis_id]
        else:
            ids = [r[0] for r in conn.execute("SELECT id FROM analyses WHERE user_id = ?", (current_user["id"],)).fetchall()]
    for aid in ids:
        if _delete_analysis_owned_by_user(aid, current_user["id"], body.reason):
            deleted_ids.append(aid)
    _update_dsar_status(req_id, "completed", {"deleted_ids": deleted_ids, "reason": body.reason})
    return {"request_id": req_id, "status": "completed", "deleted_ids": deleted_ids}


@app.get("/dsar/{request_id}")
async def dsar_status(request_id: str, current_user: dict = Depends(get_current_user)):
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT id, type, status, payload_json, created_at, updated_at FROM dsar_requests WHERE id = ? AND user_id = ?",
            (request_id, current_user["id"]),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="DSAR request not found.")
    return {
        "id": row[0],
        "type": row[1],
        "status": row[2],
        "payload": json.loads(row[3]),
        "created_at": row[4],
        "updated_at": row[5],
    }


@app.post("/admin/purge")
async def run_purge_now(current_user: dict = Depends(get_current_user)):
    # Minimal demo guard: only allow account whose email matches env admin.
    admin_email = os.getenv("PRIVACY_ADMIN_EMAIL", "").strip().lower()
    if not admin_email or current_user["email"].lower() != admin_email:
        raise HTTPException(status_code=403, detail="Admin access required.")
    deleted = purge_expired_analyses(_db_path(), RESULTS_DIR, RETENTION_POLICY)
    return {"status": "ok", "deleted": deleted}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
