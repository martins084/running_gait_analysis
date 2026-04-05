"""
app.py — FastAPI server for the Running Gait Analysis system.

Endpoints:
  POST /analyze        Upload a video → get metrics + annotated video
  GET  /results/{id}   Retrieve a previous analysis result
  GET  /poses/{id}     Full pose JSON (per-frame landmarks; large)
  GET  /download/{id}  Download the annotated video
  GET  /health         Liveness check
"""

import json
import os
import shutil
import sys
import uuid
import asyncio
import sqlite3
from pathlib import Path

# Windows: noklusējuma Proactor ciklā bieži parādās ConnectionResetError, kad pārlūks
# pārtrauc HTTP Range straumi (<video> meklēšana / jauna pieprasījuma sākšana).
# Selector politika parasti nerada šo asyncio callback trace — API darbība paliek pareiza.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from core.pose_detection import PoseDetector
from core.feature_extractor import FeatureExtractor
from utils.visualization import create_annotated_video

# ── App setup ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Running Gait Analysis API",
    version="0.1.0",
    description="Upload a running video and receive biomechanical analysis.",
)

# CORS: allow browser SPA (Vite dev on :5173) and configurable production origins.
# Set CORS_ORIGINS="https://yourapp.com,http://localhost:5173" or leave unset for defaults.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
_cors_raw = os.environ.get("CORS_ORIGINS", _default_origins).strip()
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Dev: any localhost / 127.0.0.1 port (when SPA uses VITE_API_BASE_URL instead of /api proxy).
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state / directories (project root, not process cwd) ───────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = _PROJECT_ROOT / "uploads"
RESULTS_DIR = _PROJECT_ROOT / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Instantiate once and reuse
_detector = PoseDetector()
_extractor = FeatureExtractor(fps=30)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB safety cap
ANALYZE_TIMEOUT_SEC = 300  # 5 minutes for one request


def _db_path() -> Path:
    """Resolve SQLite DB path from current results directory."""
    return RESULTS_DIR / "analysis_results.db"


def _init_db() -> None:
    """Create SQLite schema if not present."""
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _save_result_to_db(result: dict) -> None:
    """Persist full result payload as JSON for robust schema evolution."""
    _init_db()
    payload = json.dumps(result, default=str)
    with sqlite3.connect(_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO analyses (id, status, result_json)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                result_json=excluded.result_json
            """,
            (result.get("id"), result.get("status", "completed"), payload),
        )
        conn.commit()


def _load_result_from_db(analysis_id: str) -> dict | None:
    """Fetch a result from SQLite, returning None when missing."""
    _init_db()
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    """
    Upload a video file for gait analysis.

    Returns JSON with an analysis ID, computed features, and a link
    to the annotated video.
    """
    # Validate filename / extension first (cheap fast-fail).
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename in upload.")
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Use one of {ALLOWED_EXTENSIONS}.",
        )

    analysis_id = str(uuid.uuid4())

    # Persist upload with a strict size limit to avoid memory/disk abuse.
    video_path = UPLOAD_DIR / f"{analysis_id}_input{ext}"
    total_bytes = 0
    with open(video_path, "wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large (> {MAX_UPLOAD_BYTES} bytes).",
                )
            f.write(chunk)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        poses_json = RESULTS_DIR / f"{analysis_id}_poses.json"
        annotated_path = RESULTS_DIR / f"{analysis_id}_annotated.mp4"

        async def _run_pipeline():
            # Run CPU-heavy pipeline in worker thread so timeout can be enforced.
            await asyncio.to_thread(_detector.process_video, str(video_path), str(poses_json))
            features_local = await asyncio.to_thread(_extractor.extract_all_features, str(poses_json))
            await asyncio.to_thread(create_annotated_video, str(video_path), str(poses_json), str(annotated_path))
            return features_local

        try:
            features = await asyncio.wait_for(_run_pipeline(), timeout=ANALYZE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Analysis timed out after {ANALYZE_TIMEOUT_SEC} seconds.",
            )
        except Exception as exc:
            # Surface as pipeline failure (bad/corrupt video or processing failure).
            raise HTTPException(status_code=422, detail=f"Pose analysis failed: {exc}")

        # Ensure expected artifacts exist before returning success.
        if not poses_json.exists():
            raise HTTPException(status_code=422, detail="Pose detection failed: no pose output generated.")
        if not annotated_path.exists():
            raise HTTPException(status_code=422, detail="Annotation failed: no output video generated.")

        # Package result
        result = {
            "id": analysis_id,
            "status": "completed",
            "features": features,
            "annotated_video": f"/download/{analysis_id}",
            "poses": f"/results/{analysis_id}",
        }

        # Save to disk
        with open(RESULTS_DIR / f"{analysis_id}_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        _save_result_to_db(result)

        return result

    except HTTPException:
        # Re-raise known API errors untouched.
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}")
    finally:
        # Release upload handle and cleanup partial upload file.
        await video.close()
        if video_path.exists():
            video_path.unlink(missing_ok=True)


@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Retrieve a previously computed analysis."""
    # Primary source: SQLite persistent store.
    db_result = _load_result_from_db(analysis_id)
    if db_result is not None:
        return db_result

    # Backward compatibility fallback: legacy JSON-only result file.
    result_file = RESULTS_DIR / f"{analysis_id}_result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found.")

    with open(result_file) as f:
        return json.load(f)


@app.get("/download/{analysis_id}")
async def download_video(analysis_id: str):
    """Download the annotated video for a given analysis."""
    video_file = RESULTS_DIR / f"{analysis_id}_annotated.mp4"
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found.")

    # Range + inline disposition help browsers stream MP4 in <video> (seek / progressive load).
    return FileResponse(
        str(video_file),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": 'inline; filename="annotated.mp4"',
        },
    )


@app.get("/poses/{analysis_id}")
async def get_poses_json(analysis_id: str):
    """
    Serve the raw BlazePose output JSON (per-frame landmarks).

    Large payload — use for offline tools or client-side COM when needed.
    """
    poses_file = RESULTS_DIR / f"{analysis_id}_poses.json"
    if not poses_file.exists():
        raise HTTPException(status_code=404, detail="Pose data not found.")

    return FileResponse(
        str(poses_file),
        media_type="application/json",
        filename=f"{analysis_id}_poses.json",
    )


# ── Run directly ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
