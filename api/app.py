"""
app.py — FastAPI server for the Running Gait Analysis system.

Endpoints:
  POST /analyze        Upload a video → get metrics + annotated video
  GET  /results/{id}   Retrieve a previous analysis result
  GET  /download/{id}  Download the annotated video
  GET  /health         Liveness check
"""

import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
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

# ── Shared state / directories ──────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Instantiate once and reuse
_detector = PoseDetector()
_extractor = FeatureExtractor(fps=30)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


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
    # Validate extension
    ext = Path(video.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Use one of {ALLOWED_EXTENSIONS}.",
        )

    analysis_id = str(uuid.uuid4())

    # Persist upload
    video_path = UPLOAD_DIR / f"{analysis_id}_input{ext}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        # Pose detection
        poses_json = RESULTS_DIR / f"{analysis_id}_poses.json"
        _detector.process_video(str(video_path), str(poses_json))

        # Feature extraction
        features = _extractor.extract_all_features(str(poses_json))

        # Annotated video
        annotated_path = RESULTS_DIR / f"{analysis_id}_annotated.mp4"
        create_annotated_video(str(video_path), str(poses_json), str(annotated_path))

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

        return result

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/results/{analysis_id}")
async def get_results(analysis_id: str):
    """Retrieve a previously computed analysis."""
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

    return FileResponse(str(video_file), media_type="video/mp4")


# ── Run directly ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
