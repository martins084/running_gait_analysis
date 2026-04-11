# Running Gait Analysis System

Automated running gait analysis using deep learning and computer vision.  
Built as a bachelor's thesis project — analyzes running form from regular video without expensive lab equipment.

**Thesis workflow (single checklist):** see [docs/THESIS_FOLLOW_THROUGH_GUIDE.md](docs/THESIS_FOLLOW_THROUGH_GUIDE.md).

**Pilna aplikācija (API + web, LV):** soļu plāns un atjaunināšana — [docs/PALAIST_PILNU_APLIKACIJU.md](docs/PALAIST_PILNU_APLIKACIJU.md).

## What It Does

- **Pose detection** — MediaPipe BlazePose extracts 33 body landmarks per frame  
- **Feature extraction** — joint angles, stride length, cadence, symmetry, vertical oscillation  
- **Temporal smoothing** — Savitzky–Golay filtering to eliminate pose jitter  
- **ML classification** — CNN-LSTM for gait phase detection (contact / swing / push-off)  
- **REST API** — FastAPI server: upload a video, get metrics back  

## Quick Start

```bash
# Clone and enter project
cd running_gait_analysis

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode (so imports work)
pip install -e .

# Run tests
pytest tests/ -v

# Start the API server
python api/app.py
# → http://localhost:8000/docs
```

## Web frontend

The `frontend/` app is a React + Vite single-page UI for upload, progress, metrics, charts, and annotated video playback. UI fonts ship with the repo via **`@fontsource/*`** (Source Sans 3, Source Serif 4, IBM Plex Mono, with Latin Extended subsets for Latvian) — run **`npm install`** so they bundle correctly. The top bar includes a **light/dark** toggle (saved in `localStorage`) and a small **API health** indicator.

### Privacy and GDPR controls

- User account auth is required for analysis and result access (`/auth/register`, `/auth/login`).
- Analyses are user-scoped; result/pose/download endpoints enforce ownership.
- Consent is required before upload in the web UI.
- Data-subject rights endpoints:
  - `POST /dsar/export`
  - `POST /dsar/delete`
  - `GET /dsar/{request_id}`
- Retention purge is profile-based (`minimal`, `standard`, `full`) and configurable through environment variables.
- See:
  - `docs/privacy-policy.md`
  - `docs/data-retention-policy.md`
  - `docs/dsar-runbook.md`
  - `docs/dpia.md`

**Development (two terminals):**

```bash
# Terminal 1 — API (from project root, venv active)
python api/app.py
# → http://localhost:8000

# Terminal 2 — web UI
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

The browser loads the SPA from port **5173** and calls the API on port **8000**. CORS is enabled by default for local dev (`http://localhost:5173` and `http://127.0.0.1:5173`). To allow a deployed frontend or another origin, set:

```bash
set CORS_ORIGINS=https://your-frontend.example.com,http://localhost:5173
python api/app.py
```

(`CORS_ORIGINS` is a comma-separated list.)

**Configuration:** copy `frontend/.env.example` to `frontend/.env` and set `VITE_API_BASE_URL` if the API is not at `http://localhost:8000`. Optional: `VITE_DEV_ML_MOCK=true` (dev only) previews the gait-phase timeline with mock data until the classifier is wired in the API.

**Production build:** `cd frontend && npm run build` writes static files to `frontend/dist/`. Serve those with any static host and point `VITE_API_BASE_URL` at your public API. Alternatively, serve `dist/` behind the same origin as the API (reverse proxy) so CORS is not required.

**Optional Vite proxy:** in dev, `vite.config.ts` proxies `/api/*` to `http://127.0.0.1:8000` (path stripped). If you use that pattern, point the SPA at the same origin and adjust client paths—by default the app uses `VITE_API_BASE_URL` directly.

**Annotated video in the browser:** HTML5 `<video>` needs **H.264** (or similar) in most browsers. After analysis, the pipeline tries OpenCV codecs in a browser-friendly order; if **`ffmpeg` is on your PATH**, the output is automatically transcoded to **H.264 (yuv420p)** for reliable inline playback. Without `ffmpeg`, install it or rely on **Download MP4** (VLC, etc.) if the preview stays blank.

## Docker

```bash
# Build image
docker build -t running-gait-analysis:latest .

# Run container and expose API
docker run --rm -p 8000:8000 running-gait-analysis:latest

# Open docs
# http://localhost:8000/docs
```

## Postman Collection

- Import `postman_collection.json` into Postman.
- Set collection variables:
  - `baseUrl` (default `http://localhost:8000`)
  - `videoPath` (local path to test video file)
- Run requests in this order:
  1. `Health`
  2. `Analyze Video` (auto-saves `analysisId` from response)
  3. `Get Results by ID`
  4. `Download Annotated Video`

## Testing & Coverage (CI-friendly)

```bash
# Run all tests
python -m pytest tests -q

# Run tests with coverage report
python -m pytest --cov=api --cov=core --cov=utils --cov=models --cov-report=term-missing -q
```

Coverage can be improved further by adding tests for:
- `models/gait_classifier.py`

## Project Structure

```
running_gait_analysis/
├── core/                   Core processing modules
│   ├── pose_detection.py       MediaPipe wrapper
│   ├── feature_extractor.py    Biomechanical feature math
│   ├── gait_analyzer.py        End-to-end pipeline
│   └── video_processor.py      Frame extraction
├── models/                 ML model definitions & weights
├── api/                    FastAPI server
│   └── app.py
├── utils/                  Visualization, helpers
│   └── visualization.py
├── tests/                  Unit & integration tests
├── data/                   Datasets (git-ignored)
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/              Jupyter notebooks for exploration
├── docs/                   Thesis checklist, system overview, API schema
├── frontend/               React + Vite web UI (upload & results)
├── requirements.txt
├── setup.py
└── README.md
```

## Tech Stack

| Layer            | Tool                          |
|------------------|-------------------------------|
| Pose detection   | MediaPipe BlazePose           |
| Deep learning    | PyTorch + Lightning           |
| Computer vision  | OpenCV                        |
| API              | FastAPI + Uvicorn             |
| Web UI           | React + Vite + TypeScript     |
| Data processing  | NumPy, Pandas, SciPy          |
| Visualization    | Matplotlib, Plotly (backend); Recharts (web) |

## Accuracy Targets

| Metric                       | Target        |
|------------------------------|---------------|
| Pose detection (PCK)         | > 0.85        |
| Stride metrics error         | < 5%          |
| Ground contact accuracy      | > 90%         |
| Gait phase classifier        | > 85% val acc |
| Processing time (1-min video)| < 5 minutes   |

## License

Academic use — bachelor's thesis project.
