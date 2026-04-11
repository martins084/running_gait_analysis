# Project files — short guide

One-line notes for main source and config files in **running_gait_analysis**. Auto-generated or vendor dirs (`node_modules`, `venv`, `__pycache__`, lockfile internals) are omitted unless noted.

---

## Repository root

| File | What it is |
|------|------------|
| `README.md` | Project overview, setup, run API/UI, Docker, testing. |
| `requirements.txt` | Python dependency pins for the pipeline and API. |
| `setup.py` | Installs the package in editable mode (`pip install -e .`). |
| `Dockerfile` | Container image that runs the FastAPI app. |
| `postman_collection.json` | Postman requests for health → analyze → results → download. |

---

## `core/` — main processing

| File | What it is |
|------|------------|
| `core/__init__.py` | Re-exports `PoseDetector`, `FeatureExtractor`, `GaitAnalyzer`, `VideoProcessor`. |
| `core/pose_detection.py` | MediaPipe BlazePose wrapper: per-frame / full-video pose → JSON. |
| `core/feature_extractor.py` | Smoothing, joint angles, stride/cadence heuristics, symmetry, vertical oscillation. |
| `core/gait_analyzer.py` | Orchestrates pose JSON → features → `*_result.json` (library entry point). |
| `core/video_processor.py` | Utility: extract video frames to images (side path, not main API flow). |

---

## `api/`

| File | What it is |
|------|------------|
| `api/app.py` | FastAPI server: upload video, run pipeline, SQLite + JSON + annotated MP4 URLs. |

---

## `utils/`

| File | What it is |
|------|------------|
| `utils/pose_foot_sanitize.py` | Post-process pose sequences: plausible leg/foot geometry, less foot-swap noise. |
| `utils/visualization.py` | Draw skeleton + optional COM on frames; write annotated MP4; optional ffmpeg H.264 pass. |
| `utils/com_segmentation.py` | Segment-weighted 2D COM from landmarks (used in viz / features context). |
| `utils/feature_summary_csv.py` | Build one CSV summary row per clip; append to `features_summary.csv`. |

---

## `models/`

| File | What it is |
|------|------------|
| `models/__init__.py` | Re-exports `GaitPhaseClassifier`, `GaitAnomalyDetector`. |
| `models/gait_classifier.py` | PyTorch class definitions (CNN+LSTM-style phase classifier, LSTM anomaly detector). |
| `models/pretrained/pose_landmarker_full.task` | MediaPipe BlazePose landmarker asset (required at runtime; may be large). |

---

## `scripts/` — CLI tools

| File | What it is |
|------|------------|
| `scripts/process_videos.py` | Batch: pose + features + annotated video + summary CSV row per file. |
| `scripts/batch_feature_export.py` | Recompute features from existing `*_poses.json` without re-running MediaPipe. |
| `scripts/download_sample_videos.py` | Helper to fetch sample clips (URLs / paths as implemented). |
| `scripts/verify_coco_setup.py` | Validates COCO annotation JSON structure (optional benchmark context). |
| `scripts/evaluate_ground_contact.py` | Compare predicted vs labeled ground-contact frames → metrics CSV. |
| `scripts/validate_features.py` | Compare manual angle labels to exported per-frame features CSV. |
| `scripts/robustness_batch.py` | Batch robustness / quality metrics over a folder of clips → CSV. |

---

## `tests/` — automated tests

| File | What it is |
|------|------------|
| `tests/test_api_app.py` | FastAPI routes, uploads, DB, timeouts, downloads (mocked where needed). |
| `tests/test_pose_detection.py` | Pose detector behavior / geometry on controlled inputs. |
| `tests/test_feature_extraction.py` | Feature math and stride logic on synthetic landmarks. |
| `tests/test_pose_foot_sanitize.py` | Foot sanitization clamps and continuity. |
| `tests/test_visualization.py` | Drawing / annotated video helpers with synthetic data. |
| `tests/test_video_processor.py` | Frame extraction utility edge cases. |
| `tests/test_batch_feature_export.py` | Batch export script integration-style tests. |
| `tests/test_evaluate_ground_contact.py` | Ground-contact evaluator script tests. |
| `tests/test_validate_features.py` | Angle validation script tests. |
| `tests/test_scripts_edge_cases.py` | Misc. script edge cases. |

---

## `config/`

| File | What it is |
|------|------------|
| `config/dataset_manifest.json` | Tracks dataset paths, status, and thesis-related notes. |

---

## `frontend/` — web UI

| File | What it is |
|------|------------|
| `frontend/package.json` | npm scripts and dependencies (React, Vite, Recharts, fonts). |
| `frontend/package-lock.json` | Locked dependency tree for reproducible `npm ci`. |
| `frontend/vite.config.ts` | Vite build config, dev server, optional API proxy. |
| `frontend/tsconfig.json` / `tsconfig.node.json` | TypeScript compiler options. |
| `frontend/index.html` | SPA entry HTML. |
| `frontend/.env.example` | Example env vars (`VITE_API_BASE_URL`, optional dev mock flags). |
| `frontend/src/main.tsx` | React root mount. |
| `frontend/src/App.tsx` | Top-level routes and layout wiring. |
| `frontend/src/vite-env.d.ts` | Vite/client type references. |
| `frontend/src/pages/AnalyzePage.tsx` | Upload page. |
| `frontend/src/pages/ResultsPage.tsx` | Metrics, charts, video after analysis. |
| `frontend/src/components/AppShell.tsx` | App chrome: nav, theme, API health, locale. |
| `frontend/src/components/UploadZone.tsx` | Drag-and-drop / file pick upload to API. |
| `frontend/src/components/VideoPanel.tsx` | Video playback + canvas overlay (e.g. COM sync). |
| `frontend/src/components/MetricsGrid.tsx` | Summary metric cards. |
| `frontend/src/components/JointAngleChart.tsx` | Recharts plot for hip/knee angles. |
| `frontend/src/components/SymmetryChart.tsx` | Symmetry time series chart. |
| `frontend/src/components/ComOscillationChart.tsx` | Vertical oscillation / COM-related chart. |
| `frontend/src/components/MlPhasePlaceholder.tsx` | Placeholder UI for future gait-phase timeline. |
| `frontend/src/hooks/useVideoFrameSync.ts` | Syncs video current time to frame index for overlays/charts. |
| `frontend/src/api/client.ts` | Fetch helpers for `/analyze`, `/results`, `/download`, `/poses`. |
| `frontend/src/api/types.ts` | TypeScript types for API payloads. |
| `frontend/src/i18n/index.ts` | i18n message loader / setup. |
| `frontend/src/i18n/LocaleProvider.tsx` | React context for locale switching. |
| `frontend/src/i18n/locales/en.ts` | English UI strings. |
| `frontend/src/i18n/locales/lv.ts` | Latvian UI strings. |
| `frontend/src/utils/format.ts` | Number/time formatting helpers. |
| `frontend/src/utils/downsample.ts` | Downsample long series for charts. |
| `frontend/src/utils/videoContentRect.ts` | Map video element layout to content rect for canvas. |
| `frontend/src/utils/comSegmentation.ts` | Browser-side COM helper aligned with backend logic. |
| `frontend/src/styles/base.css` | Global base styles. |
| `frontend/src/styles/components.css` | Component-level styles. |
| `frontend/src/styles/tokens.css` | Design tokens (colors, spacing). |

---

## Runtime / generated folders (not source, but useful)

| Path | What it is |
|------|------------|
| `uploads/` | Temporary uploaded videos (API). |
| `results/` | `*_poses.json`, `*_result.json`, `*_annotated.mp4`, `features_summary.csv`, SQLite DB. |
| `data/` | Raw/processed datasets (often git-ignored). |

---

*Generated as a quick index; for behavior details see `README.md` and code comments in each module.*
