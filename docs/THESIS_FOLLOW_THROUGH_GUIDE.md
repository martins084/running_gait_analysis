# Thesis follow-through guide (single checklist)

One place to execute everything from **Plāns** (if you keep that folder), **TASK_REFERENCE**, and the **manual thesis tasks** plan. Work **top to bottom**; skip blocks marked *optional* if your thesis does not need them.

**Progress hygiene:** After each part, tick items here. If your repo includes [`Plāns/TASK_REFERENCE.md`](../Plāns/TASK_REFERENCE.md), sync checkboxes and “Last Updated” there too *(that path may be missing if `Plāns/` lives only on another machine or drive)*.

**Docs index (this folder):** see [Appendix D — Documentation map](#appendix-d--documentation-map).

### Where you are (update this as you go)

| Block | Status |
|-------|--------|
| **Part 0** — Environment | **Done** |
| **Part 1** — Data | §1.1 COCO verified; §1.3 **MP4s present** in `sample_videos/`; §1.4 sources Excel **deferred** until all clips collected |
| **Part 2** — Pose QA | **Stock clips done** — All **three** rows in `MediaPipe_Accuracy_Notes.md` filled (beach, urban 4K, urban 1080p) + **thesis limitations** paragraph drafted. Keep **Clip index** handy so `76555842` vs `d56ae039` is not mixed up. |
| Part 3 onward | Not started |

*Last update: 2026-04-02 — Guide synced with repo: `frontend/`, `batch_feature_export.py`, test count, API schema doc, Plāns path caveat; redundant `CURSOR_INDEPENDENT_TASKS.md` folded into Appendix D and removed.*

---

## How to use this document

| Symbol | Meaning |
|--------|--------|
| `[ ]` | Not done |
| `[x]` | Done |
| *optional* | Only if your supervisor / thesis scope requires it |

**Project root:** folder that contains `api/`, `core/`, `frontend/`, `scripts/`, `README.md`.  
**Windows:** activate venv first: `venv\Scripts\activate`  
**Run commands from project root** unless noted otherwise.

---

## Part 0 — Environment (once)

- [x] Python 3.10+ installed *(verified: 3.12 in venv)*  
- [x] `python -m venv venv` and `venv\Scripts\activate` *(venv present)*  
- [x] `pip install -r requirements.txt`  
- [x] `pip install -e .`  
- [x] `pytest tests/ -q` passes *(re-run after big changes; **46 passed** as of 2026-04-02)*  

**You:** From now on, before working, run `venv\Scripts\activate` once per terminal session.

---

## Part 1 — Data on disk + sources table (Plāns Phase 1 / Step A)

### 1.1 COCO annotations (optional — for benchmark-style validation)

- [x] Annotations unzipped under `data/raw/coco/annotations/`  
  - Expected files include at least: `instances_train2017.json`, `instances_val2017.json`  
  - **Person keypoints:** `person_keypoints_train2017.json`, `person_keypoints_val2017.json` (useful if you compare to COCO-style keypoints)  
- [ ] *If* you need image-based tests: download **val2017** images from the official COCO site and unzip to `data/raw/coco/val2017/` (folder of `.jpg` next to `annotations/`)  
- [x] Run: `python scripts/verify_coco_setup.py`  
  *(Checks files exist; validates structure using **val** JSON only so it finishes in ~20s, not minutes.)*  
- [x] Update `config/dataset_manifest.json` — created/updated with `coco_annotations: verified`  

### 1.2 Human3.6M (optional)

- [ ] Either: submit access request early **or** write in the thesis that validation uses **your videos + COCO** (not Human3.6M)  

### 1.3 Running videos for your experiments

- [x] MP4s in `data/raw/sample_videos/` *(3 stock clips; YouTube downloads removed 2026-03-28)*  

### 1.4 Sources & ethics table (required for credible thesis)

**Deferred** until your full clip list is final. Then create a table (Word / Excel / Notion) with **every** clip you analyze or show:

| File name | URL (if any) | Author / channel | License or why use is OK | How used in thesis (analysis only, figure, appendix) |
|-----------|----------------|------------------|---------------------------|--------------------------------------------------------|

- [ ] Table complete for all clips *(do this when collection is finished)*  

---

## Part 2 — Pose extraction + visual QA (Plāns Phase 2 / Step A)

### 2.1 Generate poses and/or full pipeline outputs

Pick **one** or combine:

**Option A — Batch pose JSON only** — *not in this repo.* There is **no** `scripts/batch_pose_extraction.py`; use Option B or C.

**Option B — Per-video pipeline** (always available if repo matches README):  

```bash
python scripts/process_videos.py path\to\video1.mp4 path\to\video2.mp4
python scripts/process_videos.py --input-dir data/raw/sample_videos --skip-existing
```  

The second line processes every `.mp4` in that folder; **`--skip-existing`** skips names already listed in `results/features_summary.csv`. Writes under `results/`: poses JSON, features, annotated MP4, summary.

**Option C — HTTP API**  

```bash
python api/app.py
```  

Open `http://localhost:8000/docs` → `POST /analyze` with each MP4; download annotated video from `GET /download/{id}`.

**Option D — Web UI** *(optional demo)*  

```bash
# Terminal 1: API (above). Terminal 2:
cd frontend
npm install
npm run dev
# → http://localhost:5173 — upload, results, charts, inline video (H.264 / ffmpeg helps; see README).
```

- [ ] Every important clip has been processed at least once  
- [ ] You have annotated MP4s to **watch** for each clip  

### 2.2 Visual inspection (you — cannot be automated)

For **each** video, note: camera view (side / front / other), failures (floating ankles, wrong knee, occlusion, background confusion).

- [x] Fill `docs/MediaPipe_Accuracy_Notes.md` for the **last** stock clip (`8402092…` / `d56ae039-ded`) — **done** (see checklist table + clip index)  

---

## Part 3 — Features + numeric validation (Plāns Phase 3 / Tier B)

### 3.1 Batch features from pose JSON

**Script present:** `scripts/batch_feature_export.py`.

```bash
python scripts/batch_feature_export.py
```  

Output typically under `data/processed/features/` (`*_features.json`, `*_features_frames.csv`, summary CSV). Scans `data/processed/poses/` or falls back to `results/*_poses.json` — see script `--help`.  
You can also use **`process_videos.py`** or **API** output — features are inside the result JSON from those paths.

- [ ] You have per-video or per-clip **feature** outputs you can cite  

### 3.2 Manual vs computed angles (small study)

- [ ] Copy `config/templates/manual_angle_validation_template.csv` to a working path (e.g. `data/processed/labels/my_angles.csv`) — create folders as needed  
- [ ] Choose a **small** set of frames (e.g. 3 videos × 5 frames)  
- [ ] Measure **one** joint angle manually from a still frame; copy **computed** angle from your feature CSV / JSON into the same row  
- [ ] Run:  
  `python scripts/validate_features.py --manual-csv path\to\my_angles.csv` *(see script `--help` for feature source / output dir)*  
- [ ] Record MAE/RMSE (or a short “manual vs pipeline” paragraph) in the thesis  

### 3.3 Literature reference ranges (*optional*, strong for discussion)

- [ ] Fill `docs/Running_Biomechanics_Reference.md` with typical cadence / angle ranges + **citations** *(file exists as a placeholder outline — replace `_TBD_` rows)*  

---

## Part 4 — Ground contact (*optional*)

Only if you claim sensitivity/specificity or similar.

- [ ] Label frames: copy `config/templates/ground_contact_labels_template.csv` → fill `is_contact` for 1–2 short clips  
- [ ] Run **`scripts/evaluate_ground_contact.py`** with your label CSV and matching `*_poses.json` *(predictions are computed inside the script from poses; no separate predictions CSV required)*  

```bash
python scripts/evaluate_ground_contact.py ^
  --labels-csv data/processed/labels/ground_contact_clip1.csv ^
  --poses-json results\YOUR_ID_poses.json ^
  --output-dir data/processed/validation
```

- [ ] Interpret metrics in prose (or state clearly that contact was **not** validated)  

---

## Part 5 — ML direction (Plāns Phase 4 — decisions before heavy training)

- [ ] Edit `docs/Gait_Phase_Definition.md`: number of phases (e.g. 3 vs 4), plain-language boundaries  
- [ ] Edit `docs/Training_Data_Strategy.md`: where labels come from (heuristic / manual / external dataset)  
- [ ] Short note on augmentation: create `docs/Data_Augmentation_Strategy.md` or one subsection in Training_Data_Strategy (flip, noise, time-warp — yes/no and why)  
- [ ] *Optional:* Weights & Biases (or CSV log) for experiments  

Training code (dataset class, train loop, checkpoints) comes **after** the above decisions.

---

## Part 6 — API + demo sanity check (Plāns Phase 5)

- [ ] `python api/app.py`  
- [ ] Swagger UI or Postman (`postman_collection.json`): Health → Analyze → Results → Download  
- [ ] *Optional:* Web UI — `cd frontend && npm run dev` (see README **Web frontend**)  
- [ ] Notes: what works, what confuses you, what to show the supervisor  
- [ ] Use [`docs/API_Response_Schema.md`](API_Response_Schema.md) when writing Methods / Results *(already written — cite field meanings and units; no need to recreate)*  

---

## Part 7 — Robustness + final write-up (Plāns Phase 6)

### 7.1 Small robustness set

- [ ] Curate a few clips: e.g. low light, different angle, different speed → folders under e.g. `data/test/robustness/<condition>/`  
- [ ] Add pose JSONs there (or run pipeline first), then: `python scripts/robustness_batch.py` *(writes `data/processed/robustness/robustness_metrics.csv`)*  
- [ ] CSV: `video_name`, `condition`, `pose_quality` (good/fair/poor), `notes` — filled **by you** after watching outputs  

### 7.2 Validation report + success criteria

- [ ] Outline: Methods → Pose QA → Features → Contact (if any) → Model (if any) → Robustness → **Limitations**  
- [ ] Use [`Plāns/Testing_and_Validation_Guide.md`](../Plāns/Testing_and_Validation_Guide.md) as a **menu** (you do not need every test) — *skip if `Plāns/` is not in this clone*  
- [ ] Update the success-metrics table in `Plāns/TASK_REFERENCE.md` to **measured** values (not only original targets) — *if that file exists*  

---

## Appendix A — Command cheat sheet

| Goal | Command |
|------|--------|
| Tests | `pytest tests/ -q` |
| API | `python api/app.py` → `http://localhost:8000/docs` |
| Web UI | `cd frontend && npm install && npm run dev` → `http://localhost:5173` |
| Process listed MP4s (pose + features + annotated video) | `python scripts/process_videos.py clip1.mp4 clip2.mp4` |
| Batch folder (skip already in summary CSV) | `python scripts/process_videos.py --input-dir data/raw/sample_videos --skip-existing` |
| Batch features from pose JSON | `python scripts/batch_feature_export.py` |
| Download URLs from a text file | `python scripts/download_sample_videos.py --urls-file "path\to\links.txt"` |
| Robustness batch metrics | `python scripts/robustness_batch.py` |

---

## Appendix B — Plāns folder map (deeper reading)

*Only if `Plāns/` exists next to this repo (same parent as `running_gait_analysis/` or as your university package).*

| File | Use when |
|------|----------|
| `Plāns/README_Master_Index.md` | Big-picture 17-week roadmap |
| `Plāns/TASK_REFERENCE.md` | Master checkboxes — keep in sync with this guide |
| `Plāns/START_HERE.txt` | Quick package overview |
| `Plāns/files/STRATEGIC_PLAN_HUMAN_vs_CURSOR.md` | What you do vs what to automate |
| `Plāns/Data_Collection_Guide.md` | Dataset download details |
| `Plāns/Testing_and_Validation_Guide.md` | Validation methodology ideas |
| `Plāns/Technical_Setup_Guide.md` | Code-oriented reference |

---

## Appendix C — Manual COCO JSON sanity check (no script)

1. Open `data/raw/coco/annotations/instances_val2017.json` in a text editor (or Python `json.load` once).  
2. Confirm keys exist: `images`, `annotations`, `categories` (exact names as in official COCO).  
3. If `person_keypoints_val2017.json` exists, confirm it has `annotations` with keypoint fields — that file is for **person keypoint** tasks, not the same as MediaPipe’s 33 landmarks.

---

## Appendix D — Documentation map

| Document | Role |
|----------|------|
| **This file** | Master thesis execution checklist |
| [`SYSTEM_OVERVIEW_AND_CURRENT_STATE.md`](SYSTEM_OVERVIEW_AND_CURRENT_STATE.md) | Architecture, module map, pipeline, honest “what exists” |
| [`API_Response_Schema.md`](API_Response_Schema.md) | REST `/analyze` JSON shape, units, errors |
| [`PALAIST_PILNU_APLIKACIJU.md`](PALAIST_PILNU_APLIKACIJU.md) | API + frontend palaišana, atjaunināšana pēc `git pull` |
| [`MediaPipe_Accuracy_Notes.md`](MediaPipe_Accuracy_Notes.md) | Visual QA log + clip index |
| [`Gait_Phase_Definition.md`](Gait_Phase_Definition.md) | ML phase definitions *(you fill)* |
| [`Training_Data_Strategy.md`](Training_Data_Strategy.md) | Labels, splits, augmentation plan *(you fill)* |
| [`Running_Biomechanics_Reference.md`](Running_Biomechanics_Reference.md) | Literature ranges for discussion *(you fill)* |

**Historical note:** A long Cursor task specification + execution log (`CURSOR_INDEPENDENT_TASKS.md`) was **removed** 2026-04-02 as redundant with this guide + `SYSTEM_OVERVIEW`. Landed engineering (batch feature export, validate_features, evaluate_ground_contact, robustness_batch, API schema doc, extra tests, stub thesis docs) is reflected in the repo and in `SYSTEM_OVERVIEW` §2–§7.

---

*End of guide — work Part 0 → Part 7 in order; use appendices for commands and cross-links.*
