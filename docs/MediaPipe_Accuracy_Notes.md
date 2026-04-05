# MediaPipe BlazePose — visual QA log

Watch each **annotated** MP4 (skeleton overlay), then fill **your notes** in the last columns.  
Outputs from `scripts/process_videos.py` live under `results/` with pattern `{id}_annotated.mp4`.

**Dataset note:** Low-quality **YouTube downloads** (titles with `[videoId]` in the name) were **removed** from `data/raw/sample_videos/` and their `results/*` outputs deleted. This log now tracks only the **stock** clips kept for analysis.

---

## Clip index (do not mix these up)

Use this table to map **what you see in the video** → **annotated output** → **raw file name** (matches `features_summary.csv`).

| Order | Scene (what you watch) | Annotated MP4 (`results/`) | Raw file (`data/raw/sample_videos/`) |
|-------|-------------------------|----------------------------|----------------------------------------|
| **1** | **Beach** — sand, water, male runner (barefoot) | `5d676297-eda_annotated.mp4` | `8636810-uhd_1440_2560_25fps.mp4` |
| **2** | **Urban** — pavement, buildings, railing, female runner | `76555842-79e_annotated.mp4` | `5310933-uhd_3840_2160_25fps.mp4` |
| **3** | **Urban (bridge / sidewalk)** — visually similar to clip 2 (side profile, same outfit style) but a **different** source: **1080p @ 30 fps**, shorter run; railing with cutouts, **red curb**, brick building / skyline | `d56ae039-ded_annotated.mp4` | `8402092-hd_1920_1080_30fps.mp4` |

Full paths on your machine (same clips as above):

- Beach: `...\results\5d676297-eda_annotated.mp4`
- Urban: `...\results\76555842-79e_annotated.mp4`
- Third: `...\results\d56ae039-ded_annotated.mp4`

---

## Per-video checklist

There are **three** stock videos — **three rows** below (beach, then urban 4K, then urban 1080p). If it looked like “only two videos,” the third row was still here but its **middle columns were empty** until QA text was added for `8402092…` / `d56ae039-ded`.

Rows follow **the same order as the clip index** (beach → urban → third).

| Video file (in `data/raw/sample_videos/`) | Camera view (side / front / other) | What looked good | What failed (ankles, occlusion, etc.) | Example timestamps / frame | Processed? (date + `results/` id if done) |
|--------------------------------------------|--------------------------------------|------------------|----------------------------------------|----------------------------|---------------------------------------------|
| 8636810-uhd_1440_2560_25fps.mp4 | Side | **Beach / sand / waves:** subject separates well from background; full stride is easy to read. Hips–knees–ankles and **barefoot** heel/toe landmarks often **green** and on the feet. Head and shoulders stay coherent; stance vs swing is obvious. | **Red / low-visibility** on **torso**, **elbows** (fast arms), **forearms/wrists**; **red phone armband** adds texture confusion. **Leading leg** sometimes stiff vs real thigh/knee when blurred. **Hands** can splinter into extra dots. **Motion blur** on feet and waves; one reviewed still **portrait** aspect. | **6** mid-stride stills; add MM:SS from player if needed. | 2026-03-28 — `5d676297-eda_annotated.mp4` |
| 5310933-uhd_3840_2160_25fps.mp4 | Side | **Urban / pavement:** side profile with **railing, plain stone wall, and graffiti** backdrops — subject still **pops** from background. **Legs** and **stance foot** usually **green** with heel/toe on the shoe; **swing vs stance** is easy to read in these frames. **Face** (eyes/nose/profile) and **torso box** mostly stable. | **Arm/hand confidence varies by frame (not only the far arm):** **red** dots seen on **camera-near elbow**, **forward-swing elbow + wrist cluster**, **left hand/wrist**, and **right wrist/hand cluster** in different stills — fast arm swing + overlap with torso. At least one frame shows **red near hip/waist** with busy torso lines (“pelvis jitter”). **Swing foot** sometimes slightly **off** the shoe vs the **planted** foot. **Grain/compression**; occasional **extra** hand landmarks. | **6** side-view stills reviewed from `76555842-79e_annotated.mp4`; add MM:SS in thesis if you cite exact frames. | 2026-03-28 — `76555842-79e_annotated.mp4` |
| 8402092-hd_1920_1080_30fps.mp4 | Side | **Urban bridge / sidewalk (1080p):** same **side-profile** gait read as clip 2 — **railing with cutouts**, **red curb**, asphalt, graffiti, distant buildings / skyline. **Legs, knees, ankles, and feet** mostly **green**; **stance vs swing** clear; one sampled frame is **flight** (both feet off ground) and the skeleton still holds. Face and torso generally coherent at full HD. | **Arms / hands** still the weak link: **red** markers on **wrist–hand clusters**, **torso / upper chest**, **hip–waist**, and in one frame the **entire camera-right arm** (shoulder–elbow–wrist) flips to red — same pattern as clip 2, occlusion + fast swing. **Trailing leg** can sit **slightly off** the real limb in a few frames. Slightly **less spatial detail** than the 4K urban clip but tracking behavior matches. | **6** frames sampled across the clip (indices ~56, 112, 168, 224, 280, 336 of 393 total frames @ 30 fps) from `d56ae039-ded_annotated.mp4`; add MM:SS if cited. | 2026-03-28 — `d56ae039-ded_annotated.mp4` |

## One paragraph for the thesis (limitations)

Across **three** stock clips (beach barefoot runner and **two** side-view urban runs at different resolutions), BlazePose usually keeps **hip–knee–ankle** and **foot** landmarks usable when the subject contrasts with the background and stays near profile. **Arms, wrists, and hands** are the least stable: red (low-visibility) keypoints **move between frames** and often coincide with overlap against the torso or fast swing. **Torso and pelvis** can show jitter or extra line crossings; the **swing foot** is sometimes slightly mis-anchored on the shoe. For this thesis, treat **leg-based angles and stride timing** as more reliable than **arm or trunk** metrics, and state limitations of **single-camera 2D** inference and **visibility-weighted** landmarks.

---

## How to process clips

### Batch (all MP4s in `sample_videos/`)

From project root (venv activated if you use one):

```powershell
.\venv\Scripts\python.exe scripts\process_videos.py --input-dir data\raw\sample_videos --skip-existing
```

- **`--skip-existing`** skips any file whose name is already in `results/features_summary.csv` (no duplicate CSV rows). Omit it to **re-run everything** and get new `results\*_id_*` files for each video.

### Single file

```powershell
$f = Get-Item -LiteralPath "data\raw\sample_videos\YOUR_FILE.mp4"
.\venv\Scripts\python.exe scripts\process_videos.py $f.FullName
```

Then open each new `results\XXXX_annotated.mp4` and update the table above.
