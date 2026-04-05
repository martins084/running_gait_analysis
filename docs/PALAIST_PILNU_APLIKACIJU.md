# Kā palaist aplikāciju pilnībā (API + web)

Šis dokuments ir **soļu plāns**: backend (FastAPI + MediaPipe) un frontend (React + Vite). Darbības izpildi **projekta saknē** (`running_gait_analysis/`), ja nav norādīts citādi.

---

## 0. Kas jābūt uz datora

| Komponente | Minimums | Piezīmes |
|------------|----------|-----------|
| **Python** | 3.10+ (ieteicams 3.12) | `python --version` |
| **Node.js** | 18+ (ieteicams 20 LTS) | `node --version`; frontend būvēšanai |
| **npm** | kopā ar Node | `npm --version` |
| **FFmpeg** | *ieteicams* | Ja ir `PATH`, annotētais MP4 biežāk atskaņojas pārlūkā (H.264). Nav obligāts API darbībai. |
| **Git** | *ja atjaunini no repozitorija* | `git pull` |

---

## 1. Vienreizēja iestatīšana (vai pēc ilgāka pārtraukuma)

### 1.1 Projekta mape un virtuālā vide

```powershell
cd "C:\Users\...\running_gait_analysis"   # tava ceļš

# Ja venv vēl nav:
python -m venv venv

venv\Scripts\activate
```

*(macOS / Linux: `source venv/bin/activate`)*

### 1.2 Python atkarības un pats projekts

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Tā **atjaunina** pakotnes no `requirements.txt` un nodrošina `import core`, `import api` u.c.

### 1.3 MediaPipe modeļa fails

Pose noteikšanai vajadzīgs fails:

`models/pretrained/pose_landmarker_full.task`

Ja mapes `models/pretrained/` ir tukša vai trūkst `.task`, API palaižot analīze **kritīs**. Lejupielādē BlazePose **Pose Landmarker** (full) no MediaPipe resursiem un ieliec šajā ceļā — skat. `core/pose_detection.py` un `docs/SYSTEM_OVERVIEW_AND_CURRENT_STATE.md`.

### 1.4 Frontend atkarības

Šablons izmanto lokālas fontu pakotnes (`@fontsource/source-sans-3`, `source-serif-4`, `ibm-plex-mono`, ar `latin-ext` apakškopām) — tās tiek iekļautas `npm install` laikā; papildu Google Fonts vai tīkla fontu konfigurācija nav nepieciešama.

```powershell
cd frontend
npm install
cd ..
```

Lai **atjauninātu** jau esošo `node_modules` pēc `package.json` izmaiņām:

```powershell
cd frontend
npm install
# opcionāli: npm update
cd ..
```

### 1.5 (Ieteicams) Pārbaude, ka viss salikts

Projekta saknē, ar aktīvu `venv`:

```powershell
pytest tests/ -q
```

Ja testi iziet, Python slānis un imports parasti ir kārtībā.

---

## 2. Pilna palaišana — divi termināļi

**Secība svarīga:** vispirms API, tad frontend.

### Terminālis 1 — Backend (API)

```powershell
cd "C:\Users\...\running_gait_analysis"
venv\Scripts\activate
python api/app.py
```

Gaidāmais rezultāts: serveris uz **http://0.0.0.0:8000** (lokāli: **http://localhost:8000**).

- API dokumentācija: **http://localhost:8000/docs**
- Veselība: **http://localhost:8000/health** → `{"status":"ok"}`

**Neaizver šo logu**, kamēr strādā ar UI.

### Terminālis 2 — Frontend

```powershell
cd "C:\Users\...\running_gait_analysis\frontend"
npm run dev
```

Gaidāmais rezultāts: **http://localhost:5173**

Pārlūkā atver **5173** — augšupielādē `.mp4` / `.mov` / `.avi` / `.mkv`, gaidi analīzi, skaties rezultātus un video.

---

## 3. Konfigurācija (ja nepieciešams)

| Mērķis | Ko darīt |
|--------|----------|
| API nav uz 8000 | Izveido `frontend/.env` no `frontend/.env.example` un iestati `VITE_API_BASE_URL=http://localhost:PORT` |
| COM pārklājs / `fetch` uz API (izstrāde) | Ja **nav** `VITE_API_BASE_URL`, frontend izmanto Vite **proxy** (`/api` → 8000) — tas novērš CORS, ko `<video>` nerāda. Ja `.env` norāda tiešu `http://localhost:8000`, backend jāļauj CORS (noklusējumā + regex `localhost`/`127.0.0.1`). |
| CORS kļūda no cita porta | Palaid API ar `CORS_ORIGINS` (skat. README **Web frontend**) |
| ML fāžu laika līnijas mock tikai izstrādē | `frontend/.env`: `VITE_DEV_ML_MOCK=true` |

---

## 4. Ātra problēmu diagnostika

| Simptoms | Ko pārbaudīt |
|----------|----------------|
| `ModuleNotFoundError: core` | `pip install -e .` no projekta saknes, `venv` aktivizēts |
| Analīze 422 / pose kļūda | Vai eksistē `models/pretrained/pose_landmarker_full.task` |
| Pārlūkā pelēks video | Uzstādi **FFmpeg** `PATH` un analizē atkārtoti; vai lejupielādē MP4 no UI |
| Brīdinājums par COM / nav COM pārklāja | Palaid API no **projekta saknes** (lai `results/` un `*_poses.json` ceļš sakrīt); ja vecā analīze — veic **jaunu** augšupielādi, lai API atgriež `com_xy_per_frame`. |
| `Port 8000 already in use` | Aizver citu procesu uz 8000 vai maini `uvicorn` portu `api/app.py` failā |
| `5173` aizņemts | Vite piedāvās citu portu — skat. termināļa izvadi |

---

## 5. Pēc `git pull` — īss “update” ritms

1. `git pull`
2. `venv\Scripts\activate`
3. `pip install -r requirements.txt` un `pip install -e .`
4. `pytest tests/ -q` *(ieteicams)*
5. `cd frontend && npm install && cd ..`
6. Palaid **Terminālis 1** (`python api/app.py`), tad **Terminālis 2** (`npm run dev`)

---

## 6. Tikai API (bez web)

```powershell
venv\Scripts\activate
python api/app.py
```

Tālāk: Postman (`postman_collection.json`) vai **http://localhost:8000/docs**.

---

## 7. Saistītie dokumenti

- [README.md](../README.md) — Quick Start, Docker, Postman
- [THESIS_FOLLOW_THROUGH_GUIDE.md](THESIS_FOLLOW_THROUGH_GUIDE.md) — pētniecības checklist
- [SYSTEM_OVERVIEW_AND_CURRENT_STATE.md](SYSTEM_OVERVIEW_AND_CURRENT_STATE.md) — arhitektūra
- [API_Response_Schema.md](API_Response_Schema.md) — JSON atbilžu forma

---

*Atjaunināts kopā ar repozitorija README; pārbaudi `pytest` un `npm run dev` pēc lielākām izmaiņām.*
