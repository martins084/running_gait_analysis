# Run App Locally

This guide shows how to run the full application on your local machine:

- Backend API (`FastAPI`) on `http://localhost:8000`
- Frontend UI (`Vite + React`) on `http://localhost:5173`

## 1) Prerequisites

- Python 3.10+ (recommended: 3.12)
- Node.js 18+ and npm
- Git
- Windows PowerShell (commands below are written for PowerShell)

Optional but recommended:

- `ffmpeg` in PATH (for more browser-compatible annotated MP4 output)

## 2) Clone and enter the project

```powershell
git clone https://github.com/martins084/running_gait_analysis.git
cd running_gait_analysis
```

## 3) Backend setup (API)

From project root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run the API:

```powershell
python api/app.py
```

When it starts successfully:

- API root: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health endpoint: `http://localhost:8000/health`

Keep this terminal open.

## 4) Frontend setup (UI)

Open a second terminal.

```powershell
cd frontend
npm install
```

Create env file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and confirm:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Run frontend:

```powershell
npm run dev
```

Open: `http://localhost:5173`

## 5) Quick end-to-end check

1. Open frontend at `http://localhost:5173`
2. Upload a test `.mp4` / `.mov` / `.avi` / `.mkv`
3. Wait for analysis to complete
4. Verify:
   - metrics are shown in UI,
   - annotated video appears or downloads,
   - `GET /results/{id}` returns data.

## 6) Run tests (optional)

From project root with venv active:

```powershell
pytest tests -q --tb=no
```

## 7) Common issues

### PowerShell activation blocked

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then re-run:

```powershell
.\venv\Scripts\Activate.ps1
```

### CORS error in browser

- Ensure API is running on `:8000`
- Ensure frontend `.env` has correct `VITE_API_BASE_URL`
- Restart frontend after editing `.env`

### Port already in use

- Change API port in `api/app.py` (`uvicorn.run(..., port=8000)`)
- Or stop the process currently using the port

### Annotated video does not play in browser

- Install `ffmpeg` and keep it in PATH
- Retry analysis so output video is encoded to browser-friendly format

## 8) Stop the app

- In both terminals press `Ctrl + C`
- Deactivate backend environment if needed:

```powershell
deactivate
```

