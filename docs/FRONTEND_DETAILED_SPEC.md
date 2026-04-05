# Running Gait Analysis — detailed frontend specification

**Purpose of this document:** Give a third party (supervisor, reviewer, or another AI) a **complete, implementation-accurate** picture of the **web frontend** in this repository: stack, file layout, data flow, API usage, UI behaviour, and design constraints. Everything below reflects the codebase **as of the document date**; verify against `frontend/src/` if the project evolves.

**Short summary (LV):** Viena lappušu lietotne (SPA) uz React + Vite: video augšupielāde → FastAPI analīze → rezultātu lapa ar video, COM pārklājumu uz canvas, grafikiem (Recharts) un skriešanas metrikām. Divvalodu (LV/EN), gaišs/tumšs režīms, sinhronizācija starp video kadru un grafiku atskaņotāju.

---

## 1. Technology stack

| Layer | Choice | Notes |
|--------|--------|--------|
| UI library | **React 18** | Function components only; no class components. |
| Language | **TypeScript** (~5.6) | Strict typing for API payloads in `src/api/types.ts`. |
| Build / dev server | **Vite 5** | `npm run dev` → port **5173** (see `vite.config.ts`). |
| Routing | **react-router-dom v6** | `BrowserRouter`, `Routes`, `Route`, `Navigate`. |
| Charts | **Recharts 2** | `LineChart`, `AreaChart`, `ReferenceLine`, `ResponsiveContainer`, etc. |
| Fonts (npm) | **Fontsource** | Source Sans 3, Source Serif 4, IBM Plex Mono — **Latin Extended** in `main.tsx` for Latvian diacritics. |
| Global state | **None** (no Redux/Zustand) | Local `useState` / `useRef` per page; `sessionStorage` for last analysis id only. |
| i18n | **Custom** | `LocaleProvider` + `en.ts` / `lv.ts` nested objects; `t("key.subkey")` with optional `{{var}}` interpolation. |

**Not used:** Next.js, CSS-in-JS libraries, Tailwind, React Query (fetch is plain `fetch()`).

---

## 2. Repository layout (`frontend/`)

```
frontend/
├── package.json
├── vite.config.ts
├── tsconfig.json
└── src/
    ├── main.tsx                 # Entry: fonts, CSS imports, StrictMode, LocaleProvider, App
    ├── App.tsx                  # BrowserRouter + routes
    ├── vite-env.d.ts
    ├── api/
    │   ├── client.ts            # getApiBase, resolveApiUrl, fetch wrappers
    │   └── types.ts             # Mirrors backend / docs/API_Response_Schema.md
    ├── components/
    │   ├── AppShell.tsx         # Header: brand, nav, API health, locale, theme
    │   ├── UploadZone.tsx       # Drag-drop + file input, validation
    │   ├── VideoPanel.tsx       # <video>, COM canvas overlay, download link
    │   ├── MetricsGrid.tsx      # Stride / cadence summary (dl grid)
    │   ├── JointAngleChart.tsx
    │   ├── SymmetryChart.tsx
    │   ├── ComOscillationChart.tsx
    │   └── MlPhasePlaceholder.tsx
    ├── hooks/
    │   └── useVideoFrameSync.ts # Maps video.currentTime ↔ frame index; seek helper
    ├── i18n/
    │   ├── index.ts
    │   ├── LocaleProvider.tsx
    │   └── locales/
    │       ├── en.ts
    │       └── lv.ts
    ├── pages/
    │   ├── AnalyzePage.tsx      # "/" upload flow
    │   └── ResultsPage.tsx      # "/results/:id"
    ├── styles/
    │   ├── tokens.css           # CSS variables: light/dark, fonts, chart colors
    │   ├── base.css             # Reset-ish, page chrome
    │   └── components.css       # All component/layout classes (large file)
    └── utils/
        ├── comSegmentation.ts   # Segment-weighted COM (mirrors Python)
        ├── videoContentRect.ts  # Letterbox math for object-fit: contain
        ├── downsample.ts        # Chart point budget (500 points)
        └── format.ts            # number formatting
```

---

## 3. Build, environment variables, and API connectivity

### Scripts (`package.json`)

- `npm run dev` — Vite dev server.
- `npm run build` — `tsc --noEmit && vite build` (typecheck + production bundle).
- `npm run preview` — serve production build locally.

### `VITE_API_BASE_URL`

- If **set** (e.g. `http://127.0.0.1:8000`): the SPA calls the API at that origin (requires **CORS** on FastAPI).
- If **unset in development**: `getApiBase()` returns `""` and `resolveApiUrl()` prefixes paths with **`/api`**, which Vite **proxies** to `http://127.0.0.1:8000` with path rewrite (`/api` stripped). Same-origin `fetch()` avoids CORS for API calls.
- **Production dev without proxy:** set `VITE_API_BASE_URL` explicitly.

### `VITE_DEV_ML_MOCK`

- Used only by `MlPhasePlaceholder.tsx`: if `import.meta.env.DEV && VITE_DEV_ML_MOCK === "true"`, a **mock** gait phase strip is shown for UI development when the API does not send `ml.phases_per_frame`.

### Static assets

- No `public/` images required for core UX; video URL comes from API (`annotated_video` resolved via `resolveApiUrl`).

---

## 4. Routing and page map

| Path | Component | Role |
|------|-----------|------|
| `/` | `AnalyzePage` | Hero copy, `UploadZone`, POST video, navigate to results. |
| `/results/:id` | `ResultsPage` | GET `/results/{id}`, full analysis UI. |
| `*` | `<Navigate to="/" replace />` | Fallback redirect. |

`App.tsx` wraps all routes in `AppShell` (header + `<main>`).

---

## 5. Application shell (`AppShell.tsx`)

**Always visible:** sticky **top bar** with:

- **Brand** (`Link` to `/`) — title + subtitle from i18n.
- **“New analysis”** link to `/` — **hidden on `/`** to avoid duplicate entry to the same flow (spacer div keeps layout).
- **API health indicator:** polls `GET /health` on mount and every **45s** via `getHealth()`. Dot green = OK, red = error, grey = unknown.
- **Language toggle:** LV / EN buttons; choice stored in `localStorage` (`gait-ui-locale`).
- **Theme toggle:** light/dark; stored in `localStorage` (`gait-ui-theme`); applied as `document.documentElement.dataset.theme`.
- **Skip link** to `#main-content` for keyboard users.

`LocaleProvider` updates `document.documentElement.lang` and `document.title` when locale changes.

---

## 6. Analyze page (`AnalyzePage.tsx`)

**Flow:**

1. User drops or picks a file in `UploadZone`.
2. Client validates extension (`.mp4`, `.mov`, `.avi`, `.mkv`) and max size **200 MB** (must stay aligned with server cap).
3. `postAnalyzeVideo(file)` → `POST /analyze` with `FormData` field `video`.
4. On success: `sessionStorage.setItem("gait-last-analysis-id", id)` and `navigate(/results/${id})`.
5. While busy: overlay panel with indeterminate progress + elapsed seconds (timer).

**Resume link:** If `sessionStorage` has last id, shows “resume last result” link to `/results/{id}`.

**Errors:** Shown in a `panel--error` alert; message from API `detail` when JSON, else status text.

---

## 7. Results page (`ResultsPage.tsx`)

### Data loading

- Reads `id` from `useParams()`.
- `getResults(id)` → `GET /results/{id}` → `AnalyzeResponse` (see `api/types.ts`).
- **Locale switch does not re-fetch** (dependency array is `[id]` only) to avoid duplicate GETs.

### Layout: three-column “workbench”

Implemented in CSS (`components.css`): roughly **TOC rail | video column | scrollable analysis column** at wide breakpoints (`--bp-results-3col: 1100px`). Narrow screens stack (see media queries in `components.css`).

**Left rail (`results-rail`):**

- Link back to `/` (new analysis).
- Page title, monospace **analysis id**, **copy id** button (clipboard, 2s “copied” feedback).
- Status badge (`completed` / `pending` translated).

**Center (`results-video-col`):**

- `VideoPanel` — see §9.

**Right (`results-analysis-col`):**

- **Jump nav** (`ResultsJumpNav`) — in-page anchor links: stride, vertical, COM, angles, symmetry, ML.
- **MetricsGrid** — stride metrics.
- **Vertical oscillation** — single scalar in a card (`vertical_oscillation_px`).
- **ComOscillationChart** — COM x/y vs frame.
- **JointAngleChart** — four angle series.
- **SymmetryChart** — area chart 0..1.
- **MlPhasePlaceholder** — timeline or “waiting for model” placeholder.
- **Footer:** disabled “compare” button (future feature), i18n “coming soon”.

### Video ↔ charts synchronization

- `videoRef` is created on `ResultsPage` and passed to `VideoPanel`.
- `frameCount` resolution order: `features.video_frame_count` → `com_xy_per_frame.length` → `joint_angles.length`.
- `fps` from `features.fps` (optional on legacy data).
- `useVideoFrameSync({ videoRef, frameCount, fps, onFrame: setCurrentFrameIndex })` updates `currentFrameIndex` from `video.currentTime`.
- `seekToFrame` uses `seekVideoToFrame(video, frame, frameCount, fps)` so chart clicks can move playback.

**Charts** receive `currentFrameIndex` and optional `onSeekFrame={seekToFrame}` — vertical line (`ReferenceLine`) at current frame; click on chart seeks video (shared `handleChartSeek` pattern in angle/symmetry/COM chart files).

---

## 8. API client (`api/client.ts`) and types (`api/types.ts`)

### URL resolution

- `resolveApiUrl("/download/uuid")` builds full URL for `<video src>`, `fetch`, and `<a href download>`.
- Absolute `http(s)://` paths are passed through unchanged.

### Endpoints used by the frontend

| Function | HTTP | Path | Purpose |
|----------|------|------|---------|
| `getHealth` | GET | `/health` | AppShell status dot |
| `postAnalyzeVideo` | POST | `/analyze` | Multipart upload |
| `getResults` | GET | `/results/{id}` | Feature payload + video paths |
| `getPosesJson` | GET | `/poses/{id}` | Full BlazePose JSON (large); optional client COM recompute |

### Important types (`types.ts`)

- **`FeaturePayload`:** `stride_metrics`, `joint_angles[]`, `symmetry[]`, `vertical_oscillation_px`, optional `fps`, `frame_count`, `com_xy_per_frame`, `video_frame_count`, optional `ml`.
- **`ComXY`:** `{ x, y }` in **normalized image coordinates 0–1** (not metres).
- **`AnalyzeResponse`:** `id`, `status`, `features`, `annotated_video` (relative path), `poses` (path string for result ref — UI mainly uses `annotated_video` and `id` for download).

Alignment with backend is documented in **`docs/API_Response_Schema.md`**.

---

## 9. Video panel (`VideoPanel.tsx`)

### Video source

- `src={resolveApiUrl(annotatedVideoPath)}` where `annotated_video` comes from API (e.g. `/download/{id}`).
- `controls`, `playsInline`, `preload="metadata"`.
- On `onError`: shows fallback message + link to **download** URL (same origin / resolved URL) — helps when codec/CORS prevents inline playback.

### COM overlay (canvas)

- **Primary data:** `com_xy_per_frame` from `features` (backend-computed, aligned with smoothed pipeline).
- **Fallback:** If API series missing or all-null, fetches **`getPosesJson(analysisId)`** and builds COM via **`computeComSeriesFromPosesJson`** in `utils/comSegmentation.ts` (same mass fractions as Python `utils/com_segmentation.py`; **not** identical uncertainty to server-smoothed series — UI warns via `video.comFallbackWarn`).

### Drawing

- Transparent **canvas** stacked over `<video>` (`video-shell__com-canvas`).
- **`getVideoContentRect` + `normalizedToContentPixel`** map 0–1 coords to pixels accounting for **`object-fit: contain`** letterboxing (`videoContentRect.ts`).
- **Device pixel ratio** applied to canvas backing store; CSS size matches shell.
- **ResizeObserver** on shell + `loadedmetadata` on video redraw overlay.
- Trail: last **20** frames (`TRAIL_LEN`) of COM polyline; current frame: filled circle + crosshair (stroke).
- `aria-hidden` on canvas (decorative); legends explained in text below.

### Download

- Link to `resolveApiUrl(/download/{analysisId})` with `download` attribute for MP4.

---

## 10. Charts (Recharts)

### Shared behaviour

- **Downsampling:** `downsampleIndices(len, CHART_MAX_POINTS)` where `CHART_MAX_POINTS = 500` — keeps first/last indices for long clips (performance).
- **Playhead:** `ReferenceLine` at `x={currentFrameIndex}` when data exists.
- **Seek:** If `onSeekFrame` provided, chart area is clickable; uses Recharts click state (`activeTooltipIndex` or `xValue`) to call `onSeekFrame(frame)`.
- Styling uses **CSS variables** from `tokens.css` (`--chart-grid`, `--series-*`, `--accent`, etc.).

### Components

| Component | Chart type | Series |
|-----------|------------|--------|
| `JointAngleChart` | LineChart | left/right hip + knee angles (°) |
| `SymmetryChart` | AreaChart | symmetry 0–1 |
| `ComOscillationChart` | LineChart | `cx` and `cy` vs frame; empty state if no COM data |

### ML placeholder (`MlPhasePlaceholder.tsx`)

- If API provides `ml.phases_per_frame`, renders colored timeline segments (`stance` / `swing` / `push`).
- Otherwise shows placeholder text + optional **dev mock** strip when `VITE_DEV_ML_MOCK=true`.
- **Playhead** on placeholder track: percentage = `frame / (frameCount - 1)` when `currentFrameIndex` in range.

---

## 11. Upload zone (`UploadZone.tsx`)

- Drag-and-drop + hidden file input.
- Validates extension against `ALLOWED` and **200 MB** max.
- On invalid: `alert()` with translated message (intentionally simple).

---

## 12. Hooks (`useVideoFrameSync.ts`)

- **`timeToFrameIndex`:** `round(currentTime * fps)` clamped to `[0, frameCount-1]`; `fps` from API or inferred as `frameCount / video.duration` if needed.
- **`seekVideoToFrame`:** sets `video.currentTime = frame / fps`.
- **Events:** `timeupdate`, `seeked`, `loadedmetadata`, `play`, `pause`.
- **Smooth updates while playing:** `requestAnimationFrame` loop **unless** `prefers-reduced-motion: reduce` — then relies on `timeupdate` only.

---

## 13. Internationalization

- **Default locale:** **Latvian (`lv`)** if nothing in `localStorage`.
- Bundles: `i18n/locales/lv.ts`, `en.ts` — nested keys, flat access via dot path e.g. `t("results.title")`.
- **Interpolation:** `{{name}}` in strings; `t("key", { name: value })`.
- All user-visible strings should go through `t()` (no hardcoded English in components for UI copy).

---

## 14. Styling architecture

1. **`tokens.css`** — `:root` and `[data-theme="dark"]` variables (surfaces, text, borders, chart series, phase colors, radii, breakpoints).
2. **`base.css`** — global typography, `body`, links, page background, skip link.
3. **`components.css`** — BEM-like class names: `.top-bar`, `.card`, `.results-workbench`, `.video-shell`, `.metrics-grid`, `.upload-zone`, chart wrappers, skeleton loaders, etc.

**Design intent:** Editorial serif for headings, sans for UI, mono for IDs and technical labels; warm neutral palette with ember accent; supports **reduced motion** users in sync hook (not necessarily all CSS transitions).

---

## 15. Accessibility and UX notes

- Skip link, `aria-label` / `aria-busy` / `role="alert"` / `aria-live` where appropriate.
- Chart click-to-seek is optional (class `chart-wrap--seekable` when `onSeekFrame` set).
- Results loading skeleton (`ResultsSkeleton`) while fetching.
- **No** explicit route-based code splitting in `App.tsx` (single bundle unless Vite splits dynamically — currently not used).

---

## 16. Explicit non-goals / out of scope (frontend)

- No user accounts, login, or multi-tenant UI.
- No WebSocket — analysis is request/response.
- No service worker / PWA manifest in described files.
- No unit tests in `frontend/` in current tree (verify with `glob` if added later).
- **Homography / camera calibration** — not in frontend; all coordinates are 2D normalized image space.

---

## 17. How to cite this stack in a thesis

Suggested wording (adapt as needed):

> The web interface is a single-page application implemented with **React** and **TypeScript**, built using **Vite**. Client-side routing is handled by **React Router**. Biomechanical time series and the center-of-mass trajectory are visualized with **Recharts**; video playback is synchronized with the plots via a custom hook that maps playback time to analysis frame indices. The center of mass overlay is rendered on an HTML **canvas** element aligned with the video element using letterbox-aware coordinate mapping. The backend is accessed over **REST** (`fetch`); optional development **proxy** avoids cross-origin issues. The interface supports **Latvian** and **English** and **light/dark** themes, with strings centralized in locale bundles.

---

## 18. File reference quick index

| Concern | Primary files |
|---------|----------------|
| Routes | `src/App.tsx` |
| API + env | `src/api/client.ts`, `vite.config.ts` |
| Types | `src/api/types.ts`, `docs/API_Response_Schema.md` |
| Results layout + sync | `src/pages/ResultsPage.tsx`, `src/hooks/useVideoFrameSync.ts` |
| Video + COM | `src/components/VideoPanel.tsx`, `src/utils/videoContentRect.ts`, `src/utils/comSegmentation.ts` |
| Charts | `src/components/*Chart.tsx`, `src/utils/downsample.ts` |
| i18n | `src/i18n/LocaleProvider.tsx`, `src/i18n/locales/*.ts` |
| Global chrome | `src/components/AppShell.tsx`, `src/styles/*.css` |

---

*End of specification.*
