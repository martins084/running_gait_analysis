# API response schema — `POST /analyze`

This document describes the JSON returned by the Running Gait Analysis FastAPI service (`api/app.py`). It is intended for supervisors and anyone integrating with the HTTP API.

---

## Endpoint

| Item | Value |
|------|--------|
| **Method / path** | `POST /analyze` |
| **Content-Type** | `multipart/form-data` |
| **Field name** | `video` (file upload) |
| **Allowed extensions** | `.mp4`, `.mov`, `.avi`, `.mkv` |
| **Max upload size** | 200 MB (hard cap on the server) |
| **Processing timeout** | 300 seconds (5 minutes) per request |

---

## Successful response (`200 OK`)

Content-Type: `application/json`.

### Top-level keys

| Key | Type | Meaning |
|-----|------|---------|
| `id` | string (UUID) | Unique analysis identifier. Use it with `GET /results/{id}` and `GET /download/{id}`. |
| `status` | string | Always `"completed"` on success. |
| `features` | object | Nested biomechanical metrics (see below). |
| `annotated_video` | string | Relative URL path to download the skeleton overlay video, e.g. `"/download/<id>"`. Resolve against your API base URL. |
| `poses` | string | Relative URL path to fetch the stored result payload again, e.g. `"/results/<id>"`. |

**Raw pose landmarks:** `GET /poses/{id}` returns the full BlazePose JSON (`*_poses.json`) — large. Use for research or client-side recomputation; the web UI prefers precomputed fields in `features` when available.

**Note:** The on-disk batch pipeline (`scripts/process_videos.py`) writes a slightly different JSON shape (`video`, `poses_file`, `annotated_video` as absolute paths). The API normalizes paths to URL fragments and omits raw pose file paths in the HTTP response.

---

## `features` object

### `stride_metrics` (object)

Derived from ankle trajectories (temporally smoothed). **Landmark x / y are normalized** to \([0,1]\) in MediaPipe (full image width / height), not raw sensor pixels.

#### Naming note: `stride_length_px` (normalized, not “pixels”)

| Detail | Explanation |
|--------|----------------|
| **What the number is** | Horizontal distance **in normalized x-units** between successive left-ankle “contact” events (mean of segment lengths in `x` only). Same numeric scale as `landmarks[][0]`. |
| **Range** | Typically a **small fraction** (e.g. **0.05–0.35**) for side-view phone/4K footage — it is **not** “pixels across the image” despite the `_px` suffix (legacy naming). |
| **Not absolute** | There is **no** conversion to metres or real-world stride length without camera calibration and subject distance. |

Example: `"stride_length_px": 0.2471` means the left ankle moves about **0.25 of the frame width** horizontally between successive left-foot contacts — **not** 0.25 pixels.

#### Why `stride_length_over_hip_width` can look “large” (e.g. ~40)

This ratio is:

\[
\text{stride\_length\_over\_hip\_width} = \frac{\text{stride\_length\_px (norm x)}}{\text{median hip width (norm x)}}
\]

Both numerator and denominator use the **same normalized** x-scale. Hip width in image space is often only **~0.01–0.02** (narrow segment). Stride segment is **~0.2–0.3**. So the **ratio routinely lands in roughly 10–50**, not near 1.

Example from a real export: `stride_length_px ≈ 0.247`, hip width \(\approx 0.0062\) → \(0.247 / 0.0062 \approx 40\). That is **expected** and means “stride spans ~40 hip-widths in the 2D projection,” not a bug in the tens place.

| Key | Units / type | Description |
|-----|----------------|-------------|
| `stride_length_px` | **normalized Δx** (0–1 scale) | Mean horizontal distance between successive left-foot contact events (see table above). |
| `stride_time_sec` | seconds | Mean time between successive left-foot contacts. |
| `cadence_steps_per_min` | steps/min | Estimated from left-foot stride period (two steps per stride). |
| `cadence_steps_per_min_merged` | steps/min | Cadence from merged left+right foot-strike times (deduplicated). |
| `stride_length_over_hip_width` | dimensionless ratio | Stride length (norm x) ÷ hip width (norm x); **large values (~10–50) are normal**. |
| `num_same_foot_contacts_left` | integer | Count of detected left-ankle “ground proximity” peaks. |
| `num_foot_strikes_merged` | integer | Count of merged L+R strike events. |
| `num_strides_detected` | integer | Same-foot peak count (legacy/debug). |

#### FPS and timing

| Source | Role |
|--------|------|
| **Video container** | `cv2` reads **FPS** from the file; it is written into the pose JSON as `fps` and used for timestamps (`frame_index / fps`). |
| **`FeatureExtractor` default** | If `fps` were missing, the code **defaults to 30** for angle / stride time calculations — your pipeline normally stores the **true** FPS from the video, so this is only a fallback. |
| **API / analysis** | `POST /analyze` uses the same detector path; timing metrics follow the stored `fps` in the pose sequence. |

*There is no conversion to metres without camera calibration.*

### `com_xy_per_frame` (array, optional on old rows)

| Key | Type | Description |
|-----|------|-------------|
| Each element | `{ "x": number, "y": number }` or `null` | Segment-weighted **approximate** whole-body COM in **normalized image coordinates** (same as landmarks), one entry per video frame after temporal smoothing. `null` when pose is missing or unreliable. |

### `video_frame_count` (integer, optional)

Total frames in the pose sequence (`len(poses)`). Prefer this (and `com_xy_per_frame.length`) for aligning `<video>` playback with metrics; `joint_angles.length` may differ on legacy exports that omitted frames without pose.

### `joint_angles` (array)

One entry per processed video frame (same order as the source frames). Each element:

| Key | Units | Description |
|-----|--------|-------------|
| `left_hip`, `right_hip` | degrees | Angle at hip (shoulder–hip–knee), 2D image plane. |
| `left_knee`, `right_knee` | degrees | Angle at knee (hip–knee–ankle). |

### `symmetry` (array of numbers)

One value per frame: left–right symmetry index in \([0, 1]\) (1 = more symmetric in the 2D projection).

### `vertical_oscillation_px`

Single number or `null`: approximate vertical range of the hip midpoint (normalized y), using the 5th–95th percentile spread over the clip.

### `fps` and `frame_count` (optional on legacy rows)

| Key | Type | Meaning |
|-----|------|--------|
| `fps` | integer | Frames per second from the pose pipeline (same as in the stored pose JSON). Omitted on analyses saved **before** this field was added. |
| `frame_count` | integer | Same as `joint_angles.length`; redundant but explicit for clients. |

Clients can align video playback with per-frame metrics using `fps`, or derive an effective rate as `frame_count / video_duration_sec` when `fps` is missing.

---

## Example response (truncated)

Real shape from a processed stock clip; `joint_angles` and `symmetry` arrays are shortened with `/* ... */`.

```json
{
  "id": "76555842-79e",
  "status": "completed",
  "features": {
    "stride_metrics": {
      "stride_length_px": 0.2471,
      "stride_time_sec": 1.04,
      "cadence_steps_per_min": 115.4,
      "num_same_foot_contacts_left": 5,
      "stride_length_over_hip_width": 39.9466,
      "cadence_steps_per_min_merged": 110.1,
      "num_foot_strikes_merged": 9,
      "num_strides_detected": 5
    },
    "joint_angles": [
      {
        "left_hip": 157.32,
        "right_hip": 160.25,
        "left_knee": 133.87,
        "right_knee": 151.0
      }
      /* ... one object per frame ... */
    ],
    "symmetry": [0.9981, 0.9975, 0.9962]
    /* ... one value per frame ... */,
    "vertical_oscillation_px": 0.0527,
    "fps": 30,
    "frame_count": 120
  },
  "annotated_video": "/download/76555842-79e",
  "poses": "/results/76555842-79e"
}
```

---

## Error responses

| HTTP code | When | Example `detail` |
|-----------|------|-------------------|
| **400** | Missing filename, empty file, or bad extension | `"Unsupported format '.txt'. Use one of {'.mp4', ...}."` |
| **413** | Upload exceeds 200 MB | `"File too large (> 209715200 bytes)."` |
| **422** | Pose/annotation pipeline failed (corrupt video, codec issue, etc.) | `"Pose analysis failed: ..."` or `"Pose detection failed: no pose output generated."` |
| **504** | Processing exceeded `ANALYZE_TIMEOUT_SEC` | `"Analysis timed out after 300 seconds."` |
| **500** | Unexpected server error | `"Unexpected server error: ..."` |

**404** is used on `GET /results/{id}` and `GET /download/{id}` when the analysis or annotated file does not exist.

---

## Related endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | `{"status": "ok"}` liveness check. |
| `GET` | `/results/{analysis_id}` | Same JSON as the analyze response (from SQLite or legacy `*_result.json`). |
| `GET` | `/download/{analysis_id}` | `video/mp4` file stream of the annotated video. |

---

*Generated to match `api/app.py` as of March 2026. If the server code changes, update this file alongside.*
