"""
visualization.py — Pose overlay drawing and annotated-video generation.

Draws the detected skeleton on individual frames or produces a full
annotated video with pose overlays.
"""

import json
import shutil
import subprocess
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from utils.com_segmentation import compute_segment_weighted_com_xy

# ── Video encoding (browser playback) ─────────────────────────────────────
# OpenCV default "mp4v" is MPEG-4 Part 2; Chrome / Edge / Safari often show a
# grey/black player for that codec in <video>. Prefer H.264 (AVC) when the
# local OpenCV/FFmpeg build exposes a working fourcc (varies by OS).
_MP4_FOURCC_TRY_ORDER = ("avc1", "H264", "X264", "mp4v")


def _open_mp4_writer(path: str, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a VideoWriter for MP4, preferring H.264 for HTML5 compatibility."""
    w, h = frame_size
    for tag in _MP4_FOURCC_TRY_ORDER:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if out.isOpened():
            return out
        out.release()
    raise RuntimeError(
        "Could not open any MP4 VideoWriter codec. Install a build of OpenCV with FFmpeg, "
        "or ensure an H.264 encoder is available on this system."
    )


def _try_transcode_browser_friendly_mp4(output_path: Path) -> None:
    """
    If `ffmpeg` is on PATH, re-encode to H.264 yuv420p + faststart so <video> plays
    in Chrome / Edge / Safari even when OpenCV only produced MPEG-4 Part 2 (mp4v).
    On failure, leaves the original file unchanged.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = output_path.with_suffix(".tmp_browser.mp4")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(output_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
            timeout=600,
        )
        tmp.replace(output_path)
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Skeleton connections (MediaPipe index pairs) ────────────────────────
# Grouped for color-coding: torso, arms, legs, feet
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left shoulder → left hip
    (12, 24),  # right shoulder → right hip
    (23, 24),  # hips

    # Left arm
    (11, 13), (13, 15),

    # Right arm
    (12, 14), (14, 16),

    # Left leg
    (23, 25), (25, 27),

    # Right leg
    (24, 26), (26, 28),

    # Feet
    (27, 29), (27, 31),  # left heel / toe
    (28, 30), (28, 32),  # right heel / toe
]

# Colors (BGR)
BONE_COLOR = (0, 255, 100)
JOINT_COLOR_HIGH = (0, 255, 0)    # visibility > 0.5
JOINT_COLOR_LOW = (0, 0, 255)     # visibility <= 0.5
COM_COLOR = (0, 200, 255)         # orange/cyan mix — stands out on green bones
COM_TRAIL_COLOR = (60, 140, 220)  # dimmer trail behind current COM
COM_OUTLINE = (40, 40, 40)

# Annotated MP4 overlay: ~3× thicker strokes vs the original 2px bones / 4px joints
# so the skeleton stays visible on high-res phone footage and after compression.
_BONE_THICKNESS = 6
_JOINT_RADIUS = 12
_COM_TRAIL_THICKNESS = 6
_COM_RING_THICKNESS = 6
_COM_CROSSHAIR_THICKNESS = 3


def draw_pose(
    frame: np.ndarray,
    landmarks: list | np.ndarray,
    com_trail: list[tuple[int, int]] | None = None,
    *,
    draw_com: bool = True,
) -> tuple[float, float] | None:
    """
    Draw a skeleton overlay on a single frame and optionally a segment-weighted COM marker.

    Args:
        frame: BGR image (H x W x 3).
        landmarks: (33, 3+) array or list — at minimum [x, y, z].
                   If 4 columns, the 4th is treated as visibility.
        com_trail: Optional list of pixel ``(x, y)`` for recent COM positions (oldest first);
                   drawn as a thin polyline under the skeleton for motion context.
        draw_com: If ``False``, skip COM trail + marker (use when the web UI draws COM on a
                  canvas over ``<video>`` to avoid duplicate symbols in the MP4).

    Returns:
        Normalized ``(x, y)`` of the segment-weighted COM if drawn, else ``None``.
        The frame is modified in place.
    """
    h, w = frame.shape[:2]
    lm = np.array(landmarks)

    # Recent COM path (behind skeleton so bones stay readable)
    if draw_com and com_trail and len(com_trail) >= 2:
        pts = np.array(com_trail, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            frame,
            [pts],
            isClosed=False,
            color=COM_TRAIL_COLOR,
            thickness=_COM_TRAIL_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        x1, y1 = int(lm[i][0] * w), int(lm[i][1] * h)
        x2, y2 = int(lm[j][0] * w), int(lm[j][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), BONE_COLOR, _BONE_THICKNESS, lineType=cv2.LINE_AA)

    # Draw joints
    for idx, pt in enumerate(lm):
        x, y = int(pt[0] * w), int(pt[1] * h)
        vis = pt[3] if len(pt) > 3 else 1.0
        color = JOINT_COLOR_HIGH if vis > 0.5 else JOINT_COLOR_LOW
        cv2.circle(frame, (x, y), _JOINT_RADIUS, color, -1, lineType=cv2.LINE_AA)

    # Whole-body COM (segment-weighted 2D proxy) — optional; SPA draws COM on canvas over <video>
    if not draw_com:
        return None

    com_xy = compute_segment_weighted_com_xy(lm)
    if com_xy is not None:
        cx = int(np.clip(com_xy[0] * w, 0, w - 1))
        cy = int(np.clip(com_xy[1] * h, 0, h - 1))
        r = max(5, min(w, h) // 90)
        cv2.circle(frame, (cx, cy), r + 2, COM_OUTLINE, _COM_RING_THICKNESS, lineType=cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r, COM_COLOR, -1, lineType=cv2.LINE_AA)
        d = r + 4
        cv2.line(
            frame,
            (cx - d, cy),
            (cx + d, cy),
            COM_OUTLINE,
            _COM_CROSSHAIR_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (cx, cy - d),
            (cx, cy + d),
            COM_OUTLINE,
            _COM_CROSSHAIR_THICKNESS,
            lineType=cv2.LINE_AA,
        )
        return com_xy

    return None


def create_annotated_video(
    input_video: str,
    poses_json: str,
    output_video: str,
    *,
    draw_com: bool = False,
) -> None:
    """
    Read a video and its corresponding pose JSON, then write a new
    video with the skeleton overlay on every frame.

    Args:
        input_video: Path to original video.
        poses_json:  Path to JSON produced by PoseDetector.process_video().
        output_video: Where to save the annotated video.
        draw_com: If ``True``, bake COM + trail into pixels (offline export). Default ``False``
                  so the web UI can draw a single synced COM on a canvas over ``<video>``.
    """
    with open(poses_json) as f:
        data = json.load(f)

    poses = data["poses"]

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    out = _open_mp4_writer(output_video, fps, (w, h))

    com_trail_buf: deque[tuple[int, int]] = deque(maxlen=22)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(poses) and poses[frame_idx]["landmarks"] is not None:
            lm_raw = poses[frame_idx]["landmarks"]
            if draw_com:
                com_xy = draw_pose(frame, lm_raw, list(com_trail_buf), draw_com=True)
                if com_xy is not None:
                    cx = int(np.clip(com_xy[0] * w, 0, w - 1))
                    cy = int(np.clip(com_xy[1] * h, 0, h - 1))
                    com_trail_buf.append((cx, cy))
            else:
                draw_pose(frame, lm_raw, draw_com=False)
        else:
            com_trail_buf.clear()

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    _try_transcode_browser_friendly_mp4(Path(output_video))
