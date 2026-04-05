"""Unit tests for VideoProcessor (core/video_processor.py)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.video_processor import VideoProcessor


def _write_short_mp4(path: Path, frames: int = 15, fps: float = 30.0, w: int = 320, h: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), (i * 10) % 255, dtype=np.uint8)
        out.write(img)
    out.release()


def test_get_video_info_frame_count_and_fps(tmp_path):
    vid = tmp_path / "short.mp4"
    _write_short_mp4(vid, frames=20, fps=30.0)
    vp = VideoProcessor(fps_target=30)
    info = vp.get_video_info(str(vid))
    assert info["frame_count"] == 20
    assert abs(info["fps"] - 30.0) < 1.0
    assert info["width"] == 320 and info["height"] == 240
    assert info["duration_sec"] > 0


def test_extract_frames_respects_target_fps(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_short_mp4(vid, frames=60, fps=30.0)
    out_dir = tmp_path / "frames"
    vp = VideoProcessor(fps_target=10)
    n = vp.extract_frames(str(vid), str(out_dir))
    # ~ one saved frame per 3 source frames → ~20 frames
    assert 15 <= n <= 25
    jpgs = list(Path(out_dir).glob("*.jpg"))
    assert len(jpgs) == n


def test_normalize_resolution_downscales_wide_frame():
    vp = VideoProcessor(fps_target=30)
    wide = np.zeros((720, 1920, 3), dtype=np.uint8)
    out = vp._normalize_resolution(wide)
    assert out.shape[1] == VideoProcessor.MAX_WIDTH
    assert out.shape[0] < 720


def test_cannot_open_missing_file():
    vp = VideoProcessor()
    with pytest.raises(IOError):
        vp.get_video_info(str(Path("/nonexistent/video_xyz.mp4")))


def test_sub_short_video_under_one_second(tmp_path):
    """Edge case: clip shorter than 1 s (e.g. 10 frames @ 30 FPS ≈ 0.33 s)."""
    vid = tmp_path / "tiny.mp4"
    _write_short_mp4(vid, frames=10, fps=30.0)
    vp = VideoProcessor(fps_target=30)
    info = vp.get_video_info(str(vid))
    assert info["frame_count"] == 10
    assert info["duration_sec"] < 1.0
    n = vp.extract_frames(str(vid), str(tmp_path / "out_frames"))
    assert n == 10


def test_narrow_frame_not_resized():
    """Frames narrower than MAX_WIDTH pass through unchanged height/width."""
    vp = VideoProcessor(fps_target=30)
    narrow = np.zeros((480, 640, 3), dtype=np.uint8)
    out = vp._normalize_resolution(narrow)
    assert out.shape == narrow.shape
