"""
video_processor.py — Frame extraction and video I/O utilities.

Handles reading video files, extracting frames at a target FPS,
normalizing resolution, and saving frames to disk for downstream
pose detection.
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm


class VideoProcessor:
    """
    Reads videos, extracts frames at a configurable target FPS,
    and writes them to an output directory as numbered JPEGs.
    """

    # Maximum width we'll allow — anything wider gets downscaled
    MAX_WIDTH = 1280

    # JPEG quality for saved frames (0–100)
    JPEG_QUALITY = 90

    def __init__(self, fps_target: int = 30):
        """
        Args:
            fps_target: Desired frames per second to extract.
                        If the source FPS is higher, frames are skipped.
        """
        self.fps_target = fps_target

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def extract_frames(self, video_path: str, output_dir: str) -> int:
        """
        Extract frames from a video file and save them as JPEGs.

        Args:
            video_path: Path to the input video.
            output_dir: Directory where frames will be saved.

        Returns:
            Number of frames extracted.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # Source video properties
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # How many source frames to skip between kept frames
        skip_rate = max(1, int(fps_original / self.fps_target))

        os.makedirs(output_dir, exist_ok=True)

        frame_idx = 0
        saved_count = 0

        with tqdm(total=total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Keep every nth frame to match target FPS
                if frame_idx % skip_rate == 0:
                    frame = self._normalize_resolution(frame)
                    out_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
                    saved_count += 1

                frame_idx += 1
                pbar.update(1)

        cap.release()
        return saved_count

    def get_video_info(self, video_path: str) -> dict:
        """
        Return basic metadata about a video file.

        Returns dict with keys: fps, frame_count, width, height, duration_sec.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] else 0

        cap.release()
        return info

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _normalize_resolution(self, frame):
        """Downscale the frame if it's wider than MAX_WIDTH."""
        h, w = frame.shape[:2]
        if w > self.MAX_WIDTH:
            scale = self.MAX_WIDTH / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (self.MAX_WIDTH, new_h))
        return frame
