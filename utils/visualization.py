"""
visualization.py — Pose overlay drawing and annotated-video generation.

Draws the detected skeleton on individual frames or produces a full
annotated video with pose overlays.
"""

import cv2
import json
import numpy as np
from pathlib import Path


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


def draw_pose(frame: np.ndarray, landmarks: list | np.ndarray) -> np.ndarray:
    """
    Draw a skeleton overlay on a single frame.

    Args:
        frame: BGR image (H x W x 3).
        landmarks: (33, 3+) array or list — at minimum [x, y, z].
                   If 4 columns, the 4th is treated as visibility.

    Returns:
        The frame with the skeleton drawn on it (modified in place).
    """
    h, w = frame.shape[:2]
    lm = np.array(landmarks)

    # Draw bones
    for (i, j) in SKELETON_CONNECTIONS:
        x1, y1 = int(lm[i][0] * w), int(lm[i][1] * h)
        x2, y2 = int(lm[j][0] * w), int(lm[j][1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), BONE_COLOR, 2)

    # Draw joints
    for idx, pt in enumerate(lm):
        x, y = int(pt[0] * w), int(pt[1] * h)
        vis = pt[3] if len(pt) > 3 else 1.0
        color = JOINT_COLOR_HIGH if vis > 0.5 else JOINT_COLOR_LOW
        cv2.circle(frame, (x, y), 4, color, -1)

    return frame


def create_annotated_video(
    input_video: str,
    poses_json: str,
    output_video: str,
) -> None:
    """
    Read a video and its corresponding pose JSON, then write a new
    video with the skeleton overlay on every frame.

    Args:
        input_video: Path to original video.
        poses_json:  Path to JSON produced by PoseDetector.process_video().
        output_video: Where to save the annotated video.
    """
    with open(poses_json) as f:
        data = json.load(f)

    poses = data["poses"]

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v codec for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay skeleton if we have landmarks for this frame
        if frame_idx < len(poses) and poses[frame_idx]["landmarks"] is not None:
            draw_pose(frame, poses[frame_idx]["landmarks"])

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
