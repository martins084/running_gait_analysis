"""Core processing modules for the running gait analysis system.

Imports are not re-exported here on purpose: pulling in ``pose_detection`` /
``feature_extractor`` etc. loads heavy optional deps (OpenCV, MediaPipe). Import
what you need explicitly, e.g. ``from core.pose_detection import PoseDetector``.
"""

__all__: list[str] = []
