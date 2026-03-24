"""
gait_analyzer.py — High-level orchestrator that ties together
pose detection, feature extraction, and (later) ML inference
into a single analysis pipeline.

Usage:
    analyzer = GaitAnalyzer()
    result   = analyzer.analyze_video("input.mp4")
"""

import json
import uuid
from pathlib import Path
from typing import Optional

from core.pose_detection import PoseDetector
from core.feature_extractor import FeatureExtractor


class GaitAnalyzer:
    """
    End-to-end pipeline:
      video → pose detection → feature extraction → result dict.

    Later phases will add ML model inference and annotated-video
    generation in between.
    """

    def __init__(
        self,
        results_dir: str = "results",
        fps: int = 30,
    ):
        """
        Args:
            results_dir: Where intermediate and final files are stored.
            fps: Expected source FPS (used for feature calculation).
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.detector = PoseDetector()
        self.extractor = FeatureExtractor(fps=fps)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def analyze_video(self, video_path: str, analysis_id: Optional[str] = None) -> dict:
        """
        Run the full analysis pipeline on a single video.

        Args:
            video_path: Path to the input video file.
            analysis_id: Optional unique ID; one is generated if omitted.

        Returns:
            Dict with keys: id, status, features, poses_file.
        """
        if analysis_id is None:
            analysis_id = str(uuid.uuid4())

        # Step 1 — Pose detection
        poses_json = str(self.results_dir / f"{analysis_id}_poses.json")
        self.detector.process_video(video_path, poses_json)

        # Step 2 — Feature extraction
        features = self.extractor.extract_all_features(poses_json)

        # Step 3 — Package result
        result = {
            "id": analysis_id,
            "status": "completed",
            "video": video_path,
            "features": features,
            "poses_file": poses_json,
        }

        # Persist summary
        result_json = str(self.results_dir / f"{analysis_id}_result.json")
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result
