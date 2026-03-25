"""
Batch-process one or more videos: pose JSON → features → annotated MP4 + result JSON.

Usage (from project root):
  venv\\Scripts\\python scripts\\process_videos.py path1.mp4 path2.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pose_detection import PoseDetector
from core.feature_extractor import FeatureExtractor
from utils.visualization import create_annotated_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gait pipeline on video files.")
    parser.add_argument("videos", nargs="+", help="Paths to .mp4 (or other supported) files")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for poses JSON, annotated video, and result JSON",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    detector = PoseDetector()
    extractor = FeatureExtractor(fps=30)

    for video_path in args.videos:
        vp = Path(video_path)
        if not vp.is_file():
            print(f"SKIP (not found): {vp}", file=sys.stderr)
            continue

        analysis_id = str(uuid.uuid4())[:12]
        poses_json = results_dir / f"{analysis_id}_poses.json"
        annotated_mp4 = results_dir / f"{analysis_id}_annotated.mp4"
        result_json = results_dir / f"{analysis_id}_result.json"

        print(f"\n=== {vp.name} | id={analysis_id} ===")

        detector.process_video(str(vp), str(poses_json))
        features = extractor.extract_all_features(str(poses_json))
        create_annotated_video(str(vp), str(poses_json), str(annotated_mp4))

        out = {
            "id": analysis_id,
            "status": "completed",
            "video": str(vp.resolve()),
            "features": features,
            "poses_file": str(poses_json.resolve()),
            "annotated_video": str(annotated_mp4.resolve()),
        }
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)

        print(f"  poses:     {poses_json}")
        print(f"  annotated: {annotated_mp4}")
        print(f"  result:    {result_json}")


if __name__ == "__main__":
    main()
