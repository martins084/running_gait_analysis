"""
Batch-process one or more videos: pose JSON → features → annotated MP4 + result JSON.

Usage (from project root):
  venv\\Scripts\\python scripts\\process_videos.py path1.mp4 path2.mp4
  venv\\Scripts\\python scripts\\process_videos.py --input-dir data/raw/sample_videos
  venv\\Scripts\\python scripts\\process_videos.py --input-dir data/raw/sample_videos --skip-existing
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
from utils.feature_summary_csv import (
    append_summary_csv,
    build_feature_summary_row,
    video_names_in_summary_csv,
)
from utils.visualization import create_annotated_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gait pipeline on video files.")
    parser.add_argument(
        "videos",
        nargs="*",
        default=[],
        help="Paths to .mp4 (or other supported) files",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Process every *.mp4 in this folder (sorted by path name). Combines with any paths listed as arguments.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose basename already appears as video_name in the summary CSV (avoids duplicate rows when re-running a batch).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for poses JSON, annotated video, and result JSON",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV path for one-row-per-video summary export",
    )
    parser.add_argument(
        "--storage-profile",
        choices=["minimal", "standard", "full"],
        default="minimal",
        help="Minimize persisted artifacts: minimal removes poses/video after feature extraction.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (results_dir / "features_summary.csv")

    # Build ordered, de-duplicated list: explicit args first, then --input-dir glob.
    raw_paths: list[Path] = [Path(p) for p in args.videos]
    if args.input_dir is not None:
        indir = args.input_dir
        if not indir.is_dir():
            parser.error(f"--input-dir is not a directory: {indir}")
        raw_paths.extend(sorted(indir.glob("*.mp4")))

    seen_resolved: set[str] = set()
    video_paths: list[Path] = []
    for vp in raw_paths:
        try:
            key = str(vp.resolve())
        except OSError:
            key = str(vp)
        if key in seen_resolved:
            continue
        seen_resolved.add(key)
        video_paths.append(vp)

    if not video_paths:
        parser.error("No videos to process. Pass file paths and/or --input-dir containing .mp4 files.")

    already_in_csv = video_names_in_summary_csv(summary_csv) if args.skip_existing else set()

    detector = PoseDetector()
    extractor = FeatureExtractor(fps=30)

    for video_path in video_paths:
        vp = Path(video_path)
        if not vp.is_file():
            print(f"SKIP (not found): {vp}", file=sys.stderr)
            continue

        if args.skip_existing and vp.name in already_in_csv:
            print(f"SKIP (already in {summary_csv.name}): {vp.name}")
            continue

        analysis_id = str(uuid.uuid4())[:12]
        poses_json = results_dir / f"{analysis_id}_poses.json"
        annotated_mp4 = results_dir / f"{analysis_id}_annotated.mp4"
        result_json = results_dir / f"{analysis_id}_result.json"

        print(f"\n=== {vp.name} | id={analysis_id} ===")

        detector.process_video(str(vp), str(poses_json))
        features = extractor.extract_all_features(str(poses_json))
        # draw_com=False: COM is shown only in the web UI canvas (avoids duplicate markers in MP4).
        # Use create_annotated_video(..., draw_com=True) for a fully baked COM trail in the file.
        create_annotated_video(str(vp), str(poses_json), str(annotated_mp4))

        if args.storage_profile == "minimal":
            poses_json.unlink(missing_ok=True)
            annotated_mp4.unlink(missing_ok=True)

        out = {
            "id": analysis_id,
            "status": "completed",
            "video_name": vp.name,
            "features": features,
            "poses_file": str(poses_json.resolve()) if poses_json.exists() else None,
            "annotated_video": str(annotated_mp4.resolve()) if annotated_mp4.exists() else None,
            "storage_profile": args.storage_profile,
        }
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)

        # Export compact row for thesis stats/plots.
        summary_row = build_feature_summary_row(
            analysis_id,
            vp.name,
            str(vp.resolve()),
            features,
        )
        append_summary_csv(summary_csv, summary_row)

        print(f"  poses:     {poses_json}")
        print(f"  annotated: {annotated_mp4}")
        print(f"  result:    {result_json}")
        print(f"  summary:   {summary_csv}")


if __name__ == "__main__":
    main()
