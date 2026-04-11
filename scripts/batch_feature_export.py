"""
Extract biomechanical features from existing pose JSON files (no MediaPipe re-run).

Usage (from project root):
  venv\\Scripts\\python scripts\\batch_feature_export.py
  venv\\Scripts\\python scripts\\batch_feature_export.py --input-dir results --glob "*_poses.json"
  venv\\Scripts\\python scripts\\batch_feature_export.py --output-dir data/processed/features --force

Default input: `data/processed/poses` if it exists and has JSON; otherwise `results` with `*_poses.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor
from utils.feature_summary_csv import (
    analysis_ids_in_summary_csv,
    append_summary_csv,
    build_feature_summary_row,
)


def _clip_stem(poses_path: Path) -> str:
    """`5d676297-eda_poses.json` -> `5d676297-eda`."""
    stem = poses_path.stem
    return stem[:-6] if stem.endswith("_poses") else stem


def _write_features_json(
    out_path: Path,
    source_poses: str,
    video_name: str,
    fps: float,
    features: dict,
) -> None:
    payload = {
        "source_poses_json": source_poses,
        "video_name": video_name,
        "fps": fps,
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_frames_csv(
    out_path: Path,
    video_name: str,
    fps: float,
    features: dict,
    stride: dict,
    vert_osc: float | None,
) -> None:
    """
    Per-frame angles + symmetry; global stride / cadence columns repeated for joins.
    Stride columns repeat clip-level metrics; `stride_length_px` is normalized Δx, not metres (see docs/API_Response_Schema.md).
    """
    joint_angles = features.get("joint_angles") or []
    symmetry = features.get("symmetry") or []

    cadence = stride.get("cadence_steps_per_min", "")
    cadence_m = stride.get("cadence_steps_per_min_merged", "")
    sl_px = stride.get("stride_length_px", "")
    sl_hip = stride.get("stride_length_over_hip_width", "")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "timestamp_sec",
        "video_name",
        "left_hip_deg",
        "right_hip_deg",
        "left_knee_deg",
        "right_knee_deg",
        "symmetry_index",
        "vertical_oscillation_px_clip",
        "cadence_steps_per_min",
        "cadence_steps_per_min_merged",
        "stride_length_px",
        "stride_length_over_hip_width",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        n = max(len(joint_angles), len(symmetry))
        for i in range(n):
            ja = joint_angles[i] if i < len(joint_angles) and isinstance(joint_angles[i], dict) else {}
            sym_raw = symmetry[i] if i < len(symmetry) else ""
            sym = "" if sym_raw is None else sym_raw
            w.writerow(
                {
                    "frame_id": i,
                    "timestamp_sec": round(i / fps, 6) if fps else 0.0,
                    "video_name": video_name,
                    "left_hip_deg": ja.get("left_hip", ""),
                    "right_hip_deg": ja.get("right_hip", ""),
                    "left_knee_deg": ja.get("left_knee", ""),
                    "right_knee_deg": ja.get("right_knee", ""),
                    "symmetry_index": sym,
                    "vertical_oscillation_px_clip": vert_osc if vert_osc is not None else "",
                    "cadence_steps_per_min": cadence,
                    "cadence_steps_per_min_merged": cadence_m,
                    "stride_length_px": sl_px,
                    "stride_length_over_hip_width": sl_hip,
                }
            )


def export_one(
    poses_path: Path,
    output_dir: Path,
    extractor: FeatureExtractor,
    summary_csv: Path,
    force: bool,
    force_summary: bool,
) -> None:
    clip = _clip_stem(poses_path)
    feat_json = output_dir / f"{clip}_features.json"
    feat_csv = output_dir / f"{clip}_features_frames.csv"

    if not force and feat_json.is_file() and feat_csv.is_file():
        print(f"SKIP (exists): {clip}")
        return

    try:
        with open(poses_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {poses_path}: {exc}") from exc
    if "poses" not in meta or not isinstance(meta["poses"], list):
        raise ValueError(f"Expected key 'poses' (list) in {poses_path}")
    if len(meta["poses"]) == 0:
        raise ValueError(f"Empty 'poses' list in {poses_path}")

    fps = float(meta.get("fps") or 30)
    video_path = meta.get("video", "") or ""
    video_name = Path(video_path).name if video_path else f"{clip}.mp4"

    extractor.fps = int(round(fps))
    extractor.dt = 1.0 / extractor.fps

    features = extractor.extract_all_features(str(poses_path))
    stride = features.get("stride_metrics") or {}
    vert = features.get("vertical_oscillation_px")

    _write_features_json(feat_json, str(poses_path.resolve()), video_name, fps, features)
    _write_frames_csv(feat_csv, video_name, fps, features, stride, vert)

    summary_row = build_feature_summary_row(clip, video_name, video_path or str(poses_path), features)
    existing_ids = analysis_ids_in_summary_csv(summary_csv)
    if force_summary or clip not in existing_ids:
        append_summary_csv(summary_csv, summary_row)
        print(f"OK: {poses_path.name} -> {feat_json.name}, {feat_csv.name}, summary row `{clip}`")
    else:
        print(f"OK: {poses_path.name} -> {feat_json.name}, {feat_csv.name} (summary skip: id `{clip}` already in CSV)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export features from pose JSON files (no pose re-run).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing pose JSON files (default: data/processed/poses if non-empty else results/)",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pat",
        default="*_poses.json",
        help="Glob pattern under input-dir (default: *_poses.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "features",
        help="Per-clip features JSON + frame CSV (default: data/processed/features)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Aggregated summary CSV (default: <output-dir>/features_summary.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-clip JSON/CSV exports",
    )
    parser.add_argument(
        "--force-summary",
        action="store_true",
        help="Append another summary row even if this analysis id is already in features_summary.csv",
    )
    args = parser.parse_args()

    default_poses = ROOT / "data" / "processed" / "poses"
    results_poses = ROOT / "results"
    if args.input_dir is not None:
        input_dir = args.input_dir
    elif default_poses.is_dir() and any(default_poses.glob("*.json")):
        input_dir = default_poses
    else:
        input_dir = results_poses

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(input_dir.glob(args.glob_pat))
    if not json_files:
        print(f"No files matching {args.glob_pat!r} under {input_dir}", file=sys.stderr)
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = Path(args.summary_csv) if args.summary_csv else (output_dir / "features_summary.csv")

    extractor = FeatureExtractor(fps=30)
    errors = 0
    for jp in json_files:
        try:
            export_one(jp, output_dir, extractor, summary_csv, args.force, args.force_summary)
        except Exception as exc:
            errors += 1
            print(f"ERROR {jp}: {exc}", file=sys.stderr)

    print(f"\nDone. summary: {summary_csv}  (errors: {errors})")
    if errors:
        sys.exit(3)


if __name__ == "__main__":
    main()
