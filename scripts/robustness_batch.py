"""
Scan pose JSON files under data/test/robustness/<condition>/ and aggregate stability metrics.

No labels required. For each `*_poses.json`, loads landmarks and reports:
  - mean landmark visibility (all joints, all frames)
  - cadence (if stride metrics available) as single clip value; cadence_std across bootstrap = N/A → use symmetry std as variability proxy

Outputs:
  data/processed/robustness/robustness_metrics.csv

Usage:
  python scripts/robustness_batch.py
  python scripts/robustness_batch.py --root data/test/robustness --output-csv data/processed/robustness/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor


def mean_visibility(poses: list[dict]) -> float:
    vals = []
    for p in poses:
        lm = p.get("landmarks")
        if lm is None:
            continue
        for pt in lm:
            if len(pt) > 3:
                vals.append(float(pt[3]))
    return float(np.mean(vals)) if vals else float("nan")


def run(root: Path, output_csv: Path) -> int:
    root = root.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    if not root.is_dir():
        output_csv.write_text(
            "condition,video_name,mean_pose_confidence,cadence_spm,stride_length_px,symmetry_std,notes\n",
            encoding="utf-8",
        )
        print(f"No robustness root at {root}; wrote header-only CSV.")
        return 0

    extractor = FeatureExtractor(fps=30)

    for cond_dir in sorted(root.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition = cond_dir.name
        for pose_path in sorted(cond_dir.glob("*_poses.json")):
            notes = ""
            try:
                with open(pose_path, encoding="utf-8") as f:
                    data = json.load(f)
                poses = data["poses"]
                fps = float(data.get("fps") or 30)
                extractor.fps = int(round(fps))
                features = extractor.extract_all_features(str(pose_path))
                stride = features.get("stride_metrics") or {}
                sym = features.get("symmetry") or []
                sym_f = [float(s) for s in sym if isinstance(s, (int, float))]
                sym_std = float(np.std(sym_f)) if len(sym_f) > 2 else float("nan")

                mv = mean_visibility(poses)
                cad = stride.get("cadence_steps_per_min_merged", stride.get("cadence_steps_per_min", math.nan))
                slpx = stride.get("stride_length_px", math.nan)

                rows.append(
                    {
                        "condition": condition,
                        "video_name": pose_path.name,
                        "mean_pose_confidence": round(mv, 4) if not np.isnan(mv) else "",
                        "cadence_spm": round(float(cad), 2) if cad == cad and not math.isnan(float(cad)) else "",
                        "stride_length_px": round(float(slpx), 4) if slpx == slpx and not math.isnan(float(slpx)) else "",
                        "symmetry_std": round(sym_std, 4) if not np.isnan(sym_std) else "",
                        "notes": notes,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "condition": condition,
                        "video_name": pose_path.name,
                        "mean_pose_confidence": "",
                        "cadence_spm": "",
                        "stride_length_px": "",
                        "symmetry_std": "",
                        "notes": f"error: {exc}",
                    }
                )

    fieldnames = [
        "condition",
        "video_name",
        "mean_pose_confidence",
        "cadence_spm",
        "stride_length_px",
        "symmetry_std",
        "notes",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness metrics from pose JSONs under conditions folders.")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "test" / "robustness",
        help="Folder containing subfolders per condition",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "robustness" / "robustness_metrics.csv",
    )
    args = parser.parse_args()
    sys.exit(run(args.root, args.output_csv))


if __name__ == "__main__":
    main()
