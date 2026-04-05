"""
Compare manually measured joint angles to pipeline-computed angles.

Manual CSV columns: frame_id, video_name, manual_angle, angle_name
  angle_name examples: left_hip, right_hip, left_knee, right_knee

Features source: `*_features_frames.csv` from batch_feature_export (or merged file).

Usage:
  python scripts/validate_features.py \\
    --manual-csv data/processed/labels/manual_angle_validation.csv \\
    --features-csv data/processed/features/merged_features_frames.csv \\
    --output-dir data/processed/validation
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Map logical angle name → column in features_frames export
ANGLE_TO_COL = {
    "left_hip": "left_hip_deg",
    "right_hip": "right_hip_deg",
    "left_knee": "left_knee_deg",
    "right_knee": "right_knee_deg",
}


def load_manual(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manual CSV has no header row.")
        for row in reader:
            rows.append(row)
    return rows


def load_features_rows(path: Path) -> dict[tuple[str, int, str], float]:
    """
    Index (video_name, frame_id, angle_col) -> value.
    angle_col is left_hip_deg etc.
    """
    out: dict[tuple[str, int, str], float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Features CSV has no header row.")
        for row in reader:
            vn = row.get("video_name", "")
            try:
                fid = int(row["frame_id"])
            except (KeyError, ValueError):
                continue
            for col in ANGLE_TO_COL.values():
                if col in row and row[col] not in ("", None):
                    try:
                        out[(vn, fid, col)] = float(row[col])
                    except ValueError:
                        pass
    return out


def run_validation(manual_rows: list[dict], feat_index: dict, output_dir: Path) -> int:
    """
    Returns 0 if at least one angle had 2+ matched samples for full metrics; 2 if no matches; 1 if only partial.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_angle: dict[str, list[tuple[float, float]]] = {k: [] for k in ANGLE_TO_COL}

    for row in manual_rows:
        vn = row.get("video_name", "")
        aname = row.get("angle_name", "").strip().lower().replace(" ", "_")
        if aname not in ANGLE_TO_COL:
            continue
        col = ANGLE_TO_COL[aname]
        try:
            fid = int(row["frame_id"])
            manual = float(row.get("manual_angle", row.get("manual_angle_deg", "")))
        except (KeyError, ValueError):
            continue
        key = (vn, fid, col)
        if key not in feat_index:
            continue
        by_angle[aname].append((manual, feat_index[key]))

    report_lines = []
    metrics_path = output_dir / "angle_validation_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "angle_name",
                "num_samples",
                "mean_error_deg",
                "std_error_deg",
                "mae",
                "rmse",
                "correlation",
                "pass_fail_mae_lt_5",
            ],
        )
        w.writeheader()

        for aname, pairs in by_angle.items():
            if len(pairs) < 2:
                w.writerow(
                    {
                        "angle_name": aname,
                        "num_samples": len(pairs),
                        "mean_error_deg": "",
                        "std_error_deg": "",
                        "mae": "",
                        "rmse": "",
                        "correlation": "",
                        "pass_fail_mae_lt_5": "",
                    }
                )
                continue
            m = np.array([p[0] for p in pairs])
            c = np.array([p[1] for p in pairs])
            err = c - m
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            corr = float(np.corrcoef(m, c)[0, 1]) if np.std(m) > 0 and np.std(c) > 0 else float("nan")
            pf = "PASS" if mae < 5.0 else "FAIL"
            w.writerow(
                {
                    "angle_name": aname,
                    "num_samples": len(pairs),
                    "mean_error_deg": round(float(np.mean(err)), 4),
                    "std_error_deg": round(float(np.std(err)), 4),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "correlation": round(corr, 4) if not np.isnan(corr) else "",
                    "pass_fail_mae_lt_5": pf,
                }
            )
            report_lines.append(f"{aname}: MAE={mae:.2f}°, RMSE={rmse:.2f}° → {pf}")

    (output_dir / "angle_validation_summary.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        nplots = sum(1 for pairs in by_angle.values() if len(pairs) >= 2)
        if nplots:
            fig, axes = plt.subplots(1, nplots, figsize=(4 * nplots, 4), squeeze=False)
            ax_flat = axes.ravel()
            idx = 0
            for aname, pairs in by_angle.items():
                if len(pairs) < 2:
                    continue
                m = np.array([p[0] for p in pairs])
                c = np.array([p[1] for p in pairs])
                ax = ax_flat[idx]
                ax.scatter(m, c, alpha=0.7)
                lo = min(m.min(), c.min())
                hi = max(m.max(), c.max())
                ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)
                ax.set_xlabel("Manual (°)")
                ax.set_ylabel("Computed (°)")
                ax.set_title(aname)
                idx += 1
            for j in range(idx, len(ax_flat)):
                ax_flat[j].set_visible(False)
            fig.tight_layout()
            fig.savefig(output_dir / "manual_vs_computed_angles.png", dpi=120)
            plt.close(fig)
    except Exception as exc:
        print(f"Plot skip: {exc}", file=sys.stderr)

    matched_any = any(len(by_angle[k]) > 0 for k in ANGLE_TO_COL)
    matched_enough = any(len(by_angle[k]) >= 2 for k in ANGLE_TO_COL)
    msg = "\n".join(report_lines) if report_lines else (
        "No matched rows (check video_name / frame_id / angle_name)."
    )
    print(msg)
    if not matched_any:
        return 2
    if not matched_enough:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate computed angles vs manual CSV.")
    parser.add_argument("--manual-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.manual_csv.is_file():
        print(f"Missing manual CSV: {args.manual_csv}", file=sys.stderr)
        sys.exit(1)
    if not args.features_csv.is_file():
        print(f"Missing features CSV: {args.features_csv}", file=sys.stderr)
        sys.exit(1)

    try:
        manual = load_manual(args.manual_csv)
        feat_index = load_features_rows(args.features_csv)
    except ValueError as exc:
        print(f"CSV error: {exc}", file=sys.stderr)
        sys.exit(4)

    code = run_validation(manual, feat_index, args.output_dir)
    # 2 = no rows matched any manual entry (likely wrong video_name / paths)
    if code == 2:
        sys.exit(2)


if __name__ == "__main__":
    main()
