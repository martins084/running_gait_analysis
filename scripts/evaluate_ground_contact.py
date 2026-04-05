"""
Evaluate automatic ground-contact heuristics against frame-wise labels.

Compares label CSV (`frame_id`, `is_contact`) to predictions derived from
ankle vertical position (image coordinates: larger y = lower on screen / nearer ground).

Usage:
  python scripts/evaluate_ground_contact.py \\
    --labels-csv data/processed/labels/ground_contact_clip1.csv \\
    --poses-json path/to/clip_poses.json \\
    --output-dir data/processed/validation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor

# Match FeatureExtractor defaults for foot points
LEFT_ANKLE = FeatureExtractor.JOINTS["left_ankle"]
RIGHT_ANKLE = FeatureExtractor.JOINTS["right_ankle"]
VIS_TH = 0.45


def load_labels_csv(path: Path) -> tuple[list[int], list[int]]:
    """Return parallel lists (frame_id, is_contact). Skips malformed rows; warns to stderr."""
    frames: list[int] = []
    labels: list[int] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Empty labels CSV (no header)")
        # Flexible column names
        fi = next((h for h in reader.fieldnames if h.lower().replace(" ", "_") in ("frame_id", "frame")), None)
        ci = next((h for h in reader.fieldnames if "contact" in h.lower()), None)
        if not fi or not ci:
            raise ValueError(f"Need frame_id and is_contact columns, got: {reader.fieldnames}")
        for row in reader:
            try:
                frames.append(int(row[fi]))
                labels.append(int(row[ci]))
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
    if skipped:
        print(f"Warning: skipped {skipped} malformed label row(s).", file=sys.stderr)
    if not frames:
        raise ValueError("No valid label rows after parsing (check frame_id / is_contact values).")
    return frames, labels


def ankle_ground_score_series(poses: list[dict]) -> np.ndarray:
    """
    Per frame: max(left_ankle_y, right_ankle_y) when visibility ok, else NaN; then interpolated.
    Higher score ≈ foot lower in frame ≈ closer to ground contact in side view.
    """
    ys: list[float] = []
    for p in poses:
        lm = p.get("landmarks")
        if lm is None:
            ys.append(float("nan"))
            continue
        ly = float(lm[LEFT_ANKLE][1]) if len(lm[LEFT_ANKLE]) > 3 and lm[LEFT_ANKLE][3] >= VIS_TH else float("nan")
        ry = float(lm[RIGHT_ANKLE][1]) if len(lm[RIGHT_ANKLE]) > 3 and lm[RIGHT_ANKLE][3] >= VIS_TH else float("nan")
        if np.isnan(ly) and np.isnan(ry):
            ys.append(float("nan"))
        elif np.isnan(ly):
            ys.append(ry)
        elif np.isnan(ry):
            ys.append(ly)
        else:
            ys.append(max(ly, ry))

    arr = np.array(ys, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return np.zeros(len(arr))
    return np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])


def predict_from_local_peaks(scores: np.ndarray, fps: float, neighbor: int = 2) -> np.ndarray:
    """Binary prediction: 1 within ±neighbor frames of a peak in ground score."""
    n = len(scores)
    if n < 5:
        return np.zeros(n, dtype=int)
    min_dist = max(1, int(fps * 0.25))
    peaks, _ = signal.find_peaks(scores, distance=min_dist)
    pred = np.zeros(n, dtype=int)
    for p in peaks:
        lo = max(0, p - neighbor)
        hi = min(n, p + neighbor + 1)
        pred[lo:hi] = 1
    return pred


def run_evaluation(
    labels_csv: Path,
    poses_json: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_frames, y_true = load_labels_csv(labels_csv)
    try:
        with open(poses_json, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in poses file: {exc}") from exc
    if "poses" not in data or not isinstance(data["poses"], list):
        raise ValueError("Pose JSON must contain a list key 'poses'.")
    fps = float(data.get("fps") or 30)
    poses = data["poses"]
    n = len(poses)
    if n == 0:
        raise ValueError("Pose JSON has zero frames in 'poses'.")

    # Align labels to full video length (sparse CSV → dense vectors)
    y_true_full = np.zeros(n, dtype=int)
    for fr, lb in zip(label_frames, y_true):
        if 0 <= fr < n:
            y_true_full[fr] = lb

    scores = ankle_ground_score_series(poses)
    y_pred = predict_from_local_peaks(scores, fps)

    # Restrict to frames where we have label file entries or full comparison on overlap
    mask = np.zeros(n, dtype=bool)
    for fr in label_frames:
        if 0 <= fr < n:
            mask[fr] = True
    if not mask.any():
        mask = np.ones(n, dtype=bool)

    yt = y_true_full[mask]
    yp = y_pred[mask]

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, average="binary", pos_label=1, zero_division=0
    )

    roc_auc = float("nan")
    fpr = tpr = np.array([0.0, 1.0])
    try:
        if len(np.unique(yt)) >= 2:
            roc_auc = float(roc_auc_score(yt, scores[mask]))
            fpr, tpr, _ = roc_curve(yt, scores[mask])
    except ValueError:
        pass

    metrics_path = output_dir / "ground_contact_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "sensitivity",
                "specificity",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "num_frames_evaluated",
                "tp",
                "fp",
                "tn",
                "fn",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "video_name": poses_json.stem,
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else "",
                "num_frames_evaluated": int(mask.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    # Plots (optional dependency failure → skip)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for (j, i), label in np.ndenumerate(np.array([[tn, fp], [fn, tp]])):
            ax.text(i, j, int(label), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        if len(np.unique(yt)) >= 2:
            ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "ROC needs both classes", ha="center")
        fig2.tight_layout()
        fig2.savefig(output_dir / "roc_curve.png", dpi=120)
        plt.close(fig2)
    except Exception as exc:
        print(f"Plot skip: {exc}", file=sys.stderr)

    summary = (
        f"Sensitivity={sens:.3f}, Specificity={spec:.3f}, F1={float(f1):.3f}, "
        f"ROC-AUC={roc_auc if not np.isnan(roc_auc) else 'n/a'}"
    )
    summary_path = output_dir / "ground_contact_summary.txt"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    print(summary)

    return {
        "sensitivity": sens,
        "specificity": spec,
        "f1": float(f1),
        "roc_auc": roc_auc,
        "metrics_csv": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ground contact vs labels.")
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--poses-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.labels_csv.is_file():
        print(f"Missing labels: {args.labels_csv}", file=sys.stderr)
        sys.exit(1)
    if not args.poses_json.is_file():
        print(f"Missing poses: {args.poses_json}", file=sys.stderr)
        sys.exit(1)

    try:
        run_evaluation(args.labels_csv, args.poses_json, args.output_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
