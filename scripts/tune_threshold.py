"""
Tune anomaly score threshold on validation scores and apply to test scores.

This script is intentionally post-hoc and lightweight:
- no model retraining
- no checkpoint loading
- consumes CSV outputs from `scripts/evaluate_anomaly.py`

Expected inputs (per run):
- results/<run_id>/eval_val/per_sample_scores.csv
- results/<run_id>/eval_test/per_sample_scores.csv

Main use-case:
- select threshold on validation (Youden J by default)
- report classification-style metrics at that fixed threshold on both val/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent


def _load_scores(path: Path, label_col: str) -> pd.DataFrame:
    """Load labels and reconstruction errors from evaluate_anomaly CSV output."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing score CSV: {path}")
    df = pd.read_csv(path)
    required = {label_col, "recon_error", "subject_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Score CSV missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError(f"Score CSV has zero rows: {path}")
    return df


def _to_subject_p90(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    g = df.groupby("subject_id", as_index=False).agg(
        recon_error=("recon_error", lambda s: float(np.quantile(np.asarray(s, dtype=float), 0.90))),
        label=(label_col, "max"),
    )
    y = g["label"].astype(int).to_numpy()
    s = g["recon_error"].astype(float).to_numpy()
    return y, s


def _select_threshold(y_val: np.ndarray, s_val: np.ndarray, method: str) -> float:
    """
    Select threshold from validation scores.

    Methods:
    - youden_j: maximize (TPR - FPR)
    - max_f1: threshold with highest F1 score (can be unstable with imbalance)
    """
    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation labels have a single class; cannot tune threshold with supervised objectives. "
            "Use a fixed threshold or quantile from healthy reference scores."
        )

    if method == "youden_j":
        fpr, tpr, thr = roc_curve(y_val, s_val)
        idx = int(np.argmax(tpr - fpr))
        return float(thr[idx])

    if method == "max_f1":
        best_thr = float(s_val.min())
        best_f1 = -1.0
        for thr in sorted(set(float(x) for x in s_val)):
            y_hat = (s_val >= thr).astype(int)
            f1 = float(f1_score(y_val, y_hat, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        return best_thr

    raise ValueError(f"Unsupported threshold method: {method}")


def _metrics_at_threshold(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    """Compute classification-style metrics once threshold is fixed."""
    y_hat = (s >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "predicted_positive_rate": float(y_hat.mean()),
        "num_samples": int(len(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune threshold on eval_val and apply to eval_test.")
    parser.add_argument("--run-id", required=True, help="Run id used under results/<run_id>/...")
    parser.add_argument(
        "--method",
        default="youden_j",
        choices=["youden_j", "max_f1"],
        help="Threshold selection objective on validation split.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results",
        help="Root directory that contains run result folders.",
    )
    parser.add_argument(
        "--label-col",
        default="is_injured",
        help="Label column present in evaluate per_sample_scores.csv.",
    )
    parser.add_argument(
        "--metric-level",
        choices=["session", "subject"],
        default="subject",
        help="Tune threshold on session rows or subject-level P90 aggregated scores.",
    )
    args = parser.parse_args()

    run_dir = args.results_root / args.run_id
    val_csv = run_dir / "eval_val" / "per_sample_scores.csv"
    test_csv = run_dir / "eval_test" / "per_sample_scores.csv"

    val_df = _load_scores(val_csv, label_col=args.label_col)
    test_df = _load_scores(test_csv, label_col=args.label_col)
    if args.metric_level == "subject":
        y_val, s_val = _to_subject_p90(val_df, label_col=args.label_col)
        y_test, s_test = _to_subject_p90(test_df, label_col=args.label_col)
    else:
        y_val = val_df[args.label_col].astype(int).to_numpy()
        s_val = val_df["recon_error"].astype(float).to_numpy()
        y_test = test_df[args.label_col].astype(int).to_numpy()
        s_test = test_df["recon_error"].astype(float).to_numpy()

    threshold = _select_threshold(y_val, s_val, method=args.method)

    payload = {
        "run_id": args.run_id,
        "selected_on": "eval_val",
        "selection_objective": args.method,
        "metric_level": args.metric_level,
        "aggregation": "p90_by_subject" if args.metric_level == "subject" else "none",
        "val_at_selected_threshold": _metrics_at_threshold(y_val, s_val, threshold),
        "test_at_selected_threshold": _metrics_at_threshold(y_test, s_test, threshold),
    }

    suffix = f"{args.method}_{args.metric_level}_{args.label_col}"
    out_path = run_dir / f"threshold_tuned_metrics_{suffix}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Convenience artifact for evaluate_anomaly.py --threshold-file
    (run_dir / f"selected_threshold_{suffix}.json").write_text(
        json.dumps(
            {
                "threshold": float(threshold),
                "source": "eval_val",
                "method": args.method,
                "metric_level": args.metric_level,
                "label_col": args.label_col,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
