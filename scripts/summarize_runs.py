"""
Summarize anomaly experiment metrics across multiple run IDs.

This script reads per-run JSON artifacts and produces:
- compact per-run rows
- aggregate mean/std metrics across runs
- optional CSV export for thesis tables

Supported inputs per run:
- results/<run_id>/eval_val/metrics.json
- results/<run_id>/eval_test/metrics.json
- optional threshold-tuned JSON files:
  - threshold_tuned_metrics_youden_j.json
  - threshold_tuned_metrics_max_f1.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent


def _safe_float(v: object) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_row(results_root: Path, run_id: str) -> dict:
    run_dir = results_root / run_id
    val = _read_json(run_dir / "eval_val" / "metrics.json") or {}
    test = _read_json(run_dir / "eval_test" / "metrics.json") or {}
    youden = _read_json(run_dir / "threshold_tuned_metrics_youden_j.json") or {}
    max_f1 = _read_json(run_dir / "threshold_tuned_metrics_max_f1.json") or {}

    row = {
        "run_id": run_id,
        "val_auroc": _safe_float(val.get("auroc")),
        "val_auprc": _safe_float(val.get("auprc")),
        "test_auroc": _safe_float(test.get("auroc")),
        "test_auprc": _safe_float(test.get("auprc")),
        "val_threshold_accuracy": _safe_float(val.get("threshold_accuracy")),
        "test_threshold_accuracy": _safe_float(test.get("threshold_accuracy")),
        # Optional tuned-threshold metrics (Youden J)
        "youden_val_bal_acc": _safe_float((youden.get("val_at_selected_threshold") or {}).get("balanced_accuracy")),
        "youden_test_bal_acc": _safe_float((youden.get("test_at_selected_threshold") or {}).get("balanced_accuracy")),
        "youden_val_f1": _safe_float((youden.get("val_at_selected_threshold") or {}).get("f1")),
        "youden_test_f1": _safe_float((youden.get("test_at_selected_threshold") or {}).get("f1")),
        # Optional tuned-threshold metrics (max F1)
        "maxf1_val_bal_acc": _safe_float((max_f1.get("val_at_selected_threshold") or {}).get("balanced_accuracy")),
        "maxf1_test_bal_acc": _safe_float((max_f1.get("test_at_selected_threshold") or {}).get("balanced_accuracy")),
        "maxf1_val_f1": _safe_float((max_f1.get("val_at_selected_threshold") or {}).get("f1")),
        "maxf1_test_f1": _safe_float((max_f1.get("test_at_selected_threshold") or {}).get("f1")),
    }
    return row


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if isinstance(r.get(key), float)]
    vals = [v for v in vals if v == v]  # drop NaN
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(mean(vals)), "std": float(pstdev(vals)), "n": len(vals)}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval metrics across multiple run IDs.")
    parser.add_argument("--run-id", action="append", required=True, help="Repeat for each run id.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results" / "summary_runs.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "summary_runs.csv")
    args = parser.parse_args()

    rows = [_collect_row(args.results_root, rid) for rid in args.run_id]

    summary = {
        "run_ids": args.run_id,
        "per_run": rows,
        "aggregate": {
            "val_auroc": _agg(rows, "val_auroc"),
            "val_auprc": _agg(rows, "val_auprc"),
            "test_auroc": _agg(rows, "test_auroc"),
            "test_auprc": _agg(rows, "test_auprc"),
            "youden_val_bal_acc": _agg(rows, "youden_val_bal_acc"),
            "youden_test_bal_acc": _agg(rows, "youden_test_bal_acc"),
            "youden_val_f1": _agg(rows, "youden_val_f1"),
            "youden_test_f1": _agg(rows, "youden_test_f1"),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, rows)

    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
