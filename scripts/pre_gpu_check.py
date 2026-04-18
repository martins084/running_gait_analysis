"""
Pre-GPU readiness checker for anomaly pipeline.

This performs local filesystem/config checks so GPU startup is blocked until
basic prerequisites are in place.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _exists(path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, "ok"
    return False, f"missing: {path}"


def _manifest_count(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing: {path}"
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return False, "manifest has zero rows"
    return True, f"rows={len(rows)}"


def _split_leakage(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing: {path}"
    by_split = {"train": set(), "val": set(), "test": set()}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sid = r.get("subject_id", "")
            split = r.get("split", "")
            if split in by_split:
                by_split[split].add(sid)
    if by_split["train"] & by_split["val"]:
        return False, "subject leakage train-val"
    if by_split["train"] & by_split["test"]:
        return False, "subject leakage train-test"
    if by_split["val"] & by_split["test"]:
        return False, "subject leakage val-test"
    return True, "no subject leakage"


def _run_meta(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["run_id", "seed", "config_hash", "data", "train"]
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"run metadata missing keys: {missing}"
    return True, "run metadata schema ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pre-GPU readiness (local checks).")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "anomaly_train_v1.yaml")
    parser.add_argument("--expected-manifest-rows", type=int, default=2506)
    parser.add_argument("--run-id", default="", help="Optional: validate run outputs for this run id.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dcfg = cfg["data"]
    ocfg = cfg["output"]

    checks: list[tuple[str, bool, str]] = []
    checks.append(("config_exists", *_exists(args.config)))
    checks.append(("manifest_exists_and_nonempty", *_manifest_count(ROOT / "data" / "processed" / "ric_manifest.csv")))
    checks.append(("session_split_no_leakage", *_split_leakage(ROOT / dcfg["session_split_csv"])))
    checks.append(("json_root_exists", *_exists(ROOT / dcfg["json_root"])))

    manifest_path = ROOT / "data" / "processed" / "ric_manifest.csv"
    if manifest_path.is_file():
        with open(manifest_path, newline="", encoding="utf-8") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))
        if n_rows == args.expected_manifest_rows:
            checks.append(("manifest_expected_row_count", True, f"rows={n_rows}"))
        else:
            checks.append(
                (
                    "manifest_expected_row_count",
                    False,
                    f"rows={n_rows}, expected={args.expected_manifest_rows} (ok if dataset version changed)",
                )
            )

    if args.run_id:
        run_id = args.run_id
        checks.append(("best_checkpoint_exists", *_exists(ROOT / ocfg["checkpoints_dir"] / run_id / "best.pt")))
        checks.append(("last_checkpoint_exists", *_exists(ROOT / ocfg["checkpoints_dir"] / run_id / "last.pt")))
        checks.append(("epoch_metrics_exists", *_exists(ROOT / ocfg["logs_dir"] / run_id / "epoch_metrics.csv")))
        checks.append(("train_summary_exists", *_exists(ROOT / ocfg["results_dir"] / run_id / "train_summary.json")))
        checks.append(("run_metadata_schema", *_run_meta(ROOT / ocfg["logs_dir"] / run_id / "run_metadata.json")))

    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failed += 1

    if failed:
        sys.exit(2)
    print("All checks passed.")


if __name__ == "__main__":
    main()

