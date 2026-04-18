"""
Build deterministic subject-level splits from the RIC manifest.

Outputs:
  - subject split map: one row per subject with assigned split
  - session split map: one row per session with inherited subject split
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Manifest CSV has no header row.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest CSV has no data rows.")
    if "subject_id" not in rows[0]:
        raise ValueError("Manifest CSV must contain column: subject_id")
    return rows


def _assign_subject_splits(
    subject_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    ids = sorted(set(subject_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(max(n_train, 1), max(n - 2, 1)) if n >= 3 else max(n - 1, 1)
    n_val = min(max(n_val, 1), max(n - n_train - 1, 0)) if n >= 3 else 0
    n_test = n - n_train - n_val
    if n_test <= 0 and n >= 2:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)

    split_map: dict[str, str] = {}
    for sid in ids[:n_train]:
        split_map[sid] = "train"
    for sid in ids[n_train : n_train + n_val]:
        split_map[sid] = "val"
    for sid in ids[n_train + n_val :]:
        split_map[sid] = "test"

    return split_map


def _validate_leakage(session_rows: list[dict], split_map: dict[str, str]) -> None:
    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in session_rows:
        sid = r["subject_id"]
        split = split_map[sid]
        by_split[split].add(sid)

    overlap_tv = by_split["train"] & by_split["val"]
    overlap_tt = by_split["train"] & by_split["test"]
    overlap_vt = by_split["val"] & by_split["test"]
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Subject leakage detected across splits.")


def _write_subject_split(path: Path, split_map: dict[str, str], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "split", "seed"])
        w.writeheader()
        for sid in sorted(split_map):
            w.writerow({"subject_id": sid, "split": split_map[sid], "seed": seed})


def _write_session_split(path: Path, session_rows: list[dict], split_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(session_rows[0].keys()) + ["split"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in session_rows:
            out = dict(r)
            out["split"] = split_map[r["subject_id"]]
            w.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic subject-level splits for RIC sessions.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "ric_manifest.csv",
        help="Input manifest CSV path.",
    )
    parser.add_argument(
        "--output-subject-split",
        type=Path,
        default=ROOT / "data" / "processed" / "splits" / "subject_split_v1.csv",
        help="Subject-level split output CSV.",
    )
    parser.add_argument(
        "--output-session-split",
        type=Path,
        default=ROOT / "data" / "processed" / "splits" / "session_split_v1.csv",
        help="Session-level split output CSV.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 <= args.val_ratio < 1 and (args.train_ratio + args.val_ratio) < 1):
        print("Invalid split ratios. Need: 0<train<1, 0<=val<1 and train+val<1.", file=sys.stderr)
        sys.exit(2)

    rows = _read_manifest(args.manifest_csv)
    split_map = _assign_subject_splits(
        [r["subject_id"] for r in rows],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    _validate_leakage(rows, split_map)

    _write_subject_split(args.output_subject_split, split_map, seed=args.seed)
    _write_session_split(args.output_session_split, rows, split_map)

    n_sub = len(set(r["subject_id"] for r in rows))
    n_sess = len(rows)
    c_train = sum(1 for s in split_map.values() if s == "train")
    c_val = sum(1 for s in split_map.values() if s == "val")
    c_test = sum(1 for s in split_map.values() if s == "test")

    print(f"Subjects: {n_sub} | Sessions: {n_sess}")
    print(f"Subject split counts -> train={c_train}, val={c_val}, test={c_test}")
    print(f"Subject split CSV: {args.output_subject_split}")
    print(f"Session split CSV: {args.output_session_split}")


if __name__ == "__main__":
    main()

