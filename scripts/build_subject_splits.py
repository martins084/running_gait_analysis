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

from sklearn.model_selection import train_test_split

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


def _to_int(v: object, default: int = 0) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _subject_injury_labels(session_rows: list[dict]) -> dict[str, int]:
    """
    One label per subject: 1 if any session is marked injured, else 0.
    Used for stratified splitting so train/val/test keep similar injury prevalence.
    """
    out: dict[str, int] = {}
    for r in session_rows:
        sid = r["subject_id"]
        inj = _to_int(r.get("is_injured"), 0)
        out[sid] = max(out.get(sid, 0), inj)
    return out


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


def _assign_subject_splits_stratified(
    subject_ids: list[str],
    subject_y: dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """
    Three-way split with sklearn stratification on subject-level injury label.
    Falls back to the non-stratified routine if there are not enough positives
    for each fold (sklearn would raise).
    """
    ids = sorted(set(subject_ids))
    y = [subject_y.get(s, 0) for s in ids]
    n = len(ids)
    test_size = 1.0 - train_ratio - val_ratio
    if test_size <= 0.0 or n < 3:
        return _assign_subject_splits(subject_ids, train_ratio, val_ratio, seed)

    # Both classes are required for stratify=.
    if len(set(y)) < 2 or sum(y) < 2 or (n - sum(y)) < 2:
        print(
            "Stratify skipped: need at least 2 positive and 2 negative subjects. Using random split.",
            file=sys.stderr,
        )
        return _assign_subject_splits(subject_ids, train_ratio, val_ratio, seed)

    try:
        # First: isolate test set, preserving injury ratio.
        tr_va_ids, te_ids, y_tr_va, y_te = train_test_split(
            ids,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=y,
        )
        # Second: split remaining into train and val; val is `val_ratio` of all subjects.
        rel_val = val_ratio / (train_ratio + val_ratio)
        tr_ids, va_ids, _, _ = train_test_split(
            tr_va_ids,
            y_tr_va,
            test_size=rel_val,
            random_state=seed + 1,
            shuffle=True,
            stratify=y_tr_va,
        )
    except ValueError as e:
        print(f"Stratify failed ({e}); using random subject split instead.", file=sys.stderr)
        return _assign_subject_splits(subject_ids, train_ratio, val_ratio, seed)

    split_map: dict[str, str] = {}
    for sid in tr_ids:
        split_map[sid] = "train"
    for sid in va_ids:
        split_map[sid] = "val"
    for sid in te_ids:
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
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratify subject split by injury (requires is_injured in manifest; subject positive if any session is injured).",
    )
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1 and 0 <= args.val_ratio < 1 and (args.train_ratio + args.val_ratio) < 1):
        print("Invalid split ratios. Need: 0<train<1, 0<=val<1 and train+val<1.", file=sys.stderr)
        sys.exit(2)

    rows = _read_manifest(args.manifest_csv)
    if args.stratify:
        y_by_subj = _subject_injury_labels(rows)
        split_map = _assign_subject_splits_stratified(
            [r["subject_id"] for r in rows],
            y_by_subj,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
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

    # Session-level injury rates per split (for sanity-checking stratify).
    def _rate(split_name: str) -> tuple[int, int, float]:
        sess = [r for r in rows if split_map.get(r["subject_id"]) == split_name]
        inj = sum(_to_int(r.get("is_injured"), 0) for r in sess)
        tot = len(sess)
        return inj, tot, (inj / tot) if tot else 0.0

    t_inj, t_n, t_r = _rate("train")
    v_inj, v_n, v_r = _rate("val")
    e_inj, e_n, e_r = _rate("test")

    print(f"Subjects: {n_sub} | Sessions: {n_sess}")
    print(f"Subject split counts -> train={c_train}, val={c_val}, test={c_test}")
    if args.stratify:
        print(
            f"Injured session rate: train {t_inj}/{t_n}={t_r:.1%} | "
            f"val {v_inj}/{v_n}={v_r:.1%} | test {e_inj}/{e_n}={e_r:.1%}"
        )
    print(f"Subject split CSV: {args.output_subject_split}")
    print(f"Session split CSV: {args.output_session_split}")


if __name__ == "__main__":
    main()

