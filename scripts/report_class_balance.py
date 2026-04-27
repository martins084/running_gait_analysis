"""
Print session-level and subject-level injury prevalence from a session split CSV.

Use this to verify stratified splits and to set expectations for AUROC vs AUPRC
(reconstruction + threshold models are not the same as supervised BCE with pos_weight;
class imbalance still affects rank metrics and thresholded accuracy).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _int(v: object) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Report injury class balance in a session_split CSV.")
    p.add_argument(
        "--session-split",
        type=Path,
        default=ROOT / "data" / "processed" / "splits" / "session_split_v1.csv",
        help="Session split CSV (must include is_injured, split, subject_id).",
    )
    p.add_argument(
        "--label-col",
        default="is_injured",
        help="Label column to summarize (e.g. is_injured or is_injured_strict).",
    )
    p.add_argument("--json", action="store_true", help="Print one JSON object instead of text lines.")
    args = p.parse_args()

    if not args.session_split.is_file():
        print(f"File not found: {args.session_split}", file=sys.stderr)
        sys.exit(2)

    by_split: dict[str, list[dict]] = defaultdict(list)
    with open(args.session_split, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_split[str(r.get("split", ""))].append(r)

    out: dict[str, object] = {}
    lab = str(args.label_col)
    for split in ("train", "val", "test"):
        rows = by_split.get(split, [])
        n = len(rows)
        inj = sum(1 for r in rows if _int(r.get(lab)) == 1)
        healthy = n - inj
        # One subject can have multiple sessions; mark positive if any session is injured.
        by_sub: dict[str, int] = {}
        for r in rows:
            sid = str(r.get("subject_id", ""))
            if not sid:
                continue
            v = _int(r.get(lab))
            by_sub[sid] = max(by_sub.get(sid, 0), v)
        n_subj = len(by_sub)
        subj_pos = sum(1 for v in by_sub.values() if v == 1)
        subj_ok = n_subj - subj_pos
        out[split] = {
            "sessions_total": n,
            "sessions_is_injured_1": inj,
            "sessions_is_injured_0": healthy,
            "session_rate_is_injured_1": (inj / n) if n else 0.0,
            "subjects_total": n_subj,
            "subjects_any_is_injured_1": subj_pos,
            "subjects_all_is_injured_0": subj_ok,
            "subject_rate_any_is_injured_1": (subj_pos / n_subj) if n_subj else 0.0,
        }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        if lab == "is_injured":
            print(
                "Column is_injured comes from build_ric_manifest._injury_flag: "
                "1 unless metadata explicitly matches 'no injury' (conservative default → many rows are 1).",
                file=sys.stderr,
            )
        elif lab == "is_injured_strict":
            print(
                "Column is_injured_strict comes from build_ric_manifest._injury_flag_strict: "
                "0 for explicit 'no injury' or fully empty injury fields; 1 for explicit injury metadata.",
                file=sys.stderr,
            )
        else:
            print(f"Using custom label column: {lab}", file=sys.stderr)
        print(f"Using label column: {lab}", file=sys.stderr)
        for split in ("train", "val", "test"):
            s = out[split]
            r1 = 100 * float(s["session_rate_is_injured_1"])
            rs1 = 100 * float(s["subject_rate_any_is_injured_1"])
            print(
                f"{split:5s}  sessions: is_injured=0: {s['sessions_is_injured_0']} | "
                f"is_injured=1: {s['sessions_is_injured_1']} / {s['sessions_total']} "
                f"({r1:.1f}% with label 1) | "
                f"subjects (any session=1): {s['subjects_any_is_injured_1']} / {s['subjects_total']} ({rs1:.1f}%)"
            )


if __name__ == "__main__":
    main()
