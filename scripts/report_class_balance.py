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
    for split in ("train", "val", "test"):
        rows = by_split.get(split, [])
        n = len(rows)
        inj = sum(1 for r in rows if _int(r.get("is_injured")) == 1)
        # One subject can have multiple sessions; mark positive if any session is injured.
        by_sub: dict[str, int] = {}
        for r in rows:
            sid = str(r.get("subject_id", ""))
            if not sid:
                continue
            v = _int(r.get("is_injured"))
            by_sub[sid] = max(by_sub.get(sid, 0), v)
        n_subj = len(by_sub)
        subj_pos = sum(1 for v in by_sub.values() if v == 1)
        out[split] = {
            "sessions_total": n,
            "sessions_injured": inj,
            "session_injured_rate": (inj / n) if n else 0.0,
            "subjects_total": n_subj,
            "subjects_any_injured": subj_pos,
            "subject_injured_rate": (subj_pos / n_subj) if n_subj else 0.0,
        }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for split in ("train", "val", "test"):
            s = out[split]
            print(
                f"{split:5s}  sessions: {s['sessions_injured']}/{s['sessions_total']} "
                f"({100 * float(s['session_injured_rate']):.1f}% injured) | "
                f"subjects: {s['subjects_any_injured']}/{s['subjects_total']} "
                f"({100 * float(s['subject_injured_rate']):.1f}%)"
            )


if __name__ == "__main__":
    main()
