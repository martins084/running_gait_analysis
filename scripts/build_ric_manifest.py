"""
Build a session-level manifest for the RIC dataset.

This script supports two inventory sources:
1) Local extracted folder (`reformat_data/<subject>/<session>.json`)
2) Spaces prefix via `aws s3 ls --recursive` output parsing

Usage examples:
  python scripts/build_ric_manifest.py --local-dir "/data/reformat_data"
  python scripts/build_ric_manifest.py --s3-prefix "s3://bakalaurs-ams/processed/reformat_data/" \
      --endpoint-url "https://ams3.digitaloceanspaces.com"
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class SessionKey:
    subject_id: str
    filename: str


def _norm(v: object) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _lower(v: object) -> str:
    return _norm(v).lower()


def _canon_id(v: object) -> str:
    # Canonical IDs prevent split leakage from case/whitespace aliases (e.g., "S01 " vs "s01").
    return _norm(v).lower()


_NO_INJURY_TOKENS = {"no injury", "noinjury"}
_EMPTY_TOKENS = {"", "n/a", "null"}


def _injury_flag(row: dict) -> int:
    """
    Conservative injury flag inference from metadata.
    Returns:
      1 -> injured
      0 -> not injured
    """
    inj_defn = _lower(row.get("InjDefn") or row.get("injdefn"))
    inj_joint = _lower(row.get("InjJoint") or row.get("injjoint"))
    spec = _lower(row.get("SpecInjury") or row.get("specinjury"))

    if inj_defn in _NO_INJURY_TOKENS and (
        inj_joint in (_EMPTY_TOKENS | _NO_INJURY_TOKENS) and spec in _EMPTY_TOKENS
    ):
        return 0
    if inj_joint in _NO_INJURY_TOKENS and spec in _EMPTY_TOKENS and inj_defn in _EMPTY_TOKENS:
        return 0
    return 1


def _injury_flag_strict(row: dict) -> int:
    """
    Stricter injury label:
      1 -> explicit injury-related metadata present
      0 -> explicit 'no injury' OR all relevant fields empty/unknown

    This is provided alongside the legacy conservative flag (`is_injured`) so
    downstream experiments can compare label definitions without breaking old runs.
    """
    inj_defn = _lower(row.get("InjDefn") or row.get("injdefn"))
    inj_joint = _lower(row.get("InjJoint") or row.get("injjoint"))
    spec = _lower(row.get("SpecInjury") or row.get("specinjury"))
    vals = [inj_defn, inj_joint, spec]

    # Any explicit "no injury" claim dominates.
    if any(v in _NO_INJURY_TOKENS for v in vals):
        return 0

    # Remove empty/unknown placeholders. If nothing remains, do not mark injured.
    informative = [v for v in vals if v not in _EMPTY_TOKENS]
    if not informative:
        return 0
    return 1


def _load_meta(path: Path, mode_name: str) -> dict[SessionKey, dict]:
    out: dict[SessionKey, dict] = {}
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        for row in reader:
            sid = _canon_id(row.get("sub_id") or row.get("subid"))
            fn = _norm(row.get("filename"))
            if not sid or not fn:
                continue
            key = SessionKey(subject_id=sid, filename=fn)
            merged = dict(row)
            merged["_mode_source"] = mode_name
            merged["_is_injured"] = _injury_flag(row)
            merged["_is_injured_strict"] = _injury_flag_strict(row)
            out[key] = merged
    return out


def _iter_local_sessions(local_dir: Path) -> Iterable[tuple[str, str, str]]:
    """
    Yields (subject_id, filename, source_ref)
    source_ref for local mode is path relative to local_dir.
    """
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    for p in sorted(local_dir.glob("*/*.json")):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        subject_id = _canon_id(p.parent.name)
        yield subject_id, p.name, rel


def _iter_s3_sessions(s3_prefix: str, endpoint_url: str) -> Iterable[tuple[str, str, str]]:
    """
    Yields (subject_id, filename, source_ref) from `aws s3 ls --recursive`.
    source_ref for s3 mode is the object key.
    """
    cmd = ["aws", "s3", "ls", s3_prefix, "--recursive"]
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])

    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to list S3 objects: {exc.stderr or exc.stdout}") from exc

    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        key = parts[3]
        if not key.endswith(".json"):
            continue
        key_parts = key.split("/")
        if len(key_parts) < 2:
            continue
        subject_id = _canon_id(key_parts[-2])
        filename = key_parts[-1]
        yield subject_id, filename, key


def _safe_float(v: object) -> str:
    s = _norm(v)
    if not s:
        return ""
    try:
        return f"{float(s):.6g}"
    except ValueError:
        return ""


def _build_rows(
    sessions: Iterable[tuple[str, str, str]],
    run_meta: dict[SessionKey, dict],
    walk_meta: dict[SessionKey, dict],
    inventory_type: str,
) -> list[dict]:
    rows: list[dict] = []
    missing_meta = 0
    for subject_id, filename, source_ref in sessions:
        key = SessionKey(subject_id=subject_id, filename=filename)
        run_row = run_meta.get(key)
        walk_row = walk_meta.get(key)
        if run_row is None and walk_row is None:
            missing_meta += 1

        # If both rows exist, treat as both-modes session and prefer run row for shared demographics.
        base = run_row or walk_row or {}
        has_run = 1 if run_row else 0
        has_walk = 1 if walk_row else 0

        mode = "both" if (has_run and has_walk) else ("run" if has_run else ("walk" if has_walk else "unknown"))
        label_provenance = "matched_metadata"
        is_injured = base.get("_is_injured")
        if is_injured is None:
            is_injured = 0
            label_provenance = "missing_metadata"
        is_injured_strict = base.get("_is_injured_strict")
        if is_injured_strict is None:
            is_injured_strict = 0
            label_provenance = "missing_metadata"

        if run_row is not None and walk_row is not None:
            run_lab = int(run_row.get("_is_injured", is_injured))
            walk_lab = int(walk_row.get("_is_injured", is_injured))
            if run_lab != walk_lab:
                label_provenance = "conflicting_mode_metadata"

        rows.append(
            {
                "subject_id": subject_id,
                "session_id": Path(filename).stem,
                "filename": filename,
                "inventory_type": inventory_type,
                "source_ref": source_ref,
                "mode": mode,
                "has_run": has_run,
                "has_walk": has_walk,
                "is_injured": int(is_injured),
                "is_injured_strict": int(is_injured_strict),
                "label_provenance": label_provenance,
                "speed_run_mps": _safe_float((run_row or {}).get("speed_r")),
                "speed_walk_mps": _safe_float((walk_row or {}).get("speed_w")),
                "age_years": _safe_float(base.get("age")),
                "height_cm": _safe_float(base.get("Height") or base.get("height")),
                "weight_kg": _safe_float(base.get("Weight") or base.get("weight")),
                "gender": _norm(base.get("Gender") or base.get("gender")),
                "injury_definition": _norm(base.get("InjDefn") or base.get("injdefn")),
                "injury_joint": _norm(base.get("InjJoint") or base.get("injjoint")),
                "injury_specific": _norm(base.get("SpecInjury") or base.get("specinjury")),
                "datestring": _norm(base.get("datestring")),
            }
        )

    if missing_meta:
        print(f"Warning: {missing_meta} session(s) had no metadata match.", file=sys.stderr)
    return rows


def _write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "session_id",
        "filename",
        "inventory_type",
        "source_ref",
        "mode",
        "has_run",
        "has_walk",
        "is_injured",
        "is_injured_strict",
        "label_provenance",
        "speed_run_mps",
        "speed_walk_mps",
        "age_years",
        "height_cm",
        "weight_kg",
        "gender",
        "injury_definition",
        "injury_joint",
        "injury_specific",
        "datestring",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _maybe_write_summary(rows: list[dict], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    subjects = len({r["subject_id"] for r in rows})
    run_n = sum(int(r["has_run"]) for r in rows)
    walk_n = sum(int(r["has_walk"]) for r in rows)
    both_n = sum(1 for r in rows if r["mode"] == "both")
    injured_n = sum(int(r["is_injured"]) for r in rows)
    injured_strict_n = sum(int(r["is_injured_strict"]) for r in rows)
    payload = {
        "sessions_total": n,
        "subjects_total": subjects,
        "sessions_with_run": run_n,
        "sessions_with_walk": walk_n,
        "sessions_with_both": both_n,
        "sessions_marked_injured_conservative": injured_n,
        "sessions_marked_injured_strict": injured_strict_n,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RIC manifest CSV from local folder or Spaces prefix.")
    parser.add_argument("--local-dir", type=Path, default=None, help="Local reformat_data directory.")
    parser.add_argument("--s3-prefix", default="", help="Spaces prefix, e.g. s3://bucket/processed/reformat_data/")
    parser.add_argument("--endpoint-url", default="", help="Spaces endpoint URL for aws cli.")
    parser.add_argument(
        "--run-meta-csv",
        type=Path,
        default=ROOT / "data" / "figshare" / "run_data_meta.csv",
        help="Path to run metadata CSV.",
    )
    parser.add_argument(
        "--walk-meta-csv",
        type=Path,
        default=ROOT / "data" / "figshare" / "walk_data_meta.csv",
        help="Path to walk metadata CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "ric_manifest.csv",
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "data" / "processed" / "ric_manifest_summary.json",
        help="Output summary JSON path.",
    )
    args = parser.parse_args()

    if bool(args.local_dir) == bool(args.s3_prefix):
        print("Exactly one of --local-dir or --s3-prefix must be provided.", file=sys.stderr)
        sys.exit(2)

    run_meta = _load_meta(args.run_meta_csv, "run")
    walk_meta = _load_meta(args.walk_meta_csv, "walk")

    if args.local_dir:
        sessions = list(_iter_local_sessions(args.local_dir))
        inventory_type = "local"
    else:
        sessions = list(_iter_s3_sessions(args.s3_prefix, args.endpoint_url))
        inventory_type = "s3"

    rows = _build_rows(sessions, run_meta, walk_meta, inventory_type=inventory_type)
    _write_csv(rows, args.output_csv)
    _maybe_write_summary(rows, args.summary_json)

    print(f"Manifest rows: {len(rows)}")
    print(f"Manifest CSV: {args.output_csv}")
    print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()

