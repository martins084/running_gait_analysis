"""
verify_coco_setup.py — Sanity check for local COCO annotation JSON.

Looks for instances_train2017.json / instances_val2017.json under
data/raw/coco/annotations by default. Does not download anything.

Usage (from project root):
  python scripts/verify_coco_setup.py
  python scripts/verify_coco_setup.py --annotations-dir "D:/coco/annotations"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANNOT_DIR = ROOT / "data" / "raw" / "coco" / "annotations"
EXPECTED = (
    "instances_train2017.json",
    "instances_val2017.json",
)

# Only parse the val file for structure check (train JSON is huge and slow to load).
SAMPLE_VALIDATE = "instances_val2017.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify COCO annotation files exist and load.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOT_DIR,
        help=f"Folder with COCO JSON (default: {DEFAULT_ANNOT_DIR})",
    )
    args = parser.parse_args()
    d = args.annotations_dir.resolve()

    if not d.is_dir():
        print(f"[missing] Directory does not exist: {d}")
        return 1

    missing = [name for name in EXPECTED if not (d / name).is_file()]
    if missing:
        print(f"[incomplete] Under {d}, missing: {', '.join(missing)}")
        return 2

    sample_path = d / SAMPLE_VALIDATE
    try:
        with open(sample_path, encoding="utf-8") as f:
            obj = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"[bad-json] {sample_path}: {exc}")
        return 3

    for key in ("images", "annotations"):
        if key not in obj:
            print(f"[unexpected] {sample_path} has no top-level '{key}'")
            return 4

    print(f"[ok] COCO instances JSON looks valid under {d}")
    print(f"     images: {len(obj.get('images', []))}, annotations: {len(obj.get('annotations', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
