"""
feature_summary_csv.py — One-row-per-video CSV export shared by batch pipelines.

Used by `scripts/process_videos.py` and `scripts/batch_feature_export.py` so
summary schema stays identical whether features come from a fresh run or from
existing pose JSON files.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


def safe_float(value) -> float:
    """Convert numeric-like values to float; return NaN for missing/bad values."""
    try:
        if value is None:
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def summary_stats(values: list[float], prefix: str) -> dict[str, float]:
    """
    Produce thesis-friendly summary stats for a numeric series.
    Uses NaN-safe aggregation so partially-missing series still export.
    """
    if not values:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_count": 0.0,
        }

    arr = [v for v in values if not math.isnan(v)]
    if not arr:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_count": 0.0,
        }

    mean = sum(arr) / len(arr)
    var = sum((x - mean) ** 2 for x in arr) / len(arr)
    std = math.sqrt(var)
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": min(arr),
        f"{prefix}_max": max(arr),
        f"{prefix}_count": float(len(arr)),
    }


def build_feature_summary_row(
    analysis_id: str,
    video_name: str,
    video_path: str,
    features: dict,
) -> dict[str, float | str]:
    """
    Flatten nested features into one CSV row (one analyzed video).
    """
    row: dict[str, float | str] = {
        "id": analysis_id,
        "video_name": video_name,
        "video_path": video_path,
    }

    stride = features.get("stride_metrics", {}) if isinstance(features, dict) else {}
    for k, v in stride.items():
        row[f"stride_{k}"] = safe_float(v)

    symmetry = features.get("symmetry", []) if isinstance(features, dict) else []
    symmetry_vals = [safe_float(v) for v in symmetry if isinstance(v, (int, float))]
    row.update(summary_stats(symmetry_vals, "symmetry"))

    joint_angles = features.get("joint_angles", []) if isinstance(features, dict) else []
    for joint_name in ("left_hip", "right_hip", "left_knee", "right_knee"):
        vals = [
            safe_float(frame.get(joint_name))
            for frame in joint_angles
            if isinstance(frame, dict)
        ]
        row.update(summary_stats(vals, joint_name))

    row["vertical_oscillation_px"] = safe_float(features.get("vertical_oscillation_px"))
    return row


def video_names_in_summary_csv(csv_path: Path) -> set[str]:
    """Basenames already present in features_summary.csv (column video_name)."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "video_name" not in reader.fieldnames:
            return set()
        return {row.get("video_name", "") for row in reader if row.get("video_name")}


def analysis_ids_in_summary_csv(csv_path: Path) -> set[str]:
    """Analysis ids already present (column `id`), if present in header."""
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            return set()
        return {row.get("id", "") for row in reader if row.get("id")}


def append_summary_csv(csv_path: Path, row: dict[str, float | str]) -> None:
    """
    Append one analysis row to CSV, creating file/header when needed.
    If schema expands later, it rewrites safely with merged headers.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    preferred = [
        "id",
        "video_name",
        "video_path",
    ]
    dynamic_keys = sorted(k for k in row.keys() if k not in preferred)
    fieldnames = preferred + dynamic_keys

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_fields = reader.fieldnames or []

    merged_fields = list(existing_fields)
    for fn in fieldnames:
        if fn not in merged_fields:
            merged_fields.append(fn)

    existing_rows.append({k: row.get(k, "") for k in merged_fields})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({k: r.get(k, "") for k in merged_fields})
