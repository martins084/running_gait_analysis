"""
retention.py - Retention and purge utilities.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class RetentionPolicy:
    keep_minimal_days: int
    keep_standard_days: int
    keep_full_days: int

    def cutoff_for_profile(self, profile: str) -> datetime:
        now = datetime.now(timezone.utc)
        if profile == "full":
            return now - timedelta(days=self.keep_full_days)
        if profile == "standard":
            return now - timedelta(days=self.keep_standard_days)
        return now - timedelta(days=self.keep_minimal_days)


def purge_expired_analyses(db_path: Path, results_dir: Path, policy: RetentionPolicy) -> int:
    """Delete expired analysis rows and on-disk artifacts."""
    deleted = 0
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, storage_profile, created_at FROM analyses"
        ).fetchall()
        for analysis_id, profile, created_at in rows:
            if not created_at:
                continue
            created = _parse_created_at(created_at)
            if created is None:
                continue
            if created >= policy.cutoff_for_profile(profile or "minimal"):
                continue
            _delete_artifacts(results_dir, analysis_id)
            conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            deleted += 1
        conn.commit()
    return deleted


def _parse_created_at(raw: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _delete_artifacts(results_dir: Path, analysis_id: str) -> None:
    for suffix in ("_poses.json", "_annotated.mp4", "_result.json"):
        (results_dir / f"{analysis_id}{suffix}").unlink(missing_ok=True)
