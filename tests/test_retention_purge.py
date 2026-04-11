from __future__ import annotations

import sqlite3
from pathlib import Path

from api.retention import RetentionPolicy, purge_expired_analyses


def test_purge_expired_analyses_removes_rows_and_files(tmp_path: Path):
    db = tmp_path / "analysis_results.db"
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                storage_profile TEXT NOT NULL,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO analyses (id, user_id, status, result_json, storage_profile, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("old", "u", "completed", "{}", "minimal", "2000-01-01 00:00:00"),
        )
        conn.commit()
    (results / "old_result.json").write_text("{}", encoding="utf-8")
    (results / "old_poses.json").write_text("{}", encoding="utf-8")
    (results / "old_annotated.mp4").write_bytes(b"x")
    deleted = purge_expired_analyses(
        db,
        results,
        RetentionPolicy(keep_minimal_days=1, keep_standard_days=1, keep_full_days=1),
    )
    assert deleted == 1
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT id FROM analyses WHERE id = 'old'").fetchone()
    assert row is None
