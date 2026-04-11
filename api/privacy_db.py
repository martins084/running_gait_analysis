"""
privacy_db.py - Shared SQLite helpers for privacy-aware API operations.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def init_db(db_path: Path) -> None:
    """Create tables and perform lightweight schema migrations."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT NOT NULL,
                storage_profile TEXT NOT NULL DEFAULT 'minimal',
                consent_version TEXT,
                consent_timestamp TEXT,
                consent_locale TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delete_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                deleted_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dsar_requests (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _ensure_column(conn, "analyses", "user_id", "TEXT")
        _ensure_column(conn, "analyses", "storage_profile", "TEXT NOT NULL DEFAULT 'minimal'")
        _ensure_column(conn, "analyses", "consent_version", "TEXT")
        _ensure_column(conn, "analyses", "consent_timestamp", "TEXT")
        _ensure_column(conn, "analyses", "consent_locale", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dsar_user_id ON dsar_requests(user_id)"
        )
        conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_sql: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")


def save_analysis(
    db_path: Path,
    *,
    analysis_id: str,
    user_id: str,
    status: str,
    result: dict[str, Any],
    storage_profile: str,
    consent_version: str | None,
    consent_timestamp: str | None,
    consent_locale: str | None,
) -> None:
    payload = json.dumps(result, default=str)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                id, user_id, status, result_json, storage_profile, consent_version, consent_timestamp, consent_locale
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id=excluded.user_id,
                status=excluded.status,
                result_json=excluded.result_json,
                storage_profile=excluded.storage_profile,
                consent_version=excluded.consent_version,
                consent_timestamp=excluded.consent_timestamp,
                consent_locale=excluded.consent_locale
            """,
            (
                analysis_id,
                user_id,
                status,
                payload,
                storage_profile,
                consent_version,
                consent_timestamp,
                consent_locale,
            ),
        )
        conn.commit()


def load_analysis(db_path: Path, analysis_id: str, user_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE id = ? AND user_id = ?",
            (analysis_id, user_id),
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])
