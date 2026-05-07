"""Persistent error log — full stack traces with rotation.

Per the operational-reliability audit (#7, 2026-05-07): when a cycle
errored, the orchestrator captured `str(exc)` into `report.errors` and
the stack trace was lost unless an SSH session was open at the
moment. Debugging a stale failure meant correlating GH-Actions logs
with VPS journalctl with the dashboard's stale-data badge.

This module persists the full traceback with strategy + venue context
so the dashboard / a future error-feed can show the last N. Rotation
is by row count (most recent 1000) — small, predictable disk usage.

Usage:
    from common.errors_db import record_error
    try:
        do_thing()
    except Exception:
        record_error(scope="orchestrator.run_cycle",
                       strategy="tsmom_etf",
                       venue="alpaca")
        raise
"""
from __future__ import annotations

import logging
import os
import sqlite3
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "data/errors.db"
_RETAIN_ROWS = 1000


def _db_path() -> str:
    return os.environ.get("ERRORS_DB_PATH", _DEFAULT_PATH)


@contextmanager
def _conn():
    p = _db_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    try:
        from common.sqlite_pragmas import apply_pragmas
        apply_pragmas(conn, p)
    except Exception:
        pass
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS errors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            scope       TEXT    NOT NULL,
            strategy    TEXT,
            venue       TEXT,
            exc_type    TEXT    NOT NULL,
            exc_message TEXT    NOT NULL,
            traceback   TEXT    NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(timestamp DESC)"
    )


def record_error(
    *,
    scope: str,
    strategy: str | None = None,
    venue: str | None = None,
) -> None:
    """Persist the currently-active exception's full traceback.

    Best-effort: failures here are themselves swallowed so the caller's
    own exception path isn't disturbed.
    """
    exc_type, exc_value, _tb = (None, None, None)
    import sys
    exc_info = sys.exc_info()
    if exc_info and exc_info[0] is not None:
        exc_type, exc_value, _tb = exc_info
    if exc_type is None:
        return
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, _tb))
    try:
        with _conn() as c:
            _ensure_schema(c)
            c.execute(
                "INSERT INTO errors "
                "(timestamp, scope, strategy, venue, exc_type, "
                " exc_message, traceback) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    scope,
                    strategy,
                    venue,
                    exc_type.__name__,
                    str(exc_value)[:1000],
                    tb_text[:8000],
                ),
            )
            # Rotate. Cheap because of the timestamp index.
            c.execute(
                "DELETE FROM errors WHERE id NOT IN ("
                "  SELECT id FROM errors ORDER BY id DESC LIMIT ?"
                ")",
                (_RETAIN_ROWS,),
            )
    except Exception as e:
        logger.debug(f"record_error failed: {e}")


def recent_errors(limit: int = 20) -> list[dict]:
    """Return the most recent N error rows for dashboard surfacing."""
    p = _db_path()
    if not Path(p).exists():
        return []
    try:
        with _conn() as c:
            _ensure_schema(c)
            rows = c.execute(
                "SELECT id, timestamp, scope, strategy, venue, "
                "       exc_type, exc_message, traceback "
                "  FROM errors ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.debug(f"recent_errors failed: {e}")
        return []
