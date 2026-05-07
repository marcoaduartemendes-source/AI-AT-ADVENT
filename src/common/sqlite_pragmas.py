"""Idempotent SQLite pragma application.

Per the 2026-05-07 perf audit (#4), writer DBs in this repo were never
WAL-mode and ran with default sync settings. At ~30 inserts/cycle,
performance was fine but long-tail fragmentation cost grows over months.

This module lets the writers (PerformanceTracker, EquitySnapshotDB,
StrategyRegistry, AlertDedup, etc.) opt into WAL + tuned pragmas with
a single call from `_conn()`. The pragmas are remembered per-path so
re-applying them is a cheap dict lookup.

Why these pragmas:
  journal_mode = WAL       — concurrent readers don't block the writer;
                              dashboard build can read while orchestrator
                              writes without "database is locked" errors.
  synchronous  = NORMAL    — balanced durability/speed; on a fsync-failure
                              the worst case is a lost transaction, not
                              a corrupt DB. Acceptable for trading state
                              that's also dual-written to Supabase /
                              backed up to Spaces nightly.
  temp_store   = MEMORY    — temp tables / sorts in RAM, not /tmp.
  cache_size   = -8000     — 8 MiB page cache (negative = KiB).
"""
from __future__ import annotations

import logging
import sqlite3
from threading import Lock

logger = logging.getLogger(__name__)

_APPLIED: set[str] = set()
_LOCK = Lock()


def apply_pragmas(conn: sqlite3.Connection, db_path: str) -> sqlite3.Connection:
    """Apply tuned pragmas to `conn` exactly once per `db_path` per process.

    Returns the same connection for chaining.

    Best-effort: if any pragma fails (e.g. the DB is on a filesystem that
    doesn't support WAL), the failure is logged at DEBUG and the
    connection is returned as-is. Functional correctness must not depend
    on these pragmas.
    """
    with _LOCK:
        if db_path in _APPLIED:
            return conn
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA cache_size = -8000")
    except sqlite3.Error as e:
        logger.debug(f"sqlite pragma application failed for {db_path}: {e}")
        return conn
    with _LOCK:
        _APPLIED.add(db_path)
    return conn
