"""One-shot DB migrations for the trading-performance store.

Audit-fix #4 follow-up: the orchestrator was running an UPDATE on every
init to null out phantom-loss rows from a long-fixed bug (commit
fc9640f, March 2026). The query was idempotent but ran on every
process start — wasted I/O on a hot path and hid behind a broad
`except Exception` that demoted any real schema problem to a warning.

Migrations live here now:
  * Each migration has a stable `name` and an `apply(conn)` function
  * A `schema_migrations` table tracks which have run
  * The orchestrator calls `apply_pending(conn)` exactly once at boot
  * Re-runs are no-ops (already-recorded migrations skip)

Adding a migration: append to `MIGRATIONS` with a NEW name. Never
mutate an existing migration's name or body — the marker row will
prevent it from re-running, masking the change.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    name: str
    apply: Callable[[sqlite3.Connection], int]   # returns row count touched


def _migration_001_null_phantom_pnl(conn: sqlite3.Connection) -> int:
    """Trades captured at submit (price=0) had pnl_usd computed as
    (0 − entry_price) × qty → phantom losses. Null those out so the
    dashboard / allocator metrics see consistent state.

    Tolerant of a missing `trades` table — that's the brand-new-
    install case where there's nothing to migrate. We mark the
    migration applied anyway so a future install with PerformanceTracker
    creating the table doesn't re-run this against fresh data.
    """
    has_table = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='trades'"
    ).fetchone()
    if not has_table:
        return 0
    cur = conn.execute(
        "UPDATE trades SET pnl_usd = NULL "
        "WHERE pnl_usd IS NOT NULL AND (price IS NULL OR price = 0)"
    )
    return cur.rowcount or 0


# Append-only registry. Once a migration ships, do not rename or delete
# its entry — we'd break replay consistency for VPSes that ran an
# older version. Add new ones below with a higher index in the name.
MIGRATIONS: list[Migration] = [
    Migration(
        name="001_null_phantom_pnl",
        apply=_migration_001_null_phantom_pnl,
    ),
]


def apply_pending(db_path: str) -> dict:
    """Run any migrations that haven't recorded a marker row yet.

    Returns a summary dict {migration_name: {"applied": bool,
    "rows": int}} suitable for logging.

    Errors per-migration are isolated — a failing migration logs at
    WARNING (so journalctl surfaces it) and we proceed to the next.
    Schema problems serious enough to fail every migration are visible
    via the empty summary + WARNING log.
    """
    out: dict[str, dict] = {}
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        logger.warning(f"migrations: cannot open {db_path}: {e}")
        return out
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name        TEXT PRIMARY KEY,
                applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                rows_touched INTEGER NOT NULL DEFAULT 0
            )
        """)
        applied_set = {
            r[0] for r in conn.execute(
                "SELECT name FROM schema_migrations"
            ).fetchall()
        }
        for m in MIGRATIONS:
            if m.name in applied_set:
                out[m.name] = {"applied": False, "rows": 0}
                continue
            try:
                rows = m.apply(conn)
                conn.execute(
                    "INSERT INTO schema_migrations (name, rows_touched) "
                    "VALUES (?, ?)",
                    (m.name, rows),
                )
                conn.commit()
                if rows > 0:
                    logger.info(
                        f"migration {m.name}: {rows} row(s) touched"
                    )
                out[m.name] = {"applied": True, "rows": rows}
            except sqlite3.Error as e:
                logger.warning(f"migration {m.name} failed: {e}")
                out[m.name] = {"applied": False, "error": str(e)}
    finally:
        conn.close()
    return out
