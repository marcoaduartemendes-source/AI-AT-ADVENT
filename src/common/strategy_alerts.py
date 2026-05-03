"""Per-strategy consecutive-error tracker → Pushover alert.

Sprint E1 audit fix: pre-fix the only error escalation was the
PnL-drift sanity check (orchestrator.py:680). A strategy that's
been quietly raising on every cycle for 6 hours wasn't surfaced
unless its P&L drifted enough to hit the magnitude threshold.

This module tracks consecutive cycles where a given strategy
emitted at least one error. When the count crosses CONSECUTIVE_FAIL_THRESHOLD,
alerts.alert() fires. Counter resets on the first clean cycle.

State lives in a tiny SQLite file so the count survives restarts.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, UTC
from pathlib import Path

from .alerts import alert as _alert

logger = logging.getLogger(__name__)


CONSECUTIVE_FAIL_THRESHOLD = 3   # alert after 3 cycles in a row with errors


def _db_path() -> str:
    p = os.environ.get(
        "STRATEGY_ALERTS_DB", "data/strategy_alert_state.db"
    )
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    return p


def _conn():
    p = _db_path()
    c = sqlite3.connect(p)
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE IF NOT EXISTS strategy_consecutive_errors (
            strategy            TEXT PRIMARY KEY,
            consecutive_errors  INTEGER NOT NULL DEFAULT 0,
            last_alert_count    INTEGER NOT NULL DEFAULT 0,
            last_error_text     TEXT,
            last_seen_at        TEXT NOT NULL
        )
    """)
    return c


def record_cycle_outcome(
    strategy: str,
    *,
    had_error: bool,
    error_text: str | None = None,
    threshold: int = CONSECUTIVE_FAIL_THRESHOLD,
    alert_fn=None,
) -> dict:
    """Update the consecutive-error counter for `strategy`.

    Returns a dict:
        {"strategy", "count", "alerted"}

    `alerted` is True when this call crossed the threshold and an
    alert was dispatched. Subsequent threshold-crossings within the
    same error run do NOT re-alert until the counter resets — we
    don't want a flapping strategy to spam Pushover every 5 minutes.

    `alert_fn` defaults to common.alerts.alert; injectable for tests.
    """
    fn = alert_fn or _alert
    now_iso = datetime.now(UTC).isoformat()
    with _conn() as c:
        row = c.execute(
            "SELECT consecutive_errors, last_alert_count "
            "FROM strategy_consecutive_errors WHERE strategy = ?",
            (strategy,),
        ).fetchone()
        prev_count = row["consecutive_errors"] if row else 0
        last_alert = row["last_alert_count"] if row else 0

        if had_error:
            new_count = prev_count + 1
            new_last_err = error_text or ""
            crossed = (new_count >= threshold and last_alert < threshold)
            new_last_alert = max(last_alert, new_count if crossed else last_alert)
            c.execute(
                "INSERT INTO strategy_consecutive_errors "
                "(strategy, consecutive_errors, last_alert_count, "
                " last_error_text, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(strategy) DO UPDATE SET "
                "  consecutive_errors=excluded.consecutive_errors, "
                "  last_alert_count=excluded.last_alert_count, "
                "  last_error_text=excluded.last_error_text, "
                "  last_seen_at=excluded.last_seen_at",
                (strategy, new_count, new_last_alert, new_last_err, now_iso),
            )
            c.commit()
            if crossed:
                try:
                    fn(
                        f"⚠ Strategy [{strategy}] has errored on "
                        f"{new_count} consecutive cycles. "
                        f"Last error: {new_last_err[:300]}",
                        severity="warning",
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"strategy_alerts dispatch failed: {e}"
                    )
            return {
                "strategy": strategy, "count": new_count,
                "alerted": crossed,
            }

        # Clean cycle — reset counter (and last_alert_count so future
        # error runs trigger a fresh alert)
        if prev_count > 0:
            c.execute(
                "UPDATE strategy_consecutive_errors "
                "SET consecutive_errors = 0, last_alert_count = 0, "
                "    last_seen_at = ? "
                "WHERE strategy = ?",
                (now_iso, strategy),
            )
            c.commit()
        return {"strategy": strategy, "count": 0, "alerted": False}


def all_states() -> list[dict]:
    """Return current consecutive-error state for every tracked
    strategy. Used by the dashboard for a per-strategy health column."""
    with _conn() as c:
        rows = c.execute(
            "SELECT strategy, consecutive_errors, last_alert_count, "
            "       last_error_text, last_seen_at "
            "FROM strategy_consecutive_errors"
        ).fetchall()
    return [dict(r) for r in rows]
