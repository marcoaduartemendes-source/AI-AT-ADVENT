"""Book-vitals watchdog — catches the "healthy zombie" failure mode.

THE LESSON OF MAY-JUNE 2026: the orchestrator ran perfectly for a month
— every cycle exited 0, every heartbeat pinged green — while a stale
kill-switch latch silently blocked every order. Equity drawdown never
exceeded 2.8% (threshold 15%), yet the book sat frozen for ~4 weeks and
nobody was paged, because all existing monitoring watched for CRASHES,
not for SILENCE.

This module watches for the two silent-failure conditions:

  STALE_LATCH  — kill switch latched ≥ 24h while drawdown is below half
                 the kill threshold. Almost certainly a spurious trip
                 (broker $0-equity blip, stray manual arm) that a human
                 forgot to reset.

  SILENT_BOOK  — risk state is NORMAL/WARNING (i.e. trading SHOULD be
                 happening) but no order has been recorded for more than
                 SILENT_WEEKDAY_HOURS of weekday time. Weekend hours are
                 excluded so a calm Saturday doesn't page anyone.

Findings are written to docs/watchdog.json (the dashboard renders any
non-OK status as a red banner) and pushed through common.alerts with a
20h per-condition dedupe so a persistent condition pages daily, not
every 5-minute cycle.

Trading must never block on the watchdog: every entry point swallows
its own exceptions and returns a best-effort result.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

WATCHDOG_PATH = Path("docs/watchdog.json")
VALIDATION_PATH = Path("docs/validation.json")

# A latched KILL older than this with benign drawdown → stale-latch page.
STALE_LATCH_HOURS = 24.0
# Drawdown below this fraction of kill_dd means "nowhere near a real kill".
BENIGN_DD_FRACTION = 0.5
# Weekday-hours of zero order flow before the silent-book page fires.
# 24 weekday-hours ≈ one full trading day of unexpected silence.
SILENT_WEEKDAY_HOURS = 24.0
# Re-alert cadence per condition (don't page every 5-min cycle).
REALERT_HOURS = 20.0


def _weekday_hours_between(start: datetime, end: datetime) -> float:
    """Hours between two datetimes counting only Mon-Fri.

    Coarse by design (whole-hour steps) — the watchdog cares about
    "roughly a trading day of silence", not minute precision.
    """
    if end <= start:
        return 0.0
    hours = 0.0
    cur = start
    step = timedelta(hours=1)
    # Bound the walk: anything > 14 days reads as "very silent" anyway.
    max_steps = 14 * 24
    steps = 0
    while cur < end and steps < max_steps:
        if cur.weekday() < 5:          # Mon=0 … Fri=4
            hours += 1.0
        cur += step
        steps += 1
    if steps >= max_steps:
        return float(max_steps)
    return hours


def _last_trade_at(db_path: str) -> datetime | None:
    """Timestamp of the most recent recorded order, or None."""
    try:
        if not Path(db_path).exists():
            return None
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM trades"
            ).fetchone()
        if row and row[0]:
            dt = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception as e:  # noqa: BLE001 — watchdog never raises
        logger.debug(f"[watchdog] last_trade_at failed: {e}")
    return None


def _load_state() -> dict:
    try:
        if WATCHDOG_PATH.exists():
            return json.loads(WATCHDOG_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def _should_alert(state: dict, key: str, now: datetime) -> bool:
    last = (state.get("last_alerted") or {}).get(key)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        return (now - last_dt) > timedelta(hours=REALERT_HOURS)
    except (ValueError, TypeError):
        return True


def check_book_vitals(
    *,
    kill_switch: str,
    kill_switch_at: str | None,
    drawdown_pct: float,
    kill_dd_pct: float,
    trades_db_path: str | None = None,
) -> dict:
    """Evaluate silent-failure conditions and write docs/watchdog.json.

    Returns the findings dict: {"status": "OK"|"ALERT", "findings": [...]}.
    Called from run_orchestrator.main() at cycle end; all failures are
    swallowed (the watchdog must never break a trading cycle).
    """
    now = datetime.now(UTC)
    findings: list[dict] = []
    state = _load_state()
    last_alerted: dict = state.get("last_alerted") or {}

    try:
        # ── STALE_LATCH ───────────────────────────────────────────────
        if kill_switch == "KILL" and kill_switch_at:
            try:
                armed = datetime.fromisoformat(
                    str(kill_switch_at).replace("Z", "+00:00"))
                if armed.tzinfo is None:
                    armed = armed.replace(tzinfo=UTC)
                latched_h = (now - armed).total_seconds() / 3600
            except (ValueError, TypeError):
                latched_h = None
            if (latched_h is not None
                    and latched_h >= STALE_LATCH_HOURS
                    and drawdown_pct < kill_dd_pct * BENIGN_DD_FRACTION):
                findings.append({
                    "key": "stale_latch",
                    "severity": "critical",
                    "message": (
                        f"Kill switch latched {latched_h/24:.1f} days but "
                        f"drawdown is only {drawdown_pct*100:.1f}% "
                        f"(threshold {kill_dd_pct*100:.0f}%). Latch looks "
                        f"STALE — the book is frozen for no current reason. "
                        f"Run: python src/run_orchestrator.py "
                        f"--reset-kill-switch"
                    ),
                })

        # ── SILENT_BOOK ───────────────────────────────────────────────
        if kill_switch in ("NORMAL", "WARNING"):
            db_path = trades_db_path or os.environ.get(
                "TRADING_DB_PATH", "data/trading_performance.db")
            last = _last_trade_at(db_path)
            if last is not None:
                silent_h = _weekday_hours_between(last, now)
                if silent_h >= SILENT_WEEKDAY_HOURS:
                    findings.append({
                        "key": "silent_book",
                        "severity": "warning",
                        "message": (
                            f"No order recorded for {silent_h:.0f} weekday-"
                            f"hours while risk state is {kill_switch}. "
                            f"The engine is running but nothing is "
                            f"trading — check cycle_status.json "
                            f"strategy_outcomes (BLOCKED? NO_ALLOC? "
                            f"venue creds?)."
                        ),
                    })
            # No trades EVER + NORMAL state is a cold-start, not a page.

        # ── STALE_RESEARCH ────────────────────────────────────────────
        # research.timer runs daily; validation.json silently going
        # stale (observed: 6+ days in June 2026) zeroes the
        # research_freshness grade axis AND blocks new strategies from
        # ever earning PASS. Same silent-failure family as the rest.
        try:
            vpath = VALIDATION_PATH
            if vpath.exists():
                vdata = json.loads(vpath.read_text())
                vas_of = vdata.get("as_of")
                if vas_of:
                    vdt = datetime.fromisoformat(
                        str(vas_of).replace("Z", "+00:00"))
                    if vdt.tzinfo is None:
                        vdt = vdt.replace(tzinfo=UTC)
                    stale_h = (now - vdt).total_seconds() / 3600
                    if stale_h > 48:
                        findings.append({
                            "key": "stale_research",
                            "severity": "warning",
                            "message": (
                                f"validation.json is {stale_h/24:.1f} days "
                                f"old (research.timer should refresh it "
                                f"daily at 06:30). Check: systemctl status "
                                f"research.timer && journalctl -u "
                                f"research.service -n 50"
                            ),
                        })
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[watchdog] research-staleness check failed: {e}")

        # ── Alert (deduped) + persist ─────────────────────────────────
        for f in findings:
            if _should_alert(state, f["key"], now):
                try:
                    from common.alerts import alert
                    alert(f"🐶 WATCHDOG: {f['message']}",
                          severity=f["severity"])
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[watchdog] alert dispatch failed: {e}")
                last_alerted[f["key"]] = now.isoformat()

        out = {
            "as_of": now.isoformat(),
            "status": "ALERT" if findings else "OK",
            "findings": findings,
            "last_alerted": last_alerted,
        }
        try:
            WATCHDOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            WATCHDOG_PATH.write_text(json.dumps(out, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[watchdog] persist failed: {e}")
        if findings:
            for f in findings:
                logger.warning(f"[watchdog] {f['key']}: {f['message']}")
        return out
    except Exception as e:  # noqa: BLE001 — never break the cycle
        logger.warning(f"[watchdog] check failed: {e}")
        return {"as_of": now.isoformat(), "status": "ERROR",
                "findings": [], "last_alerted": last_alerted}
