"""Tests for the book-vitals watchdog (common/watchdog.py).

Pins the invariant that the May-June 2026 month-long silent freeze can
never recur unnoticed: a stale kill latch or a silent book must produce
an ALERT finding even when every cycle exits 0 and heartbeats are green.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from common import watchdog as wd


@pytest.fixture(autouse=True)
def _isolate_watchdog_file(tmp_path, monkeypatch):
    """Redirect docs/watchdog.json + alerts so tests never touch repo
    state or fire real notifications."""
    monkeypatch.setattr(wd, "WATCHDOG_PATH", tmp_path / "watchdog.json")
    fired = []
    import common.alerts as alerts_mod
    monkeypatch.setattr(alerts_mod, "alert",
                        lambda msg, severity="info": fired.append(msg))
    return fired


def _trades_db(tmp_path, last_trade_at: datetime | None):
    db = tmp_path / "trades.db"
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, "
                  "timestamp TEXT)")
        if last_trade_at is not None:
            c.execute("INSERT INTO trades (timestamp) VALUES (?)",
                      (last_trade_at.isoformat(),))
    return str(db)


class TestStaleLatch:
    def test_month_old_latch_with_benign_dd_alerts(self, tmp_path):
        armed = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        out = wd.check_book_vitals(
            kill_switch="KILL", kill_switch_at=armed,
            drawdown_pct=0.028, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, datetime.now(UTC)),
        )
        assert out["status"] == "ALERT"
        assert any(f["key"] == "stale_latch" for f in out["findings"])

    def test_fresh_kill_does_not_alert(self, tmp_path):
        # Armed 1 hour ago — operator is presumably handling it.
        armed = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        out = wd.check_book_vitals(
            kill_switch="KILL", kill_switch_at=armed,
            drawdown_pct=0.16, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, datetime.now(UTC)),
        )
        assert not any(f["key"] == "stale_latch" for f in out["findings"])

    def test_justified_old_kill_does_not_alert(self, tmp_path):
        # Latched 3 days but drawdown is REAL (12% of a 15% threshold) —
        # the latch is doing its job; don't call it stale.
        armed = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        out = wd.check_book_vitals(
            kill_switch="KILL", kill_switch_at=armed,
            drawdown_pct=0.12, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, datetime.now(UTC)),
        )
        assert not any(f["key"] == "stale_latch" for f in out["findings"])


class TestSilentBook:
    def test_week_of_weekday_silence_alerts(self, tmp_path):
        last = datetime.now(UTC) - timedelta(days=7)
        out = wd.check_book_vitals(
            kill_switch="NORMAL", kill_switch_at=None,
            drawdown_pct=0.01, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, last),
        )
        assert any(f["key"] == "silent_book" for f in out["findings"])

    def test_recent_trade_is_quiet(self, tmp_path):
        last = datetime.now(UTC) - timedelta(hours=3)
        out = wd.check_book_vitals(
            kill_switch="NORMAL", kill_switch_at=None,
            drawdown_pct=0.01, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, last),
        )
        assert out["status"] == "OK"

    def test_cold_start_empty_db_is_quiet(self, tmp_path):
        out = wd.check_book_vitals(
            kill_switch="NORMAL", kill_switch_at=None,
            drawdown_pct=0.0, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, None),
        )
        assert out["status"] == "OK"

    def test_kill_state_suppresses_silent_book(self, tmp_path):
        # While latched, silence is EXPECTED — the stale-latch finding
        # owns that condition; silent_book must not double-page.
        last = datetime.now(UTC) - timedelta(days=10)
        out = wd.check_book_vitals(
            kill_switch="KILL", kill_switch_at=None,
            drawdown_pct=0.01, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, last),
        )
        assert not any(f["key"] == "silent_book" for f in out["findings"])


class TestWeekdayHours:
    def test_weekend_excluded(self):
        # Friday 20:00 UTC → Monday 14:00 UTC = 4h Fri + 14h Mon = 18h
        # (Sat + Sun contribute zero despite 48 wall-clock hours).
        fri = datetime(2026, 6, 5, 20, 0, tzinfo=UTC)   # Friday
        mon = datetime(2026, 6, 8, 14, 0, tzinfo=UTC)   # Monday
        h = wd._weekday_hours_between(fri, mon)
        assert 16 <= h <= 20


class TestAlertDedupe:
    def test_persistent_condition_alerts_once(self, tmp_path,
                                               _isolate_watchdog_file):
        fired = _isolate_watchdog_file
        armed = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        kwargs = dict(
            kill_switch="KILL", kill_switch_at=armed,
            drawdown_pct=0.01, kill_dd_pct=0.15,
            trades_db_path=_trades_db(tmp_path, datetime.now(UTC)),
        )
        wd.check_book_vitals(**kwargs)
        n_after_first = len(fired)
        wd.check_book_vitals(**kwargs)   # same cycle 5 minutes later
        assert len(fired) == n_after_first, (
            "persistent finding must not re-page every cycle")
