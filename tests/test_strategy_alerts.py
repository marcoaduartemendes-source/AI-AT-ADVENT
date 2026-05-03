"""Tests for the per-strategy consecutive-error alert tracker.

Sprint E1 audit fix: pre-fix, a strategy that quietly raised on
every cycle wouldn't surface unless its P&L drifted enough to
trigger the magnitude-based PnL-drift alert. The tracker here
fires Pushover after N consecutive failed cycles and resets on
the first clean one.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from common.strategy_alerts import (
    CONSECUTIVE_FAIL_THRESHOLD,
    all_states,
    record_cycle_outcome,
)


def _isolated_db(monkeypatch, tmp_path):
    """Point the tracker at a per-test SQLite file so tests don't share state."""
    monkeypatch.setenv("STRATEGY_ALERTS_DB", str(tmp_path / "alerts.db"))


def test_clean_cycle_does_not_alert(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    fake_alert = MagicMock()
    out = record_cycle_outcome("tsmom_etf", had_error=False, alert_fn=fake_alert)
    assert out == {"strategy": "tsmom_etf", "count": 0, "alerted": False}
    assert fake_alert.call_count == 0


def test_single_error_no_alert_yet(monkeypatch, tmp_path):
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    out = record_cycle_outcome("rsi_mr", had_error=True,
                                  error_text="API 500", alert_fn=fake)
    assert out["count"] == 1
    assert out["alerted"] is False
    assert fake.call_count == 0


def test_three_consecutive_errors_fire_one_alert(monkeypatch, tmp_path):
    """Threshold = 3 → 3rd error triggers exactly one alert."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    for i in range(3):
        out = record_cycle_outcome("rsi_mr", had_error=True,
                                      error_text=f"err {i}", alert_fn=fake)
    assert out["count"] == 3
    assert out["alerted"] is True
    assert fake.call_count == 1
    msg, = fake.call_args.args
    assert "rsi_mr" in msg
    assert "3 consecutive" in msg
    assert "err 2" in msg


def test_no_repeat_alert_within_same_error_run(monkeypatch, tmp_path):
    """4th, 5th errors in the same run must NOT re-alert — flapping
    strategies otherwise spam Pushover every 5 minutes."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    for i in range(5):
        record_cycle_outcome("rsi_mr", had_error=True,
                                error_text=f"err {i}", alert_fn=fake)
    assert fake.call_count == 1


def test_clean_cycle_resets_count_and_arms_next_alert(monkeypatch, tmp_path):
    """After the count resets, the next 3-in-a-row must alert again."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    # First run: 4 errors → 1 alert
    for _ in range(4):
        record_cycle_outcome("rsi_mr", had_error=True, alert_fn=fake)
    assert fake.call_count == 1

    # Clean cycle resets
    out = record_cycle_outcome("rsi_mr", had_error=False, alert_fn=fake)
    assert out["count"] == 0

    # New error run of 3 → second alert fires
    for _ in range(3):
        record_cycle_outcome("rsi_mr", had_error=True, alert_fn=fake)
    assert fake.call_count == 2


def test_per_strategy_isolation(monkeypatch, tmp_path):
    """Errors on tsmom_etf must not affect rsi_mr's counter."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    for _ in range(3):
        record_cycle_outcome("tsmom_etf", had_error=True, alert_fn=fake)
    assert fake.call_count == 1
    out = record_cycle_outcome("rsi_mr", had_error=False, alert_fn=fake)
    assert out["count"] == 0


def test_alert_dispatch_failure_does_not_raise(monkeypatch, tmp_path):
    """alert_fn raising must never propagate — alerting is best-effort."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock(side_effect=RuntimeError("Pushover down"))
    for _ in range(3):
        out = record_cycle_outcome("rsi_mr", had_error=True, alert_fn=fake)
    # Last call would have crossed threshold and tried to alert
    assert out["count"] == 3
    assert out["alerted"] is True


def test_all_states_returns_per_strategy_rows(monkeypatch, tmp_path):
    """all_states() returns one row per strategy that has EVER had an
    error. Clean-only strategies aren't tracked (saves DB churn) —
    they're trivially "healthy" and need no row."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    record_cycle_outcome("a", had_error=True, alert_fn=fake)
    record_cycle_outcome("a", had_error=True, alert_fn=fake)
    record_cycle_outcome("b", had_error=False, alert_fn=fake)
    rows = {r["strategy"]: r for r in all_states()}
    assert rows["a"]["consecutive_errors"] == 2
    # "b" never errored → no row, which is correct
    assert "b" not in rows


def test_clean_after_error_persists_zero_row(monkeypatch, tmp_path):
    """Once a strategy has errored, the row sticks around so the
    dashboard can show "currently healthy, last alert N days ago"."""
    _isolated_db(monkeypatch, tmp_path)
    fake = MagicMock()
    record_cycle_outcome("c", had_error=True, alert_fn=fake)
    record_cycle_outcome("c", had_error=False, alert_fn=fake)
    rows = {r["strategy"]: r for r in all_states()}
    assert rows["c"]["consecutive_errors"] == 0


def test_threshold_is_three():
    """Lock the contract — if we ever bump the default, audit notes
    referencing 3 consecutive failures must be updated."""
    assert CONSECUTIVE_FAIL_THRESHOLD == 3
