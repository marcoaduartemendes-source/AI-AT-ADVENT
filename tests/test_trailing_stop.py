"""Tests for the portfolio-level trailing stop in risk/manager.py.

Audit-fix follow-up: the existing drawdown thresholds measure peak-
to-now from the all-time high. That's slow when the all-time peak
was set months ago — a sharp local drop that doesn't reach
critical_dd_pct can still wipe out a month of gains. The trailing
stop uses a shorter lookback (default 14d) to catch local drops
fast.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from risk.manager import EquitySnapshotDB, RiskConfig, RiskManager
from risk.policies import KillSwitchState


def _seed_equity(db: EquitySnapshotDB, samples: list[tuple[float, int]]) -> None:
    """Seed `equity_snapshots` with (equity_usd, days_ago) tuples.
    Bypasses dual-write so the test stays fast."""
    import sqlite3
    with sqlite3.connect(db.db_path) as c:
        for equity, days_ago in samples:
            ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
            c.execute(
                "INSERT INTO equity_snapshots (timestamp, equity_usd) "
                "VALUES (?, ?)",
                (ts, equity),
            )
        c.commit()


def _broker(equity: float):
    m = MagicMock()
    m.get_account.return_value = MagicMock(
        cash_usd=equity, buying_power_usd=equity, equity_usd=equity,
    )
    m.get_positions.return_value = []
    return m


def test_trailing_high_returns_max_in_window(tmp_path):
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    _seed_equity(db, [
        (100_000, 30),    # outside 14d window
        (110_000, 10),    # inside, will be the peak
        (105_000, 5),
        (102_000, 1),
    ])
    assert db.trailing_high(14) == 110_000


def test_trailing_high_none_when_window_empty(tmp_path):
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    _seed_equity(db, [(100_000, 30)])    # only outside-window data
    assert db.trailing_high(14) is None


def test_trailing_stop_critical_escalates_kill_switch(tmp_path):
    """Sharp 8% drop from a 14-day high must escalate to CRITICAL
    even when all-time peak DD is below kill threshold."""
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    # 14d ago: $110k; today: $101k → 8.2% trailing drop
    # All-time peak ($110k) → DD = 8.2%, below 10% critical_dd_pct
    _seed_equity(db, [
        (110_000, 10),
        (108_000, 5),
        (101_000, 0),
    ])
    rm = RiskManager(
        brokers={"alpaca": _broker(101_000)},
        config=RiskConfig.from_env(),
        db=db,
    )
    state = rm.compute_state(persist=False)
    assert state.kill_switch == KillSwitchState.CRITICAL
    assert state.equity_usd == 101_000


def test_trailing_stop_warning_below_critical(tmp_path):
    """5% drop from 14-day high → WARNING (not CRITICAL)."""
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    _seed_equity(db, [
        (100_000, 10),
        (95_000, 0),    # 5% trailing drop
    ])
    rm = RiskManager(
        brokers={"alpaca": _broker(95_000)},
        config=RiskConfig.from_env(),
        db=db,
    )
    state = rm.compute_state(persist=False)
    assert state.kill_switch == KillSwitchState.WARNING


def test_trailing_stop_does_not_fire_below_threshold(tmp_path):
    """3% drop is below the 4% warning threshold → stay NORMAL."""
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    _seed_equity(db, [
        (100_000, 10),
        (97_000, 0),    # 3% drop
    ])
    rm = RiskManager(
        brokers={"alpaca": _broker(97_000)},
        config=RiskConfig.from_env(),
        db=db,
    )
    state = rm.compute_state(persist=False)
    assert state.kill_switch == KillSwitchState.NORMAL


def test_trailing_stop_disabled_when_critical_pct_zero(tmp_path):
    """Setting trailing_stop_critical_pct=0 disables the trailing
    stop entirely — drawdown thresholds still apply."""
    import dataclasses
    base = RiskConfig.from_env()
    cfg = dataclasses.replace(base, trailing_stop_critical_pct=0.0)
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    _seed_equity(db, [
        (110_000, 10),
        (101_000, 0),    # 8.2% trailing drop
    ])
    rm = RiskManager(
        brokers={"alpaca": _broker(101_000)},
        config=cfg, db=db,
    )
    state = rm.compute_state(persist=False)
    # Trailing stop disabled → falls back to normal dd evaluation.
    # 8.2% < 10% critical → NORMAL or WARNING depending on threshold.
    assert state.kill_switch != KillSwitchState.CRITICAL


def test_trailing_stop_does_not_override_kill(tmp_path):
    """KILL is sticky — trailing stop should not downgrade it."""
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    # 20% drop from peak → KILL via the absolute drawdown threshold
    _seed_equity(db, [
        (100_000, 10),
        (80_000, 0),
    ])
    rm = RiskManager(
        brokers={"alpaca": _broker(80_000)},
        config=RiskConfig.from_env(),
        db=db,
    )
    state = rm.compute_state(persist=False)
    assert state.kill_switch == KillSwitchState.KILL
