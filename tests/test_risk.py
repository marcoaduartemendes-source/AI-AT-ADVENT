"""Unit tests for risk policies + RiskManager kill-switch flow.

Risk is the layer that should NEVER fail silently. Tests here pin
down the drawdown→state mapping and the cap-resolution logic that
decides per-broker max trade size.
"""
from __future__ import annotations


import pytest

from risk.policies import KillSwitchState, RiskConfig


class TestDrawdownToState:
    """Pin the kill-switch ladder. The exact thresholds are a config
    decision; what we lock in is the *direction* and ordering."""

    def test_5pct_dd_triggers_warning(self):
        cfg = RiskConfig(
            warning_dd_pct=0.05,
            critical_dd_pct=0.10,
            kill_dd_pct=0.15,
        )
        assert cfg.state_for_drawdown(0.05) == KillSwitchState.WARNING
        assert cfg.state_for_drawdown(0.06) == KillSwitchState.WARNING
        assert cfg.state_for_drawdown(0.04) == KillSwitchState.NORMAL

    def test_10pct_dd_triggers_critical(self):
        cfg = RiskConfig(
            warning_dd_pct=0.05,
            critical_dd_pct=0.10,
            kill_dd_pct=0.15,
        )
        assert cfg.state_for_drawdown(0.10) == KillSwitchState.CRITICAL
        assert cfg.state_for_drawdown(0.13) == KillSwitchState.CRITICAL

    def test_15pct_dd_triggers_kill(self):
        cfg = RiskConfig(
            warning_dd_pct=0.05,
            critical_dd_pct=0.10,
            kill_dd_pct=0.15,
        )
        assert cfg.state_for_drawdown(0.15) == KillSwitchState.KILL
        assert cfg.state_for_drawdown(0.50) == KillSwitchState.KILL

    def test_zero_dd_is_normal(self):
        cfg = RiskConfig()
        assert cfg.state_for_drawdown(0.0) == KillSwitchState.NORMAL


class TestPerBrokerCap:
    """Per-broker max_trade_usd overrides global. The MAX_TRADE_USD_GLOBAL
    + MAX_TRADE_USD_COINBASE / _ALPACA / _KALSHI env vars are how we set
    different caps per venue."""

    def test_per_broker_override_wins_over_global(self):
        cfg = RiskConfig(
            max_trade_usd=5000,
            max_trade_usd_coinbase=100,
            max_trade_usd_alpaca=100000,
        )
        assert cfg.cap_for_venue("coinbase") == 100
        assert cfg.cap_for_venue("alpaca") == 100000

    def test_no_override_falls_back_to_global(self):
        cfg = RiskConfig(max_trade_usd=5000)
        assert cfg.cap_for_venue("coinbase") == 5000
        assert cfg.cap_for_venue("kalshi") == 5000


class TestRiskStateFromCycle:
    """Smoke test: with a mock broker and a fake equity history, the
    RiskManager produces a sensible RiskState."""

    def test_compute_state_with_mock_broker(self, tmp_path):
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker, MockPosition

        # Empty equity DB ⇒ first snapshot ⇒ no drawdown
        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        broker = MockBroker(
            venue="alpaca",
            cash_usd=80_000, equity_usd=100_000,
            positions=[MockPosition("SPY", qty=10, entry=720, mark=725)],
        )
        rm = RiskManager(
            brokers={"alpaca": broker},
            config=RiskConfig(),
            db=db,
        )
        state = rm.compute_state(persist=True)

        assert state.equity_usd == pytest.approx(100_000)
        assert state.kill_switch == KillSwitchState.NORMAL
        # Cached snapshot should be reusable for the rest of the cycle
        cached = rm.cached_positions("alpaca")
        assert len(cached) == 1
        assert cached[0].symbol == "SPY"

    def test_drawdown_drives_kill_switch(self, tmp_path):
        """Seed the equity DB with a peak, then a 16% drop; expect KILL."""
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        # Peak at $100k
        db.record_snapshot(100_000, note="peak")
        # Build a broker reporting equity of $84k → 16% drawdown
        broker = MockBroker(venue="alpaca", cash_usd=84_000, equity_usd=84_000)
        rm = RiskManager(
            brokers={"alpaca": broker},
            config=RiskConfig(
                warning_dd_pct=0.05,
                critical_dd_pct=0.10,
                kill_dd_pct=0.15,
            ),
            db=db,
        )
        state = rm.compute_state(persist=True)
        assert state.kill_switch == KillSwitchState.KILL
        assert state.drawdown_pct >= 0.15
