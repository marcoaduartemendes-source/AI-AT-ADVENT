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

    def test_monthly_loss_budget_escalates_to_critical(self, tmp_path):
        """Monthly loss > monthly_loss_limit_pct (default 4%) should
        escalate state to CRITICAL even when global drawdown hasn't
        crossed kill_dd_pct (15%). Audit fix #1."""
        from datetime import UTC, datetime
        import sqlite3
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        now = datetime.now(UTC)
        month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0,
        )
        with sqlite3.connect(db.db_path) as c:
            c.execute(
                "INSERT INTO equity_snapshots (timestamp, equity_usd, note) "
                "VALUES (?, ?, ?)",
                (month_start.isoformat(), 100_000.0, "month-start"),
            )
        # Broker reports $95k now → 5% MTD loss, exceeds 4% limit
        broker = MockBroker(venue="alpaca", cash_usd=95_000, equity_usd=95_000)
        rm = RiskManager(
            brokers={"alpaca": broker},
            config=RiskConfig(monthly_loss_limit_pct=0.04),
            db=db,
        )
        state = rm.compute_state(persist=True)
        # Drawdown alone wouldn't trigger CRITICAL (5% < 10%), but
        # the monthly-loss check should
        assert state.kill_switch == KillSwitchState.CRITICAL, (
            f"Expected CRITICAL on 5% MTD loss > 4% limit, "
            f"got {state.kill_switch}"
        )

    def test_monthly_gain_does_not_gate(self, tmp_path):
        """A profitable month should NEVER trigger the monthly loss
        budget — even if intra-month drawdown is positive."""
        from datetime import UTC, datetime
        import sqlite3
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        now = datetime.now(UTC)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        with sqlite3.connect(db.db_path) as c:
            c.execute(
                "INSERT INTO equity_snapshots (timestamp, equity_usd, note) "
                "VALUES (?, ?, ?)",
                (month_start.isoformat(), 100_000.0, "month-start"),
            )
        # Equity went UP to $105k (positive month) → MTD loss = 0
        broker = MockBroker(venue="alpaca", cash_usd=105_000, equity_usd=105_000)
        rm = RiskManager(
            brokers={"alpaca": broker},
            config=RiskConfig(monthly_loss_limit_pct=0.04),
            db=db,
        )
        state = rm.compute_state(persist=True)
        assert state.kill_switch == KillSwitchState.NORMAL


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


class TestKillSwitchLatch:
    """2026-05-22 safety fix: the kill switch was a pure function of
    current drawdown, so a KILL auto-cleared the instant equity recovered
    (re-enabling trading right after a 15% loss with no human review) and
    a manual ARM was silently overridden on the next cycle. The switch
    must now LATCH: once KILL fires it stays KILL until an explicit
    operator reset.
    """

    def _cfg(self):
        return RiskConfig(warning_dd_pct=0.05, critical_dd_pct=0.10,
                          kill_dd_pct=0.15)

    def test_latched_kill_persists_after_equity_recovers(self, tmp_path):
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        db.record_snapshot(100_000, note="peak")
        cfg = self._cfg()
        # Cycle 1: equity $84k → 16% drawdown → KILL (records the event).
        rm1 = RiskManager(
            brokers={"alpaca": MockBroker(
                venue="alpaca", cash_usd=84_000, equity_usd=84_000)},
            config=cfg, db=db)
        assert rm1.compute_state(persist=True).kill_switch == \
            KillSwitchState.KILL
        # Cycle 2: equity fully recovers to $100k → 0% drawdown. Pre-fix
        # this returned NORMAL; the latch must keep it KILL.
        rm2 = RiskManager(
            brokers={"alpaca": MockBroker(
                venue="alpaca", cash_usd=100_000, equity_usd=100_000)},
            config=cfg, db=db)
        assert rm2.compute_state(persist=True).kill_switch == \
            KillSwitchState.KILL

    def test_manual_arm_durably_holds_at_zero_drawdown(self, tmp_path):
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        db.record_snapshot(100_000, note="peak")
        cfg = self._cfg()
        rm = RiskManager(
            brokers={"alpaca": MockBroker(
                venue="alpaca", cash_usd=100_000, equity_usd=100_000)},
            config=cfg, db=db)
        # Healthy equity (0% drawdown) but operator arms the switch.
        rm.arm_kill_switch(note="manual halt")
        assert rm.compute_state(persist=True).kill_switch == \
            KillSwitchState.KILL

    def test_reset_clears_the_latch(self, tmp_path):
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        db.record_snapshot(100_000, note="peak")
        cfg = self._cfg()
        # Trip + latch a KILL.
        rm = RiskManager(
            brokers={"alpaca": MockBroker(
                venue="alpaca", cash_usd=84_000, equity_usd=84_000)},
            config=cfg, db=db)
        assert rm.compute_state(persist=True).kill_switch == \
            KillSwitchState.KILL
        # Operator resets, equity is healthy again → state follows
        # drawdown once more (NORMAL).
        rm.reset_kill_switch()
        rm2 = RiskManager(
            brokers={"alpaca": MockBroker(
                venue="alpaca", cash_usd=100_000, equity_usd=100_000)},
            config=cfg, db=db)
        assert rm2.compute_state(persist=True).kill_switch == \
            KillSwitchState.NORMAL


class TestAssetClassExposureCap:
    """Audit fix #5: per-asset-class concentration cap. Without this,
    five strategies could pile into equity-beta-1 names; one bad SPX
    day would drain the book before per-strategy freezes engage."""

    def test_equity_cap_scales_oversized_order(self, tmp_path):
        """Equity cap = 45% of $100k = $45k. Existing equity exposure
        already $40k. New order for $20k should be SCALED to $5k."""
        from risk.manager import RiskManager, EquitySnapshotDB
        from tests.mock_broker import MockBroker, MockPosition

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        # Existing equity exposure: 4 stocks × $10k each = $40k
        broker = MockBroker(
            venue="alpaca", cash_usd=60_000, equity_usd=100_000,
            positions=[
                MockPosition("AAPL", qty=50, entry=200, mark=200),
                MockPosition("MSFT", qty=25, entry=400, mark=400),
                MockPosition("GOOGL", qty=60, entry=166.67, mark=166.67),
                MockPosition("AMZN", qty=80, entry=125, mark=125),
            ],
        )
        cfg = RiskConfig(
            max_asset_class_pct={"EQUITY": 0.45, "ETF": 0.45},
        )
        rm = RiskManager(brokers={"alpaca": broker}, config=cfg, db=db)
        state = rm.compute_state(persist=True)

        # Try to buy $20k more equity — should be scaled to $5k
        from risk.manager import Decision
        decision = rm.check_order(
            notional_usd=20_000,
            symbol="NVDA",
            asset_class="EQUITY",
            state=state,
        )
        # cap=$45k - existing $40k = $5k headroom
        assert decision.decision == Decision.SCALE
        assert decision.approved_notional_usd == pytest.approx(5_000, rel=0.05)

    def test_closing_trade_bypasses_cap(self, tmp_path):
        """Closing trades reduce exposure — should ALWAYS be allowed
        even when the asset class is at/above its cap."""
        from risk.manager import RiskManager, EquitySnapshotDB, Decision
        from tests.mock_broker import MockBroker, MockPosition

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        # Equity exposure already $50k > 45% cap
        broker = MockBroker(
            venue="alpaca", cash_usd=50_000, equity_usd=100_000,
            positions=[MockPosition("AAPL", qty=250, entry=200, mark=200)],
        )
        cfg = RiskConfig(max_asset_class_pct={"EQUITY": 0.45})
        rm = RiskManager(brokers={"alpaca": broker}, config=cfg, db=db)
        state = rm.compute_state(persist=True)

        # Sell AAPL is_closing=True — must be approved
        decision = rm.check_order(
            notional_usd=10_000,
            symbol="AAPL",
            is_closing=True,
            asset_class="EQUITY",
            state=state,
        )
        assert decision.decision in (Decision.APPROVE, Decision.SCALE)
        assert decision.approved_notional_usd > 0

    def test_unconfigured_asset_class_passes(self, tmp_path):
        """An asset class not in max_asset_class_pct dict should not
        be capped (treat as unlimited)."""
        from risk.manager import RiskManager, EquitySnapshotDB, Decision
        from tests.mock_broker import MockBroker

        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        broker = MockBroker(venue="alpaca", cash_usd=100_000, equity_usd=100_000)
        # Use a high max_trade_usd so we test the asset-class cap
        # in isolation (not the per-order ceiling)
        cfg = RiskConfig(
            max_asset_class_pct={"EQUITY": 0.45},  # no NEW_CLASS
            max_trade_usd=50_000,
            max_position_pct=0.50,
        )
        rm = RiskManager(brokers={"alpaca": broker}, config=cfg, db=db)
        state = rm.compute_state(persist=True)
        decision = rm.check_order(
            notional_usd=10_000,
            symbol="WTFCOIN",
            asset_class="NEW_CLASS",
            state=state,
        )
        # No cap on NEW_CLASS → not scaled by the asset-class gate
        assert decision.decision == Decision.APPROVE
        assert decision.approved_notional_usd == pytest.approx(10_000)


class TestUnfreezeStrategies:
    """--unfreeze fixes the lifecycle latch-asymmetry: a strategy that
    auto-froze on past underperformance but has since re-validated must
    be returnable to ACTIVE (allocator never auto-recovers FROZEN)."""

    def _reg(self, tmp_path):
        from allocator.lifecycle import StrategyRegistry, StrategyState, StrategyMeta
        reg = StrategyRegistry(str(tmp_path / "alloc.db"))
        for n in ("risk_parity_etf", "loser", "active_one"):
            reg.register(StrategyMeta(name=n, asset_classes=["ETF"],
                                      venue="alpaca", target_alloc_pct=0.1,
                                      min_alloc_pct=0.0, max_alloc_pct=0.3))
        reg.set_state("risk_parity_etf", StrategyState.FROZEN, "auto-freeze")
        reg.set_state("loser", StrategyState.FROZEN, "auto-freeze")
        return reg, StrategyState

    def test_unfreeze_named_passing_strategy(self, tmp_path):
        from run_orchestrator import unfreeze_strategies
        reg, St = self._reg(tmp_path)
        out = unfreeze_strategies("risk_parity_etf", reg, {"risk_parity_etf"})
        assert out == [("risk_parity_etf", "PASS")]
        assert reg.get_state("risk_parity_etf") == St.ACTIVE
        assert reg.get_state("loser") == St.FROZEN   # untouched

    def test_unfreeze_all_flips_every_frozen(self, tmp_path):
        from run_orchestrator import unfreeze_strategies
        reg, St = self._reg(tmp_path)
        names = {n for n, _ in unfreeze_strategies("all", reg, set())}
        assert names == {"risk_parity_etf", "loser"}
        assert reg.get_state("risk_parity_etf") == St.ACTIVE
        assert reg.get_state("loser") == St.ACTIVE

    def test_unfreeze_skips_non_frozen(self, tmp_path):
        from run_orchestrator import unfreeze_strategies
        reg, St = self._reg(tmp_path)
        assert unfreeze_strategies("active_one", reg, set()) == []
        assert reg.get_state("active_one") == St.ACTIVE


class TestUnfreezeBareRegistry:
    """Regression: the CLI builds a bare StrategyRegistry() with no metas
    registered. all_states() filters by _meta (empty) → returned nothing,
    so `--unfreeze risk_parity_etf` reported 'nothing eligible'. The named
    path must read state straight from the DB via get_state()."""

    def test_named_unfreeze_works_without_registering_metas(self, tmp_path):
        from allocator.lifecycle import StrategyRegistry, StrategyState, StrategyMeta
        from run_orchestrator import unfreeze_strategies
        db = str(tmp_path / "alloc.db")
        # Seed: register + freeze in one registry instance.
        seed = StrategyRegistry(db)
        seed.register(StrategyMeta(name="risk_parity_etf", asset_classes=["ETF"],
                                   venue="alpaca", target_alloc_pct=0.2,
                                   min_alloc_pct=0.0, max_alloc_pct=0.3))
        seed.set_state("risk_parity_etf", StrategyState.FROZEN, "auto-freeze")
        # Fresh BARE registry on the same DB — mimics the CLI (no metas).
        bare = StrategyRegistry(db)
        out = unfreeze_strategies("risk_parity_etf", bare, {"risk_parity_etf"})
        assert out == [("risk_parity_etf", "PASS")]
        assert bare.get_state("risk_parity_etf") == StrategyState.ACTIVE
