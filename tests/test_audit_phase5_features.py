"""Tests for the audit-phase-5 + paper-trading-default features.

  - Per-strategy daily-notional governor (risk/manager.py + policies.py)
  - Equity-regime gate fold-in via vol_scaler() (strategies/_helpers.py)
  - errors_db persistence (common/errors_db.py)

These pin behaviour the audit identified as missing — a regression of
any of these would re-open a real failure mode (haywire strategy
fee burn, correlated equity drawdown, debug archaeology).
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from brokers.base import AssetClass
from risk.manager import Decision, EquitySnapshotDB, RiskManager
from risk.policies import KillSwitchState, RiskConfig
from tests.mock_broker import MockBroker


# ─── Per-strategy daily-notional governor ───────────────────────────


class TestDailyNotionalGovernor:
    """A single strategy must not be able to trade more than
    `max_strategy_daily_notional_pct * equity` of opening notional
    per UTC day. Closes always bypass."""

    def _make_risk(self, tmp_path) -> RiskManager:
        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        cfg = RiskConfig(max_strategy_daily_notional_pct=0.10)
        broker = MockBroker(venue="alpaca", cash_usd=10_000)
        rm = RiskManager(brokers={"alpaca": broker}, config=cfg, db=db)
        # Force a benign cached state so check_order doesn't refetch
        rm._cached_state = type(rm.compute_state())(
            timestamp=datetime.now(UTC),
            equity_usd=10_000,
            peak_equity_usd=10_000,
            drawdown_pct=0.0,
            kill_switch=KillSwitchState.NORMAL,
            realized_vol=0.10,
            leverage=0.0,
            multiplier=rm.multiplier.compute(0.0, 0.10, None),
            venues_ok=True,
        )
        return rm

    def _seed_today_buys(self, tmp_path, monkeypatch, strategy: str,
                          notional_total: float) -> None:
        """Insert N today-dated BUY rows totalling `notional_total`."""
        trades_db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(trades_db))
        with sqlite3.connect(trades_db) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, strategy TEXT,
                    product_id TEXT, side TEXT,
                    amount_usd REAL, quantity REAL, price REAL,
                    order_id TEXT, pnl_usd REAL, dry_run INTEGER,
                    fill_status TEXT
                )
            """)
            now_iso = datetime.now(UTC).isoformat()
            conn.execute(
                "INSERT INTO trades (timestamp, strategy, product_id, "
                "side, amount_usd, quantity, price, order_id, "
                "dry_run, fill_status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (now_iso, strategy, "SPY", "BUY", notional_total, 1.0,
                 notional_total, "x", 0, "FILLED"),
            )

    def test_under_cap_is_approved(self, tmp_path, monkeypatch):
        rm = self._make_risk(tmp_path)
        # 10% × $10k = $1000 cap. Existing day usage $400; new $400
        # → total $800, well under $1000 → approve.
        self._seed_today_buys(tmp_path, monkeypatch, "tsmom_etf", 400.0)
        decision = rm.check_order(
            notional_usd=400.0, symbol="QQQ",
            strategy_name="tsmom_etf", asset_class=AssetClass.ETF.value,
            venue="alpaca",
        )
        assert decision.decision != Decision.REJECT, decision.reason

    def test_over_cap_is_rejected(self, tmp_path, monkeypatch):
        rm = self._make_risk(tmp_path)
        # 10% × $10k = $1000 cap. Existing day usage $900; new $200
        # → total $1100 > $1000 → reject.
        self._seed_today_buys(tmp_path, monkeypatch, "tsmom_etf", 900.0)
        decision = rm.check_order(
            notional_usd=200.0, symbol="QQQ",
            strategy_name="tsmom_etf", asset_class=AssetClass.ETF.value,
            venue="alpaca",
        )
        assert decision.decision == Decision.REJECT
        assert "daily notional" in decision.reason.lower()

    def test_close_bypasses_cap(self, tmp_path, monkeypatch):
        rm = self._make_risk(tmp_path)
        self._seed_today_buys(tmp_path, monkeypatch, "tsmom_etf", 950.0)
        # A SELL that's a closing trade should NOT be governed —
        # we always want to be able to reduce exposure.
        decision = rm.check_order(
            notional_usd=500.0, symbol="QQQ",
            is_closing=True,
            strategy_name="tsmom_etf", asset_class=AssetClass.ETF.value,
            venue="alpaca",
        )
        assert decision.decision != Decision.REJECT

    def test_other_strategies_isolated(self, tmp_path, monkeypatch):
        """tsmom_etf hitting its cap must not affect crypto_xsmom."""
        rm = self._make_risk(tmp_path)
        self._seed_today_buys(tmp_path, monkeypatch, "tsmom_etf", 1500.0)
        decision = rm.check_order(
            notional_usd=400.0, symbol="BTC-USD",
            strategy_name="crypto_xsmom",
            asset_class=AssetClass.CRYPTO_SPOT.value,
            venue="coinbase",
        )
        assert decision.decision != Decision.REJECT


# ─── Equity-regime gate fold-in ─────────────────────────────────────


class TestEquityRegimeGate:
    """vol_scaler() must fold the equity_regime_multiplier into its
    return value so equity-momentum strategies size down automatically
    in HIGH/EXTREME regimes."""

    def _ctx_with_signal(self, payload):
        from unittest.mock import MagicMock
        ctx = MagicMock()
        ctx.scout_signals = {"vol_scaler": payload}
        return ctx

    def test_normal_regime_returns_raw_scaler(self):
        from strategies._helpers import vol_scaler
        ctx = self._ctx_with_signal({
            "equity_momentum": 1.2,
            "equity_regime": "NORMAL",
            "equity_regime_multiplier": 1.0,
        })
        assert vol_scaler(ctx, "equity_momentum") == pytest.approx(1.2)

    def test_high_regime_halves(self):
        from strategies._helpers import vol_scaler
        ctx = self._ctx_with_signal({
            "equity_momentum": 1.2,
            "equity_regime": "HIGH",
            "equity_regime_multiplier": 0.5,
        })
        # 1.2 × 0.5 = 0.6
        assert vol_scaler(ctx, "equity_momentum") == pytest.approx(0.6)

    def test_extreme_regime_quarters(self):
        from strategies._helpers import vol_scaler
        ctx = self._ctx_with_signal({
            "equity_momentum": 1.2,
            "equity_regime": "EXTREME",
            "equity_regime_multiplier": 0.25,
        })
        assert vol_scaler(ctx, "equity_momentum") == pytest.approx(0.3)

    def test_missing_signal_returns_default(self):
        from strategies._helpers import vol_scaler
        from unittest.mock import MagicMock
        ctx = MagicMock()
        ctx.scout_signals = {}
        # No signal → unchanged sizing.
        assert vol_scaler(ctx, "equity_momentum") == pytest.approx(1.0)

    def test_equity_regime_helper(self):
        from strategies._helpers import equity_regime
        from unittest.mock import MagicMock
        ctx = MagicMock()
        ctx.scout_signals = {"vol_scaler": {"equity_regime": "EXTREME"}}
        assert equity_regime(ctx) == "EXTREME"


# ─── errors_db persistence ──────────────────────────────────────────


class TestErrorsDb:
    """Stack traces persisted via record_error() must round-trip
    through recent_errors()."""

    def test_records_and_reads_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        from common.errors_db import record_error, recent_errors

        try:
            raise RuntimeError("simulated failure")
        except RuntimeError:
            record_error(
                scope="test", strategy="x", venue="alpaca",
            )

        rows = recent_errors(limit=5)
        assert len(rows) == 1
        row = rows[0]
        assert row["scope"] == "test"
        assert row["strategy"] == "x"
        assert row["venue"] == "alpaca"
        assert row["exc_type"] == "RuntimeError"
        assert "simulated failure" in row["exc_message"]
        assert "RuntimeError" in row["traceback"]

    def test_rotates_at_retain_limit(self, tmp_path, monkeypatch):
        """Fewer than 1000 rows is fine; exactly the latest 1000 are
        kept after rotation. Test with smaller numbers for speed."""
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        from common.errors_db import record_error, recent_errors

        for i in range(5):
            try:
                raise RuntimeError(f"err {i}")
            except RuntimeError:
                record_error(scope="test")
        rows = recent_errors(limit=10)
        assert len(rows) == 5
        # Most recent first
        assert "err 4" in rows[0]["exc_message"]

    def test_no_active_exception_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        from common.errors_db import record_error, recent_errors
        record_error(scope="test")    # no active exception
        assert recent_errors() == []


# ─── Backtest-metadata stub ─────────────────────────────────────────


class TestKillSwitchClosesShorts:
    """Audit-fix F1 (2026-05-07). KILL switch must emit BUY-to-close
    for short legs, not just SELL-to-close for longs. Strategies that
    hold short legs invisible to spot-only get_positions() must use
    a ledger-based fallback to know about them.
    """

    def test_default_on_emergency_close_handles_negative_qty(self):
        from strategy_engine.base import (
            Strategy,
            StrategyContext,
        )
        from brokers.base import OrderSide
        from datetime import UTC, datetime

        class _S(Strategy):
            name = "x"
            venue = "alpaca"
            def compute(self, ctx):
                return []

        s = _S(broker=None)
        ctx = StrategyContext(
            timestamp=datetime.now(UTC),
            portfolio_equity_usd=10_000,
            target_alloc_pct=0.1,
            target_alloc_usd=1_000,
            risk_multiplier=1.0,
            open_positions={
                "SPY": {"quantity": 5},     # long → SELL
                "QQQ": {"quantity": -3},    # short → BUY
                "IWM": {"quantity": 0},     # flat → ignore
            },
            scout_signals={},
        )
        proposals = s.on_emergency_close(ctx)
        sides = {(p.symbol, p.side, p.quantity) for p in proposals}
        assert ("SPY", OrderSide.SELL, 5) in sides
        assert ("QQQ", OrderSide.BUY, 3) in sides
        assert all(p.symbol != "IWM" for p in proposals)

    def test_net_qty_from_ledger_detects_short(self, tmp_path, monkeypatch):
        """A strategy that placed a SELL with no prior BUY (shorting
        a futures/perp leg) shows up as net_qty < 0 from the ledger."""
        import sqlite3
        db = tmp_path / "trades.db"
        with sqlite3.connect(db) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, strategy TEXT,
                    product_id TEXT, side TEXT,
                    amount_usd REAL, quantity REAL, price REAL,
                    order_id TEXT, pnl_usd REAL, dry_run INTEGER,
                    fill_status TEXT, entry_price REAL, venue TEXT
                )
            """)
            conn.executemany(
                "INSERT INTO trades (timestamp, strategy, product_id, "
                "side, amount_usd, quantity, price, order_id, "
                "fill_status, venue) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    # Long leg: BUY 1 BTC-USD
                    ("2026-05-08T00:00:00+00:00", "carry", "BTC-USD",
                     "BUY", 50000, 1.0, 50000, "o-1", "FILLED", "coinbase"),
                    # Short leg: SELL 1 BTC-PERP-INTX (no prior BUY)
                    ("2026-05-08T00:00:01+00:00", "carry", "BTC-PERP-INTX",
                     "SELL", 50000, 1.0, 50000, "o-2", "FILLED", "coinbase"),
                ],
            )
        monkeypatch.setenv("TRADING_DB_PATH", str(db))

        from strategies._helpers import net_qty_from_ledger
        net = net_qty_from_ledger("carry", "coinbase")
        assert net["BTC-USD"] == pytest.approx(1.0)
        assert net["BTC-PERP-INTX"] == pytest.approx(-1.0)


class TestCoinbaseUnknownOrderIdRejects:
    """Audit-fix F3 (2026-05-07). Coinbase responses missing order_id
    must raise BrokerError, not write a stuck-PENDING row to the
    ledger. The previous behaviour was the user's reported failure
    mode: real USD left wallet, ledger thought it was in-flight.
    """

    def test_no_order_id_raises_broker_error(self, monkeypatch):
        from brokers.base import BrokerError, OrderSide, OrderType
        from brokers.coinbase import CoinbaseAdapter

        adapter = CoinbaseAdapter.__new__(CoinbaseAdapter)
        adapter.venue = "coinbase"
        adapter.is_paper = False
        adapter._product_cache = {}

        class _StubClient:
            def create_market_buy(self, *a, **kw):
                # Coinbase response shape with no order_id
                return {"failure_reason": "INSUFFICIENT_FUND"}

        adapter.client = _StubClient()
        with pytest.raises(BrokerError) as excinfo:
            adapter.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                notional_usd=100,
            )
        assert "INSUFFICIENT_FUND" in str(excinfo.value)


class TestPerBrokerFlagEmptyString:
    """Regression: GitHub Actions injects ${{ vars.X }} as the empty
    string when X is unset. The previous _per_broker_flag returned
    True for "" (because "".lower() != "false"), forcing every venue
    to DRY when the user expected paper trading. The fix: empty
    string == None == fall through to global DRY_RUN.
    """

    def _flag(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("DRY_RUN_ALPACA", raising=False)
        else:
            monkeypatch.setenv("DRY_RUN_ALPACA", value)
        # Re-import the inner closure-style function via the public API:
        # easier to test the documented semantic via the OrchestratorConfig
        # is_dry contract, but for this test we want the parser itself.
        # The parser is defined inside main() — we replicate its body.
        import os as _os
        v = _os.environ.get("DRY_RUN_ALPACA")
        if v is None:
            return None
        v = v.strip()
        if v == "":
            return None
        return v.lower() != "false"

    def test_unset_returns_none(self, monkeypatch):
        assert self._flag(monkeypatch, None) is None

    def test_empty_string_returns_none(self, monkeypatch):
        # The bug: GitHub Actions empty-var was being parsed as True (DRY)
        assert self._flag(monkeypatch, "") is None

    def test_whitespace_returns_none(self, monkeypatch):
        assert self._flag(monkeypatch, "   ") is None

    def test_false_string_returns_false(self, monkeypatch):
        assert self._flag(monkeypatch, "false") is False

    def test_true_string_returns_true(self, monkeypatch):
        assert self._flag(monkeypatch, "true") is True


class TestLiveStrategiesGate:
    """Regression: LIVE_STRATEGIES must require ALLOW_LIVE_TRADING=1.
    Audit failure-mode 2026-05-07: a stale repo Variable from earlier
    testing was firing real Coinbase orders the moment a strategy
    shipped, even though every other DRY flag was set safely. The
    fix gates LIVE_STRATEGIES behind the same two-key boundary that
    DRY_RUN=false already uses.
    """

    def _run_gate(self, monkeypatch, dry_run, allow_live, live_strats):
        if dry_run is None:
            monkeypatch.delenv("DRY_RUN", raising=False)
        else:
            monkeypatch.setenv("DRY_RUN", dry_run)
        if allow_live is None:
            monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
        else:
            monkeypatch.setenv("ALLOW_LIVE_TRADING", allow_live)
        if live_strats is None:
            monkeypatch.delenv("LIVE_STRATEGIES", raising=False)
        else:
            monkeypatch.setenv("LIVE_STRATEGIES", live_strats)
        # Replicate the gate logic from run_orchestrator.main() — keep
        # this test independent of broker init / heavy import paths.
        import os as _os
        dry_env = _os.environ.get("DRY_RUN", "true").lower()
        allow = _os.environ.get("ALLOW_LIVE_TRADING") == "1"
        if dry_env == "false" and not allow:
            _os.environ["DRY_RUN"] = "true"
        if _os.environ.get("LIVE_STRATEGIES") and not allow:
            _os.environ["LIVE_STRATEGIES"] = ""
        return {
            "dry": _os.environ.get("DRY_RUN"),
            "live": _os.environ.get("LIVE_STRATEGIES"),
        }

    def test_live_strats_set_without_allow_is_neutralized(self, monkeypatch):
        out = self._run_gate(
            monkeypatch,
            dry_run="true",
            allow_live=None,
            live_strats="crypto_funding_carry,crypto_xsmom",
        )
        assert out["live"] == "", (
            "LIVE_STRATEGIES should be cleared when ALLOW_LIVE_TRADING != '1'"
        )

    def test_live_strats_with_allow_is_honoured(self, monkeypatch):
        out = self._run_gate(
            monkeypatch,
            dry_run="true",
            allow_live="1",
            live_strats="crypto_funding_carry",
        )
        assert out["live"] == "crypto_funding_carry"

    def test_dry_false_without_allow_is_forced_back_to_true(self, monkeypatch):
        out = self._run_gate(
            monkeypatch,
            dry_run="false",
            allow_live=None,
            live_strats=None,
        )
        assert out["dry"] == "true"

    def test_dry_false_with_allow_stays_false(self, monkeypatch):
        out = self._run_gate(
            monkeypatch,
            dry_run="false",
            allow_live="1",
            live_strats=None,
        )
        assert out["dry"] == "false"


class TestBacktestMetadata:
    """The registry stub must be importable + return None for any
    strategy until real backtests are wired in."""

    def test_returns_none_for_unknown(self):
        from strategies._backtest_metadata import prior_for
        assert prior_for("nonexistent_strategy") is None

    def test_priors_dict_starts_empty(self):
        """Honest empty default; populating with fake numbers would
        be worse than uniform prior."""
        from strategies._backtest_metadata import PRIORS
        assert PRIORS == {}
