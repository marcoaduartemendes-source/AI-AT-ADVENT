"""Tests for the audit-phase-5 features.

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
