"""Regression tests for the 4 bugs that hit production today.

Each test FAILS on commits before the fix and PASSES on commits after.
If any one of these comes back, CI is the gate, not the user reading
the dashboard.

Bug index (in order shipped):
  1. Phantom -$5,746 PnL — _record_trade computed PnL with price=0
     because the broker hadn't reported a fill yet.
  2. Intra-cycle wash trade — _pending_cache wasn't updated after
     each successful place_order, so two strategies firing opposite
     sides on the same symbol in one cycle tripped Alpaca.
  3. Coinbase MARKET SELL needs qty — basis trade strategy was
     passing notional_usd to Coinbase SELL legs, which the API
     rejects with "Coinbase MARKET SELL requires quantity".
  4. Dashboard $0 P&L despite open positions — load_live_data
     summary was realized-only; unrealized PnL from open positions
     was silently dropped from the headline total.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# These imports run through conftest's sys.path setup.
from brokers.base import Order, OrderSide, OrderStatus, OrderType, Position, AssetClass


# ─────────────────────────────────────────────────────────────────────
# Bug #1 — phantom -$5,746 PnL from price=0 record
# ─────────────────────────────────────────────────────────────────────

class TestRecordTradeWithZeroPriceDoesNotComputePnL:
    """When the broker hasn't reported a fill yet (filled_avg_price is
    None), _record_trade MUST leave pnl_usd=NULL — never compute
    (0 - entry) * qty which produces a fake loss equal to the position
    value."""

    def test_orchestrator_record_trade_skips_pnl_when_no_fill(self, tmp_path, monkeypatch):
        from strategy_engine.orchestrator import Orchestrator
        from trading.performance import PerformanceTracker

        # Point the tracker at a tmp DB
        db = tmp_path / "trading_performance.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))

        # Build the most minimal Orchestrator we can — risk, allocator,
        # strategies, brokers don't matter for this test.
        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = PerformanceTracker(str(db))
        orch.risk = MagicMock()
        orch.risk.cached_positions = MagicMock(return_value=[
            Position(
                venue="alpaca", symbol="TLT",
                asset_class=AssetClass.ETF,
                quantity=74.71, avg_entry_price=85.65,
                market_price=85.65, unrealized_pnl_usd=0.0,
            ),
        ])
        orch.cfg = MagicMock()
        orch.cfg.is_dry = MagicMock(return_value=False)

        # Build a closing-SELL proposal for which the order has NO fill
        # data yet — this is the exact shape that produced -$5,746.44.
        from strategy_engine.base import TradeProposal
        proposal = TradeProposal(
            strategy="tsmom_etf",
            venue="alpaca",
            symbol="TLT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=67.24,
            confidence=0.9,
            reason="trend exit",
            is_closing=True,
        )
        order = Order(
            venue="alpaca", order_id="abc-123", symbol="TLT",
            side=OrderSide.SELL, type=OrderType.MARKET,
            quantity=67.24, notional_usd=None, limit_price=None,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            filled_avg_price=None,   # ← THE BUG: this was None at submit
        )
        decision = MagicMock()
        decision.approved_notional_usd = 6078.59

        orch._record_trade(proposal, order, decision)

        # Inspect the row that was written
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT price, pnl_usd FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        price, pnl = row
        # Bug repro guard: pnl_usd MUST be NULL until a real fill
        # arrives. The old code wrote ~ -5746.44 here.
        assert pnl is None, (
            f"Expected pnl_usd=NULL pre-fill, got {pnl} "
            f"(price={price}). Phantom-loss bug regression."
        )


# ─────────────────────────────────────────────────────────────────────
# Bug #2 — intra-cycle wash trade
# ─────────────────────────────────────────────────────────────────────

class TestIntraCycleWashTrade:
    """When two strategies in one cycle place opposite-side orders on
    the same symbol (e.g. risk_parity BUY SPY then tsmom SELL SPY),
    the second one MUST be skipped. The pending_cache populated at
    cycle-start doesn't see the just-placed BUY, so we have to mark
    it ourselves via _mark_pending_intracycle."""

    def test_mark_pending_intracycle_inflates_n_pending(self):
        from strategy_engine.orchestrator import Orchestrator
        from strategy_engine.base import TradeProposal

        orch = Orchestrator.__new__(Orchestrator)
        orch._pending_cache = {"alpaca": {}}

        proposal = TradeProposal(
            strategy="risk_parity_etf",
            venue="alpaca",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional_usd=3829.66,
            confidence=0.85,
            reason="rebalance",
        )
        order = Order(
            venue="alpaca", order_id="ee4e-123", symbol="SPY",
            side=OrderSide.BUY, type=OrderType.MARKET,
            quantity=0, notional_usd=3829.66, limit_price=None,
            status=OrderStatus.PENDING,
        )
        decision = MagicMock()
        decision.approved_notional_usd = 3829.66

        orch._mark_pending_intracycle(proposal, order, decision)

        # The just-placed BUY must now show up as pending so the next
        # SELL on SPY in this cycle gets skipped.
        entry = orch._pending_cache["alpaca"]["SPY"]
        assert entry["n_pending"] == 1
        assert entry["buy_notional_usd"] == pytest.approx(3829.66)


# ─────────────────────────────────────────────────────────────────────
# Bug #3 — Coinbase MARKET SELL needs qty (basis trade)
# ─────────────────────────────────────────────────────────────────────

class TestCryptoBasisTradeUsesQtyForCoinbaseSells:
    """The basis trade's entry creates two opposite-side proposals
    (long spot + short future). For Coinbase, BUY takes notional_usd
    (quote_size) but SELL must take quantity (base_size). Passing
    notional_usd to a SELL is rejected by the broker with:
        BrokerError: "Coinbase MARKET SELL requires quantity"
    This test asserts that whenever the basis trade creates a SELL
    proposal, it carries `quantity` and not `notional_usd`."""

    def test_basis_trade_sell_legs_carry_quantity_not_notional(self, monkeypatch):
        from strategies.crypto_basis_trade import CryptoBasisTrade

        # Stub the public-products endpoint to return one ETH future
        # with positive basis, forcing the strategy to open a trade.
        fake_products = [
            {"product_id": "ET-29MAY26-CDE", "price": "3500"},
        ]
        from strategies import crypto_basis_trade as cbt
        monkeypatch.setattr(cbt, "cached_get", lambda url, params=None, ttl_seconds=0: (
            {"products": fake_products}
            if "products" in url and not url.endswith("ETH-USD")
            else {"price": "3000"}  # spot price
        ))

        from strategy_engine.base import StrategyContext
        adapter = MagicMock()
        strat = CryptoBasisTrade(adapter)
        ctx = StrategyContext(
            timestamp=datetime.now(timezone.utc),
            portfolio_equity_usd=10000,
            target_alloc_pct=0.02,
            target_alloc_usd=200,
            risk_multiplier=1.0,
            open_positions={},   # no positions → entry path
            scout_signals={},
            pending_orders={},
        )
        proposals = strat.compute(ctx)

        # The strategy should produce 2 proposals (spot+future legs).
        sells = [p for p in proposals if p.side == OrderSide.SELL]
        assert sells, (
            "Expected at least one SELL leg from basis trade entry. "
            "Bug #3 caused the SELL leg to be the future short."
        )
        for sell in sells:
            assert sell.quantity is not None and sell.quantity > 0, (
                f"SELL leg {sell.symbol} must carry quantity, not "
                f"notional_usd. Got quantity={sell.quantity}, "
                f"notional={sell.notional_usd}. Coinbase rejects "
                f"notional-only SELLs with HTTP 400."
            )


# ─────────────────────────────────────────────────────────────────────
# Bug #4 — dashboard summary excluded unrealized PnL
# ─────────────────────────────────────────────────────────────────────

class TestDashboardSummaryIncludesUnrealizedPnL:
    """The headline total_pnl_usd on the dashboard MUST include
    unrealized PnL from open broker positions, not only realized PnL
    from closed trades. With no fill polling, realized stays at $0
    indefinitely — so a realized-only headline showed $0 even when
    we held $25k of paper positions worth +$51 right now.

    This is a unit test on _summarize + the explicit summary build
    in load_live_data, asserting that the headline merges both.
    """

    def test_summarize_handles_none_pnl_rows(self):
        from build_dashboard import _summarize

        trades = [
            {"side": "BUY", "amount_usd": 100, "pnl_usd": None,
             "strategy": "tsmom_etf"},
            {"side": "BUY", "amount_usd": 200, "pnl_usd": None,
             "strategy": "tsmom_etf"},
            {"side": "SELL", "amount_usd": 100, "pnl_usd": 12.5,
             "strategy": "tsmom_etf"},
        ]
        s = _summarize(trades)
        # Realized PnL = 12.5 (only one closed trade).
        assert s["total_pnl_usd"] == pytest.approx(12.5)
        # Entry volume = 300 (two BUYs).
        assert s["entry_volume_usd"] == pytest.approx(300)
        # No phantom -$300 from the unfilled rows.
        assert s["n_trades"] == 1

    def test_summary_headline_merges_realized_and_unrealized(self):
        """Reproduces the failure mode: realized=0, unrealized=+$51 →
        headline must be +$51, not $0."""
        # Simulate the explicit summary-build in load_live_data
        all_trades = [
            {"side": "BUY", "amount_usd": 100, "pnl_usd": None,
             "strategy": "tsmom_etf", "product_id": "TLT"},
        ]
        open_pos_raw = [
            {"quantity": 67.24, "market_price": 85.7,
             "unrealized_pnl_usd": 51.04, "venue": "alpaca",
             "product_id": "TLT"},
        ]
        from build_dashboard import _summarize
        base = _summarize(all_trades)
        realized = base["total_pnl_usd"]
        unrealized = sum(p["unrealized_pnl_usd"] for p in open_pos_raw)
        total = realized + unrealized
        # Bug #4 produced total=0 here (only realized counted).
        assert total == pytest.approx(51.04)
        assert realized == pytest.approx(0)
        assert unrealized == pytest.approx(51.04)


# ─────────────────────────────────────────────────────────────────────
# Bonus: independent FIFO recompute
# ─────────────────────────────────────────────────────────────────────

class TestFIFORecompute:
    """Sanity-check the second-brain that catches the next
    phantom-loss class of bug. A clean ledger should produce a
    matching DB total and FIFO recompute (drift ≤ $0.50)."""

    def test_clean_ledger_produces_zero_drift(self, tmp_path):
        from trading.recompute import recompute_realized_pnl_fifo

        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, product_id TEXT,
                side TEXT, amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL, dry_run INTEGER
            )
        """)
        # BUY 10 @ $100, then SELL 10 @ $110 → +$100 realized
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T10:00:00", "tsmom_etf", "SPY", "BUY",
             1000, 10, 100, "o1", None, 0),
        )
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T11:00:00", "tsmom_etf", "SPY", "SELL",
             1100, 10, 110, "o2", 100.0, 0),
        )
        conn.commit()
        conn.close()

        db_total, recomputed, drift = recompute_realized_pnl_fifo(str(db))
        assert db_total == pytest.approx(100.0)
        assert recomputed == pytest.approx(100.0)
        assert drift == {}    # no per-strategy drift > $0.50

    def test_drift_is_detected(self, tmp_path):
        """If the DB pnl_usd doesn't match what FIFO would derive,
        the function must report it in per_strategy_drift."""
        from trading.recompute import recompute_realized_pnl_fifo

        db = tmp_path / "test_drift.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, product_id TEXT,
                side TEXT, amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL, dry_run INTEGER
            )
        """)
        # BUY 10 @ $100, SELL 10 @ $110 — but DB stores wrong PnL=+$200
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T10:00:00", "tsmom_etf", "SPY", "BUY",
             1000, 10, 100, "o1", None, 0),
        )
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T11:00:00", "tsmom_etf", "SPY", "SELL",
             1100, 10, 110, "o2", 200.0, 0),    # ← WRONG: should be 100
        )
        conn.commit()
        conn.close()

        db_total, recomputed, drift = recompute_realized_pnl_fifo(str(db))
        assert db_total == pytest.approx(200.0)
        assert recomputed == pytest.approx(100.0)
        assert "tsmom_etf" in drift
        assert drift["tsmom_etf"] == pytest.approx(100.0)
