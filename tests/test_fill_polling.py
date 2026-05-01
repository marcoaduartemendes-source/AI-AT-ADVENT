"""End-to-end test of the fill-polling loop.

Records a trade with no fill, then transitions the mock broker's
order to FILLED, then runs _poll_pending_fills and asserts that the
trade row was backfilled with the real fill price + qty + PnL.

This is the test that would have caught the phantom-loss bug AND
proves the new polling loop actually backfills data correctly.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from brokers.base import OrderSide, OrderStatus, OrderType
from tests.mock_broker import MockBroker, MockPosition
from trading.performance import PerformanceTracker
from trading.portfolio import TradeRecord


def _seed_unfilled_trade(tracker: PerformanceTracker, order_id: str,
                          strategy: str, symbol: str, side: str = "SELL"):
    """Insert a row that looks like a trade recorded at submit-time —
    price=0, pnl_usd=NULL, has an order_id."""
    tracker.record_trade(TradeRecord(
        timestamp=datetime.now(timezone.utc),
        strategy=strategy,
        product_id=symbol,
        side=side,
        amount_usd=0.0,
        quantity=10.0,
        price=0.0,         # ← The submit-time placeholder
        order_id=order_id,
        pnl_usd=None,
        dry_run=False,
    ))


class TestPollPendingFills:

    def test_fill_polling_backfills_real_price_and_qty(self, tmp_path, monkeypatch):
        from strategy_engine.orchestrator import Orchestrator

        # Set up a tracker on a fresh DB
        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))

        # Mock broker holds an SPY position at $720 entry; we'll
        # simulate a SELL filling at $725 → +$5 × 10 qty = +$50 PnL.
        broker = MockBroker(
            venue="alpaca",
            cash_usd=10_000,
            positions=[MockPosition("SPY", qty=10, entry=720, mark=725)],
        )

        # Place a SELL order; mock reports it as PENDING (not yet filled)
        broker.fill_next = "PENDING"
        order = broker.place_order(
            symbol="SPY", side=OrderSide.SELL, type=OrderType.MARKET,
            quantity=10,
        )
        assert order.status == OrderStatus.PENDING

        # Seed the ledger with a row matching that order
        _seed_unfilled_trade(
            tracker, order.order_id, "tsmom_etf", "SPY", side="SELL",
        )

        # Build a stub orchestrator with the bits _poll_pending_fills uses
        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = tracker
        orch.brokers = {"alpaca": broker}
        orch.strategies = {"tsmom_etf": MagicMock(venue="alpaca")}
        orch.risk = MagicMock()
        orch.risk.cached_positions = MagicMock(
            return_value=broker.get_positions()
        )

        # The order is still PENDING; polling should be a no-op
        report = MagicMock()
        orch._poll_pending_fills(report)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT price, pnl_usd FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == 0           # price still 0 — unfilled
        assert row[1] is None        # PnL still NULL

        # Now simulate the broker filling at $725
        broker.fill_order(order.order_id, price=725.0, quantity=10.0)

        # Refresh cached positions to reflect post-fill state (the
        # original SPY position is now closed by the SELL — but
        # _poll_pending_fills uses the position avg_entry_price BEFORE
        # the fill, so we keep it at 720 in the cached snapshot.
        orch.risk.cached_positions = MagicMock(
            return_value=[
                MockPosition("SPY", qty=10, entry=720, mark=725).to_position("alpaca")
            ]
        )

        orch._poll_pending_fills(report)
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT price, quantity, amount_usd, pnl_usd "
                "FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        price, qty, amount, pnl = row
        assert price == pytest.approx(725.0)
        assert qty == pytest.approx(10.0)
        assert amount == pytest.approx(7250.0)
        assert pnl == pytest.approx(50.0)   # (725 - 720) × 10

    def test_fill_polling_skips_canceled_orders(self, tmp_path, monkeypatch):
        from strategy_engine.orchestrator import Orchestrator

        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))

        broker = MockBroker(venue="alpaca")
        broker.fill_next = "PENDING"
        order = broker.place_order(
            symbol="TLT", side=OrderSide.BUY, type=OrderType.MARKET,
            quantity=5,
        )
        _seed_unfilled_trade(tracker, order.order_id, "tsmom_etf", "TLT", "BUY")
        # Now cancel
        broker.cancel_order(order.order_id)

        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = tracker
        orch.brokers = {"alpaca": broker}
        orch.strategies = {"tsmom_etf": MagicMock(venue="alpaca")}
        orch.risk = MagicMock()
        orch.risk.cached_positions = MagicMock(return_value=[])

        orch._poll_pending_fills(MagicMock())
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT price, pnl_usd FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        # Sentinel marker so the polling loop stops re-checking
        assert row[0] == -1
        assert row[1] is None
