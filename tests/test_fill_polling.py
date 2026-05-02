"""End-to-end test of the fill-polling loop.

Records a trade with no fill, then transitions the mock broker's
order to FILLED, then runs _poll_pending_fills and asserts that the
trade row was backfilled with the real fill price + qty + PnL.

This is the test that would have caught the phantom-loss bug AND
proves the new polling loop actually backfills data correctly.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, UTC
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
        timestamp=datetime.now(UTC),
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
                "SELECT price, pnl_usd, fill_status FROM trades "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        # Audit fix #2: authoritative status is now fill_status,
        # not the price=-1 sentinel.
        assert row[2] == "CANCELED"
        assert row[1] is None


# ─── Audit fix #2: fill_status invariant tests ─────────────────────────


class TestFillStatusInvariants:
    """Audit fix #2: PnL must NEVER be persisted on rows whose
    fill_status is not 'FILLED'. Eliminates the phantom-loss bug class."""

    def test_record_trade_defaults_to_pending(self, tmp_path, monkeypatch):
        """Every newly-recorded trade with no PnL starts as PENDING."""
        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))
        tracker.record_trade(TradeRecord(
            timestamp=datetime.now(UTC),
            strategy="tsmom_etf", product_id="SPY", side="BUY",
            amount_usd=1000, quantity=2, price=0.0,    # ← no fill yet
            order_id="abc-1", pnl_usd=None, dry_run=False,
        ))
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT fill_status FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "PENDING"

    def test_record_trade_with_pnl_is_filled(self, tmp_path, monkeypatch):
        """If caller passes a non-null pnl_usd AND price>0 (synthetic
        backtest path or DRY auto-fill), the row is auto-marked FILLED."""
        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))
        tracker.record_trade(TradeRecord(
            timestamp=datetime.now(UTC),
            strategy="tsmom_etf", product_id="SPY", side="SELL",
            amount_usd=1100, quantity=2, price=550.0,
            order_id="abc-2", pnl_usd=100.0, dry_run=False,
        ))
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT fill_status, pnl_usd FROM trades "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert row[0] == "FILLED"
        assert row[1] == 100.0

    def test_update_with_canceled_nulls_pnl(self, tmp_path, monkeypatch):
        """Even if a buggy caller tries to pass pnl_usd alongside
        fill_status='CANCELED', update_trade_fill MUST null the pnl
        — enforces the invariant at the storage layer."""
        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))
        # Insert a PENDING row
        tracker.record_trade(TradeRecord(
            timestamp=datetime.now(UTC),
            strategy="x", product_id="y", side="SELL",
            amount_usd=0, quantity=10, price=0.0,
            order_id="o-1", pnl_usd=None, dry_run=False,
        ))
        with sqlite3.connect(db) as conn:
            tid = conn.execute(
                "SELECT id FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()[0]
        # Buggy update — pnl_usd present despite CANCELED
        tracker.update_trade_fill(
            trade_id=tid,
            price=0.0, quantity=0.0, amount_usd=0.0,
            pnl_usd=999.99,        # ← deliberately wrong
            fill_status="CANCELED",
        )
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT fill_status, pnl_usd FROM trades WHERE id = ?",
                (tid,),
            ).fetchone()
        assert row[0] == "CANCELED"
        assert row[1] is None    # pnl was nulled by the storage guard

    def test_get_unfilled_uses_fill_status(self, tmp_path, monkeypatch):
        """get_unfilled_trades returns only PENDING + PARTIALLY_FILLED.
        Previously matched on price=0 which had edge cases."""
        db = tmp_path / "trades.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))
        tracker = PerformanceTracker(str(db))
        # Three rows: PENDING, FILLED, CANCELED
        for status, price, oid in [
            (None,     0.0,  "p-1"),    # → PENDING (default)
            (50.0,     100.0, "p-2"),   # → FILLED (auto)
        ]:
            tracker.record_trade(TradeRecord(
                timestamp=datetime.now(UTC),
                strategy="x", product_id="y", side="BUY",
                amount_usd=100, quantity=1, price=price,
                order_id=oid, pnl_usd=status, dry_run=False,
            ))
        # Manually mark another as CANCELED
        tracker.record_trade(TradeRecord(
            timestamp=datetime.now(UTC),
            strategy="x", product_id="y", side="BUY",
            amount_usd=100, quantity=1, price=0.0,
            order_id="p-3", pnl_usd=None, dry_run=False,
        ))
        with sqlite3.connect(db) as conn:
            cancel_id = conn.execute(
                "SELECT id FROM trades WHERE order_id='p-3'"
            ).fetchone()[0]
        tracker.update_trade_fill(
            trade_id=cancel_id, price=0.0, quantity=0.0, amount_usd=0.0,
            pnl_usd=None, fill_status="CANCELED",
        )
        unfilled = tracker.get_unfilled_trades()
        order_ids = {row["order_id"] for row in unfilled}
        assert "p-1" in order_ids       # still PENDING
        assert "p-2" not in order_ids   # FILLED → not unfilled
        assert "p-3" not in order_ids   # CANCELED → not unfilled
