"""Tests for the trades-side Supabase failover in PerformanceTracker.

Audit-fix follow-up: trades were dual-written to Supabase but never
read from there. If the local SQLite gets wiped (corruption, bad
restore, accidental rm) the dashboard rendered empty trade history
even though Supabase preserved everything. The new failover path
consults Supabase only when SQLite is empty — non-empty SQLite
trumps Supabase, since dual-writes keep them in sync under normal
operation.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from trading.performance import PerformanceTracker
from trading.portfolio import TradeRecord
from datetime import UTC, datetime


def test_sqlite_with_rows_does_not_consult_supabase(tmp_path):
    """Happy path: SQLite has data → Supabase is not queried."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.recent_trades.side_effect = AssertionError(
        "Should not be called when SQLite has data"
    )
    tracker = PerformanceTracker(db_path=str(tmp_path / "trades.db"))
    tracker._supabase = fake    # inject after init

    # Add one trade
    tracker.record_trade(TradeRecord(
        timestamp=datetime.now(UTC), strategy="x", product_id="SPY",
        side="BUY", amount_usd=1000, quantity=2.0, price=500.0,
        order_id="o-1", pnl_usd=None, fill_status="FILLED",
    ))
    rows = tracker.get_recent_trades(limit=10)
    assert len(rows) == 1
    # Supabase NOT queried (would have raised)


def test_empty_sqlite_falls_back_to_supabase(tmp_path):
    """Disaster recovery: SQLite empty → Supabase wins."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.recent_trades.return_value = [
        {"order_id": "sb-1", "strategy": "x", "product_id": "SPY",
         "pnl_usd": 25.0, "fill_status": "FILLED"},
        {"order_id": "sb-2", "strategy": "x", "product_id": "QQQ",
         "pnl_usd": -5.0, "fill_status": "FILLED"},
    ]
    tracker = PerformanceTracker(db_path=str(tmp_path / "fresh.db"))
    tracker._supabase = fake

    rows = tracker.get_recent_trades(limit=10)
    assert len(rows) == 2
    assert {r["order_id"] for r in rows} == {"sb-1", "sb-2"}
    assert fake.recent_trades.call_count == 1


def test_strategy_filter_passes_to_supabase(tmp_path):
    """Strategy filter must propagate to the Supabase failover read."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.recent_trades.return_value = []
    tracker = PerformanceTracker(db_path=str(tmp_path / "fresh.db"))
    tracker._supabase = fake

    tracker.get_recent_trades(strategy="tsmom_etf", limit=5)
    fake.recent_trades.assert_called_once_with(
        strategy="tsmom_etf", limit=5,
    )


def test_supabase_failover_failure_returns_empty(tmp_path):
    """Supabase raising on the failover read must not propagate."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.recent_trades.side_effect = RuntimeError("postgrest 503")
    tracker = PerformanceTracker(db_path=str(tmp_path / "fresh.db"))
    tracker._supabase = fake

    rows = tracker.get_recent_trades(limit=10)
    # SQLite empty + Supabase failed → empty list, no crash
    assert rows == []


def test_no_supabase_means_sqlite_only(tmp_path):
    """No Supabase configured → SQLite-only behaviour, unchanged."""
    tracker = PerformanceTracker(db_path=str(tmp_path / "trades.db"))
    tracker._supabase = None
    rows = tracker.get_recent_trades(limit=10)
    assert rows == []
