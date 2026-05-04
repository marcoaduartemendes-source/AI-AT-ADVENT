"""Tests for per-strategy lifetime P&L attribution.

User-reported bug: "lifetime P&L per strategy not updating". Root
cause was that broker-side open positions never had their unrealized
P&L attributed back to the strategy that opened them. Paper-trading
strategies that had entered but not closed showed $0 lifetime P&L
because realized=$0 and (without this attribution) unrealized=$0.

Fix: load_live_data now derives net open quantity per (strategy,
symbol) from the trade ledger, then walks broker positions and
splits each position's unrealized P&L proportionally across the
strategies that have a claim on that symbol.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from build_dashboard import load_live_data


def _seed(path: str, rows: list[dict]) -> None:
    with sqlite3.connect(path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT, product_id TEXT, side TEXT,
                amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL,
                dry_run INTEGER DEFAULT 0,
                fill_status TEXT DEFAULT 'FILLED'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS open_positions (
                strategy TEXT, product_id TEXT, quantity REAL,
                cost_basis_usd REAL, entry_price REAL, entry_time TEXT
            )
        """)
        for r in rows:
            c.execute(
                "INSERT INTO trades(timestamp, strategy, product_id, side, "
                "amount_usd, quantity, price, order_id, pnl_usd, fill_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r["timestamp"], r["strategy"], r["product_id"],
                 r.get("side", "BUY"), r.get("amount_usd", 100.0),
                 r.get("quantity", 1.0), r.get("price", 100.0),
                 r.get("order_id", f"o-{r['timestamp']}"),
                 r.get("pnl_usd"), r.get("fill_status", "FILLED")),
            )
        c.commit()


def test_per_strategy_unrealized_attributed_from_open_position(
    monkeypatch, tmp_path,
):
    """Paper strategy bought 2 SPY @ $500; broker now reports
    unrealized +$25. Strategy summary should reflect that:
      realized = $0  (no SELL)
      unrealized = +$25  (from broker)
      lifetime = $25
    """
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed(db, [
        {"timestamp": (now - timedelta(hours=2)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "BUY",
         "amount_usd": 1000.0, "quantity": 2.0, "price": 500.0,
         "order_id": "o-1", "pnl_usd": None},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)

    # Stub the broker to return one open SPY position with +$25 unrealized
    fake_broker = MagicMock()
    fake_broker.venue = "alpaca"
    fake_broker.get_account.return_value = MagicMock(
        cash_usd=98000.0, buying_power_usd=98000.0, equity_usd=100025.0,
    )

    # Build the broker snapshot _broker_snapshot would normally produce
    fake_position = MagicMock()
    fake_position.symbol = "SPY"
    fake_position.quantity = 2.0
    fake_position.avg_entry_price = 500.0
    fake_position.market_price = 512.50
    fake_position.unrealized_pnl_usd = 25.0
    fake_broker.get_positions.return_value = [fake_position]
    fake_broker.get_open_orders.return_value = []
    fake_broker.list_supported_asset_classes.return_value = []

    with patch("brokers.registry.build_brokers",
                return_value={"alpaca": fake_broker}):
        live = load_live_data()

    s = live["by_strategy"]["tsmom_etf"]["summary"]
    assert s["total_pnl_usd"] == 0.0   # realized: no closes
    assert abs(s["unrealized_pnl_usd"] - 25.0) < 0.01
    assert abs(s["lifetime_pnl_usd"] - 25.0) < 0.01


def test_strategy_with_no_open_position_unaffected(
    monkeypatch, tmp_path,
):
    """A strategy that never opened a position should have
    unrealized = 0 and lifetime = realized."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed(db, [
        # Closed round-trip: realized $5 profit
        {"timestamp": (now - timedelta(days=1)).isoformat(),
         "strategy": "rsi_mean_reversion", "product_id": "AAPL",
         "side": "BUY", "amount_usd": 100, "quantity": 1.0,
         "price": 100.0, "order_id": "o-buy", "pnl_usd": None},
        {"timestamp": now.isoformat(),
         "strategy": "rsi_mean_reversion", "product_id": "AAPL",
         "side": "SELL", "amount_usd": 105, "quantity": 1.0,
         "price": 105.0, "order_id": "o-sell", "pnl_usd": 5.0},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()

    s = live["by_strategy"]["rsi_mean_reversion"]["summary"]
    assert s["total_pnl_usd"] == 5.0
    assert s["unrealized_pnl_usd"] == 0.0
    assert s["lifetime_pnl_usd"] == 5.0


def test_pnl_sparkline_populated(monkeypatch, tmp_path):
    """Each strategy summary gets a `pnl_sparkline` array of cumulative
    realized P&L per closed trade. Drives the dashboard's mini chart."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed(db, [
        {"timestamp": (now - timedelta(days=3)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "SELL",
         "amount_usd": 105, "quantity": 1.0, "price": 105.0,
         "order_id": "s1", "pnl_usd": 5.0},
        {"timestamp": (now - timedelta(days=2)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "SELL",
         "amount_usd": 110, "quantity": 1.0, "price": 110.0,
         "order_id": "s2", "pnl_usd": -2.0},
        {"timestamp": (now - timedelta(days=1)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "SELL",
         "amount_usd": 108, "quantity": 1.0, "price": 108.0,
         "order_id": "s3", "pnl_usd": 8.0},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()

    spark = live["by_strategy"]["tsmom_etf"]["summary"]["pnl_sparkline"]
    # Cumulative: 5.0 → 3.0 → 11.0
    assert spark == [5.0, 3.0, 11.0]


def test_open_symbols_tracked_per_strategy(monkeypatch, tmp_path):
    """The summary's `open_symbols` dict reflects net open quantity
    per (strategy, symbol). Used by the attribution code to decide
    which strategies own which broker positions."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed(db, [
        # tsmom_etf: 2 BUYs of SPY, 1 partial SELL → net 1 SPY open
        {"timestamp": (now - timedelta(days=3)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "BUY",
         "amount_usd": 1000, "quantity": 2.0, "price": 500.0,
         "order_id": "b1", "pnl_usd": None},
        {"timestamp": (now - timedelta(days=2)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "SELL",
         "amount_usd": 510, "quantity": 1.0, "price": 510.0,
         "order_id": "s1", "pnl_usd": 10.0},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()

    s = live["by_strategy"]["tsmom_etf"]["summary"]
    assert s["open_symbols"] == {"SPY": 1.0}


def test_strategies_meta_in_config(monkeypatch, tmp_path):
    """Dashboard needs strategies_meta to render mode badges. Verify
    the run_orchestrator.ALL_STRATEGIES export carries through."""
    # Build the config dict the same way main() does
    from run_orchestrator import ALL_STRATEGIES
    meta = {
        m.name: {"venue": m.venue} for m in ALL_STRATEGIES
    }
    # tsmom_etf must be on alpaca; crypto_funding_carry on coinbase;
    # kalshi_calibration_arb on kalshi
    assert meta["tsmom_etf"]["venue"] == "alpaca"
    assert meta["crypto_funding_carry"]["venue"] == "coinbase"
    assert meta["kalshi_calibration_arb"]["venue"] == "kalshi"
