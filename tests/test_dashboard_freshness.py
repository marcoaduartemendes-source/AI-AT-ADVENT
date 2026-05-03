"""Tests for dashboard freshness + completeness invariants.

Bug fixed by this commit: `all_strategy_names` was hardcoded to 13
strategies in `load_live_data`. When 11 new strategies shipped
(Phase 4/4b/5) the dashboard silently rendered only the original
13 — every new strategy was invisible to the user.

These tests pin down the contract:
  - All registered strategies appear in by_strategy
  - Every entry has a last_trade_at field (None ok for new strategies)
  - days_since_last_trade is computed when a last_trade_at exists
  - Sort order in the JS layer is by total_pnl_usd desc
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from build_dashboard import load_live_data


def _seed_trades_db(path: str, rows: list[dict]) -> None:
    with sqlite3.connect(path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT,
                product_id TEXT,
                side TEXT,
                amount_usd REAL,
                quantity REAL,
                price REAL,
                order_id TEXT,
                pnl_usd REAL,
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
                (r["timestamp"], r["strategy"], r.get("product_id", "X"),
                 r.get("side", "BUY"), r.get("amount_usd", 100.0),
                 r.get("quantity", 1.0), r.get("price", 100.0),
                 r.get("order_id", f"o-{r['timestamp']}"),
                 r.get("pnl_usd"), r.get("fill_status", "FILLED")),
            )
        c.commit()


def test_all_registered_strategies_appear_in_by_strategy(monkeypatch, tmp_path):
    """Every strategy registered in ALL_STRATEGIES must show up in
    by_strategy, even when it has zero trades. This is the regression
    that hid 11 strategies from the dashboard."""
    db = str(tmp_path / "trades.db")
    _seed_trades_db(db, [])    # empty trades — strategies still must appear
    monkeypatch.setenv("TRADING_DB_PATH", db)

    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()

    by_strat = live["by_strategy"]
    # Pull the canonical list from the registry — same source the
    # production dashboard uses.
    from run_orchestrator import ALL_STRATEGIES
    expected = {meta.name for meta in ALL_STRATEGIES}
    missing = expected - set(by_strat.keys())
    assert not missing, (
        f"Strategies registered but missing from dashboard: {missing}. "
        f"This is the bug that hid Phase 4/4b/5 strategies for "
        f"weeks until the user noticed."
    )


def test_legacy_strategies_always_present(monkeypatch, tmp_path):
    """Momentum/MeanReversion/VolatilityBreakout pre-date the
    orchestrator registry; they must be hardcoded as visible."""
    db = str(tmp_path / "trades.db")
    _seed_trades_db(db, [])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()
    for legacy in ("Momentum", "MeanReversion", "VolatilityBreakout"):
        assert legacy in live["by_strategy"], (
            f"Legacy strategy {legacy} missing — historical Coinbase "
            f"trades would become invisible"
        )


def test_unknown_strategy_with_trades_still_appears(monkeypatch, tmp_path):
    """If a strategy was renamed in the registry but historical rows
    reference the old name, those rows must still be visible."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed_trades_db(db, [
        {"timestamp": (now - timedelta(hours=2)).isoformat(),
         "strategy": "RenamedStrategyXYZ", "side": "SELL",
         "amount_usd": 100.0, "quantity": 1.0, "price": 100.0,
         "order_id": "o-1", "pnl_usd": 5.0, "fill_status": "FILLED"},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()
    assert "RenamedStrategyXYZ" in live["by_strategy"]
    assert live["by_strategy"]["RenamedStrategyXYZ"]["summary"]["n_trades"] == 1


def test_last_trade_at_populated_per_strategy(monkeypatch, tmp_path):
    """Every strategy summary must include last_trade_at — null when
    the strategy has never traded; ISO timestamp when it has."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    older = (now - timedelta(days=2)).isoformat()
    newer = (now - timedelta(hours=3)).isoformat()
    _seed_trades_db(db, [
        {"timestamp": older, "strategy": "Momentum", "side": "BUY",
         "amount_usd": 100, "quantity": 1.0, "price": 100.0,
         "order_id": "o-1", "pnl_usd": None},
        {"timestamp": newer, "strategy": "Momentum", "side": "SELL",
         "amount_usd": 105, "quantity": 1.0, "price": 105.0,
         "order_id": "o-2", "pnl_usd": 5.0},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()

    momentum = live["by_strategy"]["Momentum"]["summary"]
    # Most-recent timestamp wins
    assert momentum["last_trade_at"] == newer
    assert momentum["days_since_last_trade"] is not None
    assert 0 <= momentum["days_since_last_trade"] <= 1   # ~3 hours

    # A strategy that has never traded → null
    no_trade = live["by_strategy"].get("MeanReversion", {}).get("summary", {})
    assert no_trade.get("last_trade_at") is None
    assert no_trade.get("days_since_last_trade") is None


def test_every_strategy_has_freshness_fields(monkeypatch, tmp_path):
    """Every strategy summary must include the fields the JS layer
    expects: last_trade_at + days_since_last_trade. Missing either
    would render '—' for every row in the Last-trade column."""
    db = str(tmp_path / "trades.db")
    _seed_trades_db(db, [])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()
    for st in live["by_strategy"].values():
        assert "summary" in st
        assert "trades" in st
        assert "last_trade_at" in st["summary"]
        assert "days_since_last_trade" in st["summary"]


def test_total_pnl_field_present_for_sort_invariant(monkeypatch, tmp_path):
    """The JS layer sorts rows by total_pnl_usd desc. Every summary
    must have that field or the sort silently breaks."""
    db = str(tmp_path / "trades.db")
    _seed_trades_db(db, [])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        live = load_live_data()
    for st in live["by_strategy"].values():
        s = st["summary"]
        assert "total_pnl_usd" in s, (
            f"{s.get('strategy')} missing total_pnl_usd — strategy "
            f"would mis-rank in the dashboard sort"
        )
