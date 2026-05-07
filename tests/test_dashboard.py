"""End-to-end test of the minimal dashboard.

The dashboard is a single-page server-rendered HTML file. We don't
unit-test private helpers (_per_strategy_pnl etc.) — we test the
contract: given a ledger with closed trades + a risk snapshot, the
rendered HTML contains the strategy name, mode badge, P&L value, and
kill-switch banner. This is the smallest possible test that catches
the kinds of regressions that break the dashboard for users.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def populated_dbs(tmp_path, monkeypatch):
    """Build a tiny trades.db + risk_state.db so render_dashboard has
    real data to work with."""
    trades_db = tmp_path / "trades.db"
    risk_db = tmp_path / "risk.db"

    # trades.db — one BUY + one closing SELL with known PnL
    with sqlite3.connect(trades_db) as conn:
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                product_id TEXT NOT NULL,
                side TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                order_id TEXT, pnl_usd REAL,
                dry_run INTEGER NOT NULL DEFAULT 1,
                fill_status TEXT NOT NULL DEFAULT 'PENDING',
                entry_price REAL, venue TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO trades "
            "(timestamp, strategy, product_id, side, amount_usd, quantity, "
            " price, order_id, pnl_usd, dry_run, fill_status, venue) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("2026-04-01T12:00:00+00:00", "tsmom_etf", "SPY",
                 "BUY", 1000, 2.0, 500.0, "o-1", None, 0, "FILLED", "alpaca"),
                ("2026-04-15T12:00:00+00:00", "tsmom_etf", "SPY",
                 "SELL", 1100, 2.0, 550.0, "o-2", 100.0, 0, "FILLED", "alpaca"),
                ("2026-04-20T12:00:00+00:00", "rsi_mean_reversion", "AAPL",
                 "BUY", 500, 5.0, 100.0, "o-3", None, 0, "FILLED", "alpaca"),
                ("2026-04-25T12:00:00+00:00", "rsi_mean_reversion", "AAPL",
                 "SELL", 400, 5.0, 80.0, "o-4", -100.0, 0, "FILLED", "alpaca"),
            ],
        )

    # risk_state.db — equity snapshot + a CRITICAL kill-switch row
    with sqlite3.connect(risk_db) as conn:
        conn.execute("""
            CREATE TABLE equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, equity_usd REAL NOT NULL,
                note TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE kill_switch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, state TEXT NOT NULL,
                drawdown_pct REAL, note TEXT
            )
        """)
        conn.execute(
            "INSERT INTO equity_snapshots (timestamp, equity_usd) "
            "VALUES (?, ?)",
            ("2026-05-01T00:00:00+00:00", 12345.67),
        )
        conn.execute(
            "INSERT INTO kill_switch_events (timestamp, state, drawdown_pct) "
            "VALUES (?, ?, ?)",
            ("2026-05-01T00:00:00+00:00", "WARNING", 0.05),
        )

    monkeypatch.setenv("TRADING_DB_PATH", str(trades_db))
    monkeypatch.setenv("RISK_DB_PATH", str(risk_db))
    monkeypatch.setenv("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("DRY_RUN_ALPACA", "false")
    monkeypatch.setenv("DRY_RUN", "true")
    return tmp_path


def test_dashboard_renders_strategies_ranked_by_pnl(populated_dbs, tmp_path):
    from build_dashboard import render_dashboard

    out = tmp_path / "out.html"
    render_dashboard(out)
    html = out.read_text(encoding="utf-8")

    # Header banner shows the kill-switch state + portfolio equity
    assert "WARNING" in html, "kill-switch state must appear in banner"
    assert "$12,345.67" in html, "portfolio equity must render formatted"

    # Both strategies render
    assert "tsmom_etf" in html
    assert "rsi_mean_reversion" in html

    # Mode badge: paper Alpaca + DRY_RUN_ALPACA=false → PAPER
    assert "PAPER" in html

    # Realized PnL values render (one positive, one negative)
    assert "$100.00" in html      # tsmom_etf
    assert "-$100.00" in html     # rsi_mean_reversion

    # Win/loss aggregation: two closed trades, one win, one loss
    assert "Closed trades" in html

    # Ordering: tsmom_etf (+$100) ranks above rsi_mean_reversion (-$100).
    # Both rows are tabular; the strategy with higher P&L should appear
    # first in the document.
    pos_tsmom = html.find(">tsmom_etf<")
    pos_rsi = html.find(">rsi_mean_reversion<")
    assert 0 < pos_tsmom < pos_rsi, (
        "Higher-P&L strategy must appear above lower-P&L strategy."
    )


def test_dashboard_handles_empty_dbs(tmp_path, monkeypatch):
    """No trade rows + no equity snapshots — the dashboard must still
    render a valid HTML page rather than crashing or producing junk."""
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("RISK_DB_PATH", str(tmp_path / "missing-risk.db"))
    monkeypatch.delenv("LIVE_STRATEGIES", raising=False)
    monkeypatch.setenv("DRY_RUN", "true")

    from build_dashboard import render_dashboard

    out = tmp_path / "empty.html"
    render_dashboard(out)
    html = out.read_text(encoding="utf-8")

    assert "<!DOCTYPE html>" in html
    # 24 strategies still listed from the registry, just with 0 trades each
    assert "DRY" in html
    # Kill-switch defaults to NO-DATA / UNKNOWN when risk_state.db is missing
    assert ("NO-DATA" in html) or ("UNKNOWN" in html)
