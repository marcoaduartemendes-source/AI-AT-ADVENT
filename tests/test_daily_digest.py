"""Tests for the daily digest builder.

Sprint E2 audit fix. The digest reads from SQLite + the alerts DB +
strategy_alerts state + risk manager — all of which are mocked here
so the test runs in an isolated tmp_path with no network.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from common.daily_digest import (
    build_digest,
    send_digest,
)


def _seed_trading_db(path: str, trades: list[dict]) -> None:
    with sqlite3.connect(path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                strategy TEXT,
                product_id TEXT,
                side TEXT,
                amount_usd REAL,
                quantity REAL,
                price REAL,
                order_id TEXT,
                pnl_usd REAL,
                fill_status TEXT
            )
        """)
        for t in trades:
            c.execute(
                "INSERT INTO trades(timestamp, strategy, product_id, side, "
                " amount_usd, quantity, price, order_id, pnl_usd, fill_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (t["timestamp"], t["strategy"], t["product_id"], t["side"],
                 t["amount_usd"], t["quantity"], t["price"], t["order_id"],
                 t["pnl_usd"], t["fill_status"]),
            )
        c.commit()


def test_digest_runs_with_no_db(monkeypatch, tmp_path):
    """Fresh deployment with no DB yet — digest must still build
    without crashing, returning a placeholder note."""
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "missing.db"))
    # Stub the broker registry to avoid hitting real APIs
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager") as fake_rm:
            fake_state = MagicMock()
            fake_state.kill_switch.value = "OK"
            fake_state.drawdown_pct = 0.0
            fake_state.peak_equity_usd = 0.0
            fake_state.equity_usd = 0.0
            fake_state.venues_ok = True
            fake_state.multiplier.effective = 1.0
            fake_rm.return_value.compute_state.return_value = fake_state
            text = build_digest()
    assert "AAA daily digest" in text
    assert "P&L" in text
    assert "Brokers" in text


def test_digest_includes_recent_pnl(monkeypatch, tmp_path):
    """Trades within last 24h appear in the headline P&L."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    _seed_trading_db(db, [
        {"timestamp": (now - timedelta(hours=2)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "SPY", "side": "SELL",
         "amount_usd": 1000.0, "quantity": 2.0, "price": 500.0,
         "order_id": "o1", "pnl_usd": 25.50, "fill_status": "FILLED"},
        {"timestamp": (now - timedelta(hours=10)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "QQQ", "side": "SELL",
         "amount_usd": 800.0, "quantity": 2.0, "price": 400.0,
         "order_id": "o2", "pnl_usd": -8.0, "fill_status": "FILLED"},
        # Older than 24h — should NOT appear
        {"timestamp": (now - timedelta(hours=48)).isoformat(),
         "strategy": "tsmom_etf", "product_id": "OLD", "side": "SELL",
         "amount_usd": 100.0, "quantity": 1.0, "price": 100.0,
         "order_id": "o3", "pnl_usd": 999.0, "fill_status": "FILLED"},
    ])
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager"):
            text = build_digest(now=now)

    # 24h window: 25.50 + (-8.0) = 17.50, the 999.0 row excluded
    assert "$+17.50" in text
    assert "999" not in text


def test_digest_top_movers(monkeypatch, tmp_path):
    """Winners and losers section lists the top 3 by realized P&L."""
    db = str(tmp_path / "trades.db")
    now = datetime.now(UTC)
    rows = []
    # 5 winners + 5 losers, all in last 24h
    for i, pnl in enumerate([100, 80, 60, 40, 20]):
        rows.append({
            "timestamp": (now - timedelta(hours=1 + i)).isoformat(),
            "strategy": f"win{i}", "product_id": f"WIN{i}", "side": "SELL",
            "amount_usd": 100.0, "quantity": 1.0, "price": 100.0,
            "order_id": f"w{i}", "pnl_usd": float(pnl),
            "fill_status": "FILLED",
        })
    for i, pnl in enumerate([-50, -40, -30, -20, -10]):
        rows.append({
            "timestamp": (now - timedelta(hours=10 + i)).isoformat(),
            "strategy": f"los{i}", "product_id": f"LOS{i}", "side": "SELL",
            "amount_usd": 100.0, "quantity": 1.0, "price": 100.0,
            "order_id": f"l{i}", "pnl_usd": float(pnl),
            "fill_status": "FILLED",
        })
    _seed_trading_db(db, rows)
    monkeypatch.setenv("TRADING_DB_PATH", db)
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager"):
            text = build_digest(now=now)
    assert "WIN0" in text     # top winner
    assert "WIN1" in text
    assert "WIN2" in text
    assert "LOS0" in text     # worst loser
    # 4th-best winner shouldn't make the top-3 list
    assert "WIN4" not in text


def test_digest_strategy_health_flags_failing(monkeypatch, tmp_path):
    """A strategy with 4 consecutive errors must show in the health section."""
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("STRATEGY_ALERTS_DB", str(tmp_path / "alerts.db"))
    # Pre-seed the alerts DB with a failing strategy
    from common.strategy_alerts import record_cycle_outcome
    fake_alert = MagicMock()
    for i in range(4):
        record_cycle_outcome("rsi_mr", had_error=True,
                                error_text=f"err {i}", alert_fn=fake_alert)
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager"):
            text = build_digest()
    assert "rsi_mr" in text
    assert "count=4" in text or "consecutive" in text


def test_send_digest_dispatches_alert_and_heartbeat(monkeypatch, tmp_path):
    """send_digest fires alert + heartbeat; failures don't propagate."""
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "missing.db"))
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager"):
            with patch("common.alerts.alert") as fake_alert:
                with patch("common.heartbeat.ping_success") as fake_hb:
                    fake_alert.return_value = True
                    fake_hb.return_value = True
                    ok = send_digest()
    assert ok is True
    assert fake_alert.call_count == 1
    assert fake_hb.call_count == 1


def test_send_digest_alert_failure_does_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "missing.db"))
    with patch("brokers.registry.build_brokers", return_value={}):
        with patch("risk.manager.RiskManager"):
            with patch("common.alerts.alert",
                         side_effect=RuntimeError("smtp down")):
                with patch("common.heartbeat.ping_success",
                             return_value=True):
                    ok = send_digest()
    # Build succeeded; alert dispatch failed but didn't propagate
    assert ok is True
