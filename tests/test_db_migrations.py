"""Tests for src/trading/migrations.py.

Audit-fix follow-up: the prior init-time UPDATE ran on every
orchestrator process start, masked schema problems behind a broad
`except Exception` at debug level, and had no replay safety. The
new migration runner uses a marker-row table so each migration
applies at most once.
"""
from __future__ import annotations

import sqlite3

from trading.migrations import MIGRATIONS, apply_pending


def _seed_trades_table(db_path: str, rows: list[dict]) -> None:
    with sqlite3.connect(db_path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
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
        for r in rows:
            c.execute(
                "INSERT INTO trades(timestamp, strategy, product_id, side, "
                "amount_usd, quantity, price, order_id, pnl_usd, fill_status) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (r["timestamp"], r["strategy"], r["product_id"], r["side"],
                 r["amount_usd"], r["quantity"], r["price"], r["order_id"],
                 r["pnl_usd"], r["fill_status"]),
            )
        c.commit()


def test_phantom_pnl_rows_get_nulled(tmp_path):
    """Trades with price=0 + non-null pnl_usd are the phantom-loss
    rows. Migration nulls them out."""
    db = str(tmp_path / "trades.db")
    _seed_trades_table(db, [
        {"timestamp": "2026-04-01T10:00:00", "strategy": "tsmom",
         "product_id": "SPY", "side": "SELL", "amount_usd": 1000.0,
         "quantity": 2.0, "price": 0.0,    # phantom!
         "order_id": "o1", "pnl_usd": -5746.0,
         "fill_status": "FILLED"},
        # Healthy row — should NOT be touched
        {"timestamp": "2026-04-01T11:00:00", "strategy": "tsmom",
         "product_id": "QQQ", "side": "SELL", "amount_usd": 2000.0,
         "quantity": 5.0, "price": 400.0,
         "order_id": "o2", "pnl_usd": 25.0,
         "fill_status": "FILLED"},
    ])
    summary = apply_pending(db)
    assert summary["001_null_phantom_pnl"]["applied"] is True
    assert summary["001_null_phantom_pnl"]["rows"] == 1

    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT order_id, pnl_usd FROM trades ORDER BY id"
        ).fetchall()
    assert rows[0] == ("o1", None)         # nulled
    assert rows[1] == ("o2", 25.0)         # untouched


def test_migration_does_not_re_run(tmp_path):
    """Second invocation must be a no-op — the marker row prevents replay."""
    db = str(tmp_path / "trades.db")
    _seed_trades_table(db, [
        {"timestamp": "2026-04-01T10:00:00", "strategy": "tsmom",
         "product_id": "SPY", "side": "SELL", "amount_usd": 1000.0,
         "quantity": 2.0, "price": 0.0,
         "order_id": "o1", "pnl_usd": -5746.0,
         "fill_status": "FILLED"},
    ])
    apply_pending(db)
    # Insert another phantom row AFTER first migration ran — this row
    # should NOT be touched on re-run (idempotency by name, not state).
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO trades (timestamp, strategy, product_id, side, "
            "amount_usd, quantity, price, order_id, pnl_usd, fill_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-04-02T10:00", "tsmom", "IWM", "SELL", 1000.0, 2.0,
             0.0, "o2", -100.0, "FILLED"),
        )
        c.commit()

    summary = apply_pending(db)
    assert summary["001_null_phantom_pnl"]["applied"] is False
    assert summary["001_null_phantom_pnl"]["rows"] == 0

    with sqlite3.connect(db) as c:
        # The new phantom row stays as-is — migrations are name-gated,
        # not state-gated. Replay safety > eager cleanup.
        row = c.execute(
            "SELECT pnl_usd FROM trades WHERE order_id = 'o2'"
        ).fetchone()
    assert row[0] == -100.0


def test_marker_table_is_created_idempotently(tmp_path):
    """Running on a brand-new DB must work without pre-creating
    schema_migrations."""
    db = str(tmp_path / "fresh.db")
    # Empty DB — no trades table
    summary = apply_pending(db)
    # Migration ran, touched 0 rows (no trades to fix)
    assert "001_null_phantom_pnl" in summary
    # Marker table now exists
    with sqlite3.connect(db) as c:
        rows = c.execute("SELECT name FROM schema_migrations").fetchall()
    assert ("001_null_phantom_pnl",) in rows


def test_unopenable_db_returns_empty(tmp_path):
    """If the DB path is unwritable, return empty summary instead of
    raising — the orchestrator must not abort init on a broken DB."""
    bad_path = str(tmp_path / "nonexistent_dir" / "trades.db")
    summary = apply_pending(bad_path)
    # Empty dict on outer failure — sqlite couldn't even open
    assert summary == {}


def test_registry_is_append_only():
    """Defensive: the audit's anti-pattern is mutating an existing
    migration in-place. The list must only ever grow."""
    names = [m.name for m in MIGRATIONS]
    # No duplicates
    assert len(names) == len(set(names))
    # First migration must always be 001 (the phantom-PnL fix); newer
    # migrations append. If you remove this one, you'll re-replay it
    # against any production DB whose marker row references its name.
    assert names[0] == "001_null_phantom_pnl"
