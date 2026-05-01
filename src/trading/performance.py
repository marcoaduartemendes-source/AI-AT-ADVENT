import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from .portfolio import TradeRecord

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks all trades in SQLite and computes strategy metrics."""

    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        logger.info(f"Performance DB: {self.db_path}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    strategy       TEXT NOT NULL,
                    product_id     TEXT NOT NULL,
                    quantity       REAL NOT NULL,
                    cost_basis_usd REAL NOT NULL,
                    entry_price    REAL NOT NULL,
                    entry_time     TEXT NOT NULL,
                    PRIMARY KEY (strategy, product_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    strategy     TEXT    NOT NULL,
                    product_id   TEXT    NOT NULL,
                    side         TEXT    NOT NULL,
                    amount_usd   REAL    NOT NULL,
                    quantity     REAL    NOT NULL,
                    price        REAL    NOT NULL,
                    order_id     TEXT,
                    pnl_usd      REAL,
                    dry_run      INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT    NOT NULL,
                    strategy     TEXT    NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning      INTEGER NOT NULL,
                    total_pnl    REAL    NOT NULL,
                    win_rate     REAL,
                    sharpe       REAL,
                    max_drawdown REAL
                )
            """)
            # Performance indexes — strategies query by strategy name +
            # side='SELL' on every cycle; without these, full-table scan.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy_side "
                          "ON trades(strategy, side, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp "
                          "ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_strategy "
                          "ON snapshots(strategy, timestamp)")

    # ── Recording ────────────────────────────────────────────────────────────

    def record_trade(self, record: TradeRecord):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades
                  (timestamp, strategy, product_id, side, amount_usd, quantity, price, order_id, pnl_usd, dry_run)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.strategy,
                    record.product_id,
                    record.side,
                    record.amount_usd,
                    record.quantity,
                    record.price,
                    record.order_id,
                    record.pnl_usd,
                    1 if record.dry_run else 0,
                ),
            )

    # ── Fill backfill (Phase 0 — fill polling) ───────────────────────────────

    def get_unfilled_trades(self, max_age_hours: int = 48) -> list[dict]:
        """Return trade rows where the broker hadn't reported a fill at
        record-time (price=0 or NULL), still within the polling window.

        These rows are candidates for a fill-status check the next time
        the orchestrator polls. Older rows are presumed dead — orders
        almost always either fill or get cancelled within a few hours;
        anything older than `max_age_hours` is skipped to keep the
        polling loop bounded.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, strategy, product_id, side,
                       amount_usd, quantity, price, order_id, pnl_usd
                FROM trades
                WHERE order_id IS NOT NULL
                  AND order_id != ''
                  AND order_id != 'unknown'
                  AND (price IS NULL OR price = 0)
                  AND timestamp >= datetime('now', ?)
                ORDER BY id ASC
                """,
                (f"-{max_age_hours} hours",),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_trade_fill(
        self, trade_id: int, price: float, quantity: float,
        amount_usd: float, pnl_usd: float | None,
    ) -> None:
        """Backfill a trade row with real fill data once the broker
        reports the order as filled. Sets price, quantity, amount_usd,
        and (for closing SELLs that we can attribute) pnl_usd."""
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE trades
                   SET price      = ?,
                       quantity   = ?,
                       amount_usd = ?,
                       pnl_usd    = ?
                 WHERE id = ?
                """,
                (price, quantity, amount_usd, pnl_usd, trade_id),
            )

    # ── Position persistence ─────────────────────────────────────────────────

    def save_positions(self, positions: dict):
        """Persist open positions to SQLite so they survive between bot restarts."""
        with self._conn() as conn:
            conn.execute("DELETE FROM open_positions")
            for strategy, prod_map in positions.items():
                for product_id, pos in prod_map.items():
                    conn.execute(
                        """INSERT INTO open_positions
                           (strategy, product_id, quantity, cost_basis_usd, entry_price, entry_time)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (strategy, product_id, pos.quantity,
                         pos.cost_basis_usd, pos.entry_price,
                         pos.entry_time.isoformat()),
                    )

    def load_positions(self) -> dict:
        """Reload persisted positions on startup."""
        from datetime import datetime
        from .portfolio import Position
        positions: dict = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM open_positions").fetchall()
        for r in rows:
            positions.setdefault(r["strategy"], {})[r["product_id"]] = Position(
                product_id=r["product_id"],
                quantity=r["quantity"],
                cost_basis_usd=r["cost_basis_usd"],
                entry_price=r["entry_price"],
                entry_time=datetime.fromisoformat(r["entry_time"]),
                strategy=r["strategy"],
            )
        return positions

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_metrics(self, strategy: str | None = None) -> dict:
        with self._conn() as conn:
            if strategy:
                rows = conn.execute(
                    "SELECT pnl_usd FROM trades WHERE strategy=? AND side='SELL' ORDER BY timestamp",
                    (strategy,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT pnl_usd FROM trades WHERE side='SELL' ORDER BY timestamp"
                ).fetchall()

        pnls = [r["pnl_usd"] for r in rows if r["pnl_usd"] is not None]

        base = {
            "strategy": strategy or "ALL",
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "sharpe": None,
            "max_drawdown": 0.0,
        }
        if not pnls:
            return base

        arr = np.array(pnls)
        wins = int((arr > 0).sum())

        sharpe = None
        if len(arr) > 1 and arr.std() > 0:
            # Annualise using sqrt(252) — approximate for hourly-to-daily comparison
            sharpe = round(float(arr.mean() / arr.std() * np.sqrt(252)), 3)

        cumulative = np.cumsum(arr)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float((running_max - cumulative).max())

        return {
            "strategy": strategy or "ALL",
            "closed_trades": len(pnls),
            "wins": wins,
            "losses": len(pnls) - wins,
            "win_rate": wins / len(pnls),
            "total_pnl": float(arr.sum()),
            "avg_pnl": float(arr.mean()),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    def get_recent_trades(self, strategy: str | None = None, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            if strategy:
                rows = conn.execute(
                    "SELECT * FROM trades WHERE strategy=? ORDER BY timestamp DESC LIMIT ?",
                    (strategy, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Dashboard ────────────────────────────────────────────────────────────

    def print_dashboard(self, strategies: list[str]):
        W = 62
        print("\n" + "═" * W)
        print("  CRYPTO TRADING DASHBOARD  ·  " + datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        print("═" * W)

        for name in strategies:
            m = self.get_metrics(name)
            sign = "+" if m["total_pnl"] >= 0 else ""
            print(f"\n  ▶  {name}")
            print(f"  {'─' * (W - 4)}")
            print(f"  Closed trades : {m['closed_trades']:>6}   "
                  f"(W={m['wins']} / L={m['losses']})")
            print(f"  Win rate      : {m['win_rate'] * 100:>5.1f}%")
            print(f"  Total P&L     : ${sign}{m['total_pnl']:>8.2f}")
            print(f"  Avg P&L/trade : ${m['avg_pnl']:>+8.2f}")
            if m["sharpe"] is not None:
                print(f"  Sharpe ratio  : {m['sharpe']:>8.3f}")
            print(f"  Max drawdown  : ${m['max_drawdown']:>8.2f}")

        print(f"\n  {'─' * (W - 4)}")
        total = self.get_metrics()
        sign = "+" if total["total_pnl"] >= 0 else ""
        print(f"  COMBINED      : ${sign}{total['total_pnl']:.2f} "
              f"across {total['closed_trades']} closed trades  "
              f"(win rate {total['win_rate'] * 100:.1f}%)")
        print("═" * W + "\n")

    def save_snapshot(self, strategies: list[str]):
        for name in strategies:
            m = self.get_metrics(name)
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO snapshots
                      (timestamp, strategy, total_trades, winning, total_pnl, win_rate, sharpe, max_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.utcnow().isoformat(),
                        name,
                        m["closed_trades"],
                        m["wins"],
                        m["total_pnl"],
                        m["win_rate"],
                        m["sharpe"],
                        m["max_drawdown"],
                    ),
                )
