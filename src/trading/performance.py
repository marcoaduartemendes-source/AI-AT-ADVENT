import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from .portfolio import TradeRecord

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks all trades in SQLite — and, when SUPABASE_URL +
    SUPABASE_SERVICE_KEY are set, dual-writes every insert/update
    to Supabase Postgres as well.

    Dual-write rationale: SQLite stays the source of truth until we've
    seen ≥1 week of clean Postgres writes. Reads still come from
    SQLite during the migration window. After cutover (a separate
    explicit step), Postgres becomes primary and SQLite drops to
    a local cache."""

    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        logger.info(f"Performance DB: {self.db_path}")

        # Optional secondary sink — Supabase. Lazily-imported so the
        # tracker doesn't pull `requests` at module-load time when
        # nothing's configured.
        self._supabase = None
        try:
            from common.supabase_store import SupabaseStore
            store = SupabaseStore()
            if store.is_configured():
                self._supabase = store
                logger.info("Supabase dual-write enabled (PerformanceTracker)")
        except ImportError:
            pass    # supabase_store missing — older deployment

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
                    dry_run      INTEGER NOT NULL DEFAULT 1,
                    fill_status  TEXT    NOT NULL DEFAULT 'PENDING'
                )
            """)
            # Audit fix #2: fill_status migration for existing rows.
            # SQLite ALTER TABLE ADD COLUMN is supported; we wrap in
            # try/except so re-running on a fresh DB doesn't error.
            try:
                conn.execute(
                    "ALTER TABLE trades ADD COLUMN fill_status TEXT "
                    "NOT NULL DEFAULT 'PENDING'"
                )
                # Backfill fill_status from price for legacy rows:
                #   price > 0 AND pnl_usd IS NOT NULL → FILLED
                #   price == -1                       → CANCELED (sentinel)
                #   price == 0                        → PENDING (default)
                conn.execute(
                    "UPDATE trades SET fill_status = 'FILLED' "
                    "WHERE price > 0 AND fill_status = 'PENDING'"
                )
                conn.execute(
                    "UPDATE trades SET fill_status = 'CANCELED' "
                    "WHERE price < 0 AND fill_status = 'PENDING'"
                )
                logger.info("trades.fill_status column added + backfilled")
            except sqlite3.OperationalError:
                pass    # column already exists
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
        # Audit fix #2 + post-deploy fix: prefer the explicit
        # fill_status from the orchestrator (which knows the actual
        # broker order status) over the auto-detect heuristic. The
        # heuristic was conservative (PENDING unless pnl_usd was
        # set) which incorrectly stranded opening BUYs with non-zero
        # price as PENDING — they never got picked up by the polling
        # loop because their order_id was already attached, leading
        # to 770 stuck rows in production.
        if record.fill_status is not None:
            initial_status = record.fill_status
        else:
            # Legacy callers (synthetic / DRY) get the auto-detect:
            # if there's a real pnl_usd, the order was clearly filled.
            # Otherwise default to PENDING and let the poller transition.
            initial_status = ("FILLED"
                              if (record.pnl_usd is not None
                                  and record.price > 0)
                              else "PENDING")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO trades
                  (timestamp, strategy, product_id, side, amount_usd,
                   quantity, price, order_id, pnl_usd, dry_run, fill_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    initial_status,
                ),
            )
        # Dual-write to Supabase (if configured). Failure logs but
        # never raises — SQLite is still the source of truth.
        if self._supabase is not None:
            self._supabase.insert_trade({
                "timestamp": record.timestamp.isoformat(),
                "strategy": record.strategy,
                "product_id": record.product_id,
                "side": record.side,
                "amount_usd": record.amount_usd,
                "quantity": record.quantity,
                "price": record.price,
                "order_id": record.order_id,
                "pnl_usd": record.pnl_usd,
                "dry_run": bool(record.dry_run),
                "fill_status": initial_status,
                "recorded_via": os.environ.get("DEPLOY_ENV", "github-actions"),
            })

    # ── Fill backfill (Phase 0 — fill polling) ───────────────────────────────

    def get_unfilled_trades(self, max_age_hours: int = 48) -> list[dict]:
        """Return trade rows the broker hasn't reported as filled yet.

        Audit fix #2: query is now keyed on fill_status='PENDING'
        (or 'PARTIALLY_FILLED' — partial fills should be re-polled
        until they complete). Previously we matched on price=0,
        which conflated "broker didn't fill yet" with "broker
        reported a $0 fill" with "we never updated the price".
        fill_status disambiguates.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, strategy, product_id, side,
                       amount_usd, quantity, price, order_id, pnl_usd,
                       fill_status
                FROM trades
                WHERE order_id IS NOT NULL
                  AND order_id != ''
                  AND order_id != 'unknown'
                  AND fill_status IN ('PENDING', 'PARTIALLY_FILLED')
                  AND timestamp >= datetime('now', ?)
                ORDER BY id ASC
                """,
                (f"-{max_age_hours} hours",),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_trade_fill(
        self, trade_id: int, price: float, quantity: float,
        amount_usd: float, pnl_usd: float | None,
        fill_status: str = "FILLED",
    ) -> None:
        """Backfill a trade row with real fill data once the broker
        reports the order as filled.

        Audit fix #2: now sets fill_status as well. Caller passes one
        of FILLED / PARTIALLY_FILLED / CANCELED / REJECTED. The
        invariant that PnL is only meaningful for fill_status='FILLED'
        is enforced here — if anyone passes a non-FILLED status with
        a non-null pnl_usd, we null the PnL out so the dashboard
        never displays speculative numbers.
        """
        if fill_status != "FILLED" and pnl_usd is not None:
            logger.warning(
                f"update_trade_fill: nulling pnl_usd ${pnl_usd:.2f} "
                f"for non-FILLED status={fill_status} (id={trade_id})"
            )
            pnl_usd = None

        # Read the row first so we can identify it on Postgres by
        # order_id (different primary key spaces between SQLite and
        # Postgres). Tiny extra read; keeps writes idempotent.
        order_id = None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT order_id FROM trades WHERE id = ?", (trade_id,),
            ).fetchone()
            if row:
                order_id = row["order_id"]
            conn.execute(
                """
                UPDATE trades
                   SET price       = ?,
                       quantity    = ?,
                       amount_usd  = ?,
                       pnl_usd     = ?,
                       fill_status = ?
                 WHERE id = ?
                """,
                (price, quantity, amount_usd, pnl_usd, fill_status, trade_id),
            )
        # Dual-write the backfill to Supabase by order_id
        if self._supabase is not None and order_id and order_id != "unknown":
            self._supabase.update_trade_fill(
                order_id=order_id,
                price=price, quantity=quantity,
                amount_usd=amount_usd, pnl_usd=pnl_usd,
                fill_status=fill_status,
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
        """SQLite-primary, Supabase failover.

        SQLite is the read primary under normal conditions. If SQLite
        returns 0 rows AND Supabase is configured, we consult Supabase
        as a disaster-recovery fallback — covers the "VPS DB wiped,
        dashboard now empty" scenario where Supabase still has the
        full history. Same failover pattern as EquitySnapshotDB.peak_equity."""
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
        local = [dict(r) for r in rows]

        # Supabase failover: only when SQLite is empty (the suspect
        # disaster-recovery case). When SQLite has any rows we trust
        # it — dual-writes keep Supabase in sync, so an empty SQLite
        # is the only state where Supabase could legitimately have
        # more than us.
        if not local and self._supabase is not None:
            try:
                sb_rows = self._supabase.recent_trades(
                    strategy=strategy, limit=limit,
                )
            except Exception as e:    # noqa: BLE001
                logger.warning(f"Supabase trades failover read failed: {e}")
                sb_rows = []
            if sb_rows:
                logger.warning(
                    f"PerformanceTracker.get_recent_trades: SQLite empty, "
                    f"using Supabase ({len(sb_rows)} rows) — "
                    f"disaster-recovery path"
                )
                return sb_rows
        return local

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
