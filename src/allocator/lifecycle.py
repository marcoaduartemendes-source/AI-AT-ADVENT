"""Strategy lifecycle state machine.

Each strategy is a pod that moves through states based on rolling
performance:

    ACTIVE  ─┬─ 30d Sharpe < 0 OR DD > warning_dd ─→ WATCH
             │   (allocation halved; flagged on dashboard)
             │
             ├─ 30d Sharpe < freeze_sharpe OR DD > freeze_dd ─→ FROZEN
             │   (allocation = 0; positions closed; signals still computed)
             │
             └─ manual ─→ FROZEN | RETIRED

    WATCH   ─┬─ next 14d Sharpe > recovery_sharpe ─→ ACTIVE
             ├─ continued underperformance        ─→ FROZEN
             └─ manual                            ─→ ACTIVE | FROZEN | RETIRED

    FROZEN  ─── manual unfreeze only ───→ ACTIVE
                (no auto-resurrection — too easy to chase recoveries)

    RETIRED ─── manual revival only ───→ ACTIVE

Every transition is logged to SQLite with timestamp + reason.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
from enum import Enum

logger = logging.getLogger(__name__)


class StrategyState(str, Enum):
    ACTIVE = "ACTIVE"
    WATCH = "WATCH"
    FROZEN = "FROZEN"
    RETIRED = "RETIRED"


@dataclass
class StrategyMeta:
    """Static metadata about a strategy. Stored in the registry."""

    name: str
    asset_classes: list[str]              # e.g. ["CRYPTO_PERP"]
    venue: str                            # primary broker
    target_alloc_pct: float               # baseline % of capital (allocator may scale)
    max_alloc_pct: float = 0.30           # hard ceiling
    min_alloc_pct: float = 0.05           # floor while ACTIVE/WATCH (0 if FROZEN)
    description: str = ""
    enabled: bool = True                  # master kill — disabled won't even compute signals


class StrategyRegistry:
    """Runtime registry of strategies + lifecycle persistence."""

    def __init__(self, db_path: str | None = None):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("ALLOCATOR_DB_PATH", "data/allocator.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._meta: dict[str, StrategyMeta] = {}
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS strategy_state (
                    name       TEXT PRIMARY KEY,
                    state      TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    note       TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS lifecycle_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    from_state TEXT,
                    to_state   TEXT NOT NULL,
                    reason     TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS allocations (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp     TEXT NOT NULL,
                    name          TEXT NOT NULL,
                    target_pct    REAL NOT NULL,
                    target_usd    REAL NOT NULL,
                    state         TEXT NOT NULL,
                    sharpe        REAL,
                    drawdown_pct  REAL,
                    reason        TEXT
                )
            """)

    @contextmanager
    def _conn(self):
        from common.sqlite_pragmas import apply_pragmas
        conn = sqlite3.connect(self.db_path)
        apply_pragmas(conn, self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Registration -----------------------------------------------------

    def register(self, meta: StrategyMeta) -> None:
        self._meta[meta.name] = meta
        # Initialize state in DB if not present
        with self._conn() as c:
            row = c.execute(
                "SELECT state FROM strategy_state WHERE name=?", (meta.name,)
            ).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO strategy_state (name, state, updated_at, note) VALUES (?,?,?,?)",
                    (meta.name, StrategyState.ACTIVE.value,
                     datetime.now(UTC).isoformat(), "registered"),
                )

    def list_names(self) -> list[str]:
        return list(self._meta.keys())

    def meta(self, name: str) -> StrategyMeta | None:
        return self._meta.get(name)

    # ── State -----------------------------------------------------------

    def get_state(self, name: str) -> StrategyState:
        with self._conn() as c:
            row = c.execute(
                "SELECT state FROM strategy_state WHERE name=?", (name,)
            ).fetchone()
        return StrategyState(row["state"]) if row else StrategyState.ACTIVE

    def set_state(self, name: str, new: StrategyState, reason: str = "") -> None:
        prev = self.get_state(name)
        if prev == new:
            return
        ts = datetime.now(UTC).isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO strategy_state (name, state, updated_at, note) "
                "VALUES (?,?,?,?)", (name, new.value, ts, reason),
            )
            c.execute(
                "INSERT INTO lifecycle_events (timestamp, name, from_state, to_state, reason) "
                "VALUES (?,?,?,?,?)", (ts, name, prev.value, new.value, reason),
            )
        logger.info(f"[lifecycle] {name}: {prev.value} → {new.value} ({reason})")

    def all_states(self) -> dict[str, StrategyState]:
        with self._conn() as c:
            rows = c.execute("SELECT name, state FROM strategy_state").fetchall()
        return {r["name"]: StrategyState(r["state"]) for r in rows
                if r["name"] in self._meta}

    # ── Allocation history ---------------------------------------------

    def record_allocation(self, name: str, *, target_pct: float, target_usd: float,
                           state: StrategyState, sharpe: float | None = None,
                           drawdown_pct: float | None = None, reason: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO allocations (timestamp, name, target_pct, target_usd, "
                "state, sharpe, drawdown_pct, reason) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), name, target_pct, target_usd,
                 state.value, sharpe, drawdown_pct, reason),
            )

    def latest_allocations(self) -> dict[str, dict]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT a.* FROM allocations a INNER JOIN (
                    SELECT name, MAX(id) AS max_id FROM allocations GROUP BY name
                ) m ON a.id = m.max_id
            """).fetchall()
        return {r["name"]: dict(r) for r in rows}

    def lifecycle_events(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM lifecycle_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
