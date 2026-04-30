"""Signal bus — SQLite pub/sub used by scouts and strategies.

Each row is a single signal payload produced by a scout. Strategies query
recent rows for their venue/asset_class. Old signals expire after a TTL
(default 24h) so the bus stays fast.

Design choice: SQLite (not a queue) because:
  • we already use SQLite for everything else
  • signals are read MORE often than written (each strategy cycle reads N
    signals; each scout writes ~once an hour) — DB scan beats consuming
    a queue once per strategy
  • survives across runs without a persistent broker
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalRow:
    id: int
    scout: str                 # which scout produced it
    venue: str                 # e.g. "coinbase", "alpaca", "kalshi", "macro"
    signal_type: str           # e.g. "funding_rates", "mispriced", "vix_regime"
    payload: Dict              # arbitrary JSON
    ttl_seconds: int
    created_at: datetime

    def is_fresh(self) -> bool:
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age <= self.ttl_seconds


class SignalBus:
    """SQLite-backed shared bus."""

    def __init__(self, db_path: Optional[str] = None,
                  default_ttl_seconds: int = 24 * 3600):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("SIGNAL_BUS_DB", "data/signal_bus.db")
        )
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.default_ttl = default_ttl_seconds
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    scout        TEXT NOT NULL,
                    venue        TEXT NOT NULL,
                    signal_type  TEXT NOT NULL,
                    payload      TEXT NOT NULL,
                    ttl_seconds  INTEGER NOT NULL,
                    created_at   TEXT NOT NULL
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_signals_lookup "
                       "ON signals(venue, signal_type, created_at DESC)")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Publish ─────────────────────────────────────────────────────────

    def publish(self, *, scout: str, venue: str, signal_type: str,
                 payload: Dict, ttl_seconds: Optional[int] = None) -> int:
        """Write one signal row. Returns the inserted id."""
        ts = datetime.now(timezone.utc).isoformat()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO signals (scout, venue, signal_type, payload, "
                "ttl_seconds, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (scout, venue, signal_type,
                 json.dumps(payload, default=str), ttl, ts),
            )
            return cur.lastrowid

    # ── Query ───────────────────────────────────────────────────────────

    def latest(self, *, venue: Optional[str] = None,
                signal_type: Optional[str] = None,
                limit: int = 50) -> List[SignalRow]:
        """Return most-recent signals (fresh-first, includes stale)."""
        sql = "SELECT * FROM signals"
        clauses = []
        params: List[Any] = []
        if venue:
            clauses.append("venue=?")
            params.append(venue)
        if signal_type:
            clauses.append("signal_type=?")
            params.append(signal_type)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._conn() as c:
            rows = c.execute(sql, params).fetchall()
        return [self._row_to_signal(r) for r in rows]

    def get_fresh_for_strategy(self, venue: str) -> Dict[str, Any]:
        """Return a payload-shaped dict ready for StrategyContext.scout_signals.

        Groups latest fresh signal of each signal_type for the given venue,
        plus the macro signals (which apply to all venues).
        """
        out: Dict[str, Any] = {}
        # Per-venue signals
        for row in self.latest(venue=venue, limit=200):
            if not row.is_fresh():
                continue
            out.setdefault(row.signal_type, row.payload)
        # Always include macro
        if venue != "macro":
            for row in self.latest(venue="macro", limit=50):
                if not row.is_fresh():
                    continue
                key = f"macro_{row.signal_type}"
                out.setdefault(key, row.payload)
        return out

    # ── Maintenance ─────────────────────────────────────────────────────

    def vacuum_expired(self) -> int:
        """Delete rows past their TTL. Returns count deleted."""
        cutoff = time.time()
        with self._conn() as c:
            cur = c.execute("""
                DELETE FROM signals
                WHERE strftime('%s', created_at) IS NOT NULL
                  AND CAST(strftime('%s', created_at) AS INTEGER) + ttl_seconds < ?
            """, (int(cutoff),))
            return cur.rowcount

    # ── Helpers ─────────────────────────────────────────────────────────

    def _row_to_signal(self, r: sqlite3.Row) -> SignalRow:
        return SignalRow(
            id=r["id"], scout=r["scout"], venue=r["venue"],
            signal_type=r["signal_type"], payload=json.loads(r["payload"]),
            ttl_seconds=int(r["ttl_seconds"]),
            created_at=datetime.fromisoformat(r["created_at"]),
        )
