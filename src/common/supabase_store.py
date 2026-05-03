"""Supabase REST-API client for the trading state.

Uses the Supabase PostgREST endpoint (https://<project>.supabase.co/rest/v1)
authenticated with the service_role key. Returns plain dicts. No
ORM, no connection pool — just HTTP POST/GET, which is sufficient
at our cadence (a few hundred rows/day).

Why HTTP and not psycopg/Postgres-direct:
  - Service-role key works directly with the REST API, no separate
    DB password setup
  - Trivially survives Supabase IP rotation, project upgrades, etc.
  - Latency-bound on insert (~80-150 ms per call) is negligible at
    10 trades/cycle × every 5 min

Design:
  - Used by storage.py as the "secondary" sink in dual-write mode.
  - Each method maps 1:1 to a SQLite table operation in
    PerformanceTracker / EquitySnapshotDB / StrategyRegistry.
  - Failures NEVER raise — log + return False. SQLite stays the
    source of truth until cutover.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class SupabaseStore:
    """Thin wrapper around Supabase's PostgREST. Every method returns
    True on successful insert/update, False on any failure — never
    raises so dual-write callers can keep going."""

    def __init__(
        self,
        url: str | None = None,
        service_key: str | None = None,
        timeout_seconds: float = 10.0,
    ):
        raw = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        # Be forgiving about the user's URL format. The Supabase
        # dashboard sometimes shows the URL with `/rest/v1` already
        # appended (in the API quickstart panel), and pasting that
        # value into the secret causes us to duplicate the path:
        #   https://xxx.supabase.co/rest/v1/rest/v1/trades  → 404
        # Strip any trailing /rest/v1 (or /rest) to normalize.
        for suffix in ("/rest/v1", "/rest"):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)].rstrip("/")
                break
        self.url = raw
        self.key = service_key or os.environ.get("SUPABASE_SERVICE_KEY", "")
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.url) and bool(self.key)

    def _headers(self, prefer: str = "return=minimal") -> dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }

    def _post(self, table: str, payload: dict | list[dict]) -> bool:
        if not self.is_configured():
            return False
        url = f"{self.url}/rest/v1/{table}"
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code >= 300:
                # Include the constructed URL in the warning so a
                # malformed SUPABASE_URL is obvious from CI logs.
                # The key never appears here; only the URL path.
                logger.warning(
                    f"supabase POST {url} HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
                return False
            return True
        except requests.RequestException as e:
            logger.warning(f"supabase POST {url} failed: {e}")
            return False

    def _patch(self, table: str, query: str, payload: dict) -> bool:
        if not self.is_configured():
            return False
        url = f"{self.url}/rest/v1/{table}?{query}"
        try:
            resp = requests.patch(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            if resp.status_code >= 300:
                logger.warning(
                    f"supabase PATCH {table} HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
                return False
            return True
        except requests.RequestException as e:
            logger.warning(f"supabase PATCH {table} failed: {e}")
            return False

    def _select(self, table: str, query: str = "") -> list[dict]:
        if not self.is_configured():
            return []
        url = f"{self.url}/rest/v1/{table}"
        if query:
            url += f"?{query}"
        try:
            resp = requests.get(
                url,
                headers=self._headers("return=representation"),
                timeout=self.timeout,
            )
            if resp.status_code >= 300:
                logger.warning(
                    f"supabase SELECT {table} HTTP {resp.status_code}: "
                    f"{resp.text[:300]}"
                )
                return []
            return resp.json() or []
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"supabase SELECT {table} failed: {e}")
            return []

    # ── Domain-specific helpers ───────────────────────────────────────

    def insert_trade(self, row: dict[str, Any]) -> bool:
        """Insert one trade. `row` keys map 1:1 to columns in the
        `trades` table — see deploy/supabase_schema.sql."""
        return self._post("trades", row)

    def update_trade_fill(
        self, *, order_id: str, price: float, quantity: float,
        amount_usd: float, pnl_usd: float | None,
        fill_status: str | None = None,
    ) -> bool:
        """Backfill a row when the broker reports a fill. Matches by
        order_id which is unique per trade."""
        payload: dict = {
            "price": price,
            "quantity": quantity,
            "amount_usd": amount_usd,
            "pnl_usd": pnl_usd,
        }
        if fill_status is not None:
            payload["fill_status"] = fill_status
        # PostgREST query: order_id=eq.<value>
        return self._patch("trades", f"order_id=eq.{order_id}", payload)

    def insert_equity_snapshot(self, *, equity_usd: float,
                                 timestamp: str, note: str = "") -> bool:
        return self._post("equity_snapshots", {
            "timestamp": timestamp,
            "equity_usd": equity_usd,
            "note": note,
        })

    def insert_kill_switch_event(
        self, *, timestamp: str, state: str,
        drawdown_pct: float, note: str = "",
    ) -> bool:
        return self._post("kill_switch_events", {
            "timestamp": timestamp,
            "state": state,
            "drawdown_pct": drawdown_pct,
            "note": note,
        })

    # ── Read helpers — used by EquitySnapshotDB so the kill-switch
    # baseline can survive a corrupted local SQLite. SQLite stays the
    # write-and-read primary; Supabase is consulted ONLY if SQLite
    # appears to be missing or has materially less history (audit-fix
    # for the #1 audit-flagged single-point-of-failure).

    def peak_equity_since(self, since_iso: str | None = None) -> float | None:
        """MAX(equity_usd) over all snapshots, or just those at/after
        `since_iso`. Returns None if Supabase is unconfigured or the
        table is empty/unreachable."""
        if not self.is_configured():
            return None
        # PostgREST sort + limit — pull the single largest row.
        query = "select=equity_usd&order=equity_usd.desc&limit=1"
        if since_iso:
            query += f"&timestamp=gte.{since_iso}"
        rows = self._select("equity_snapshots", query)
        if not rows:
            return None
        try:
            return float(rows[0].get("equity_usd"))
        except (TypeError, ValueError):
            return None

    def recent_equity_snapshots(self, n: int = 60) -> list[float]:
        """Last `n` equity_usd values, oldest first. Returns empty list
        on any error so callers can fall back to SQLite cleanly."""
        if not self.is_configured():
            return []
        query = f"select=equity_usd,timestamp&order=timestamp.desc&limit={n + 1}"
        rows = self._select("equity_snapshots", query)
        out: list[float] = []
        for r in reversed(rows):    # oldest first
            try:
                out.append(float(r.get("equity_usd")))
            except (TypeError, ValueError):
                continue
        return out

    def insert_allocation(self, row: dict[str, Any]) -> bool:
        """Row matches `allocations` table columns."""
        return self._post("allocations", row)

    def insert_lifecycle_event(self, row: dict[str, Any]) -> bool:
        return self._post("lifecycle_events", row)

    def upsert_strategy_state(self, row: dict[str, Any]) -> bool:
        """`strategy_state` is keyed by name — Postgres UPSERT via
        Prefer: resolution=merge-duplicates header."""
        if not self.is_configured():
            return False
        url = f"{self.url}/rest/v1/strategy_state"
        try:
            resp = requests.post(
                url,
                headers={
                    **self._headers(),
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                json=row,
                timeout=self.timeout,
            )
            if resp.status_code >= 300:
                logger.warning(
                    f"supabase upsert strategy_state HTTP "
                    f"{resp.status_code}: {resp.text[:300]}"
                )
                return False
            return True
        except requests.RequestException as e:
            logger.warning(f"supabase upsert strategy_state failed: {e}")
            return False

    def insert_signal(self, *, venue: str, name: str,
                       payload: dict, expires_at: str | None = None,
                       timestamp: str) -> bool:
        return self._post("signals", {
            "timestamp": timestamp,
            "venue": venue,
            "name": name,
            "payload": payload,
            "expires_at": expires_at,
        })

    def insert_strategic_review(self, row: dict[str, Any]) -> bool:
        return self._post("strategic_review", row)

    # ── Schema bootstrap (one-shot) ───────────────────────────────────

    def ensure_schema(self) -> bool:
        """Probe the trades table to confirm schema is in place.

        Returns True if the table exists and is queryable. False if
        not (run deploy/supabase_schema.sql first). Designed to be
        called once at orchestrator startup with a clear log line if
        the schema is missing — no auto-create, that's a manual step
        through the Supabase SQL editor.
        """
        if not self.is_configured():
            return False
        try:
            # Tiny query: count=exact returns a Content-Range header
            # with the row count. limit=1 keeps the body minimal.
            resp = requests.get(
                f"{self.url}/rest/v1/trades?select=id&limit=1",
                headers=self._headers(),
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                logger.info(
                    f"Supabase schema OK at {self.url}"
                )
                return True
            if resp.status_code == 404:
                logger.warning(
                    "Supabase reachable but `trades` table missing. "
                    "Run deploy/supabase_schema.sql in the SQL Editor."
                )
                return False
            logger.warning(
                f"Supabase schema probe HTTP {resp.status_code}: "
                f"{resp.text[:200]}"
            )
            return False
        except requests.RequestException as e:
            logger.warning(f"Supabase schema probe failed: {e}")
            return False
