"""Kalshi historical-market fetcher for backtests.

Reuses the existing src/brokers/kalshi.py auth wrapper (RSA-PSS-SHA256
signed requests) but with read-only methods that don't place orders.
Strategy backtests need:

    1. List of resolved markets in the lookback window
       (settlement_value, ticker, close_ts)
    2. Each market's price-history snapshots leading up to resolution
       (so we can backtest "buy at p=0.10, settles to 1.00" trades)
    3. The metadata describing each market type (binary, range, etc)

Endpoints (auth via existing KalshiAdapter):
  GET /trade-api/v2/markets?status=settled         list resolved markets
  GET /trade-api/v2/markets/{ticker}                market + settlement
  GET /trade-api/v2/markets/{ticker}/orderbook      historical book snap

Bounded by Kalshi's API rate limits — we cache 24h on resolved markets
(they don't change after settlement).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ResolvedMarket:
    """A Kalshi market that has settled."""
    ticker: str                    # e.g. "INX-23DEC29-T4400"
    title: str                     # human-readable
    open_ts: datetime
    close_ts: datetime
    settlement_value: float        # 0.0 (NO won) or 1.0 (YES won)
    yes_close_price: float         # last YES price before settlement (0-1)
    raw: dict


@dataclass
class MarketSnapshot:
    """One observation of a market's prices at a moment in time."""
    ticker: str
    timestamp: datetime
    yes_bid: float                 # 0-1
    yes_ask: float
    yes_mid: float
    volume_24h: int


# ─── Client ───────────────────────────────────────────────────────────


class KalshiHistoryClient:
    """Read-only Kalshi history client, reusing the live adapter's
    auth so we don't have to reimplement RSA signing here."""

    def __init__(
        self,
        adapter=None,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        import os as _os
        self.adapter = adapter
        # Lazy-init the adapter from env if not passed
        if self.adapter is None:
            try:
                from brokers.kalshi import KalshiAdapter
                self.adapter = KalshiAdapter()
            except Exception as e:
                logger.info(f"KalshiHistoryClient: adapter init failed: {e}")
        self.cache_dir = Path(
            cache_dir or _os.environ.get("KALSHI_CACHE_DIR", "data/cache/kalshi")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return self.adapter is not None and getattr(self.adapter, "_configured", False)

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_key(self, path: str, params: dict) -> str:
        scrubbed = {k: v for k, v in sorted(params.items())}
        h = hashlib.sha256(
            f"{path}|{json.dumps(scrubbed, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return f"{path.strip('/').replace('/', '_')}_{h}"

    def _read_cache(self, key: str, ttl_seconds: int):
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"kalshi cache write {key} failed: {e}")

    def _request(self, path: str, params: dict | None = None,
                  cache_ttl_seconds: int = 24 * 3600) -> dict:
        """Call adapter._request (which handles RSA auth). Disk-cached."""
        if not self.is_configured():
            return {}
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds)
        if cached is not None:
            return cached
        try:
            data = self.adapter._request("GET", path, params=params)
            self._write_cache(cache_key, data)
            return data or {}
        except Exception as e:
            logger.warning(f"kalshi history {path} failed: {e}")
            return {}

    # ── Public API ────────────────────────────────────────────────────

    def settled_markets(
        self,
        from_date: date,
        to_date: date | None = None,
        limit: int = 200,
    ) -> list[ResolvedMarket]:
        """List Kalshi markets that have settled in the window.
        Used by both kalshi_arb and macro_kalshi backtests as the
        candidate set."""
        to_date = to_date or date.today()
        params = {
            "status": "settled",
            "limit": min(limit, 1000),
            # Kalshi API uses unix-second filter
            "min_close_ts": int(datetime(from_date.year, from_date.month,
                                          from_date.day, tzinfo=UTC).timestamp()),
        }
        data = self._request("/markets", params)
        rows = data.get("markets") or []
        out: list[ResolvedMarket] = []
        for m in rows:
            try:
                ticker = m["ticker"]
                title = m.get("title") or ticker
                open_ts = datetime.fromtimestamp(m.get("open_ts") or 0, tz=UTC)
                close_ts = datetime.fromtimestamp(m.get("close_ts") or 0, tz=UTC)
                settlement = float(m.get("settlement_value", 0))
                # Settlement value is 0 or 100 cents on Kalshi; normalize to 0-1
                if settlement > 1:
                    settlement = settlement / 100
                yes_close = float(m.get("yes_close") or 0)
                if yes_close > 1:
                    yes_close = yes_close / 100
            except (KeyError, ValueError, TypeError):
                continue
            out.append(ResolvedMarket(
                ticker=ticker,
                title=title,
                open_ts=open_ts,
                close_ts=close_ts,
                settlement_value=settlement,
                yes_close_price=yes_close,
                raw=m,
            ))
        return out

    def market_detail(self, ticker: str) -> ResolvedMarket | None:
        """Single-market detail for a known ticker."""
        data = self._request(f"/markets/{ticker}")
        m = data.get("market") if isinstance(data, dict) else None
        if not m:
            return None
        try:
            settlement = float(m.get("settlement_value", 0))
            if settlement > 1:
                settlement = settlement / 100
            yes_close = float(m.get("yes_close") or 0)
            if yes_close > 1:
                yes_close = yes_close / 100
            return ResolvedMarket(
                ticker=ticker,
                title=m.get("title") or ticker,
                open_ts=datetime.fromtimestamp(m.get("open_ts") or 0, tz=UTC),
                close_ts=datetime.fromtimestamp(m.get("close_ts") or 0, tz=UTC),
                settlement_value=settlement,
                yes_close_price=yes_close,
                raw=m,
            )
        except (KeyError, ValueError, TypeError):
            return None
