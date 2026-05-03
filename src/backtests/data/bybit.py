"""Bybit public-data fetcher for crypto perp + futures backtests.

Why Bybit (and not Coinbase): the Coinbase International funding-history
endpoint requires authenticated access. Bybit publishes everything
publicly with no auth, no rate-limit issues at our cadence, and ~5
years of history.

Funding rates are highly correlated across venues (Coinbase, Bybit,
Binance, OKX) because perpetuals arbitrage between them. Using
Bybit's history as a proxy for what a Coinbase-trading strategy
would have earned is acceptable for backtest validation — directional
signal is preserved.

Endpoints (all `https://api.bybit.com`):
  GET /v5/market/funding/history?category=linear&symbol=BTCUSDT
      → 8h funding-rate snapshots back to 2020
  GET /v5/market/kline?category=linear&symbol=BTCUSDT&interval=D
      → daily candles (perp price)
  GET /v5/market/kline?category=spot&symbol=BTCUSDT&interval=D
      → daily candles (spot price)
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

BYBIT_BASE = "https://api.bybit.com"
DEFAULT_CACHE_TTL = 6 * 3600   # 6h — funding rates only change every 8h


@dataclass
class FundingPoint:
    """One 8-hour funding-rate snapshot."""
    timestamp: datetime          # exact 8h boundary the rate paid
    symbol: str                  # e.g. "BTCUSDT"
    funding_rate: float          # decimal (e.g. 0.0001 = 1bp / 8h)


@dataclass
class BybitCandle:
    """Daily OHLCV bar for a Bybit perp or spot instrument."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# ─── Symbol mapping ───────────────────────────────────────────────────


# Coinbase uses BTC-USD; Bybit uses BTCUSDT (USD-margined perp).
# We normalize on the Coinbase side ("BTC-USD") and map here.
COINBASE_TO_BYBIT_SPOT = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "MATIC-USD": "MATICUSDT",
    "DOT-USD": "DOTUSDT",
    "LINK-USD": "LINKUSDT",
}


def coinbase_to_bybit(symbol: str, kind: str = "spot") -> str:
    """Normalize Coinbase symbol (e.g. BTC-USD) to Bybit (BTCUSDT).
    `kind` is "spot" or "linear" (perp); both use the same suffix
    on Bybit so no special handling needed."""
    return COINBASE_TO_BYBIT_SPOT.get(symbol, symbol.replace("-", "") + "T"
                                       if "USD" in symbol else symbol)


# ─── Client ───────────────────────────────────────────────────────────


class BybitClient:
    """Disk-cached public-data Bybit client. No auth needed; just
    HTTP GET to the documented public endpoints."""

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        import os as _os
        self.cache_dir = Path(
            cache_dir or _os.environ.get("BYBIT_CACHE_DIR", "data/cache/bybit")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """Always True — Bybit's public endpoints require no key.
        Method exists so backtests can polymorphically check whether
        their data source is ready (same shape as PolygonClient.is_configured)."""
        return True

    # ── Generic GET with cache ────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
              cache_ttl_seconds: int | None = None) -> dict:
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds or DEFAULT_CACHE_TTL)
        if cached is not None:
            return cached

        url = f"{BYBIT_BASE}{path}"
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"bybit GET {path} attempt {attempt} failed: {e}")
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                wait = attempt * 5
                logger.info(f"bybit 429 rate-limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning(f"bybit {path} HTTP {resp.status_code}: "
                               f"{resp.text[:200]}")
                return {}
            try:
                data = resp.json()
            except ValueError:
                return {}
            self._write_cache(cache_key, data)
            return data
        return {}

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
            logger.debug(f"bybit cache write {key} failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def funding_history(
        self,
        symbol: str,
        limit: int = 200,
        from_ms: int | None = None,
    ) -> list[FundingPoint]:
        """Recent 8h funding-rate snapshots for a Bybit perp.

        `symbol` accepts either Bybit-native ("BTCUSDT") or
        Coinbase-style ("BTC-USD") and normalizes. Bybit caps `limit`
        at 200; for longer histories, paginate with `from_ms` set to
        the oldest result's timestamp - 1ms.

        Returns oldest-first list (we reverse Bybit's newest-first order).
        """
        bybit_sym = coinbase_to_bybit(symbol, kind="linear")
        params: dict = {
            "category": "linear",
            "symbol": bybit_sym,
            "limit": min(limit, 200),
        }
        if from_ms is not None:
            params["startTime"] = from_ms

        data = self._get("/v5/market/funding/history", params)
        rows = (data.get("result") or {}).get("list") or []
        out: list[FundingPoint] = []
        for r in rows:
            try:
                ts = datetime.fromtimestamp(int(r["fundingRateTimestamp"]) / 1000,
                                              tz=UTC)
                rate = float(r["fundingRate"])
            except (KeyError, ValueError, TypeError):
                continue
            out.append(FundingPoint(
                timestamp=ts, symbol=bybit_sym, funding_rate=rate,
            ))
        out.sort(key=lambda f: f.timestamp)   # oldest-first
        return out

    def daily_bars(
        self,
        symbol: str,
        kind: str = "spot",
        days: int = 365,
    ) -> list[BybitCandle]:
        """Daily OHLCV bars. `kind` = "spot" or "linear" (perp).

        Returns oldest-first.
        """
        bybit_sym = coinbase_to_bybit(symbol, kind=kind)
        params = {
            "category": kind,
            "symbol": bybit_sym,
            "interval": "D",
            "limit": min(days, 1000),
        }
        data = self._get("/v5/market/kline", params)
        rows = (data.get("result") or {}).get("list") or []
        out: list[BybitCandle] = []
        for r in rows:
            try:
                # r = [openTime, open, high, low, close, volume, turnover]
                ts = datetime.fromtimestamp(int(r[0]) / 1000, tz=UTC)
                o = float(r[1])
                h = float(r[2])
                lo = float(r[3])
                c = float(r[4])
                v = float(r[5])
            except (IndexError, ValueError, TypeError):
                continue
            out.append(BybitCandle(
                timestamp=ts, open=o, high=h, low=lo, close=c, volume=v,
            ))
        out.sort(key=lambda b: b.timestamp)
        return out
