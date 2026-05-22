"""OKX public-data fetcher — drop-in replacement for the Bybit client.

WHY: Bybit fronts its API with CloudFront and geo-blocks whole countries
(observed 2026-05-22 on the DigitalOcean droplet: every Bybit call →
HTTP 403 "blocked from your country"), which left crypto_funding_carry
and crypto_basis_trade permanently NO_DATA — and crypto_basis_trade
trades LIVE on Coinbase, so it was running unvalidated.

OKX publishes the same data (8h funding-rate history + daily klines) on
unauthenticated public endpoints that are reachable from US/EU
datacenters. Funding rates arbitrage to within ~1bp across venues, so
OKX is as faithful a proxy for a Coinbase-trading strategy as Bybit was.

Same interface as data.bybit.BybitClient (funding_history / daily_bars /
is_configured) so it's a drop-in. Pair it with the Bybit client behind
FallbackCryptoClient so a future geo-shift on either venue self-heals.

Endpoints (https://www.okx.com):
  GET /api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100
  GET /api/v5/market/history-candles?instId=BTC-USDT&bar=1D&limit=100
  GET /api/v5/market/history-candles?instId=BTC-USDT-SWAP&bar=1D&limit=100
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from .bybit import BybitCandle, FundingPoint

logger = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com"
DEFAULT_CACHE_TTL = 6 * 3600       # 6h — funding only changes every 8h
_MAX_PAGES = 20                    # bound pagination (≤ ~2000 rows)


# ─── Symbol mapping ───────────────────────────────────────────────────
# Coinbase "BTC-USD" → OKX spot "BTC-USDT" / perp "BTC-USDT-SWAP".
_COINBASE_TO_OKX_BASE = {
    "BTC-USD": "BTC-USDT", "ETH-USD": "ETH-USDT", "SOL-USD": "SOL-USDT",
    "ADA-USD": "ADA-USDT", "AVAX-USD": "AVAX-USDT", "MATIC-USD": "MATIC-USDT",
    "DOT-USD": "DOT-USDT", "LINK-USD": "LINK-USDT",
}


def coinbase_to_okx(symbol: str, kind: str = "spot") -> str:
    """Coinbase symbol → OKX instId. kind 'spot' → BTC-USDT;
    'linear'/'perp' → BTC-USDT-SWAP."""
    base = _COINBASE_TO_OKX_BASE.get(symbol)
    if base is None:
        # Best-effort: BTC-USD → BTC-USDT
        base = symbol.replace("-USD", "-USDT") if "-USD" in symbol else symbol
    if kind in ("linear", "perp", "swap"):
        return f"{base}-SWAP"
    return base


class OKXClient:
    """Disk-cached public-data OKX client. No auth required."""

    def __init__(self, cache_dir: str | None = None,
                 timeout_seconds: float = 20.0):
        import os as _os
        self.cache_dir = Path(
            cache_dir or _os.environ.get("OKX_CACHE_DIR", "data/cache/okx")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return True

    # ── Generic GET with cache ────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
             cache_ttl_seconds: int | None = None) -> dict:
        params = dict(params or {})
        key = self._cache_key(path, params)
        cached = self._read_cache(key, cache_ttl_seconds or DEFAULT_CACHE_TTL)
        if cached is not None:
            return cached
        url = f"{OKX_BASE}{path}"
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"okx GET {path} attempt {attempt} failed: {e}")
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                time.sleep(attempt * 5)
                continue
            if resp.status_code != 200:
                logger.warning(f"okx {path} HTTP {resp.status_code}: "
                               f"{resp.text[:160]}")
                return {}
            try:
                data = resp.json()
            except ValueError:
                return {}
            self._write_cache(key, data)
            return data
        return {}

    def _cache_key(self, path: str, params: dict) -> str:
        h = hashlib.sha256(
            f"{path}|{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return f"{path.strip('/').replace('/', '_')}_{h}"

    def _read_cache(self, key: str, ttl_seconds: int):
        p = self.cache_dir / f"{key}.json"
        if not p.exists() or time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data) -> None:
        try:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"okx cache write {key} failed: {e}")

    # ── Public API (mirrors BybitClient) ──────────────────────────────

    def funding_history(self, symbol: str, limit: int = 200,
                        from_ms: int | None = None) -> list[FundingPoint]:
        """8h funding-rate snapshots, oldest-first. OKX caps each page at
        100; paginate backwards with `after` until we've covered
        `limit` rows or `from_ms`."""
        inst = coinbase_to_okx(symbol, kind="linear")
        out: list[FundingPoint] = []
        after: str | None = None
        for _ in range(_MAX_PAGES):
            params = {"instId": inst, "limit": 100}
            if after is not None:
                params["after"] = after
            data = self._get("/api/v5/public/funding-rate-history", params)
            rows = data.get("data") or []
            if not rows:
                break
            for r in rows:
                try:
                    ts_ms = int(r["fundingTime"])
                    out.append(FundingPoint(
                        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                        symbol=inst,
                        funding_rate=float(r["fundingRate"]),
                    ))
                except (KeyError, ValueError, TypeError):
                    continue
            after = rows[-1].get("fundingTime")   # page older
            oldest_ms = int(rows[-1].get("fundingTime") or 0)
            if from_ms is not None and oldest_ms <= from_ms:
                break
            if len(out) >= limit and from_ms is None:
                break
        out.sort(key=lambda f: f.timestamp)
        return out

    def daily_bars(self, symbol: str, kind: str = "spot",
                   days: int = 365) -> list[BybitCandle]:
        """Daily OHLCV bars, oldest-first. kind 'spot' or 'linear'."""
        inst = coinbase_to_okx(symbol, kind=kind)
        seen: dict[int, BybitCandle] = {}
        after: str | None = None
        for _ in range(_MAX_PAGES):
            params = {"instId": inst, "bar": "1D", "limit": 100}
            if after is not None:
                params["after"] = after
            data = self._get("/api/v5/market/history-candles", params)
            rows = data.get("data") or []
            if not rows:
                break
            for r in rows:
                try:
                    # r = [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                    ts_ms = int(r[0])
                    seen[ts_ms] = BybitCandle(
                        timestamp=datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                        open=float(r[1]), high=float(r[2]), low=float(r[3]),
                        close=float(r[4]), volume=float(r[5]),
                    )
                except (IndexError, ValueError, TypeError):
                    continue
            after = rows[-1][0]            # page older
            if len(seen) >= days:
                break
        return sorted(seen.values(), key=lambda b: b.timestamp)


class FallbackCryptoClient:
    """Try each underlying client in order, returning the first non-empty
    result per call. Default chain is OKX → Bybit, so a geo-block on
    either venue self-heals as long as one is reachable from the host."""

    def __init__(self, clients: list | None = None):
        self._clients = clients or [OKXClient(), _lazy_bybit()]

    def is_configured(self) -> bool:
        return True

    def funding_history(self, *a, **k) -> list[FundingPoint]:
        for c in self._clients:
            try:
                rows = c.funding_history(*a, **k)
            except Exception as e:
                logger.debug(f"{type(c).__name__}.funding_history failed: {e}")
                continue
            if rows:
                return rows
        return []

    def daily_bars(self, *a, **k) -> list[BybitCandle]:
        for c in self._clients:
            try:
                rows = c.daily_bars(*a, **k)
            except Exception as e:
                logger.debug(f"{type(c).__name__}.daily_bars failed: {e}")
                continue
            if rows:
                return rows
        return []


def _lazy_bybit():
    from .bybit import BybitClient
    return BybitClient()


def crypto_data_client() -> FallbackCryptoClient:
    """Default historical crypto-data source for the funding/basis
    backtests: OKX first (reachable where Bybit is geo-blocked), Bybit
    as fallback."""
    return FallbackCryptoClient()
