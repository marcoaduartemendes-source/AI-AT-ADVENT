"""Binance public futures + spot data — multi-venue funding consensus
and broader alt coverage.

Why this matters:
    - crypto_funding_carry currently uses Bybit alone as the funding
      reference. Binance is the largest crypto perp venue by volume;
      adding it as a second source gives:
        * Cross-venue consensus (only fire when Bybit AND Binance agree
          funding is elevated → fewer false positives)
        * Broader alt coverage (Binance lists ~300 perps vs Bybit ~150)
    - crypto_xsmom benefits from a second daily-bar feed for tickers
      Coinbase doesn't list (ARB, OP, INJ, RNDR, etc).

Free-tier sufficiency:
    Binance public REST API: 2400 weight/min on /fapi (futures) and
    6000/min on /api (spot). At 30-min scout cadence we use < 50/min.
    No auth needed for read-only endpoints.

    Paid tier (Binance VIP, requires KYC + $1M+ account size) doesn't
    add anything for read-only data — only execution-side benefits.
    Coinglass ($30-200/mo) aggregates Binance + Bybit + OKX + Deribit
    into one endpoint; with two free venues already wired, the marginal
    value of Coinglass is mostly the third venue (OKX) which we can
    always add separately for free.

Endpoints:
    GET https://fapi.binance.com/fapi/v1/fundingRate  (perp funding)
    GET https://fapi.binance.com/fapi/v1/klines       (perp candles)
    GET https://api.binance.com/api/v3/klines         (spot candles)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_API = "https://api.binance.com"
DEFAULT_CACHE_TTL = 6 * 3600   # 6h — funding only changes every 8h


@dataclass
class BinanceFundingPoint:
    """One 8-hour Binance perp funding-rate snapshot."""
    timestamp: datetime
    symbol: str
    funding_rate: float


@dataclass
class BinanceCandle:
    """OHLCV bar from spot or perp."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# Coinbase ↔ Binance ticker map. Binance uses BTCUSDT (USDT-margined).
COINBASE_TO_BINANCE = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "MATIC-USD": "MATICUSDT",
    "DOT-USD": "DOTUSDT",
    "LINK-USD": "LINKUSDT",
    "DOGE-USD": "DOGEUSDT",
    "LTC-USD": "LTCUSDT",
    "XRP-USD": "XRPUSDT",
    "BCH-USD": "BCHUSDT",
    # alts not on Coinbase but widely traded
    "ARB-USD": "ARBUSDT",
    "OP-USD": "OPUSDT",
    "INJ-USD": "INJUSDT",
}


def coinbase_to_binance(symbol: str) -> str:
    """Normalize a Coinbase-style symbol (BTC-USD) to Binance (BTCUSDT)."""
    if symbol in COINBASE_TO_BINANCE:
        return COINBASE_TO_BINANCE[symbol]
    if "-" in symbol and symbol.endswith("-USD"):
        return symbol.replace("-USD", "USDT")
    return symbol


class BinanceClient:
    """Disk-cached Binance public data fetcher. Read-only, no auth."""

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.cache_dir = Path(
            cache_dir or os.environ.get(
                "BINANCE_CACHE_DIR", "data/cache/binance"
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """Always True — Binance public read endpoints need no key."""
        return True

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_key(self, base: str, path: str, params: dict) -> str:
        scrubbed = dict(sorted(params.items()))
        h = hashlib.sha256(
            f"{base}{path}|{json.dumps(scrubbed, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return f"{path.strip('/').replace('/', '_')}_{h}"

    def _read_cache(self, key: str, ttl_seconds: int):
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data) -> None:
        p = self.cache_dir / f"{key}.json"
        try:
            p.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"binance cache write {key} failed: {e}")

    def _get(self, base: str, path: str, params: dict | None = None,
              cache_ttl_seconds: int = DEFAULT_CACHE_TTL):
        params = dict(params or {})
        cache_key = self._cache_key(base, path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds)
        if cached is not None:
            return cached
        url = f"{base}{path}"
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(
                    f"binance GET {path} attempt {attempt} failed: {e}"
                )
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                wait = attempt * 5
                logger.info(f"binance 429, sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning(
                    f"binance {path} HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return None
            try:
                data = resp.json()
            except ValueError:
                return None
            self._write_cache(cache_key, data)
            return data
        return None

    # ── Public API ────────────────────────────────────────────────────

    def funding_history(
        self,
        symbol: str,
        limit: int = 200,
        from_ms: int | None = None,
    ) -> list[BinanceFundingPoint]:
        """8h perp-funding history for a Binance USDT-margined perp."""
        bsym = coinbase_to_binance(symbol)
        params: dict = {"symbol": bsym, "limit": min(limit, 1000)}
        if from_ms is not None:
            params["startTime"] = from_ms
        data = self._get(BINANCE_FAPI, "/fapi/v1/fundingRate", params)
        if not data:
            return []
        out: list[BinanceFundingPoint] = []
        for r in data:
            try:
                ts = datetime.fromtimestamp(
                    int(r["fundingTime"]) / 1000, tz=UTC
                )
                rate = float(r["fundingRate"])
            except (KeyError, ValueError, TypeError):
                continue
            out.append(BinanceFundingPoint(
                timestamp=ts, symbol=bsym, funding_rate=rate,
            ))
        out.sort(key=lambda x: x.timestamp)
        return out

    def daily_bars(
        self,
        symbol: str,
        kind: str = "spot",
        days: int = 365,
    ) -> list[BinanceCandle]:
        """Daily OHLCV. `kind` = "spot" or "perp"."""
        bsym = coinbase_to_binance(symbol)
        if kind == "spot":
            base, path = BINANCE_API, "/api/v3/klines"
        else:
            base, path = BINANCE_FAPI, "/fapi/v1/klines"
        params = {
            "symbol": bsym, "interval": "1d", "limit": min(days, 1500),
        }
        data = self._get(base, path, params)
        if not data:
            return []
        out: list[BinanceCandle] = []
        for r in data:
            try:
                ts = datetime.fromtimestamp(int(r[0]) / 1000,
                                              tz=UTC)
                o = float(r[1])
                h = float(r[2])
                lo = float(r[3])
                c = float(r[4])
                v = float(r[5])
            except (IndexError, ValueError, TypeError):
                continue
            out.append(BinanceCandle(
                timestamp=ts, open=o, high=h, low=lo, close=c, volume=v,
            ))
        out.sort(key=lambda b: b.timestamp)
        return out

    def consensus_funding_apr(
        self,
        symbol: str,
        bybit_apr: float,
        annualization_factor: int = 365 * 3,
    ) -> tuple[float, bool]:
        """Cross-venue funding consensus: pull Binance funding, return
        (consensus_apr, agree). `agree` is True iff Bybit and Binance
        APRs are within 1% of each other AND both above 1% APR (i.e.
        both venues say "funding is elevated"). Used to gate
        crypto_funding_carry on multi-venue confirmation."""
        bn = self.funding_history(symbol, limit=1)
        if not bn:
            return (bybit_apr, False)
        binance_apr = bn[-1].funding_rate * annualization_factor * 100
        consensus = (bybit_apr + binance_apr) / 2
        agree = (
            abs(bybit_apr - binance_apr) <= 1.0
            and bybit_apr > 1.0
            and binance_apr > 1.0
        )
        return (consensus, agree)
