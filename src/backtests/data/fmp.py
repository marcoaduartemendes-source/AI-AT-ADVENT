"""Financial Modeling Prep (FMP) API wrapper for backtest data.

Drop-in alternative to data/polygon.py — returns the same DailyBar
and EarningsRecord shapes so the PEAD/commodity backtests work
unchanged regardless of which vendor is configured.

Endpoints we use (all on the `/stable` path):
  - /historical-price-eod/full?symbol=AAPL                 daily OHLCV
  - /historical/earning_calendar?symbol=AAPL               EPS actual + estimate

Auth: pass `FMP_API_KEY` env var or `api_key=` to the constructor.
Rate limits depend on plan (Starter = 300 req/min). On 429 we back
off linearly.

Caching: every successful response is disk-cached under
`data/cache/fmp/{endpoint}/{params_hash}.json` for 24h. Backtests
re-run hundreds of times; the cache is what makes that feasible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import requests

# Reuse the typed return shapes from polygon.py so callers don't
# need to know which vendor produced the data.
from .polygon import DailyBar, EarningsRecord

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"
DEFAULT_CACHE_TTL = 24 * 3600   # 24h


class FMPError(RuntimeError):
    """Raised on FMP API errors. Includes HTTP status if available."""


# ─── Client ───────────────────────────────────────────────────────────


class FMPClient:
    """Disk-cached FMP API client. Same surface as PolygonClient.

    Usage:
        client = FMPClient()  # reads FMP_API_KEY from env
        bars = client.daily_bars("AAPL", "2024-01-01", "2024-12-31")
        earn = client.recent_earnings("AAPL", limit=8)
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            logger.info("FMPClient: FMP_API_KEY not set — "
                        "backtests requiring FMP will skip")
        self.cache_dir = Path(
            cache_dir or os.environ.get("FMP_CACHE_DIR", "data/cache/fmp")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ── Generic GET with cache + retry ────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
              cache_ttl_seconds: int | None = None) -> list | dict:
        if not self.api_key:
            raise FMPError(
                "FMP_API_KEY env var not set. Backtests requiring "
                "FMP data cannot run without it."
            )
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds or DEFAULT_CACHE_TTL)
        if cached is not None:
            return cached

        params["apikey"] = self.api_key
        url = f"{FMP_BASE}{path}"
        for attempt in range(1, 5):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"FMP GET {path} attempt {attempt} failed: {e}")
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                wait = attempt * 6
                logger.info(f"FMP 429 (rate limit) on {path}, "
                            f"sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise FMPError(
                    f"FMP {path} returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
            data = resp.json()
            self._write_cache(cache_key, data)
            return data
        raise FMPError(f"FMP {path} retries exhausted")

    def _cache_key(self, path: str, params: dict) -> str:
        # Strip apikey before hashing so key rotation doesn't bust cache
        scrubbed = {k: v for k, v in sorted(params.items()) if k != "apikey"}
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
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"FMP cache miss for {key}: {e}")
            return None

    def _write_cache(self, key: str, data) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"FMP cache write {key} failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def daily_bars(
        self,
        symbol: str,
        from_date,
        to_date,
        adjusted: bool = True,
    ) -> list[DailyBar]:
        """Daily OHLCV bars for one ticker. `from_date`/`to_date` accept
        ISO strings (YYYY-MM-DD) or date objects. Always adjusted on FMP."""
        f = from_date.isoformat() if hasattr(from_date, "isoformat") else from_date
        t = to_date.isoformat() if hasattr(to_date, "isoformat") else to_date
        data = self._get(
            "/historical-price-eod/full",
            {"symbol": symbol, "from": f, "to": t},
        )
        rows = data if isinstance(data, list) else data.get("historical", [])
        out: list[DailyBar] = []
        for r in rows:
            try:
                d = datetime.strptime(r["date"][:10], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            out.append(DailyBar(
                date=d,
                open=float(r.get("open", 0)),
                high=float(r.get("high", 0)),
                low=float(r.get("low", 0)),
                close=float(r.get("close", 0)),
                volume=float(r.get("volume", 0)),
                vwap=float(r["vwap"]) if r.get("vwap") is not None else None,
            ))
        # FMP returns newest-first; we standardize on oldest-first
        out.sort(key=lambda b: b.date)
        return out

    def recent_earnings(self, ticker: str, limit: int = 8) -> list[EarningsRecord]:
        """Last N quarterly earnings reports with EPS actual + estimate.

        FMP's /historical/earning_calendar?symbol=… returns the entire
        history; we slice to `limit` (newest-first by date in their
        response).

        IMPORTANT: FMP reports a `date` field which is the announcement
        date — equivalent to Polygon's filing_date. We map it into
        `filing_date` so the backtest's no-look-ahead guard works the
        same way.
        """
        data = self._get(
            "/historical/earning_calendar",
            {"symbol": ticker},
        )
        if not isinstance(data, list):
            return []

        out: list[EarningsRecord] = []
        for r in data[:limit]:
            try:
                announce = datetime.strptime(r["date"][:10], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            eps_actual = _f(r.get("epsActual") or r.get("eps"))
            eps_estimate = _f(r.get("epsEstimated") or r.get("epsEstimate"))
            rev_actual = _f(r.get("revenueActual") or r.get("revenue"))
            rev_estimate = _f(r.get("revenueEstimated") or r.get("revenueEstimate"))
            out.append(EarningsRecord(
                ticker=ticker,
                period_end=announce,       # FMP doesn't separate period_end clearly
                filing_date=announce,      # the announcement date IS public
                eps_actual=eps_actual,
                eps_estimate=eps_estimate,
                revenue_actual=rev_actual,
                revenue_estimate=rev_estimate,
                raw=r,
            ))
        return out


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── Vendor auto-selection ────────────────────────────────────────────


def get_data_client():
    """Return whichever data vendor is configured.

    Preference order:
      1. FMP    (FMP_API_KEY)        — cheapest tier with EPS estimates
      2. Polygon (POLYGON_API_KEY)   — fallback if migrating
      3. Raise — neither configured

    Both clients implement the same surface (is_configured, daily_bars,
    recent_earnings) so the PEAD backtest doesn't care which one
    answered.
    """
    if os.environ.get("FMP_API_KEY"):
        return FMPClient()
    if os.environ.get("POLYGON_API_KEY"):
        from .polygon import PolygonClient
        return PolygonClient()
    # Return an unconfigured FMP client so the backtest can return
    # its placeholder summary instead of crashing.
    return FMPClient()
