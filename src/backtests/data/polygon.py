"""Polygon.io API wrapper for backtest data.

Endpoints we use:
  - /v2/aggs/ticker/{symbol}/range/1/day/{from}/{to}      daily bars (equity + crypto)
  - /vX/reference/financials                              quarterly financials (PEAD)
  - /v3/reference/dividends                               dividend history
  - /v3/reference/splits                                  split history (for adjustments)
  - /v1/open-close/{symbol}/{date}                        single-day OHLC

Auth: pass `POLYGON_API_KEY` as an env var. Rate limits depend on plan
(Starter = 5 req/min; we cache and respect that). On 429 we back off
linearly by attempt × 12s.

Caching: every successful response is cached on disk under
`data/cache/polygon/{symbol}/{endpoint}/{params_hash}.json` for 24h
unless `cache_ttl_seconds=` is explicitly passed lower. Backtests
re-run hundreds of times; the disk cache is what makes that feasible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"
DEFAULT_CACHE_TTL = 24 * 3600   # 24h


class PolygonError(RuntimeError):
    """Raised on Polygon API errors. Includes HTTP status if available."""


# ─── Typed return shapes ──────────────────────────────────────────────


@dataclass
class DailyBar:
    date: date          # session date (UTC for crypto, ET-local for equity)
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


@dataclass
class EarningsRecord:
    """One reported quarter from /vX/reference/financials.

    Crucially includes `filing_date` so the backtest can position
    AFTER the filing was public — no look-ahead bias."""
    ticker: str
    period_end: date              # fiscal period end
    filing_date: date | None      # date filed with SEC (>= report date)
    eps_actual: float | None
    eps_estimate: float | None
    revenue_actual: float | None
    revenue_estimate: float | None
    raw: dict = None              # type: ignore[assignment]

    @property
    def eps_surprise_pct(self) -> float | None:
        """Reported EPS vs estimate, as % of |estimate|. None if either
        is missing."""
        if (self.eps_actual is None or self.eps_estimate is None
                or self.eps_estimate == 0):
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate) * 100


# ─── Client ───────────────────────────────────────────────────────────


class PolygonClient:
    """Disk-cached Polygon API client.

    Usage:
        client = PolygonClient()  # reads POLYGON_API_KEY from env
        bars = client.daily_bars("AAPL", "2024-01-01", "2024-12-31")
        earn = client.recent_earnings("AAPL", limit=8)
    """

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            # Don't raise here — instantiation must succeed in CI even
            # without the secret. Methods raise PolygonError when called
            # without a key, so callers can catch and fall back.
            logger.info("PolygonClient: POLYGON_API_KEY not set — "
                        "backtests requiring Polygon will skip")
        self.cache_dir = Path(
            cache_dir or os.environ.get("POLYGON_CACHE_DIR", "data/cache/polygon")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    # ── Generic GET with cache + retry ────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
              cache_ttl_seconds: int | None = None) -> dict:
        if not self.api_key:
            raise PolygonError(
                "POLYGON_API_KEY env var not set. Backtests requiring "
                "Polygon data cannot run without it."
            )
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds or DEFAULT_CACHE_TTL)
        if cached is not None:
            return cached

        params["apiKey"] = self.api_key
        url = f"{POLYGON_BASE}{path}"
        for attempt in range(1, 5):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"polygon GET {path} attempt {attempt} failed: {e}")
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                wait = attempt * 12
                logger.info(f"polygon 429 (rate limit) on {path}, "
                            f"sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise PolygonError(
                    f"polygon {path} returned HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
            data = resp.json()
            self._write_cache(cache_key, data)
            return data
        raise PolygonError(f"polygon {path} retries exhausted")

    def _cache_key(self, path: str, params: dict) -> str:
        # Strip apiKey before hashing so key rotation doesn't bust cache
        scrubbed = {k: v for k, v in sorted(params.items()) if k != "apiKey"}
        h = hashlib.sha256(
            f"{path}|{json.dumps(scrubbed, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return f"{path.strip('/').replace('/', '_')}_{h}"

    def _read_cache(self, key: str, ttl_seconds: int) -> dict | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"polygon cache miss for {key}: {e}")
            return None

    def _write_cache(self, key: str, data: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"polygon cache write {key} failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def daily_bars(
        self,
        symbol: str,
        from_date: str | date,
        to_date: str | date,
        adjusted: bool = True,
    ) -> list[DailyBar]:
        """Daily OHLCV bars for one ticker. `from_date`/`to_date` are
        inclusive ISO strings (YYYY-MM-DD) or date objects.

        For equities: Polygon returns adjusted bars by default
        (split + dividend). For crypto: adjustment doesn't apply.
        """
        f = from_date.isoformat() if isinstance(from_date, date) else from_date
        t = to_date.isoformat() if isinstance(to_date, date) else to_date
        path = f"/v2/aggs/ticker/{symbol}/range/1/day/{f}/{t}"
        data = self._get(path, {"adjusted": str(adjusted).lower(),
                                  "sort": "asc", "limit": 50000})
        out: list[DailyBar] = []
        for r in data.get("results") or []:
            ts_ms = r.get("t")
            if ts_ms is None:
                continue
            d = datetime.fromtimestamp(ts_ms / 1000).date()
            out.append(DailyBar(
                date=d,
                open=float(r["o"]),
                high=float(r["h"]),
                low=float(r["l"]),
                close=float(r["c"]),
                volume=float(r.get("v", 0)),
                vwap=float(r["vw"]) if "vw" in r else None,
            ))
        return out

    def recent_earnings(self, ticker: str, limit: int = 8) -> list[EarningsRecord]:
        """Last N quarterly earnings reports for `ticker`.

        Polygon's /vX/reference/financials returns a flat blob; we extract
        EPS actual + estimate from the income_statement section. Filing
        date matters: a backtest must position only AFTER filing_date so
        we don't introduce look-ahead bias.
        """
        path = "/vX/reference/financials"
        data = self._get(path, {
            "ticker": ticker,
            "timeframe": "quarterly",
            "order": "desc",
            "limit": limit,
            "sort": "period_of_report_date",
        })
        out: list[EarningsRecord] = []
        for r in data.get("results") or []:
            try:
                period_end = datetime.strptime(
                    r.get("period_of_report_date") or r.get("end_date", ""),
                    "%Y-%m-%d",
                ).date()
            except ValueError:
                continue
            filing_str = r.get("filing_date") or r.get("acceptance_datetime")
            filing_date = None
            if filing_str:
                try:
                    filing_date = datetime.strptime(filing_str[:10], "%Y-%m-%d").date()
                except ValueError:
                    pass
            fin = r.get("financials", {})
            inc = fin.get("income_statement", {})
            eps_actual = _xpath_value(inc, "basic_earnings_per_share")
            eps_estimate = _xpath_value(inc, "basic_earnings_per_share_estimate")
            rev_actual = _xpath_value(inc, "revenues")
            rev_estimate = _xpath_value(inc, "revenues_estimate")
            out.append(EarningsRecord(
                ticker=ticker,
                period_end=period_end,
                filing_date=filing_date,
                eps_actual=eps_actual,
                eps_estimate=eps_estimate,
                revenue_actual=rev_actual,
                revenue_estimate=rev_estimate,
                raw=r,
            ))
        return out

    def is_configured(self) -> bool:
        return bool(self.api_key)


def _xpath_value(node: dict, key: str) -> float | None:
    """Helper: extract `node[key]['value']` defensively."""
    leaf = node.get(key) if node else None
    if not isinstance(leaf, dict):
        return None
    val = leaf.get("value")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
