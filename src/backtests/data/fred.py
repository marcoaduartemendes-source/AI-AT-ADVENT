"""FRED (Federal Reserve Economic Data) public API wrapper.

Free public API from the St. Louis Fed. Provides historical macro
series: Fed funds rate, CPI, unemployment, GDP, treasury yields,
VIX, etc. We use it for the macro_kalshi backtest to rebuild the
known-at-time-T state of macro variables that drive Kalshi's
binary markets ("will Fed cut rates this meeting?", etc).

Auth: pass `FRED_API_KEY` as env var. Free tier is 120 requests/min,
which is far above what we need at backtest cadence.

Endpoint: https://api.stlouisfed.org/fred/series/observations
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

FRED_BASE = "https://api.stlouisfed.org"
DEFAULT_CACHE_TTL = 24 * 3600   # 24h — macro series update daily at most


@dataclass
class FREDObservation:
    """One FRED time-series observation."""
    date: date                   # observation date
    value: float | None          # None when FRED reports "." (missing)


# Common series IDs for trading strategies
COMMON_SERIES = {
    # Rates
    "FEDFUNDS":     "Effective federal funds rate (monthly avg)",
    "DFF":          "Federal funds effective rate (daily)",
    "DGS10":        "10-year treasury yield (daily)",
    "DGS2":         "2-year treasury yield (daily)",
    "T10Y2Y":       "10y - 2y spread (yield curve)",
    "DGS3MO":       "3-month treasury",
    # Inflation
    "CPIAUCSL":     "CPI all urban consumers (monthly)",
    "CPILFESL":     "Core CPI (monthly)",
    # Labor
    "UNRATE":       "Unemployment rate",
    "PAYEMS":       "Nonfarm payrolls",
    # Markets
    "VIXCLS":       "VIX close",
    "DCOILWTICO":   "WTI oil spot",
    "GOLDAMGBD228NLBM": "Gold London PM fix",
    # Growth
    "GDP":          "GDP (quarterly)",
}


class FREDClient:
    """Disk-cached FRED API client. Failure-tolerant — returns
    empty list on any error so backtests skip cleanly."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self.cache_dir = Path(
            cache_dir or os.environ.get("FRED_CACHE_DIR", "data/cache/fred")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        if not self.api_key:
            logger.info("FREDClient: FRED_API_KEY not set — "
                        "macro backtests will skip")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ── Generic GET with cache + retry ────────────────────────────────

    def _get(self, path: str, params: dict | None = None,
              cache_ttl_seconds: int | None = None) -> dict:
        if not self.api_key:
            return {}
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds or DEFAULT_CACHE_TTL)
        if cached is not None:
            return cached

        params["api_key"] = self.api_key
        params["file_type"] = "json"
        url = f"{FRED_BASE}{path}"
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                logger.warning(f"FRED GET {path} attempt {attempt} failed: {e}")
                time.sleep(attempt * 2)
                continue
            if resp.status_code == 429:
                wait = attempt * 5
                logger.info(f"FRED 429 rate-limited, sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning(f"FRED {path} HTTP {resp.status_code}: "
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
        scrubbed = {k: v for k, v in sorted(params.items()) if k != "api_key"}
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
            logger.debug(f"FRED cache write {key} failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def series_observations(
        self,
        series_id: str,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> list[FREDObservation]:
        """Time-series observations for a FRED series.

        `series_id` is the FRED code (e.g. "FEDFUNDS", "VIXCLS").
        Date params accept ISO strings or date objects.
        """
        params: dict = {"series_id": series_id, "sort_order": "asc"}
        if from_date:
            f = from_date.isoformat() if hasattr(from_date, "isoformat") else from_date
            params["observation_start"] = f
        if to_date:
            t = to_date.isoformat() if hasattr(to_date, "isoformat") else to_date
            params["observation_end"] = t
        data = self._get("/fred/series/observations", params)
        rows = data.get("observations") or []
        out: list[FREDObservation] = []
        for r in rows:
            try:
                d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            except (KeyError, ValueError):
                continue
            v = r.get("value")
            try:
                val = float(v) if v not in (None, "", ".") else None
            except (TypeError, ValueError):
                val = None
            out.append(FREDObservation(date=d, value=val))
        return out

    def latest_value(self, series_id: str) -> float | None:
        """Convenience: most-recent non-null observation."""
        obs = self.series_observations(series_id)
        for o in reversed(obs):
            if o.value is not None:
                return o.value
        return None
