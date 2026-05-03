"""CME FedWatch — Fed Funds futures-implied rate-cut probabilities.

Why this matters:
    macro_kalshi trades Kalshi Fed-rate markets but the live strategy
    has no CME reference price. CME's Fed Funds futures are the gold
    standard for market-implied probabilities — when Kalshi diverges
    from CME, the divergence is the trade.

Data source decision:
    CME publishes the FedWatch tool data as a free public JSON
    endpoint at cmegroup.com. No auth, no rate limits at our cadence
    (every 30 minutes via the macro scout). For paid data:
        - CME DataMine ($$$ — actual full futures order book), only
          worth it for HFT-tier execution which we don't do.
        - Bloomberg / Refinitiv: ~$24k/yr terminal subscription.
    Free CME public is sufficient for our 5-minute orchestrator
    cadence. We document the freshness assumption (~5 min stale) and
    the strategy is robust to that.

Failure mode: CME website occasionally rejects the JSON endpoint with
a 403 or returns an HTML challenge page. Scout treats absence of
signal as no-info — strategies that need the CME context skip the
trade rather than firing on stale data.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


CME_FEDWATCH_URL = (
    "https://www.cmegroup.com/services/cme/json/cme_meeting_data.json"
)
DEFAULT_CACHE_TTL = 30 * 60   # 30 minutes — matches scout cadence


@dataclass
class FedMeetingProb:
    """Implied probabilities for one upcoming FOMC meeting."""
    meeting_date: date
    # Each entry: (lo_bps, hi_bps, probability)
    # e.g. (425, 450, 0.85) = 85% probability target rate stays 4.25-4.50%
    target_rate_probs: list[tuple[int, int, float]]
    raw: dict


class CMEFedWatchClient:
    """Disk-cached CME FedWatch fetcher. No auth needed; free public JSON.

    Pattern matches BybitClient / FREDClient / KalshiHistoryClient so
    scouts and backtests can polymorphically check `is_configured()`."""

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.cache_dir = Path(
            cache_dir or os.environ.get("CME_CACHE_DIR", "data/cache/cme")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """No auth required; always True. Method exists for symmetry
        with other data clients."""
        return True

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        return self.cache_dir / "fedwatch_meetings.json"

    def _read_cache(self, ttl_seconds: int):
        import json
        import time
        p = self._cache_path()
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_cache(self, data) -> None:
        import json
        try:
            self._cache_path().write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"CME cache write failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def upcoming_meetings(
        self,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL,
    ) -> list[FedMeetingProb]:
        """Pull the latest FedWatch implied probabilities for upcoming
        FOMC meetings. Returns oldest-first (next meeting first).

        Returns empty list on any error — callers should treat absence
        as no-info, not stale info.
        """
        cached = self._read_cache(cache_ttl_seconds)
        if cached is None:
            try:
                resp = requests.get(
                    CME_FEDWATCH_URL,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; trading-bot/1.0)",
                        "Accept": "application/json",
                    },
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    logger.warning(f"CME FedWatch HTTP {resp.status_code}")
                    return []
                data = resp.json()
                self._write_cache(data)
            except (requests.RequestException, ValueError) as e:
                logger.warning(f"CME FedWatch fetch failed: {e}")
                return []
        else:
            data = cached

        return _parse_fedwatch(data)


def _parse_fedwatch(data: dict) -> list[FedMeetingProb]:
    """Normalize CME's nested JSON into FedMeetingProb rows.

    The CME response shape varies; we defensively pluck `meetings` /
    `meetingProbabilities` / similar top-level keys and skip rows
    missing required fields rather than crashing the scout.
    """
    if not isinstance(data, dict):
        return []
    meetings_raw = (
        data.get("meetings")
        or data.get("meetingProbabilities")
        or data.get("data", {}).get("meetings", [])
        or []
    )
    out: list[FedMeetingProb] = []
    for m in meetings_raw:
        if not isinstance(m, dict):
            continue
        try:
            meeting_str = (m.get("meetingDate") or m.get("date") or
                           m.get("meeting"))
            if not meeting_str:
                continue
            md = _parse_meeting_date(meeting_str)
            if md is None:
                continue
            probs_raw = (m.get("probabilities") or m.get("targetRates") or
                         m.get("rates") or [])
            probs: list[tuple[int, int, float]] = []
            for p in probs_raw:
                if not isinstance(p, dict):
                    continue
                lo = _coerce_int(p.get("lo") or p.get("lower") or
                                  p.get("rate_lo"))
                hi = _coerce_int(p.get("hi") or p.get("upper") or
                                  p.get("rate_hi"))
                pr = _coerce_float(p.get("probability") or p.get("prob") or
                                    p.get("p"))
                if lo is None or hi is None or pr is None:
                    continue
                # Normalize 0-100 percent to 0-1 probability
                if pr > 1:
                    pr = pr / 100
                probs.append((lo, hi, pr))
            if probs:
                out.append(FedMeetingProb(
                    meeting_date=md, target_rate_probs=probs, raw=m,
                ))
        except (TypeError, ValueError):
            continue
    out.sort(key=lambda x: x.meeting_date)
    return out


def _parse_meeting_date(s: str) -> date | None:
    """CME publishes meeting dates as YYYY-MM-DD or MM/DD/YYYY."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _coerce_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _coerce_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
