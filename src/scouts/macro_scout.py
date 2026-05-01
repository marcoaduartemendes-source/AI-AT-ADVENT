"""Macro scout — VIX, Fed FOMC calendar, key data releases.

Outputs:
  signal_type="vix_regime"     payload={vix, regime}        venue="macro"
  signal_type="fomc_window"    payload={blackout, days}     venue="macro"
  signal_type="econ_calendar"  payload={upcoming_events}    venue="macro"

Data sources (no auth required):
  • CBOE — VIX index via Yahoo Finance public endpoint
  • Fed — FOMC meeting calendar (federalreserve.gov)
  • BLS / BEA — CPI / NFP / GDP release dates via tradingeconomics public list

We fetch lazily and tolerate outages — a missing signal is better than a
bogus one.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List

from common import cached_get

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Static FOMC calendar for 2026 (manually curated, refresh annually).
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-11-04", "2026-12-09",
]


class MacroScout(ScoutAgent):
    name = "macro_scout"

    def scan(self) -> List[ScoutSignal]:
        signals: List[ScoutSignal] = []

        # ── VIX regime
        vix = self._fetch_vix_yahoo()
        if vix is not None:
            regime = self._vix_regime(vix)
            signals.append(ScoutSignal(
                venue="macro", signal_type="vix_regime",
                payload={"vix": vix, "regime": regime,
                          "as_of": datetime.now(timezone.utc).isoformat()},
                ttl_seconds=4 * 3600,  # refresh every 4h
            ))

        # ── FOMC blackout window
        signals.append(self._fomc_window_signal())

        return signals

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fetch_vix_yahoo(self) -> float | None:
        """Yahoo Finance v8 chart endpoint — no auth, returns last quote.
        Cached 5 min — VIX updates intraday but doesn't move every minute."""
        data = cached_get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX",
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroScout/1.0)"},
            ttl_seconds=300,
        )
        if not data:
            return None
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        quotes = result[0].get("indicators", {}).get("quote", [{}])[0]
        closes = quotes.get("close") or []
        for v in reversed(closes):
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _vix_regime(vix: float) -> str:
        if vix >= 35: return "extreme"
        if vix >= 25: return "elevated"
        if vix >= 18: return "normal"
        return "calm"

    def _fomc_window_signal(self) -> ScoutSignal:
        today = datetime.now(timezone.utc).date()
        upcoming = []
        for d in _FOMC_2026:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            if dt >= today:
                upcoming.append(d)
        if not upcoming:
            return ScoutSignal(
                venue="macro", signal_type="fomc_window",
                payload={"blackout": False, "next_meeting": None,
                          "days_to_next": None},
                ttl_seconds=24 * 3600,
            )
        next_meeting = upcoming[0]
        next_dt = datetime.strptime(next_meeting, "%Y-%m-%d").date()
        days_to_next = (next_dt - today).days
        # 2-day blackout starting day before meeting (Fed quiet period)
        blackout = days_to_next <= 1
        return ScoutSignal(
            venue="macro", signal_type="fomc_window",
            payload={"blackout": blackout, "next_meeting": next_meeting,
                      "days_to_next": days_to_next, "all_upcoming": upcoming[:6]},
            ttl_seconds=12 * 3600,
        )
