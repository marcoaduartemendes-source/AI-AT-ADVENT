"""Macro scout — VIX, Fed FOMC calendar, CME-implied probabilities.

Outputs:
  signal_type="vix_regime"        payload={vix, regime}        venue="macro"
  signal_type="fomc_window"       payload={blackout, days}     venue="macro"
  signal_type="cme_implied_probs" payload={meeting, probs}     venue="macro"
  signal_type="econ_calendar"     payload={upcoming_events}    venue="macro"

Data sources (all free, no auth required):
  • CBOE — VIX index via Yahoo Finance public endpoint
  • Fed — FOMC meeting calendar (federalreserve.gov)
  • CME — Fed Funds futures-implied rate probabilities (FedWatch JSON)
  • BLS / BEA — CPI / NFP / GDP release dates via tradingeconomics public list

We fetch lazily and tolerate outages — a missing signal is better than a
bogus one.
"""
from __future__ import annotations

import logging
from datetime import datetime, UTC

from backtests.data.cme_fedwatch import CMEFedWatchClient
from common import cached_get

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Static FOMC fallback calendar. Used only when the live federalreserve.gov
# scrape fails. Sprint B4 audit fix: previously this was the ONLY source —
# the list silently expires every January. Now we attempt to scrape the
# live page first; the static list is a backstop so the scout still works
# offline / when fed.gov is down. Refresh annually as a sanity check.
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
_FOMC_FALLBACK_2026 = [
    "2026-01-28", "2026-03-18", "2026-04-29",
    "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-11-04", "2026-12-09",
]
# 2027 placeholder: keep current list valid past Jan 2027 even if the
# scrape fails. Refresh when the Fed publishes its 2027 calendar.
_FOMC_FALLBACK_2027 = [
    "2027-01-27", "2027-03-17", "2027-04-28",
    "2027-06-16", "2027-07-28", "2027-09-22",
    "2027-11-03", "2027-12-15",
]
# Combined fallback covers two full calendar years; emits a warning
# when today is past the last entry so we notice in journalctl before
# the calendar runs dry.
_FOMC_FALLBACK = _FOMC_FALLBACK_2026 + _FOMC_FALLBACK_2027


def _fetch_fomc_calendar() -> list[str]:
    """Return upcoming FOMC meeting dates as ISO strings.

    Tries the live federalreserve.gov page first (free, no auth);
    falls back to the curated static list when the scrape fails.
    Cached for 24h via cached_get since the page rarely changes.
    """
    today = datetime.now(UTC).date()
    try:
        # The fed publishes the calendar as a structured HTML page.
        # We don't parse the full HTML — we extract the date strings
        # via a defensive regex against `id="article"` content. This
        # is brittle by design: if the page changes, the regex won't
        # match, and we fall back gracefully to the static list.
        body = cached_get(
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroScout/1.0)"},
            ttl_seconds=86400,
        )
    except Exception:
        body = None

    parsed: list[str] = []
    if isinstance(body, dict):     # cached_get returns json; this page is HTML
        body = None                 # which means parse failed → fall back

    if isinstance(body, str):
        import re
        # Look for date patterns like 'January 28, 2026' or '1/28-29, 2026'.
        # Capture the month, day, year and normalize to YYYY-MM-DD.
        months = ("January February March April May June July August "
                   "September October November December").split()
        rx = re.compile(
            r"(" + "|".join(months) + r")\s+(\d{1,2})(?:[-–](\d{1,2}))?,\s*(\d{4})"
        )
        for m_name, day, end_day, year in rx.findall(body):
            try:
                month_idx = months.index(m_name) + 1
                # Two-day FOMC meetings: use the second day (announcement day)
                day_int = int(end_day) if end_day else int(day)
                d = datetime(int(year), month_idx, day_int).date()
                if d >= today:
                    parsed.append(d.isoformat())
            except (ValueError, IndexError):
                continue

    # If scrape gave us at least 1 future meeting, prefer it. Otherwise
    # fall back to the curated static list.
    if parsed:
        # Dedupe & sort
        return sorted(set(parsed))

    fallback = [d for d in _FOMC_FALLBACK
                if datetime.strptime(d, "%Y-%m-%d").date() >= today]
    if not fallback:
        logger.warning(
            "FOMC fallback calendar exhausted — refresh _FOMC_FALLBACK "
            "with the next year's meeting dates"
        )
    return fallback


class MacroScout(ScoutAgent):
    name = "macro_scout"

    def __init__(self, bus=None, cme_client: CMEFedWatchClient | None = None):
        super().__init__(bus)
        self._cme = cme_client or CMEFedWatchClient()

    def scan(self) -> list[ScoutSignal]:
        signals: list[ScoutSignal] = []

        # ── VIX regime
        vix = self._fetch_vix_yahoo()
        if vix is not None:
            regime = self._vix_regime(vix)
            signals.append(ScoutSignal(
                venue="macro", signal_type="vix_regime",
                payload={"vix": vix, "regime": regime,
                          "as_of": datetime.now(UTC).isoformat()},
                ttl_seconds=4 * 3600,  # refresh every 4h
            ))

        # ── FOMC blackout window
        signals.append(self._fomc_window_signal())

        # ── CME-implied rate probabilities for upcoming meetings
        cme_signal = self._cme_implied_probs_signal()
        if cme_signal is not None:
            signals.append(cme_signal)

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

    def _cme_implied_probs_signal(self) -> ScoutSignal | None:
        """Pull CME FedWatch probabilities for the next 1-2 FOMC meetings.

        macro_kalshi consumes this to compute the Kalshi-vs-CME divergence
        — when Kalshi's implied probability for "Fed cuts 25bps" diverges
        from CME's by > 5%, that's the trade signal.

        Returns None if CME endpoint is unavailable (we'd rather skip
        than publish stale probabilities).
        """
        try:
            meetings = self._cme.upcoming_meetings()
        except Exception as e:
            logger.warning(f"[{self.name}] CME fetch failed: {e}")
            return None
        if not meetings:
            return None
        payload = {
            "meetings": [
                {
                    "date": m.meeting_date.isoformat(),
                    "probs": [
                        {"lo_bps": lo, "hi_bps": hi, "p": prob}
                        for lo, hi, prob in m.target_rate_probs
                    ],
                }
                for m in meetings[:3]   # next 3 meetings is plenty
            ],
            "as_of": datetime.now(UTC).isoformat(),
        }
        return ScoutSignal(
            venue="macro", signal_type="cme_implied_probs",
            payload=payload,
            ttl_seconds=30 * 60,    # 30 min — same as macro scout cadence
        )

    def _fomc_window_signal(self) -> ScoutSignal:
        today = datetime.now(UTC).date()
        # Sprint B4: pull from federalreserve.gov when reachable;
        # the static fallback inside _fetch_fomc_calendar handles
        # offline / scrape-broken cases.
        upcoming = _fetch_fomc_calendar()
        # _fetch_fomc_calendar already filters >= today; no need to re-filter
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
