"""Activist 13D scout — polls SEC EDGAR for new Schedule 13D filings
and publishes the underlying ticker to the signal bus so the
`activist_13d` strategy can buy the target before the drift completes.

REAL EDGE:
    Brav-Jiang-Partnoy-Thomas (JF 2008, updated 2019): +7-8% abnormal
    return in the (-10, +20) day window around a 13D filing, with NO
    long-term reversal (Bebchuk-Brav-Jiang 2015). The drift continues
    for months in many cases.

    Edge persistence: SEC mandates 13D filing within 10 calendar days of
    crossing 5%. The activist has typically built the position over
    weeks at lower prices — the filing is a credibly signaled commitment
    to push for change. Mimicked systematically by Two Sigma's event
    sleeve; played discretionarily by Senator, Sachem Head, JANA,
    ValueAct, Engine No. 1, etc.

DATA:
    SEC EDGAR full-text search for form type "SC 13D" — free, no API
    key. Returns the most recent filings; we filter for those in the
    last N hours (defaults to 24h to catch the day's drift window).
"""
from __future__ import annotations

import json
import logging
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)

# SEC requires a descriptive UA on EDGAR scrapes. Reusing the project's
# canonical contact string per their fair-use guidelines.
_UA = ("ai-at-advent activist scout "
       "(contact: marcoaduartemendes@gmail.com)")

# EDGAR full-text search endpoint. The "type=SC+13D" filter returns
# initial Schedule 13D filings (activist stakes); "SC 13D/A" amendments
# are noisier and excluded.
_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index?"
    "q=&dateRange=custom&forms=SC+13D&startdt={start}&enddt={end}"
)


def _fetch(url: str, timeout: int = 12) -> dict | None:
    try:
        req = Request(url, headers={"User-Agent": _UA,
                                     "Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (URLError, json.JSONDecodeError, TimeoutError, OSError) as e:
        logger.warning(f"[activist_13d] EDGAR fetch failed: {e}")
        return None


def _normalize_ticker(raw: str | None) -> str | None:
    """Best-effort extraction of a US-equity ticker from EDGAR metadata.
    EDGAR fields are inconsistent; returns None on anything ambiguous so
    the strategy never trades a guess."""
    if not raw:
        return None
    s = raw.strip().upper()
    # Strip exchange suffixes like "(NYSE: ABC)"
    m = re.search(r"\b([A-Z]{1,5})(?:\.[A-Z])?\b", s)
    if not m:
        return None
    tic = m.group(1)
    # Filter obvious false positives
    if tic in {"SEC", "USA", "INC", "LLC", "LTD", "CIK", "SC", "13D"}:
        return None
    return tic


class Activist13DScout(ScoutAgent):
    """Daily check for new SC 13D filings (5%+ activist stakes)."""

    name = "activist_13d"

    def scan(self) -> list[ScoutSignal]:
        # Pull last 2 calendar days of SC 13D filings so we don't miss
        # weekend / cron-skip windows. The signal bus de-dupes by
        # (venue, signal_type, payload['ticker']) within the TTL.
        from datetime import UTC, datetime, timedelta
        end = datetime.now(UTC).date()
        start = end - timedelta(days=2)
        url = _SEARCH_URL.format(start=start.isoformat(),
                                    end=end.isoformat())
        data = _fetch(url)
        if not data:
            return []
        hits = (data.get("hits") or {}).get("hits") or []
        signals: list[ScoutSignal] = []
        for hit in hits[:50]:        # bound work — drift lasts days
            src = hit.get("_source") or {}
            tickers = src.get("tickers") or []
            for tic_raw in tickers[:1]:    # one ticker per filing
                tic = _normalize_ticker(tic_raw)
                if not tic:
                    continue
                filed = src.get("file_date") or src.get("filed_date")
                signals.append(ScoutSignal(
                    venue="alpaca",      # consumer is an Alpaca strategy
                    signal_type="activist_13d_new",
                    payload={
                        "ticker": tic,
                        "filed_date": filed,
                        "accession": src.get("adsh") or src.get("accession"),
                        "cik": src.get("ciks") or src.get("cik"),
                    },
                    # 5 trading days — the bulk of the drift completes
                    # in the first ~2 days but holding through day-5
                    # captures most of the Brav-Jiang documented effect.
                    ttl_seconds=5 * 24 * 3600,
                ))
        if signals:
            logger.info(f"[activist_13d] published {len(signals)} new "
                        f"13D signals")
        return signals
