"""Merger Arb scout — fetch announced M&A deals from FMP and publish
target + acquirer + announced deal price to the signal bus.

REAL EDGE:
    Merger arbitrage has 25y Sharpe ~1.46 (S&P Merger Arb Index, per
    Accelerate Shares 2023). Pershing Square, Paulson, Water Island,
    Pentwater all run it. Edge comes from compensation for deal-break
    tail risk + index funds dumping targets at announcement.

DATA:
    FMP `/stable/mergers-acquisitions` — list of announced deals with
    target, acquirer, deal price, status. If FMP isn't configured, the
    scout silently emits nothing and the strategy sits flat.
"""
from __future__ import annotations

import logging

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


class MergerArbScout(ScoutAgent):
    """Daily check for announced M&A deals (target, acquirer, deal $)."""

    name = "merger_arb"

    def scan(self) -> list[ScoutSignal]:
        try:
            from backtests.data.fmp import FMPClient
            client = FMPClient()
        except Exception as e:
            logger.debug(f"[merger_arb] FMP unavailable: {e}")
            return []
        if not client.is_configured():
            return []

        # FMP exposes pending M&A under multiple endpoints across plan
        # tiers. Try the modern `/stable/mergers-acquisitions-latest`
        # first; fall back to `/v3/mergers-and-acquisitions-rss-feed`.
        deals = []
        for path in (
            "mergers-acquisitions-latest",
            "mergers-and-acquisitions-rss-feed",
        ):
            try:
                resp = client._get(path, cache_ttl_seconds=3600)
                if isinstance(resp, list) and resp:
                    deals = resp
                    break
            except Exception as e:
                logger.debug(f"[merger_arb] {path} returned: {e}")
                continue

        if not deals:
            return []

        signals: list[ScoutSignal] = []
        for d in deals[:50]:    # cap work
            if not isinstance(d, dict):
                continue
            target = (d.get("targetedCompanyTicker")
                      or d.get("symbol") or "").upper()
            if not target:
                continue
            # Deal-price field varies by FMP plan tier; check the
            # common ones and skip if missing.
            deal_price = (d.get("transactionValue") or d.get("dealValue")
                          or d.get("price_per_share")
                          or d.get("pricePerShare"))
            if not deal_price:
                continue
            try:
                deal_price = float(deal_price)
            except (TypeError, ValueError):
                continue
            if deal_price <= 0:
                continue
            signals.append(ScoutSignal(
                venue="alpaca",
                signal_type="merger_arb_deal",
                payload={
                    "target": target,
                    "acquirer": d.get("companyName") or d.get("acquirer"),
                    "deal_price": deal_price,
                    "announced": d.get("transactionDate")
                                  or d.get("announced"),
                    "status": d.get("status") or "PENDING",
                },
                # Deals close on weeks-to-months horizons; refresh daily.
                ttl_seconds=36 * 3600,
            ))
        if signals:
            logger.info(f"[merger_arb] published {len(signals)} deals")
        return signals
