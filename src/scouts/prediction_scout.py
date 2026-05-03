"""Prediction-market scout — Kalshi mispriced markets + Polymarket cross-check.

Outputs:
  signal_type="mispriced"        payload=[{ticker, yes_price, fair_value,
                                            edge, polymarket_yes, …}, …]
  signal_type="cross_venue_arb"  payload=[{kalshi_ticker, kalshi_yes,
                                            polymarket_yes, divergence}, …]

Uses the existing recalibration table from the kalshi_calibration_arb
strategy (Snowberg-Wolfers 2010, Said 2024). Pulls open Kalshi markets
through the authenticated KalshiAdapter, screens for liquid contracts
where the recalibrated fair-value differs from the market price by more
than the configured threshold, and publishes a ranked list.

Additionally, for each Kalshi market we attempt a token-overlap match
against Polymarket's active markets. When both venues have liquid
contracts on the same underlying event, large divergences (≥ 5 cents)
are cleaner arb signals than the calibration-table fade alone — they
don't depend on the Snowberg-Wolfers prior holding.
"""
from __future__ import annotations

import logging

from backtests.data.polymarket import PolymarketClient, PolymarketContract
from brokers.registry import get_broker
from strategies.kalshi_calibration_arb import (
    DEFAULT_RECALIBRATION,
    ENTRY_EDGE_CENTS,
    MIN_OPEN_INTEREST_USD,
)

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Pull at most this many markets per scan to keep API usage bounded.
MAX_MARKETS_PER_SCAN = 200


class PredictionScout(ScoutAgent):
    name = "prediction_scout"

    def __init__(self, bus=None,
                  recalibration=None,
                  entry_edge_cents: float = ENTRY_EDGE_CENTS,
                  min_open_interest_usd: float = MIN_OPEN_INTEREST_USD,
                  polymarket: PolymarketClient | None = None,
                  cross_venue_min_divergence: float = 0.05):
        super().__init__(bus=bus)
        self.recalibration = recalibration or DEFAULT_RECALIBRATION
        self.entry_edge_cents = entry_edge_cents
        self.min_oi_usd = min_open_interest_usd
        self._polymarket = polymarket or PolymarketClient()
        self._cross_venue_min_divergence = cross_venue_min_divergence

    def scan(self) -> list[ScoutSignal]:
        broker = get_broker("kalshi")
        if broker is None:
            logger.warning(f"[{self.name}] Kalshi adapter not configured")
            return []

        try:
            # Kalshi /markets endpoint returns market list with bid/ask/oi
            d = broker._request("GET", f"/markets?status=open&limit={MAX_MARKETS_PER_SCAN}")
        except Exception as e:
            logger.warning(f"[{self.name}] markets fetch failed: {e}")
            return []

        # Pre-fetch Polymarket universe once per scan for cross-venue match.
        # If Polymarket is unreachable we fall back to no enrichment.
        polymarket_markets = self._fetch_polymarket_safely()

        markets = d.get("markets", [])
        ranked: list[dict] = []
        cross_venue: list[dict] = []
        for m in markets:
            ticker = m.get("ticker")
            yes_bid = m.get("yes_bid")          # cents
            yes_ask = m.get("yes_ask")          # cents
            oi = m.get("open_interest")         # contracts (1 contract = $1)
            if ticker is None or yes_bid is None or yes_ask is None:
                continue
            mid_cents = (yes_bid + yes_ask) / 2.0
            yes_price = mid_cents / 100.0
            if yes_price <= 0 or yes_price >= 1:
                continue

            fair_value = self._fair_value(yes_price)
            edge = fair_value - yes_price
            edge_cents = edge * 100

            # Open interest in dollars (contracts × $1 settlement value)
            oi_usd = float(oi) if oi else 0.0

            # Try to find a Polymarket counterpart for this Kalshi market
            pm_match: PolymarketContract | None = None
            kalshi_title = m.get("title") or ""
            if polymarket_markets and kalshi_title:
                pm_match = self._polymarket.find_kalshi_match(
                    kalshi_title, polymarket_markets,
                )

            # Cross-venue arbitrage signal: if Polymarket and Kalshi
            # disagree by ≥ min_divergence on the same event, that's a
            # cleaner trade than the calibration fade.
            if pm_match is not None:
                divergence = pm_match.yes_price - yes_price
                if abs(divergence) >= self._cross_venue_min_divergence:
                    cross_venue.append({
                        "kalshi_ticker": ticker,
                        "kalshi_yes": yes_price,
                        "polymarket_id": pm_match.market_id,
                        "polymarket_yes": pm_match.yes_price,
                        "polymarket_question": pm_match.question,
                        "polymarket_volume_24h": pm_match.volume_24h_usd,
                        "divergence": round(divergence, 4),
                        "abs_divergence_cents": round(abs(divergence) * 100, 1),
                        "title": kalshi_title,
                    })

            # Calibration-arb signal (the existing fade)
            if abs(edge_cents) < self.entry_edge_cents:
                continue
            if oi_usd < self.min_oi_usd:
                continue
            ranked.append({
                "ticker": ticker,
                "yes_price": yes_price,
                "fair_value": fair_value,
                "edge_cents": round(edge_cents, 2),
                "open_interest_usd": oi_usd,
                "category": m.get("category"),
                "title": kalshi_title,
                "yes_bid": yes_bid / 100.0,
                "yes_ask": yes_ask / 100.0,
                "polymarket_yes": pm_match.yes_price if pm_match else None,
                "polymarket_volume_24h":
                    pm_match.volume_24h_usd if pm_match else None,
            })

        # Sort by absolute edge (best opportunities first), cap to top 50
        ranked.sort(key=lambda r: abs(r["edge_cents"]), reverse=True)
        ranked = ranked[:50]

        cross_venue.sort(key=lambda r: r["abs_divergence_cents"], reverse=True)
        cross_venue = cross_venue[:50]

        out: list[ScoutSignal] = []
        if ranked:
            out.append(ScoutSignal(
                venue="kalshi", signal_type="mispriced",
                payload=ranked,
                ttl_seconds=2 * 3600,
            ))
        if cross_venue:
            out.append(ScoutSignal(
                venue="kalshi", signal_type="cross_venue_arb",
                payload=cross_venue,
                ttl_seconds=2 * 3600,
            ))
        return out

    def _fetch_polymarket_safely(self) -> list[PolymarketContract]:
        """Fetch the active Polymarket universe; tolerate failure."""
        try:
            return self._polymarket.active_markets(
                limit=300, min_volume_usd=1000.0,
            )
        except Exception as e:
            logger.info(f"[{self.name}] polymarket fetch failed (skipping "
                         f"cross-venue enrichment): {e}")
            return []

    def _fair_value(self, market_price: float) -> float:
        for lo, hi, shift in self.recalibration:
            if lo <= market_price < hi:
                return max(0.0, min(1.0, market_price + shift))
        return market_price
