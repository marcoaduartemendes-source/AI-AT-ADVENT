"""Prediction-market scout — Kalshi mispriced markets via favorite-longshot
bias recalibration.

Outputs:
  signal_type="mispriced"  payload=[{ticker, yes_price, fair_value, edge,
                                       category, open_interest_usd}, …]

Uses the existing recalibration table from the kalshi_calibration_arb
strategy (Snowberg-Wolfers 2010, Said 2024). Pulls open Kalshi markets
through the authenticated KalshiAdapter, screens for liquid contracts
where the recalibrated fair-value differs from the market price by more
than the configured threshold, and publishes a ranked list.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

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
                  min_open_interest_usd: float = MIN_OPEN_INTEREST_USD):
        super().__init__(bus=bus)
        self.recalibration = recalibration or DEFAULT_RECALIBRATION
        self.entry_edge_cents = entry_edge_cents
        self.min_oi_usd = min_open_interest_usd

    def scan(self) -> List[ScoutSignal]:
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

        markets = d.get("markets", [])
        ranked: List[Dict] = []
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
            if abs(edge_cents) < self.entry_edge_cents:
                continue

            # Open interest in dollars (contracts × $1 settlement value)
            oi_usd = float(oi) if oi else 0.0
            if oi_usd < self.min_oi_usd:
                continue

            ranked.append({
                "ticker": ticker,
                "yes_price": yes_price,
                "fair_value": fair_value,
                "edge_cents": round(edge_cents, 2),
                "open_interest_usd": oi_usd,
                "category": m.get("category"),
                "title": m.get("title"),
                "yes_bid": yes_bid / 100.0,
                "yes_ask": yes_ask / 100.0,
            })

        # Sort by absolute edge (best opportunities first), cap to top 50
        ranked.sort(key=lambda r: abs(r["edge_cents"]), reverse=True)
        ranked = ranked[:50]

        if not ranked:
            return []
        return [ScoutSignal(
            venue="kalshi", signal_type="mispriced",
            payload=ranked,
            ttl_seconds=2 * 3600,
        )]

    def _fair_value(self, market_price: float) -> float:
        for lo, hi, shift in self.recalibration:
            if lo <= market_price < hi:
                return max(0.0, min(1.0, market_price + shift))
        return market_price
