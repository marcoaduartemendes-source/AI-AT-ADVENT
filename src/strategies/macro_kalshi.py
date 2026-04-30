"""Macro probability mispricing — Kalshi macroeconomic events vs CME FedWatch.

Edge thesis: Kalshi prices for FOMC / CPI / NFP outcomes occasionally
diverge from the CME-implied probabilities (derived from Fed Funds
futures). When divergence > entry threshold AND the CME signal is more
liquid/efficient, fade the Kalshi mispricing.

V1 implementation: piggybacks on the prediction scout's mispriced list
but additionally filters for tickers in the macro categories (Fed,
inflation, jobs). Where the Kalshi market is in a known macro category,
we apply a slightly higher size since these markets are more liquid and
the divergence signal is statistically cleaner.

Future work (W3+): pull live CME FedWatch implied probabilities and
compare directly to Kalshi.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Macro market category keywords on Kalshi (approximate — refines as we
# learn the actual category strings used in the API responses)
MACRO_KEYWORDS = ("fed", "fomc", "rate", "cpi", "inflation", "nfp",
                   "jobs", "unemployment", "gdp", "pce", "pmi", "ism")

ENTRY_EDGE_CENTS = 2.5      # Tighter than calibration arb because liquid
MAX_PER_TRADE_PCT = 0.20    # Bigger size — more confident
KELLY_FRACTION = 0.30


class MacroKalshi(Strategy):
    name = "macro_kalshi"
    venue = "kalshi"

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        candidates: List[Dict] = ctx.scout_signals.get("mispriced", []) or []
        if not candidates:
            return []

        # Filter for macro events
        macro_candidates = [
            c for c in candidates
            if self._is_macro(c.get("ticker", ""), c.get("category", ""),
                                c.get("title", ""))
        ]
        if not macro_candidates:
            return []

        proposals: List[TradeProposal] = []
        per_trade_cap = ctx.target_alloc_usd * MAX_PER_TRADE_PCT

        for m in macro_candidates:
            ticker = m["ticker"]
            yes_price = m["yes_price"]
            fair_value = m["fair_value"]
            edge = fair_value - yes_price
            edge_cents = edge * 100
            if abs(edge_cents) < ENTRY_EDGE_CENTS:
                continue

            if edge > 0:
                p, cost = fair_value, yes_price
                side = OrderSide.BUY
            else:
                p, cost = 1 - fair_value, 1 - yes_price
                side = OrderSide.SELL

            if cost <= 0:
                continue
            b = (1 - cost) / cost
            full_kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0
            sized_usd = min(ctx.target_alloc_usd * full_kelly * KELLY_FRACTION,
                             per_trade_cap)
            if sized_usd < 1.0:
                continue
            n_contracts = int(sized_usd / cost)
            if n_contracts < 1:
                continue

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=ticker,
                side=side, order_type=OrderType.LIMIT,
                quantity=float(n_contracts), limit_price=yes_price,
                confidence=min(0.92, abs(edge) * 5),
                reason=(f"macro mispricing: market={yes_price:.3f}, "
                        f"fair={fair_value:.3f}, edge={edge_cents:+.1f}c"),
                metadata={"category": m.get("category"),
                          "title": m.get("title")},
            ))
        return proposals

    @staticmethod
    def _is_macro(ticker: str, category: str, title: str) -> bool:
        haystack = f"{ticker} {category} {title}".lower()
        return any(k in haystack for k in MACRO_KEYWORDS)
