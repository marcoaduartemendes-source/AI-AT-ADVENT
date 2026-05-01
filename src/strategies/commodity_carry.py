"""Commodity term-structure carry — Coinbase commodity futures.

Erb-Harvey 2006 "Tactical & Strategic Value of Commodity Returns": rank
commodities by annualized roll yield (front vs deferred). Long the top
quintile (backwardated), short the bottom quintile (contango).

Implementation here is a long-only retail-friendly variant — the
allocator's max_alloc_pct caps total exposure, so we go long the top 2
backwardated commodities. Shorting commodity futures at $1k–$25k account
size is operationally awkward; long-only carry still captures most of
the documented edge.

Inputs from scout:
    coinbase.term_structure → {<root>: {front, second, annualized_carry_pct}}
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Minimum annualized carry to enter (after estimated friction)
ENTRY_CARRY_PCT = 4.0       # 4% APR
EXIT_CARRY_PCT = 1.0        # exit when carry decays below 1% APR
TOP_N = 2                   # long top-N most-backwardated commodities


class CommodityCarry(Strategy):
    name = "commodity_carry"
    venue = "coinbase"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        term: dict = ctx.scout_signals.get("term_structure", {})
        if not term:
            logger.debug(f"[{self.name}] no term_structure signal yet")
            return []

        # Rank by annualized carry (positive = backwardated = long-able)
        ranked = sorted(
            ((root, info.get("annualized_carry_pct", 0), info)
             for root, info in term.items()),
            key=lambda r: r[1], reverse=True,
        )
        # Pick the top-N where carry exceeds entry threshold
        long_set = [(root, carry, info) for root, carry, info in ranked[:TOP_N]
                     if carry >= ENTRY_CARRY_PCT]
        long_roots = {root for root, _, _ in long_set}

        proposals: list[TradeProposal] = []
        per_leg = ctx.target_alloc_usd / max(1, TOP_N)

        # Open positions for top-N where we're not already in
        for root, carry, info in long_set:
            front_id = info["front"]["id"]
            existing_qty = ctx.open_positions.get(front_id, {}).get("quantity", 0)
            if existing_qty > 0:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=front_id,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_leg,
                confidence=min(0.9, carry / 20),
                reason=(f"{info['name']} backwardation "
                        f"{info['backwardation_pct']:.2f}% / "
                        f"{carry:.1f}% APR — open carry"),
                metadata={"root": root, "carry_apr_pct": carry},
            ))

        # Close positions whose carry has decayed (or aren't in top-N anymore)
        for root, info in term.items():
            carry = info.get("annualized_carry_pct", 0)
            front_id = info.get("front", {}).get("id")
            if not front_id:
                continue
            existing_qty = ctx.open_positions.get(front_id, {}).get("quantity", 0)
            if existing_qty <= 0:
                continue
            should_close = (root not in long_roots) or (carry < EXIT_CARRY_PCT)
            if should_close:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=front_id,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=existing_qty, confidence=0.95,
                    reason=(f"{info['name']} carry {carry:.1f}% APR — close"),
                    is_closing=True,
                    metadata={"root": root, "carry_apr_pct": carry},
                ))

        return proposals
