"""Commodity momentum — cross-sectional CTA trend on commodity ETFs.

Commodities are the textbook home of time-series + cross-sectional
momentum: trends are long, persistent, and driven by slow-moving supply/
demand and inventory cycles (Erb-Harvey 2006; Moskowitz-Ooi-Pedersen
2012; the entire managed-futures industry). The book already has
commodity_carry (futures backwardation) but no commodity *momentum*
sleeve — a distinct, complementary premium.

CONSTRUCTION
  Universe: GLD (gold), SLV (silver), DBC (broad), USO (oil),
    UNG (natgas), DBA (agriculture), CPER (copper).
  Signal: 12-1m momentum (252d lookback, skip last 21d).
  Selection: long the TOP_K=3 by momentum, but only those with positive
    absolute momentum (don't fight a falling commodity). Equal-weight;
    failed slots stay in cash (long-only — no commodity shorts here).
  Monthly rebalance → fee-efficient.

Genuinely uncorrelated to equities and to commodity_carry (different
signal: trend vs term-structure). Stays DRY until validation PASS;
auto_demote freezes it if the edge doesn't hold.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

UNIVERSE = ["GLD", "SLV", "DBC", "USO", "UNG", "DBA", "CPER"]
LOOKBACK_DAYS = 252
SKIP_DAYS = 21
TOP_K = 3


class CommodityMomentum(Strategy):
    name = "commodity_momentum"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        book = ctx.target_alloc_usd

        momentum: dict[str, float] = {}
        prices: dict[str, float] = {}
        for symbol in UNIVERSE:
            try:
                candles = self.broker.get_candles(
                    symbol, "ONE_DAY", num_candles=LOOKBACK_DAYS + 30)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {symbol}: {e}")
                continue
            if len(candles) < LOOKBACK_DAYS:
                continue
            closes = np.array([c.close for c in candles])
            prices[symbol] = float(closes[-1])
            window = closes[-LOOKBACK_DAYS:-SKIP_DAYS] if SKIP_DAYS else closes[-LOOKBACK_DAYS:]
            if len(window) < 30:
                continue
            momentum[symbol] = (window[-1] - window[0]) / window[0]

        if not momentum:
            return []

        # Top-K by momentum, gated by positive absolute momentum.
        ranked = sorted(momentum, key=lambda s: momentum[s], reverse=True)[:TOP_K]
        winners = [s for s in ranked if momentum[s] > 0]
        per_slot = book / TOP_K     # fixed denominator → de-risks when <K winners
        target_usd: dict[str, float] = {s: 0.0 for s in UNIVERSE}
        for s in winners:
            target_usd[s] = per_slot

        proposals: list[TradeProposal] = []
        for symbol in UNIVERSE:
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            tgt = target_usd[symbol]
            cur_qty = ctx.open_positions.get(symbol, {}).get("quantity", 0.0)
            cur_usd = cur_qty * price
            pending = ctx.pending_orders.get(symbol, {})
            committed = cur_usd + pending.get("buy_notional_usd", 0.0)
            delta = tgt - committed
            if abs(delta) < max(50.0, per_slot * 0.10):
                continue
            if delta > 0 and pending.get("n_pending", 0) > 0:
                continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            mom = momentum.get(symbol)
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=side, order_type=OrderType.MARKET,
                notional_usd=abs(delta), confidence=0.65,
                reason=(f"commodity momentum: target=${tgt:.0f}, "
                        f"current=${cur_usd:.0f}"
                        + (f", 12-1m={mom*100:+.1f}%" if mom is not None else "")),
                is_closing=(side == OrderSide.SELL and cur_qty > 0),
                metadata={"momentum_12_1m": mom, "target_usd": tgt},
            ))
        return proposals
