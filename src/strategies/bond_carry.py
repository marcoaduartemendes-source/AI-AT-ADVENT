"""Bond carry — harvest the term + credit premium, trend-gated.

The book holds bonds only as a defensive leg (risk_parity, dual_momentum's
risk-off sleeve). This is a dedicated sleeve that actively harvests the
two structural bond premia:

  • Term premium  — longer-duration Treasuries (TLT/IEF) pay you to bear
    duration risk.
  • Credit spread — IG/HY/EM bonds (LQD/HYG/EMB) pay a spread over
    Treasuries for default risk.

Both are genuine *carry* (positive expected drift from yield), but both
get crushed when rates spike or spreads blow out (2022 was the worst
Treasury year in history). So we don't hold them blindly — each sleeve is
included only while it's above its 100-day SMA (the trend confirms the
carry isn't being overwhelmed by a rate/spread move). Any sleeve that
fails the trend gate parks its capital in SHY (1-3y Treasuries — minimal
duration, still earns short-rate carry). Full risk-off → 100% SHY.

This is the classic "carry with a trend overlay" that managed-futures and
fixed-income relative-value desks run. Monthly rebalance keeps turnover
(and fees) low. Uncorrelated to the book's equity beta.

Stays DRY until docs/validation.json records PASS — auto_demote freezes
it if the live edge doesn't hold, so there's no capital risk to prove it.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

# Carry-bearing sleeves (term + credit premium).
CARRY_UNIVERSE = ["TLT", "IEF", "LQD", "HYG", "EMB"]
SAFE_ASSET = "SHY"                 # 1-3y Treasuries — duration-light parking
TREND_SMA = 100                    # include a sleeve only above its 100d SMA


class BondCarry(Strategy):
    name = "bond_carry"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        book = ctx.target_alloc_usd

        prices: dict[str, float] = {}
        above_trend: dict[str, bool] = {}
        for symbol in CARRY_UNIVERSE + [SAFE_ASSET]:
            try:
                candles = self.broker.get_candles(
                    symbol, "ONE_DAY", num_candles=TREND_SMA + 30)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {symbol}: {e}")
                continue
            if len(candles) < TREND_SMA:
                continue
            closes = np.array([c.close for c in candles])
            prices[symbol] = float(closes[-1])
            above_trend[symbol] = float(closes[-1]) >= float(closes[-TREND_SMA:].mean())

        if SAFE_ASSET not in prices:
            return []   # need the parking leg to act safely

        # Equal-weight slots across the carry universe; each slot holds
        # its sleeve if above trend, else rotates to SHY.
        n_slots = len(CARRY_UNIVERSE)
        per_slot = book / n_slots
        target_usd: dict[str, float] = {s: 0.0 for s in CARRY_UNIVERSE}
        target_usd[SAFE_ASSET] = 0.0
        for s in CARRY_UNIVERSE:
            if s in prices and above_trend.get(s):
                target_usd[s] += per_slot       # carry on
            else:
                target_usd[SAFE_ASSET] += per_slot  # risk-off → SHY

        proposals: list[TradeProposal] = []
        for symbol, tgt in target_usd.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
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
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=side, order_type=OrderType.MARKET,
                notional_usd=abs(delta), confidence=0.65,
                reason=(f"bond carry: {'SAFE' if symbol == SAFE_ASSET else 'carry'} "
                        f"target=${tgt:.0f}, current=${cur_usd:.0f}"),
                is_closing=(side == OrderSide.SELL and cur_qty > 0),
                metadata={"target_usd": tgt,
                          "above_trend": above_trend.get(symbol)},
            ))
        return proposals
