"""Sector rotation — long the top-N momentum sector ETFs.

Documented edge (Asness 2013, Faber 2007 GTAA): sector returns exhibit
3-12 month momentum. Rotating monthly into the top-3 by trailing return
captures most of that. Defensive bias: only enters sectors with positive
6-month trend (long-only momentum, no shorts).

Universe: the 11 SPDR sector ETFs covering the entire S&P 500.

Rules:
  - Each cycle, compute trailing 90-day returns for all 11 sectors.
  - Long the top-3 by return; equal-weight.
  - Rebalance: if a current holding drops out of the top-3 OR its 90d
    return goes negative, close it and rotate to the new winner.
  - No more than 3 concurrent positions.

Less reactive than RSI (rotates monthly, not daily) so allocator
naturally sees this as a longer-horizon sleeve.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


LOOKBACK_DAYS = 90
TOP_N = 3
MIN_RETURN_PCT = 0.0       # don't long sectors with negative trend
REBAL_COOLDOWN_DAYS = 7    # max once-a-week rotations per slot


SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLI",   # Industrials
    "XLB",   # Materials
    "XLU",   # Utilities
    "XLRE",  # Real Estate
    "XLC",   # Communications
]


class SectorRotation(Strategy):
    name = "sector_rotation"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Compute 90-day return per sector
        rankings = []
        for sym in SECTOR_ETFS:
            ret = self._lookback_return_pct(sym, LOOKBACK_DAYS)
            if ret is None:
                continue
            rankings.append((sym, ret))
        rankings.sort(key=lambda r: r[1], reverse=True)

        # Pick top-N where return meets threshold
        target_set = {sym for sym, ret in rankings[:TOP_N]
                       if ret >= MIN_RETURN_PCT}

        proposals: list[TradeProposal] = []
        size_per_slot = ctx.target_alloc_usd / max(1, TOP_N)

        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # Exit holdings not in target_set
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0:
                continue
            if sym in target_set:
                continue
            # Cooldown: don't churn a slot we just opened/closed
            if not self._past_cooldown(pos):
                continue
            ret = next((r for s, r in rankings if s == sym), None)
            ret_str = f"{ret:.1f}%" if ret is not None else "n/a"
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85,
                reason=f"{sym} 90d ret {ret_str} dropped from top-{TOP_N}",
                is_closing=True,
            ))

        # Enter new top-N picks not currently held
        for sym, ret in rankings[:TOP_N]:
            if ret < MIN_RETURN_PCT:
                continue
            if sym in held:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=size_per_slot, confidence=0.8,
                reason=f"{sym} top-{TOP_N} 90d {ret:+.1f}%",
                metadata={"lookback_return_pct": ret, "rank": 1},
            ))
        return proposals

    # ── Helpers ───────────────────────────────────────────────────────

    def _lookback_return_pct(self, symbol: str, days: int) -> float | None:
        from ._helpers import lookback_return_pct
        return lookback_return_pct(self.broker, self.name, symbol, days)

    def _past_cooldown(self, pos: dict) -> bool:
        from ._helpers import past_cooldown
        return past_cooldown(pos, REBAL_COOLDOWN_DAYS)
