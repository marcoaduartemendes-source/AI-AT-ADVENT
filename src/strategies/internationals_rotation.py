"""International / regional ETF momentum rotation.

US equities have dominated for 15 years, but periods of US under-
performance vs international ETFs (2002-2007, late 2025) deliver
significant alpha to the regions that lead. This strategy provides
diversification away from the US-equity-only Phase-4 sleeve:

  Universe: 8 country / region ETFs covering major developed and
  emerging markets:
    EFA   - iShares MSCI EAFE (Europe + Asia + Far East developed)
    EEM   - iShares MSCI Emerging Markets
    EWJ   - iShares MSCI Japan
    EWG   - iShares MSCI Germany
    EWU   - iShares MSCI United Kingdom
    INDA  - iShares MSCI India
    EWZ   - iShares MSCI Brazil
    FXI   - iShares China Large-Cap

Each cycle, rank by trailing 90d return; long the top-2 if their
return is > a US-baseline (SPY 90d return) by ≥ 2%. The relative-
strength filter prevents simply piling into all-equity beta when
the US is leading.

Position sizing: $4k × 2 = $8k peak deployment. 14d cooldown.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


LOOKBACK_DAYS = 90
TOP_N = 2
RELATIVE_STRENGTH_BPS = 200    # international must beat SPY by ≥ 2%
COOLDOWN_DAYS = 14
TRADE_SIZE_USD = 10000.0   # per-position cap raised for paper-trading experimentation


INTERNATIONAL_ETFS = [
    "EFA",   # MSCI EAFE
    "EEM",   # Emerging Markets
    "EWJ",   # Japan
    "EWG",   # Germany
    "EWU",   # UK
    "INDA",  # India
    "EWZ",   # Brazil
    "FXI",   # China Large-Cap
]

US_BASELINE = "SPY"


class InternationalsRotation(Strategy):
    name = "internationals_rotation"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # 90-day return for the US baseline
        us_return = self._lookback_return_pct(US_BASELINE, LOOKBACK_DAYS)
        if us_return is None:
            logger.debug(f"[{self.name}] no SPY baseline — sitting out")
            return []

        rankings = []
        for sym in INTERNATIONAL_ETFS:
            ret = self._lookback_return_pct(sym, LOOKBACK_DAYS)
            if ret is None:
                continue
            rankings.append((sym, ret))
        rankings.sort(key=lambda r: r[1], reverse=True)

        # Filter: must beat US baseline by RELATIVE_STRENGTH_BPS
        threshold = us_return + RELATIVE_STRENGTH_BPS / 100
        winners = [(sym, ret) for sym, ret in rankings[:TOP_N]
                   if ret > threshold]
        target_set = {sym for sym, _ in winners}

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # Exits: held but not in target_set + past cooldown
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0 or sym in target_set:
                continue
            if not self._past_cooldown(pos):
                continue
            ret = next((r for s, r in rankings if s == sym), None)
            ret_str = f"{ret:.1f}%" if ret is not None else "n/a"
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85,
                reason=f"{sym} 90d {ret_str} no longer beats SPY+{RELATIVE_STRENGTH_BPS/100:.0f}%",
                is_closing=True,
            ))

        # Entries: target set members not currently held.
        # Sizing: respect ctx.target_alloc_usd (allocator's verdict)
        # divided across the slots, capped by TRADE_SIZE_USD which
        # serves as a per-position max. Vol-managed overlay scaler
        # (Moreira-Muir 2017) multiplies the per-slot size.
        from ._helpers import vol_scaler
        overlay = vol_scaler(ctx, "equity_momentum", 1.0)
        slots = max(1, TOP_N)
        per_slot_alloc = (ctx.target_alloc_usd * overlay) / slots
        per_slot = min(per_slot_alloc, TRADE_SIZE_USD)
        for sym, ret in winners:
            if sym in held:
                continue
            spread = ret - us_return
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_slot, confidence=0.75,
                reason=f"{sym} {ret:+.1f}% (vs SPY {us_return:+.1f}%, "
                       f"spread +{spread:.1f}%)",
                metadata={"return_pct": ret, "spread_pct": spread},
            ))
        return proposals

    def _lookback_return_pct(self, symbol: str, days: int) -> float | None:
        from ._helpers import lookback_return_pct
        return lookback_return_pct(self.broker, self.name, symbol, days)

    def _past_cooldown(self, pos: dict) -> bool:
        from ._helpers import past_cooldown
        return past_cooldown(pos, COOLDOWN_DAYS)
