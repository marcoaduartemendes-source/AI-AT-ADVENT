"""cross_venue_arb — Kalshi vs Polymarket cross-venue arbitrage.

Difference from kalshi_calibration_arb (intra-Kalshi favorite-longshot
fade) and macro_kalshi_v2 (Kalshi vs CME Fed Funds futures):
    cross_venue_arb trades the divergence between two LIQUID PREDICTION
    MARKETS pricing the same underlying event. When Kalshi YES is at
    0.62 and Polymarket YES is at 0.45 on "Will the Fed cut rates in
    June 2026?", one of those venues is wrong. Buy the cheap side on
    Kalshi (the only side we can actually trade today — Polymarket is
    on-chain and we don't have wallet integration).

The edge thesis:
    Cross-venue divergences between liquid prediction markets close
    quickly via professional arbitrageurs. By the time we see ≥5¢
    divergence, the slower-moving side is mispriced and reverts to
    fair value (the fair value being the volume-weighted average of
    the two venues, anchored by the more-liquid side).

    We deliberately bet only on the Kalshi side — we read both
    venues, write only on Kalshi. A future iteration can add on-chain
    Polymarket execution for true two-sided arb.

Data path:
    PredictionScout publishes signal_type="cross_venue_arb" on
    venue="kalshi" with payload [{kalshi_ticker, kalshi_yes,
    polymarket_yes, polymarket_volume_24h, divergence, ...}, …].
    The scout already filters to ≥5¢ divergence and ranks by
    abs(divergence). We just need to size + place the trade.

Sizing: like kalshi_calibration_arb's Kelly × 0.25, but with a
TIGHTER per-trade cap because cross-venue arbs are rarer and we
want to discover correlation properties before letting them dominate.

Failure modes:
    - Polymarket volume too low → scout already filters at $1k/24h;
      strategy adds a defensive 2nd check.
    - Same direction as v1 calibration arb on the same ticker → that's
      fine, both fire the same trade. Risk gate dedupes via the
      pending-orders cache.
    - Kalshi yes_price moved since scout snapshot → limit order at
      original price; if the market moved, the order rests until
      filled or cancelled by next cycle.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Scout filters at 5¢ — strategy uses same threshold for clarity.
# Smaller divergences would be unprofitable after the 5% Kalshi fee.
ENTRY_DIVERGENCE = 0.05

# Polymarket volume floor — sub-$1k/24h means the cheap side is
# probably stale, not informationally cheap. Scout already enforces
# but we double-check.
MIN_POLYMARKET_VOLUME_USD = 5_000.0

# Kelly fraction — conservative because cross-venue arb is correlated
# with v1 calibration fade.
KELLY_FRACTION = 0.20

# Per-trade cap — tighter than v1/v2 (10%) until we have correlation
# data. Champion-tier auto-promotion will lift it after Sharpe ≥ 1.0.
MAX_PER_TRADE_PCT = 0.05


class CrossVenueArb(Strategy):
    name = "cross_venue_arb"
    venue = "kalshi"

    def __init__(self, broker,
                  entry_divergence: float = ENTRY_DIVERGENCE,
                  kelly_fraction: float = KELLY_FRACTION,
                  max_per_trade_pct: float = MAX_PER_TRADE_PCT,
                  min_polymarket_volume_usd: float = MIN_POLYMARKET_VOLUME_USD):
        super().__init__(broker)
        self.entry_divergence = entry_divergence
        self.kelly_fraction = kelly_fraction
        self.max_per_trade_pct = max_per_trade_pct
        self.min_polymarket_volume_usd = min_polymarket_volume_usd

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        candidates: list[dict] = ctx.scout_signals.get(
            "cross_venue_arb", []
        ) or []
        if not candidates:
            logger.debug(f"[{self.name}] no cross-venue arb signals")
            return []

        proposals: list[TradeProposal] = []
        per_trade_cap_usd = ctx.target_alloc_usd * self.max_per_trade_pct

        for c in candidates:
            ticker = c.get("kalshi_ticker")
            kalshi_yes = c.get("kalshi_yes")
            polymarket_yes = c.get("polymarket_yes")
            pm_volume = c.get("polymarket_volume_24h", 0.0) or 0.0
            divergence = c.get("divergence")
            if (ticker is None or kalshi_yes is None
                    or polymarket_yes is None or divergence is None):
                continue
            if abs(divergence) < self.entry_divergence:
                continue
            if pm_volume < self.min_polymarket_volume_usd:
                continue
            if kalshi_yes <= 0 or kalshi_yes >= 1:
                continue

            # divergence = polymarket_yes - kalshi_yes
            #   +ve → Kalshi YES is cheap relative to Polymarket → BUY YES
            #   -ve → Kalshi YES is rich → BUY NO (= sell YES)
            if divergence > 0:
                # Polymarket says fair value > kalshi_yes → Kalshi too cheap
                # Fair value approximation: vol-weighted-mean lifted toward
                # the higher-volume Polymarket side. We lean toward
                # Polymarket since it's the deeper market.
                fair_value = polymarket_yes * 0.7 + kalshi_yes * 0.3
                p, cost, side = fair_value, kalshi_yes, OrderSide.BUY
            else:
                fair_value = polymarket_yes * 0.7 + kalshi_yes * 0.3
                # Buying NO at 1 - kalshi_yes
                p = 1 - fair_value
                cost = 1 - kalshi_yes
                side = OrderSide.SELL

            if cost <= 0:
                continue
            b = (1 - cost) / cost
            full_kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
            kelly = full_kelly * self.kelly_fraction
            position_usd = min(ctx.target_alloc_usd * kelly, per_trade_cap_usd)
            if position_usd < 1.0:
                continue
            n_contracts = int(position_usd / cost)
            if n_contracts < 1:
                continue

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=ticker,
                side=side, order_type=OrderType.LIMIT,
                quantity=float(n_contracts), limit_price=kalshi_yes,
                confidence=min(0.95, abs(divergence) * 5),
                reason=(
                    f"kalshi {kalshi_yes:.2f} vs polymarket "
                    f"{polymarket_yes:.2f} (Δ={divergence:+.2f}); "
                    f"pm_vol=${pm_volume:.0f}"
                ),
                metadata={
                    "kalshi_yes": kalshi_yes,
                    "polymarket_yes": polymarket_yes,
                    "polymarket_volume_24h": pm_volume,
                    "divergence": divergence,
                    "fair_value": fair_value,
                    "polymarket_question": c.get("polymarket_question"),
                },
            ))
        return proposals
