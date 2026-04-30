"""Kalshi calibration arbitrage — favorite-longshot bias fader.

Research bottom-line: Kalshi market prices are not unbiased probability
estimates. Longshots (e.g. 5–15% YES) tend to be systematically overpriced;
heavy favorites (85–95%) systematically underpriced. Snowberg & Wolfers
(2010) document this across betting markets; Polymarket calibration
studies (Said, 2024) confirm modest mispricing in the 60–85% bucket.

Algorithm:
    1. Pull every open market via the Kalshi adapter.
    2. For each, compute a recalibrated fair-value probability using a
       per-category isotonic regression curve (built offline from
       resolution history; for now we use a fixed table from published
       calibration studies).
    3. If |market price − fair value| > entry threshold AND market has
       enough liquidity (open interest > $500), submit a small position.
    4. Position size scales with edge: edge_cents × kelly_fraction × alloc.
    5. Hold to expiry (no early-exit logic in V1; binary settlement).

V1 stub: emits proposals only when ctx.scout_signals['mispriced'] flags
markets — wired in W2 with the prediction-market scout. This file ships
the Kelly-sizing scaffold and proposal generation.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Default per-category recalibration table.
# Each row: (market_price_bucket_lo, market_price_bucket_hi, fair_value_offset)
# A negative offset means "market is overpricing YES" — fade by selling YES.
# Source: Snowberg & Wolfers (2010), Said (2024). Replace with isotonic
# regression on Kalshi-specific resolution history once the scout pulls it.
DEFAULT_RECALIBRATION = [
    # (market_lo, market_hi, fair_value_shift_cents)
    (0.02, 0.10, -0.020),   # heavy longshots overpriced ~2 cents
    (0.10, 0.20, -0.015),
    (0.20, 0.40, -0.005),
    (0.40, 0.60,  0.000),   # midfield is well-calibrated
    (0.60, 0.80, +0.010),
    (0.80, 0.90, +0.015),
    (0.90, 0.98, +0.020),   # heavy favorites underpriced ~2 cents
]

# Trade only if mispricing exceeds this in cents.
ENTRY_EDGE_CENTS = 3        # 3¢ on a $1 contract = 3% gross edge

# Kelly fraction — bet only this proportion of full Kelly to manage variance.
KELLY_FRACTION = 0.25

# Per-trade ceiling as % of strategy alloc.
MAX_PER_TRADE_PCT = 0.10

# Minimum open interest in dollars to consider a market liquid enough.
MIN_OPEN_INTEREST_USD = 500


class KalshiCalibrationArb(Strategy):
    name = "kalshi_calibration_arb"
    venue = "kalshi"

    def __init__(self, broker, recalibration=None,
                  entry_edge_cents: float = ENTRY_EDGE_CENTS,
                  kelly_fraction: float = KELLY_FRACTION,
                  max_per_trade_pct: float = MAX_PER_TRADE_PCT):
        super().__init__(broker)
        self.recalibration = recalibration or DEFAULT_RECALIBRATION
        self.entry_edge_cents = entry_edge_cents
        self.kelly_fraction = kelly_fraction
        self.max_per_trade_pct = max_per_trade_pct

    # ── Public ----------------------------------------------------------

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Mispriced-market list comes from the prediction-market scout (W2).
        candidates: List[Dict] = ctx.scout_signals.get("mispriced", [])
        if not candidates:
            logger.debug(f"[{self.name}] no scout candidates this cycle")
            return []

        proposals: List[TradeProposal] = []
        per_trade_cap_usd = ctx.target_alloc_usd * self.max_per_trade_pct

        for market in candidates:
            ticker = market.get("ticker")
            yes_price = market.get("yes_price")
            open_interest_usd = market.get("open_interest_usd", 0)
            if not ticker or yes_price is None:
                continue
            if open_interest_usd < MIN_OPEN_INTEREST_USD:
                continue

            fair_value = self._fair_value(yes_price)
            edge = fair_value - yes_price                # positive = buy YES
            edge_cents = edge * 100

            if abs(edge_cents) < self.entry_edge_cents:
                continue

            # Kelly-fraction sizing on a binary contract:
            #   p = fair_value, b = (payoff - cost) / cost
            # For a YES contract bought at yes_price, payoff if YES is $1:
            #   b = (1 - yes_price) / yes_price
            # Kelly fraction: f* = p - (1 - p) / b
            if edge > 0:
                # Buying YES at yes_price; payoff = $1 if YES
                p = fair_value
                cost = yes_price
                b = (1 - cost) / cost if cost > 0 else 0
                full_kelly = p - (1 - p) / b if b > 0 else 0
                side = OrderSide.BUY
            else:
                # Buying NO at (1 - yes_price); equivalent to selling YES
                p = 1 - fair_value
                cost = 1 - yes_price
                b = (1 - cost) / cost if cost > 0 else 0
                full_kelly = p - (1 - p) / b if b > 0 else 0
                side = OrderSide.SELL

            kelly = max(0.0, full_kelly) * self.kelly_fraction
            position_usd = min(ctx.target_alloc_usd * kelly, per_trade_cap_usd)
            if position_usd < 1.0:
                continue
            # Convert USD to integer YES contracts (each costs `cost` dollars)
            n_contracts = int(position_usd / cost)
            if n_contracts < 1:
                continue

            proposals.append(TradeProposal(
                strategy=self.name,
                venue=self.venue,
                symbol=ticker,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=float(n_contracts),
                limit_price=yes_price,
                confidence=min(0.95, abs(edge) * 5),     # 20¢ edge → 1.0
                expected_return_pct=edge / cost * 100 if cost else None,
                reason=(f"mispriced: market={yes_price:.3f}, fair={fair_value:.3f}, "
                        f"edge={edge_cents:+.1f}¢, kelly={kelly:.3f}"),
                metadata={"category": market.get("category"),
                          "open_interest_usd": open_interest_usd},
            ))

        return proposals

    # ── Helpers ---------------------------------------------------------

    def _fair_value(self, market_price: float) -> float:
        """Apply the recalibration table to translate a market price into
        an estimate of the true probability."""
        for lo, hi, shift in self.recalibration:
            if lo <= market_price < hi:
                return max(0.0, min(1.0, market_price + shift))
        return market_price
