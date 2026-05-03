"""Crypto funding-rate carry — Coinbase US perpetuals.

Top-ranked strategy in our research: long spot + short perp (or vice versa)
captures the funding payment that flows from longs to shorts twice daily.
Coinbase US perp funding has averaged ~5–11% APR in 2024–25.

Algorithm:
    1. For each (spot, perp) pair, fetch latest funding rate.
    2. If funding > +threshold (longs pay shorts):
         long spot + short perp → collect funding, delta-neutral.
       If funding < -threshold (shorts pay longs):
         short spot + long perp → collect funding.
    3. Size each leg to half the strategy's target_alloc_usd.
    4. Rebalance every funding period (~8h on Coinbase US perps).

V1 stub: emits proposals only when ctx.scout_signals reports a funding rate
extreme. Wiring the actual funding-rate data feed lands in W2 with the
crypto scout. This file ships the scaffolding so the orchestrator + risk
gating can be exercised end-to-end.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Pairs we'll trade. (spot_symbol, perp_symbol)
#
# Sprint B1 audit fix: the previous tickers (BIT/ETP/SLP-PERP-INTX)
# were placeholder strings that don't exist on Coinbase International
# Exchange — orders against them would be rejected immediately. The
# real Coinbase Intl perp tickers preserve the spot symbol prefix:
#   BTC-PERP-INTX  ETH-PERP-INTX  SOL-PERP-INTX
# Source: https://api.coinbase.com/api/v3/brokerage/market/products
# filtered to product_type=FUTURE & contract_expiry_type=PERPETUAL.
DEFAULT_PAIRS = [
    ("BTC-USD", "BTC-PERP-INTX"),
    ("ETH-USD", "ETH-PERP-INTX"),
    ("SOL-USD", "SOL-PERP-INTX"),
]

# Annualized funding-rate thresholds for entry / exit.
# Funding rate is published per 8h period; annualize with 3 × 365 = 1095.
ENTRY_APR_BPS = 500    # 5% APR
EXIT_APR_BPS = 100     # 1% APR


class CryptoFundingCarry(Strategy):
    name = "crypto_funding_carry"
    venue = "coinbase"

    def __init__(self, broker, pairs=None,
                  entry_apr_bps: int = ENTRY_APR_BPS,
                  exit_apr_bps: int = EXIT_APR_BPS):
        super().__init__(broker)
        self.pairs = pairs or DEFAULT_PAIRS
        self.entry_apr_bps = entry_apr_bps
        self.exit_apr_bps = exit_apr_bps

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Funding rates come from the crypto scout (W2). Until that's wired,
        # the strategy emits no proposals — empty list is safe behavior.
        funding_signals = ctx.scout_signals.get("funding_rates", {})
        if not funding_signals:
            logger.debug(f"[{self.name}] no funding-rate signals from scout yet")
            return []

        proposals: list[TradeProposal] = []
        per_leg_usd = ctx.target_alloc_usd / 2  # half spot, half perp

        for spot_sym, perp_sym in self.pairs:
            apr_bps = funding_signals.get(perp_sym, {}).get("apr_bps")
            if apr_bps is None:
                continue

            spot_pos = ctx.open_positions.get(spot_sym, {})
            perp_pos = ctx.open_positions.get(perp_sym, {})
            in_position = (spot_pos.get("quantity", 0) > 0 or
                           perp_pos.get("quantity", 0) > 0)

            if not in_position and abs(apr_bps) >= self.entry_apr_bps:
                # Open carry: positive APR → long spot, short perp
                #             negative APR → short spot, long perp
                spot_side = OrderSide.BUY if apr_bps > 0 else OrderSide.SELL
                perp_side = OrderSide.SELL if apr_bps > 0 else OrderSide.BUY
                reason = f"funding APR {apr_bps:.0f}bps — open carry"
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=spot_sym,
                    side=spot_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg_usd, confidence=0.85, reason=reason,
                    metadata={"leg": "spot", "apr_bps": apr_bps},
                ))
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=perp_sym,
                    side=perp_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg_usd, confidence=0.85, reason=reason,
                    metadata={"leg": "perp", "apr_bps": apr_bps},
                ))
            elif in_position and abs(apr_bps) < self.exit_apr_bps:
                # Close: funding has decayed below exit threshold
                reason = f"funding APR {apr_bps:.0f}bps — close carry"
                if spot_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=spot_sym,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=spot_pos["quantity"],
                        confidence=0.95, reason=reason, is_closing=True,
                        metadata={"leg": "spot"},
                    ))
                if perp_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=perp_sym,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=perp_pos["quantity"],
                        confidence=0.95, reason=reason, is_closing=True,
                        metadata={"leg": "perp"},
                    ))

        return proposals
