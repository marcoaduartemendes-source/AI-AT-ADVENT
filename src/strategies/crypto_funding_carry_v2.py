"""crypto_funding_carry_v2 — multi-venue consensus on funding spikes.

Difference from v1 (crypto_funding_carry):
    v1 fires on Coinbase funding APR > 5% threshold. Single-venue
    signal — vulnerable to Coinbase-specific glitches (low-volume
    flash spikes, exchange maintenance windows that distort the
    funding calc).

    v2 ALSO requires Binance to confirm. The CryptoScout publishes a
    "cross_venue_funding" signal with venues_agree=True only when:
      • Coinbase APR > 100bps (1% — same threshold as v1's exit)
      • Binance APR > 100bps
      • |Coinbase APR - Binance APR| <= 200bps (2% drift window)
    Plus v1's existing entry threshold of 500bps APR on the funding
    rates signal.

    Result: false-positive rate drops materially (any single-venue
    spike that isn't market-wide is filtered out). When both venues
    agree the funding is hot, the carry trade is more reliable.

    Catch: misses true Coinbase-only opportunities. Acceptable —
    we have v1 still running for those, and v1's per-trade cap is
    smaller specifically because it's higher-noise.

Same execution mechanics as v1: long spot, short perp at half-and-
half allocation, close when funding decays below exit threshold.

Failure modes:
    - cross_venue_funding signal absent → no proposals (v1 may still
      fire on its own funding_rates signal)
    - venues_agree=False → skip; lets v1 handle if its own signal
      qualifies, but with smaller size
    - Binance API unreachable → scout marks venues_agree=False; v2
      skips entirely. v1 keeps running.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategies.crypto_funding_carry import DEFAULT_PAIRS
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Same as v1 — 5% APR entry, 1% APR exit. Only the *gate* changes.
ENTRY_APR_BPS = 500
EXIT_APR_BPS = 100

# When both venues agree, we run a tighter book size than v1 because
# we expect higher fill rates and lower noise. Per-leg notional
# scales with strategy alloc — half goes to spot leg, half to perp.
# v1 used target_alloc_usd / 2; v2 uses target_alloc_usd / 2 too,
# but the strategy itself is allocated less by the meta-allocator
# until it earns its place.


class CryptoFundingCarryV2(Strategy):
    name = "crypto_funding_carry_v2"
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

        # Both signals must be present. funding_rates carries Coinbase
        # APRs; cross_venue_funding carries the agreement flag.
        funding_signals = ctx.scout_signals.get("funding_rates", {}) or {}
        cross_venue = ctx.scout_signals.get("cross_venue_funding", []) or []
        if not funding_signals:
            logger.debug(f"[{self.name}] no funding signals")
            return []

        # Build a {spot_symbol: agree_bool} lookup
        agree_by_spot: dict[str, bool] = {}
        for row in cross_venue:
            sym = row.get("symbol")
            if sym:
                agree_by_spot[sym] = bool(row.get("agree", False))

        proposals: list[TradeProposal] = []
        per_leg_usd = ctx.target_alloc_usd / 2

        for spot_sym, perp_sym in self.pairs:
            apr_bps = funding_signals.get(perp_sym, {}).get("apr_bps")
            if apr_bps is None:
                continue
            agree = agree_by_spot.get(spot_sym, False)

            spot_pos = ctx.open_positions.get(spot_sym, {})
            perp_pos = ctx.open_positions.get(perp_sym, {})
            in_position = (spot_pos.get("quantity", 0) > 0 or
                           perp_pos.get("quantity", 0) > 0)

            # Entry: BOTH funding > entry threshold AND venues agree
            if (not in_position and abs(apr_bps) >= self.entry_apr_bps
                    and agree):
                spot_side = OrderSide.BUY if apr_bps > 0 else OrderSide.SELL
                perp_side = OrderSide.SELL if apr_bps > 0 else OrderSide.BUY
                reason = (f"funding APR {apr_bps:.0f}bps + venues_agree "
                          f"— open consensus carry")
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=spot_sym,
                    side=spot_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg_usd, confidence=0.92, reason=reason,
                    metadata={"leg": "spot", "apr_bps": apr_bps,
                              "venues_agree": agree},
                ))
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=perp_sym,
                    side=perp_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg_usd, confidence=0.92, reason=reason,
                    metadata={"leg": "perp", "apr_bps": apr_bps,
                              "venues_agree": agree},
                ))
            # Exit: funding decayed OR venues stop agreeing (regime change)
            elif in_position and (
                    abs(apr_bps) < self.exit_apr_bps or not agree):
                exit_reason = (
                    f"funding APR {apr_bps:.0f}bps — close carry"
                    if abs(apr_bps) < self.exit_apr_bps
                    else "venues stopped agreeing — close carry"
                )
                if spot_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=spot_sym,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=spot_pos["quantity"],
                        confidence=0.95, reason=exit_reason, is_closing=True,
                        metadata={"leg": "spot"},
                    ))
                if perp_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=perp_sym,
                        side=OrderSide.BUY, order_type=OrderType.MARKET,
                        quantity=perp_pos["quantity"],
                        confidence=0.95, reason=exit_reason, is_closing=True,
                        metadata={"leg": "perp"},
                    ))

        return proposals
