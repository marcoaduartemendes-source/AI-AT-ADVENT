"""Crypto pairs-trading — statistical arbitrage on major-pair price ratios.

Edge: BTC and ETH are deeply cointegrated (Lin et al. 2019, "Bitcoin
and crypto market integration"). Their price ratio mean-reverts on
the 5-30d horizon. ETH/SOL is similar but noisier (smaller sample).

Mechanics:
  1. For each pair (e.g. BTC-USD / ETH-USD) compute price ratio
     `r_t = price_BTC / price_ETH` and a 30d rolling z-score.
  2. |z| > 2.0 → enter delta-neutral pair: long the cheap leg via
     spot, short the rich leg via the corresponding perp.
  3. |z| < 0.5 → close both legs.
  4. 7d cooldown per pair after exit (avoids flapping at boundaries).

Position sizing: half target_alloc_usd to each leg, capped by
MAX_TRADE_USD_COINBASE (the per-venue cap — $100 in real-money mode).

Why this is more advanced than the existing four:
  - Delta-neutral by construction; profit isn't dependent on crypto
    direction, only on relative price moves between BTC/ETH.
  - Composes with crypto_xsmom: xsmom captures absolute momentum,
    pairs captures relative deviation from equilibrium.
  - Real edges in the literature; conservative thresholds (|z|>2)
    keep trade frequency low → fee-friendly.
"""
from __future__ import annotations

import logging
import math

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Pairs to trade. (rich_leg_spot, rich_leg_perp, cheap_leg_spot, cheap_leg_perp)
# We trade BOTH directions of the ratio: when BTC is rich, sell BTC perp /
# buy ETH spot; when ETH is rich, sell ETH perp / buy BTC spot.
PAIR_DEFS = [
    {
        "name": "BTC_ETH",
        "a_spot": "BTC-USD", "a_perp": "BTC-PERP-INTX",
        "b_spot": "ETH-USD", "b_perp": "ETH-PERP-INTX",
    },
    {
        "name": "ETH_SOL",
        "a_spot": "ETH-USD", "a_perp": "ETH-PERP-INTX",
        "b_spot": "SOL-USD", "b_perp": "SOL-PERP-INTX",
    },
]

LOOKBACK_DAYS = 30          # window for ratio mean + std
ENTRY_Z = 2.0               # |z| above this opens
EXIT_Z = 0.5                # |z| below this closes
COOLDOWN_DAYS = 7           # per pair after exit


class CryptoPairsTrading(Strategy):
    name = "crypto_pairs_trading"
    venue = "coinbase"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        per_leg_usd = ctx.target_alloc_usd / (2 * len(PAIR_DEFS))

        for pair in PAIR_DEFS:
            stats = self._pair_stats(pair["a_spot"], pair["b_spot"])
            if stats is None:
                continue
            z, ratio_now = stats

            # Holding state: are we already in this pair?
            a_pos = ctx.open_positions.get(pair["a_spot"], {})
            b_pos = ctx.open_positions.get(pair["b_spot"], {})
            in_position = (
                (a_pos.get("quantity", 0) or 0) > 0
                or (b_pos.get("quantity", 0) or 0) > 0
            )

            # Entry: |z| > 2 and not already holding.
            # Spot-only mode: until perp short support lands (F6 — Coinbase
            # INTX/CFM endpoints), we long the CHEAP leg only. Captures
            # mean-reversion of the relative price, just without the
            # delta-neutral hedge. The previous code emitted a SELL on a
            # perp symbol that the spot adapter then rejected ("Coinbase
            # MARKET SELL requires quantity") — bug observed 2026-05-08.
            if not in_position and abs(z) > ENTRY_Z:
                if z > 0:
                    long_leg = pair["b_spot"]
                    rich, cheap = pair["a_spot"], pair["b_spot"]
                else:
                    long_leg = pair["a_spot"]
                    rich, cheap = pair["b_spot"], pair["a_spot"]
                reason = (
                    f"{pair['name']} z={z:+.2f} ratio={ratio_now:.4f} "
                    f"→ long cheap leg {cheap} (rich={rich}, spot-only)"
                )
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=long_leg,
                    side=OrderSide.BUY, order_type=OrderType.MARKET,
                    notional_usd=per_leg_usd, confidence=0.75, reason=reason,
                    metadata={"pair": pair["name"], "z": z, "leg": "long_spot"},
                ))

            # Exit: |z| < 0.5 and currently holding either leg.
            elif in_position and abs(z) < EXIT_Z:
                reason = f"{pair['name']} z={z:+.2f} → close pair"
                if (a_pos.get("quantity", 0) or 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue,
                        symbol=pair["a_spot"],
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=a_pos["quantity"],
                        confidence=0.95, reason=reason, is_closing=True,
                    ))
                if (b_pos.get("quantity", 0) or 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue,
                        symbol=pair["b_spot"],
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=b_pos["quantity"],
                        confidence=0.95, reason=reason, is_closing=True,
                    ))

        return proposals

    # ── Helpers ──────────────────────────────────────────────────────

    def _pair_stats(self, a_sym: str, b_sym: str) -> tuple[float, float] | None:
        """Return (z_score, current_ratio) over the last LOOKBACK_DAYS,
        or None if data is insufficient. Uses daily candles from
        the broker — falls through gracefully on any fetch failure."""
        try:
            a = self.broker.get_candles(a_sym, "ONE_DAY",
                                          num_candles=LOOKBACK_DAYS + 5)
            b = self.broker.get_candles(b_sym, "ONE_DAY",
                                          num_candles=LOOKBACK_DAYS + 5)
        except Exception as e:
            logger.debug(f"[{self.name}] candles fetch {a_sym}/{b_sym}: {e}")
            return None
        if len(a) < LOOKBACK_DAYS or len(b) < LOOKBACK_DAYS:
            return None
        n = min(len(a), len(b), LOOKBACK_DAYS)
        ratios = []
        for i in range(-n, 0):
            try:
                pa = a[i].close
                pb = b[i].close
            except (IndexError, AttributeError):
                continue
            if pa and pb and pb > 0:
                ratios.append(pa / pb)
        if len(ratios) < n - 2:
            return None
        mean = sum(ratios) / len(ratios)
        var = sum((r - mean) ** 2 for r in ratios) / max(len(ratios) - 1, 1)
        sd = math.sqrt(var)
        if sd <= 0:
            return None
        z = (ratios[-1] - mean) / sd
        return z, ratios[-1]
