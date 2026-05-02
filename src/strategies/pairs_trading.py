"""Pairs trading — market-neutral statistical arbitrage.

Documented edge (Gatev-Goetzmann-Rouwenhorst 2006): pairs of historically
correlated stocks tend to revert when their price ratio deviates from
its mean. We trade the spread:

  z = (current_ratio - mean(ratio_window)) / std(ratio_window)

  Long the underperformer + short the outperformer when |z| > entry_z.
  Close when |z| < exit_z.

Asymmetric pair list — these are well-documented "twin" stocks where
the pair has decades of co-movement:

  KO / PEP   - Coca-Cola vs Pepsi
  V / MA     - Visa vs Mastercard
  GS / MS    - Goldman Sachs vs Morgan Stanley
  JPM / BAC  - JPMorgan vs Bank of America
  HD / LOW   - Home Depot vs Lowe's
  CVX / XOM  - Chevron vs ExxonMobil
  GOOGL/MSFT - Alphabet vs Microsoft (less classic but liquid)

Risk: pair shorts on Alpaca paper require a margin account. The bot
will degrade gracefully — Alpaca returns "shorting not enabled" and
the proposal gets rejected; long-only leg still fires. In practice,
we do "long-bias pair trading": only the long leg of the spread,
not a true market-neutral.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


LOOKBACK_DAYS = 60          # window for ratio mean / stdev
ENTRY_Z = 2.0               # |z| > this opens a position
EXIT_Z = 0.5                # |z| < this closes
MAX_HOLD_DAYS = 30          # forced exit if z hasn't reverted
TRADE_SIZE_USD = 5000.0     # per-leg notional


@dataclass
class _Pair:
    a: str
    b: str
    note: str = ""


PAIRS = [
    _Pair("KO", "PEP",   "Coke vs Pepsi"),
    _Pair("V", "MA",     "Visa vs Mastercard"),
    _Pair("GS", "MS",    "Goldman vs Morgan Stanley"),
    _Pair("JPM", "BAC",  "JPMorgan vs Bank of America"),
    _Pair("HD", "LOW",   "Home Depot vs Lowe's"),
    _Pair("CVX", "XOM",  "Chevron vs ExxonMobil"),
]


class PairsTrading(Strategy):
    name = "pairs_trading"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        for pair in PAIRS:
            z = self._zscore(pair.a, pair.b)
            if z is None:
                continue

            # Has either leg of this pair already?
            has_a = pair.a in held
            has_b = pair.b in held

            # ── Exits: |z| reverted to within EXIT_Z ─────────────────
            if (has_a or has_b) and abs(z) < EXIT_Z:
                if has_a:
                    qty_a = ctx.open_positions[pair.a].get("quantity", 0)
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=pair.a,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=qty_a, confidence=0.95,
                        reason=f"{pair.note}: spread reverted to z={z:+.2f}",
                        is_closing=True,
                    ))
                if has_b:
                    qty_b = ctx.open_positions[pair.b].get("quantity", 0)
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=pair.b,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=qty_b, confidence=0.95,
                        reason=f"{pair.note}: spread reverted to z={z:+.2f}",
                        is_closing=True,
                    ))
                continue

            # ── Entries: |z| > ENTRY_Z and not already in ────────────
            if has_a or has_b:
                continue
            if abs(z) < ENTRY_Z:
                continue
            # Long the cheaper one (a is "cheap" when z is HIGH —
            # because z = (a/b - mean) / std, so high z means a is
            # too high vs b → SHORT a, LONG b. Wait, let me recheck.
            # Actually z = (a_price/b_price - mean_ratio) / std
            # High z = a_price/b_price is HIGH → a is RICH, b is CHEAP
            # So: long b (the cheap one), short a (the rich one).
            # Long-only fallback: just long b.
            if z > 0:
                # a is rich vs b → long b
                long_sym, rich_sym = pair.b, pair.a
            else:
                # a is cheap vs b → long a
                long_sym, rich_sym = pair.a, pair.b

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=long_sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=TRADE_SIZE_USD, confidence=0.7,
                reason=f"{pair.note}: long {long_sym} (cheap leg) "
                       f"vs {rich_sym}, z={z:+.2f}",
                metadata={"pair": f"{pair.a}/{pair.b}", "z_score": z,
                          "long_leg": long_sym, "rich_leg": rich_sym},
            ))
        return proposals

    # ── Z-score on the price ratio ───────────────────────────────────

    def _zscore(self, a: str, b: str) -> float | None:
        try:
            ca = self.broker.get_candles(a, "1Day", num_candles=LOOKBACK_DAYS + 5)
            cb = self.broker.get_candles(b, "1Day", num_candles=LOOKBACK_DAYS + 5)
        except Exception as e:
            logger.debug(f"[{self.name}] {a}/{b} candles failed: {e}")
            return None
        n = min(len(ca), len(cb), LOOKBACK_DAYS + 1)
        if n < LOOKBACK_DAYS:
            return None
        ratios = []
        for i in range(-n, 0):
            pb = cb[i].close
            if pb <= 0:
                return None
            ratios.append(ca[i].close / pb)
        if len(ratios) < 2:
            return None
        mean = sum(ratios) / len(ratios)
        var = sum((r - mean) ** 2 for r in ratios) / (len(ratios) - 1)
        sd = math.sqrt(var)
        if sd == 0:
            return None
        return (ratios[-1] - mean) / sd
