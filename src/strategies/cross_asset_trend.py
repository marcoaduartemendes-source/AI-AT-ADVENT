"""Cross-asset trend (CTA-style) — diversification away from equity beta.

USER MANDATE (2026-05-20)
"Think as an obsessed performance-driven Wall Street hedge fund
manager constantly looking for alpha."

WHY THIS STRATEGY
The book is dominated by equity-beta strategies: multifactor,
sector, dividend, thematic, leveraged, intraday, pairs, low-vol,
tsmom_etf, risk_parity_etf, gap, turn-of-month. Every one of
those rises and falls with the S&P. A 2008/2020/2022 SPX
drawdown drains everything at once.

CTAs (commodity trading advisors) like AQR's Managed Futures and
Man AHL's Diversified harvest trend across MULTIPLE asset classes
— bonds, currencies, commodities, energy — and historically have
LOW or NEGATIVE correlation to equities. Sharpes are modest (0.5-
0.8) but the diversification is real: 2022 was the worst US 60/40
year in 50+ years AND CTAs had their best year of the decade.

UNIVERSE — five liquid, deep ETFs spanning four asset classes:
  TLT   Long US Treasuries          (duration / bonds)
  GLD   Gold                         (safe-haven / inflation)
  DBC   Broad commodities            (commodity beta)
  USO   Crude oil                    (energy)
  UUP   US dollar bullish            (FX)

SIGNAL — classic 12-1 month time-series momentum (Moskowitz, Ooi,
Pedersen 2012). Long an asset if its 12m total return (skipping
the most recent 21 days to avoid 1m mean-reversion) is positive.
Long-only — short-the-loser sleeve adds whipsaw without much
edge for retail-grade execution.

POSITION SIZING — equal-weight across eligible names, vol-scaled
by the sleeve's overall vol overlay. Rebalances monthly (21 bars
of cooldown). Low turnover by design — keeps fee bleed under 10bps
round-trip.

LIVE PROMOTION
Registered DRY at 3% target / 6% cap. Stays paper until validation
PASS, walk-forward ROBUST, AND 30+ days of paper Sharpe ≥ 0.3.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


UNIVERSE = ["TLT", "GLD", "DBC", "USO", "UUP"]

MOM_LOOKBACK = 252
MOM_SKIP = 21
REBALANCE_COOLDOWN_DAYS = 21
MIN_NAMES = 1


class CrossAssetTrend(Strategy):
    name = "cross_asset_trend"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        eligible: list[tuple[str, float]] = []  # (sym, 12-1 momentum)
        for sym in UNIVERSE:
            try:
                candles = self.broker.get_candles(
                    sym, "ONE_DAY",
                    num_candles=MOM_LOOKBACK + MOM_SKIP + 10)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {sym}: {e}")
                continue
            if len(candles) < MOM_LOOKBACK + MOM_SKIP:
                continue
            closes = np.array([c.close for c in candles], dtype=float)
            if closes[-1] <= 0:
                continue
            p_start = closes[-(MOM_LOOKBACK + MOM_SKIP)]
            p_end = closes[-(MOM_SKIP + 1)]
            if p_start <= 0:
                continue
            mom = p_end / p_start - 1.0
            if mom > 0:
                eligible.append((sym, float(mom)))

        if len(eligible) < MIN_NAMES:
            logger.info(f"[{self.name}] no positive-trend assets; "
                        f"sitting flat (cash is a position)")
            return self._maybe_close_exits(ctx, set())

        target_set = {sym for sym, _ in eligible}
        per_name_usd = sleeve_usd / len(eligible)
        open_pos = ctx.open_positions or {}
        held = {
            s for s, p in open_pos.items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }

        proposals = self._maybe_close_exits(ctx, target_set)
        for sym, mom in eligible:
            if sym in held:
                continue
            pos = open_pos.get(sym, {})
            if pos and not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name_usd,
                confidence=min(0.6 + abs(mom) * 0.4, 0.9),
                is_closing=False,
                reason=f"12-1m TSMOM {mom*100:+.1f}% — positive trend",
                metadata={"model": "cross_asset_trend",
                          "asset_class": "diversified",
                          "mom_12_1": round(mom, 4)},
            ))
        return proposals

    def _maybe_close_exits(self, ctx: StrategyContext,
                            target_set: set[str]
                            ) -> list[TradeProposal]:
        """Close held names that are no longer in the trend-positive set."""
        from ._helpers import past_cooldown
        proposals: list[TradeProposal] = []
        for sym, pos in (ctx.open_positions or {}).items():
            if sym not in UNIVERSE:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            if sym in target_set:
                continue
            if not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85, is_closing=True,
                reason="12-1m trend turned negative — exit",
                metadata={"model": "cross_asset_trend", "leg": "exit"},
            ))
        return proposals
