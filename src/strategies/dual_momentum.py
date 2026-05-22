"""Dual Momentum — Antonacci-style tactical allocation with a bond hedge.

The book's bench is heavy on long-only equity-beta momentum (tsmom,
sector_rotation, multifactor), all of which are ~correlated to SPX in a
bull market. This sleeve is the genuine *crisis-alpha diversifier* that
was missing: it concentrates in the strongest risk assets when trends are
up, but rotates the whole book into Treasuries when momentum turns
negative — capturing the flight-to-quality bond rally exactly when
equities sell off.

EDGE (well-documented, 90+ years of evidence):
  • Absolute (time-series) momentum — Moskowitz/Ooi/Pedersen 2012;
    Hurst/Ooi/Pedersen 2017 "A Century of Evidence on Trend-Following".
  • Relative (cross-sectional) momentum — Jegadeesh/Titman 1993.
  • The two combined ("Dual Momentum") — Antonacci 2014: hold the
    strongest assets, but only while their own absolute momentum is
    positive; otherwise sit in bonds. Higher return AND lower drawdown
    than either leg alone.

CONSTRUCTION
  Risk universe (6): SPY, QQQ, EFA, EEM, VNQ, GLD — US large/tech,
    intl DM/EM, REITs, gold. Safe asset: IEF (7-10y Treasuries).
  Signal: 12-1m momentum (252d lookback, skip last 21d to dodge
    short-term reversal) on each risk asset.
  Selection: rank risk assets by momentum, take the TOP_K=3. Each top-K
    slot holds its asset only if that asset's absolute momentum > 0;
    otherwise the slot's capital rotates to IEF. All three negative →
    100% IEF (full risk-off).
  Sizing: equal-weight across the K slots (book / K each).
  Turnover: monthly rebalance via the allocator cadence + a delta
    threshold, so fee math survives (the validation gate charges 10bps
    round-trip and demands return_on_volume > 0).

Stays in DRY until docs/validation.json records PASS — see CLAUDE.md and
common/strategy_validation.py. If the live edge fails to materialise,
auto_demote freezes it on the next cycle: no capital risk to prove it.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

RISK_UNIVERSE = ["SPY", "QQQ", "EFA", "EEM", "VNQ", "GLD"]
SAFE_ASSET = "IEF"

LOOKBACK_DAYS = 252      # 12 months
SKIP_DAYS = 21           # skip last month (12-1m momentum)
TOP_K = 3               # concentrate in the 3 strongest risk assets


class DualMomentum(Strategy):
    name = "dual_momentum"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        book = ctx.target_alloc_usd

        # 1) Momentum for each risk asset.
        momentum: dict[str, float] = {}
        prices: dict[str, float] = {}
        for symbol in RISK_UNIVERSE + [SAFE_ASSET]:
            try:
                candles = self.broker.get_candles(
                    symbol, "ONE_DAY", num_candles=LOOKBACK_DAYS + 30)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {symbol}: {e}")
                continue
            if len(candles) < LOOKBACK_DAYS:
                continue
            closes = np.array([c.close for c in candles])
            prices[symbol] = float(closes[-1])
            window = closes[-LOOKBACK_DAYS:-SKIP_DAYS] if SKIP_DAYS else closes[-LOOKBACK_DAYS:]
            if len(window) < 30:
                continue
            momentum[symbol] = (window[-1] - window[0]) / window[0]

        # Need prices for the safe asset and at least some risk assets to act.
        risk_with_mom = {s: momentum[s] for s in RISK_UNIVERSE if s in momentum}
        if SAFE_ASSET not in prices or not risk_with_mom:
            return []

        # 2) Dual-momentum selection: top-K by relative momentum, each
        # gated by its own absolute momentum; failed slots → bonds.
        ranked = sorted(risk_with_mom, key=lambda s: risk_with_mom[s],
                        reverse=True)[:TOP_K]
        per_slot = book / TOP_K
        target_usd: dict[str, float] = {s: 0.0 for s in RISK_UNIVERSE}
        target_usd[SAFE_ASSET] = 0.0
        for s in ranked:
            if risk_with_mom[s] > 0:
                target_usd[s] += per_slot          # risk-on slot
            else:
                target_usd[SAFE_ASSET] += per_slot  # risk-off → Treasuries

        # 3) Diff against current holdings and emit BUY/SELL deltas.
        proposals: list[TradeProposal] = []
        for symbol, tgt in target_usd.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            cur_qty = ctx.open_positions.get(symbol, {}).get("quantity", 0.0)
            cur_usd = cur_qty * price
            pending = ctx.pending_orders.get(symbol, {})
            committed_usd = cur_usd + pending.get("buy_notional_usd", 0.0)
            delta_usd = tgt - committed_usd
            # Rebalance band: ignore dust to keep turnover (and fees) low.
            if abs(delta_usd) < max(50.0, per_slot * 0.10):
                continue
            if delta_usd > 0 and pending.get("n_pending", 0) > 0:
                continue   # don't stack on an in-flight BUY
            side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
            mom = momentum.get(symbol)
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=side, order_type=OrderType.MARKET,
                notional_usd=abs(delta_usd),
                confidence=0.7,
                reason=(
                    f"dual-momentum: {'SAFE' if symbol == SAFE_ASSET else 'risk'} "
                    f"target=${tgt:.0f}, current=${cur_usd:.0f}"
                    + (f", 12-1m={mom*100:+.1f}%" if mom is not None else "")
                ),
                is_closing=(side == OrderSide.SELL and cur_qty > 0),
                metadata={"momentum_12_1m": mom, "target_usd": tgt},
            ))
        return proposals
