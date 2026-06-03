"""Six institutional-grade alpha sleeves — Citadel/Jane Street caliber.

Each strategy here represents a SEPARATE, ACADEMICALLY-DOCUMENTED return
premium that does NOT overlap with the existing equity-momentum heavy
roster. Built deliberately so the cross-strategy correlation matrix
(see common/portfolio_intel.py) finally has uncorrelated names — which
is what actually drives Sharpe up, not adding more equity-trend sleeves.

Each one shares the same disciplined skeleton:
  • Liquid ETF universe (no single-name idiosyncratic risk)
  • Regime gate (only deploy when the underlying premium is harvestable)
  • Monthly rebalance (fee-efficient — survives the 10bps gate)
  • Volatility-overlay aware
  • DRY until validation PASS — auto_demote freezes if the live edge fails

THE SIX STRATEGIES

  1. global_macro_momentum  — Country-ETF cross-sectional momentum
     (Asness-Liew-Stevens 2013). Rotate to the top countries by 12-1m
     return; uncorrelated to US-equity momentum because country dispersion
     dominates US-stock dispersion at 1-12m horizons.

  2. quality_factor          — Long QUAL + USMV (Asness-Frazzini-Pedersen
     "Quality Minus Junk"; Frazzini-Pedersen "Betting Against Beta").
     Defensive equity tilt; outperforms in selloffs.

  3. defensive_value         — Long VTV + IDV (intl div) + USMV. Classic
     value-defensive composite (AQR's "Style Premia"); negatively
     correlated with momentum, true diversifier.

  4. high_yield_carry        — HYG/JNK above 200d SMA, otherwise SHY.
     Captures the credit-spread premium with a trend gate to avoid
     getting steamrolled in credit selloffs.

  5. reit_income_carry       — VNQ/SCHH above 200d SMA, otherwise SHY.
     Real-estate yield + a trend filter. Distinct asset class.

  6. size_premium_trend      — IWM (small cap) above 200d SMA, else SHY.
     The size premium (Fama-French SMB) only in regimes where small caps
     are actually trending — avoids the long-running "size factor died"
     drawdowns of pure SMB.

All six register SMALL and DRY until validated, like every new sleeve.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Shared helpers ───────────────────────────────────────────────────


def _candles(broker, symbol, n):
    try:
        c = broker.get_candles(symbol, "ONE_DAY", num_candles=n)
        return np.array([x.close for x in c], dtype=float) if len(c) >= 30 else None
    except Exception as e:
        logger.debug(f"candles {symbol}: {e}")
        return None


def _above_sma(closes: np.ndarray, days: int) -> bool:
    return len(closes) >= days and closes[-1] >= closes[-days:].mean()


def _delta_to_target(ctx: StrategyContext, name: str, venue: str,
                     target_usd: dict[str, float], prices: dict[str, float],
                     per_slot: float, label: str
                     ) -> list[TradeProposal]:
    """Diff broker positions against target_usd and emit BUY/SELL
    proposals — the shared rebalance plumbing all six use."""
    out: list[TradeProposal] = []
    for sym, tgt in target_usd.items():
        price = prices.get(sym)
        if not price or price <= 0:
            continue
        cur_qty = ctx.open_positions.get(sym, {}).get("quantity", 0.0)
        cur_usd = cur_qty * price
        pending = ctx.pending_orders.get(sym, {})
        committed = cur_usd + pending.get("buy_notional_usd", 0.0)
        delta = tgt - committed
        if abs(delta) < max(50.0, per_slot * 0.10):
            continue
        if delta > 0 and pending.get("n_pending", 0) > 0:
            continue
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        out.append(TradeProposal(
            strategy=name, venue=venue, symbol=sym, side=side,
            order_type=OrderType.MARKET, notional_usd=abs(delta),
            confidence=0.65, reason=f"{label}: target=${tgt:.0f}, current=${cur_usd:.0f}",
            is_closing=(side == OrderSide.SELL and cur_qty > 0),
            metadata={"target_usd": tgt, "as_of": datetime.now(UTC).isoformat()},
        ))
    return out


# ─── 1. global_macro_momentum ─────────────────────────────────────────


GMM_UNIVERSE = ["SPY", "EFA", "EWJ", "EWG", "EWU", "EWZ",
                "INDA", "FXI", "VWO", "EWY", "EWA", "EWC"]
GMM_TOP_K = 3
GMM_LOOKBACK = 252
GMM_SKIP = 21


class GlobalMacroMomentum(Strategy):
    name = "global_macro_momentum"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, mom = {}, {}
        for sym in GMM_UNIVERSE:
            c = _candles(self.broker, sym, GMM_LOOKBACK + 30)
            if c is None or len(c) < GMM_LOOKBACK:
                continue
            prices[sym] = float(c[-1])
            w = c[-GMM_LOOKBACK:-GMM_SKIP]
            if len(w) >= 30:
                mom[sym] = (w[-1] - w[0]) / w[0]
        if not mom:
            return []
        winners = sorted([s for s in mom if mom[s] > 0],
                         key=lambda s: mom[s], reverse=True)[:GMM_TOP_K]
        per_slot = ctx.target_alloc_usd / GMM_TOP_K
        tgt = {s: 0.0 for s in GMM_UNIVERSE}
        for s in winners:
            tgt[s] = per_slot
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                per_slot, "country momentum")


# ─── 2. quality_factor ────────────────────────────────────────────────


QF_BASKET = ["QUAL", "USMV"]
QF_SAFE = "SHY"
QF_TREND = 200


class QualityFactor(Strategy):
    name = "quality_factor"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, above = {}, {}
        for sym in QF_BASKET + [QF_SAFE]:
            c = _candles(self.broker, sym, QF_TREND + 30)
            if c is None or len(c) < QF_TREND:
                continue
            prices[sym] = float(c[-1])
            above[sym] = _above_sma(c, QF_TREND)
        if QF_SAFE not in prices:
            return []
        per_slot = ctx.target_alloc_usd / len(QF_BASKET)
        tgt = {s: 0.0 for s in QF_BASKET + [QF_SAFE]}
        for s in QF_BASKET:
            if s in prices and above.get(s):
                tgt[s] = per_slot
            else:
                tgt[QF_SAFE] += per_slot
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                per_slot, "quality factor")


# ─── 3. defensive_value ───────────────────────────────────────────────


DV_BASKET = ["VTV", "IDV", "USMV"]
DV_SAFE = "SHY"
DV_TREND = 200


class DefensiveValue(Strategy):
    name = "defensive_value"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, above = {}, {}
        for sym in DV_BASKET + [DV_SAFE]:
            c = _candles(self.broker, sym, DV_TREND + 30)
            if c is None or len(c) < DV_TREND:
                continue
            prices[sym] = float(c[-1])
            above[sym] = _above_sma(c, DV_TREND)
        if DV_SAFE not in prices:
            return []
        per_slot = ctx.target_alloc_usd / len(DV_BASKET)
        tgt = {s: 0.0 for s in DV_BASKET + [DV_SAFE]}
        for s in DV_BASKET:
            if s in prices and above.get(s):
                tgt[s] = per_slot
            else:
                tgt[DV_SAFE] += per_slot
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                per_slot, "defensive value")


# ─── 4. high_yield_carry ──────────────────────────────────────────────


HY_BASKET = ["HYG", "JNK"]
HY_SAFE = "SHY"
HY_TREND = 200


class HighYieldCarry(Strategy):
    name = "high_yield_carry"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, above = {}, {}
        for sym in HY_BASKET + [HY_SAFE]:
            c = _candles(self.broker, sym, HY_TREND + 30)
            if c is None or len(c) < HY_TREND:
                continue
            prices[sym] = float(c[-1])
            above[sym] = _above_sma(c, HY_TREND)
        if HY_SAFE not in prices:
            return []
        per_slot = ctx.target_alloc_usd / len(HY_BASKET)
        tgt = {s: 0.0 for s in HY_BASKET + [HY_SAFE]}
        for s in HY_BASKET:
            if s in prices and above.get(s):
                tgt[s] = per_slot
            else:
                tgt[HY_SAFE] += per_slot
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                per_slot, "credit carry")


# ─── 5. reit_income_carry ─────────────────────────────────────────────


REIT_BASKET = ["VNQ", "SCHH"]
REIT_SAFE = "SHY"
REIT_TREND = 200


class ReitIncomeCarry(Strategy):
    name = "reit_income_carry"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, above = {}, {}
        for sym in REIT_BASKET + [REIT_SAFE]:
            c = _candles(self.broker, sym, REIT_TREND + 30)
            if c is None or len(c) < REIT_TREND:
                continue
            prices[sym] = float(c[-1])
            above[sym] = _above_sma(c, REIT_TREND)
        if REIT_SAFE not in prices:
            return []
        per_slot = ctx.target_alloc_usd / len(REIT_BASKET)
        tgt = {s: 0.0 for s in REIT_BASKET + [REIT_SAFE]}
        for s in REIT_BASKET:
            if s in prices and above.get(s):
                tgt[s] = per_slot
            else:
                tgt[REIT_SAFE] += per_slot
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                per_slot, "REIT carry")


# ─── 6. size_premium_trend ────────────────────────────────────────────


SP_RISK = "IWM"
SP_SAFE = "SHY"
SP_TREND = 200


class SizePremiumTrend(Strategy):
    name = "size_premium_trend"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        prices, above = {}, {}
        for sym in (SP_RISK, SP_SAFE):
            c = _candles(self.broker, sym, SP_TREND + 30)
            if c is None or len(c) < SP_TREND:
                continue
            prices[sym] = float(c[-1])
            above[sym] = _above_sma(c, SP_TREND)
        if SP_SAFE not in prices:
            return []
        risk_on = prices.get(SP_RISK) and above.get(SP_RISK)
        tgt = {SP_RISK: ctx.target_alloc_usd if risk_on else 0.0,
               SP_SAFE: 0.0 if risk_on else ctx.target_alloc_usd}
        return _delta_to_target(ctx, self.name, self.venue, tgt, prices,
                                ctx.target_alloc_usd,
                                "size premium" if risk_on else "size-safe")
