"""High Vol-Target Managed-Futures Trend on a cross-asset ETF basket.

REAL EDGE, REAL FIRMS:
    Time-series momentum (TSMOM, Moskowitz-Ooi-Pedersen 2012) is the
    canonical strategy at Man AHL ($20B+ in trend), Winton, Aspect,
    AQR Managed Futures HV, Transtrend, Lynx, ISAM. AQR's "Century of
    Evidence on Trend-Following" reports net Sharpe ~1.0 over 100+
    years across 67 markets. AQR Managed Futures HV: 14.2% 3yr
    annualized return, Sharpe 0.89, vol ~17%. CTAs ran +25-40% in 2022.

    THIS IS THE AGGRESSIVE LEVERAGE PLAY the user requested. CTAs scale
    to a fixed vol target by levering 3-5x notional on low-vol assets
    (rates, FX) and de-levering on high-vol ones (equity, energy).

ENTRY:
    For each ETF in the cross-asset basket:
      • 12-1 month return > 0  → long; otherwise → flat.
      • Position notional = sleeve_usd × min(VOL_TARGET / realized_vol_60d,
        MAX_LEVERAGE_PER_NAME).
      • The leverage cap per name is 4x (matches AQR HV's actual per-leg
        cap; their headline "5x" gross is across positions).

EXIT:
    • Trend flip (12-1m return ≤ 0) → exit.
    • Hard -10% per-name stop (CTAs don't usually use stops, but on
      retail leverage we add one to bound single-name drawdown).
    • Vol-scaler overlay halves/zeros in HIGH/CRISIS vol regimes.

UNIVERSE (10 ETFs, cross-asset diversified):
    Equity:    SPY, QQQ, EFA, EEM      (US large, US tech, intl dev, EM)
    Rates:     TLT, IEF                (long US treasury, intermediate)
    Commodities: GLD, DBC              (gold, broad commodities)
    FX/Dollar: UUP                     (USD index)
    Real:      VNQ                     (REITs)

    Coverage is the diversification engine — a CTA's edge is that on
    average ~60% of legs are profitable in any given quarter while ~40%
    are losers, and the leverage scales the winners up enough to net
    7-10% annualized.

SAFETY:
    • Per-name leverage cap 4x.
    • Total sleeve gross leverage cap = 4x (so this sleeve alone can't
      blow the portfolio leverage_cap=2.0x — the orchestrator's risk
      manager will scale further if needed).
    • Vol regime gate via vol_scaler.
    • Hard -10% per-name stop.

FAILURE MODES:
    Sharp reversals / whipsaws (2009 commodities, 2018 risk-off,
    early-2022 rates) — historical CTA -20% drawdowns. The diversification
    is the structural shock-absorber; periods where ALL trends fail
    simultaneously are rare (~once a decade).
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

UNIVERSE = [
    "SPY", "QQQ", "EFA", "EEM",      # equity (US/intl/EM)
    "TLT", "IEF",                    # rates
    "GLD", "DBC",                    # commodities (precious + broad)
    "UUP",                           # FX (USD)
    "VNQ",                           # real estate
]
TREND_LOOKBACK = 252                 # 12 months
SKIP_RECENT = 21                     # skip last month (12-1 spec)
VOL_LOOKBACK = 60                    # realized-vol window
VOL_TARGET_ANN = 0.17                # 17% annualized vol target
MAX_LEVERAGE_PER_NAME = 4.0
MIN_ENTRY_USD = 200.0
STOP_PCT = 0.10                      # -10% per-name hard stop
_ANN = np.sqrt(252)


class HighVolTrend(Strategy):
    name = "high_vol_trend"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        from ._helpers import vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        # Per-name budget (equal-weight before vol scaling)
        per_name_base = sleeve_usd / len(UNIVERSE)
        if per_name_base < MIN_ENTRY_USD:
            return []

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Hard stops first.
        for sym, pos in open_pos.items():
            if sym not in UNIVERSE or not hasattr(pos, "get"):
                continue
            qty = pos.get("quantity", 0) or 0
            if qty <= 0:
                continue
            entry = (pos.get("avg_entry_price")
                     or pos.get("entry_price") or 0)
            try:
                last = float(self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue
            if entry and last and last <= entry * (1 - STOP_PCT):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.99, is_closing=True,
                    reason=(f"hard stop: ${last:.2f} ≤ ${entry:.2f} "
                            f"× {1-STOP_PCT:.0%}"),
                    metadata={"model": self.name, "leg": "stop"},
                ))

        # ── 2) Per-symbol trend signal + vol scaling.
        for sym in UNIVERSE:
            try:
                cs = self.broker.get_candles(
                    sym, "ONE_DAY",
                    num_candles=TREND_LOOKBACK + SKIP_RECENT + 5)
            except Exception:
                continue
            if len(cs) < TREND_LOOKBACK + SKIP_RECENT:
                continue
            closes = np.array([c.close for c in cs], dtype=float)
            if closes[-1] <= 0:
                continue
            # 12-1m return: price 1 month ago vs price 13 months ago.
            t1_close = closes[-SKIP_RECENT - 1]
            t12_close = closes[-TREND_LOOKBACK - SKIP_RECENT]
            if t12_close <= 0:
                continue
            trend_ret = t1_close / t12_close - 1.0

            pos = open_pos.get(sym, {})
            qty = ((pos.get("quantity", 0) or 0)
                   if hasattr(pos, "get") else 0)
            held = qty > 0

            # Trend flip → exit.
            if held and trend_ret <= 0:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.85, is_closing=True,
                    reason=f"trend flip: 12-1m return {trend_ret:+.2%}",
                    metadata={"model": self.name, "leg": "exit"},
                ))
                continue

            # Already held in uptrend → leave alone (no churn).
            if held:
                continue

            # New entry on positive trend.
            if trend_ret <= 0:
                continue

            # Realized vol → leverage scaling.
            rets = np.diff(np.log(closes[-VOL_LOOKBACK - 1:]))
            sd = float(rets.std(ddof=1))
            if sd <= 0:
                continue
            realized_vol_ann = sd * _ANN
            lev = min(VOL_TARGET_ANN / realized_vol_ann,
                      MAX_LEVERAGE_PER_NAME)
            if lev <= 0:
                continue
            notional = per_name_base * lev
            if notional < MIN_ENTRY_USD:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=notional, confidence=0.7, is_closing=False,
                reason=(f"TSMOM 12-1m {trend_ret:+.2%}, vol "
                        f"{realized_vol_ann:.1%} → {lev:.1f}× lev "
                        f"(${notional:.0f})"),
                metadata={"model": self.name, "leg": "entry",
                          "trend_12_1m_pct": trend_ret * 100,
                          "realized_vol_ann": realized_vol_ann,
                          "leverage_applied": lev,
                          "stop_pct": STOP_PCT,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
        return proposals
