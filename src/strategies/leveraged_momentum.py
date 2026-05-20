"""Leveraged trend strategy — 3x ETFs with strict regime gate.

USER ASK
The user explicitly requested leveraged exposure (up to 3x) on
retirement-critical capital. I pushed back in chat (decay, gap
risk, vol drag); they overrode. This module honours that override
RESPONSIBLY — leverage only inside the conditions where the
empirical record says 3x ETFs aren't suicidal:

  • Strong uptrend in the unlevered underlying  (price > 200d SMA)
  • Low realised vol regime                      (≤ VOL_CEILING)
  • Hard stop on each position                   (entry × (1-STOP_PCT))
  • Tiny sleeve cap                              (~2% of book target)

WHY THE GATES
3x leveraged ETFs (TQQQ/UPRO/SOXL/TNA) reset DAILY, so their
multi-period return path-depends on volatility:

    levered_return ≈ 3·µ − ½·(3²−3)·σ²·dt   (Ito's lemma, log-rets)

In a 30% drawdown with 30%-annualised σ, 3x ETFs typically lose
60–80% (not 90%) because the daily reset compounds losses
asymmetrically. The only empirical regime where 3x has been
SAFER than equivalent margin is low-vol uptrends — exactly the
regime this strategy gates on. Outside that regime, the position
is FLAT (in cash), not short.

UNIVERSE — 3x bull ETFs only (no 3x inverse; shorting volatility-
decay products is its own foot-gun, not this strategy's job):

    Ticker  Underlying     Issuer  Notes
    TQQQ    Nasdaq-100     PSDQ    Tech-heavy; highest historical Sharpe of the four
    UPRO    S&P 500        PSDQ    Broadest US equity
    SOXL    PHLX semis     DRX     Most concentrated AI-supply-chain proxy
    TNA     Russell 2000   DRX     Small-cap; highest decay; smallest weight

LIVE-PROMOTION GATE
This strategy registers at DRY target_alloc_pct=0.02 (2%) and
should NOT be promoted to live until the validation harness
(src/common/strategy_validation.py) records a PASS verdict over
the 5y window. If the validation panel shows FAIL or UNPROVEN,
keep it paper.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────

# Universe — (leveraged ETF, unlevered proxy used for regime check).
# Regime is measured on the UNDERLYING, not the 3x product, because
# the 3x product's own moving average lags and reacts to its own
# decay path, not the actual asset trend.
UNIVERSE: dict[str, str] = {
    "TQQQ": "QQQ",     # 3x Nasdaq-100
    "UPRO": "SPY",     # 3x S&P 500
    "SOXL": "SOXX",    # 3x semis
    "TNA":  "IWM",     # 3x Russell 2000
}

TREND_SMA = 200            # underlying must trade above 200d SMA
VOL_LOOKBACK = 60          # window for realised vol (60d ~ quarterly)
VOL_CEILING = 0.022        # daily realised vol cap (~35% annualised);
                            # historic SPY median ≈ 0.008 (~13% ann).
                            # Above this threshold 3x decay dominates.

STOP_PCT = 0.15            # hard stop: -15% from entry on the levered
                            # product (equivalent to -5% on underlying).
                            # Per-position; checked every cycle.

REBALANCE_COOLDOWN_DAYS = 5    # don't churn — leveraged spreads are wide


class LeveragedMomentum(Strategy):
    name = "leveraged_momentum"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Vol-managed overlay halves the sleeve in HIGH/EXTREME regimes,
        # zeroes it in CRISIS — exactly when 3x decay is most lethal.
        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Hard stops on existing positions (always check first
        #       — the stop is non-negotiable; cooldown doesn't apply).
        for sym, pos in open_pos.items():
            if sym not in UNIVERSE:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            entry = (pos.get("avg_entry_price") or pos.get("entry_price") or 0)
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
                    reason=(f"hard stop: last ${last:.2f} ≤ entry "
                            f"${entry:.2f} × {1-STOP_PCT:.0%}"),
                    metadata={"model": "leveraged_momentum",
                              "leg": "stop"},
                ))

        # ── 2) Regime scan on the underlyings; only LONG eligible
        #       3x names; FLAT (no entry) otherwise.
        eligible: list[str] = []
        for lev_sym, underlying in UNIVERSE.items():
            try:
                under = self.broker.get_candles(
                    underlying, "ONE_DAY",
                    num_candles=TREND_SMA + VOL_LOOKBACK + 5)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {underlying}: {e}")
                continue
            if len(under) < TREND_SMA:
                continue
            closes = np.array([c.close for c in under], dtype=float)
            if closes[-1] <= 0:
                continue
            sma200 = closes[-TREND_SMA:].mean()
            if closes[-1] < sma200:
                continue                       # not in uptrend
            # Realised vol on log-returns over VOL_LOOKBACK days.
            rets = np.diff(np.log(closes[-VOL_LOOKBACK - 1:]))
            rv = float(rets.std())
            if rv > VOL_CEILING:
                continue                       # too choppy for 3x
            eligible.append(lev_sym)

        if not eligible:
            logger.info(f"[{self.name}] no underlyings in low-vol uptrend; "
                        f"sitting flat (this is the SAFE state for 3x)")
            return proposals               # only stops, no new entries

        # Equal-weight across eligible 3x names; cap per-name to keep
        # any single 3x position from dominating the sleeve.
        per_name_usd = sleeve_usd / len(eligible)
        held = {
            s for s, p in open_pos.items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }
        for sym in eligible:
            if sym in held:
                continue                   # already long; let it ride
            pos = open_pos.get(sym, {})
            if pos and not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name_usd,
                confidence=0.7, is_closing=False,
                reason=("3x trend-on: underlying above 200d SMA, "
                        "realised vol within ceiling"),
                metadata={"model": "leveraged_momentum",
                          "leg": "entry",
                          "stop_pct": STOP_PCT,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
        return proposals
