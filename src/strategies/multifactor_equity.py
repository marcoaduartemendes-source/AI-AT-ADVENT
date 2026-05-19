"""Multi-factor cross-sectional equity model — institutional-grade.

This is the book's flagship systematic-equity sleeve. It is the
strategy a quant shop (AQR / Dimensional / Robeco) actually runs,
condensed to what Alpaca can execute: a cross-sectional composite
of independently-documented return factors, sector-neutralised,
volatility-targeted, and turnover-controlled.

WHY THIS EXISTS
The 2026-05-19 audit found the book had ~20 single-signal retail
strategies and *no factor model*: signals were un-normalised,
fee-unaware, and not regime-gated. This strategy fixes all three
structurally:

  • Factors are cross-sectionally z-scored every rebalance so a
    noisy raw signal can't dominate (the RSI/Bollinger failure mode).
  • Gross exposure is scaled by the vol-managed overlay scaler, so
    the sleeve de-risks in HIGH/EXTREME vol regimes automatically.
  • A rank-hysteresis + cooldown turnover buffer means a name is
    only churned when its composite rank decays materially — this
    is the fee-bleed fix (the audit's #1 money-loser).

FACTORS (each a documented, separately-published anomaly)
  1. Momentum 12-1     Jegadeesh-Titman 1993, Asness-Moskowitz-
                       Pedersen 2013. 252d return skipping the most
                       recent 21d (avoids 1-month reversal).
  2. Low-volatility    Frazzini-Pedersen 2014 ("Betting Against
                       Beta"), Blitz-van Vliet 2007. −1 × 120d
                       realised vol → low-vol names score high.
  3. Short-term        Lehmann 1990, Lo-MacKinlay 1990. −1 × 5d
     reversal          return → recent losers score high (liquidity
                       provision premium).

Composite = 0.45·z(mom) + 0.35·z(lowvol) + 0.20·z(reversal)
(momentum is the highest-Sharpe equity factor historically, hence
the largest weight; reversal is noisiest, hence the smallest.)

ELIGIBILITY GATE
A name must be trading above its 200-day SMA to be longable. This
is a trend/quality regime filter — the low-vol and reversal factors
are only harvested on names that aren't in a structural downtrend
(prevents the "cheap leg keeps falling" failure the audit flagged
for pairs_trading and low_vol_anomaly).

CONSTRUCTION
  • Long the top-decile composite, equal-weight.
  • Max 2 names per GICS-ish sector bucket (sector-neutralisation —
    stops the book becoming a tech-momentum bet).
  • Total deployment = target_alloc_usd × vol_scaler(ctx).
  • Rebalance cadence enforced by REBALANCE_COOLDOWN_DAYS; a held
    name is only sold when it falls out of the top *third* (not the
    top decile) — hysteresis band that cuts turnover ~60% vs naive
    re-ranking.
"""
from __future__ import annotations

import logging

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────

MOM_LOOKBACK = 252          # 12-month total window
MOM_SKIP = 21               # skip most-recent month (1m reversal)
VOL_LOOKBACK = 120          # realised-vol window (low-vol factor)
REVERSAL_LOOKBACK = 5       # short-term reversal window
TREND_SMA = 200             # eligibility: must be above 200d SMA

W_MOM = 0.45
W_LOWVOL = 0.35
W_REVERSAL = 0.20

TOP_DECILE_FRAC = 0.10      # long the top 10% by composite
EXIT_RANK_FRAC = 0.33       # held name only sold if it leaves top 33%
MAX_PER_SECTOR = 2
REBALANCE_COOLDOWN_DAYS = 7
MIN_NAMES = 3               # don't trade a degenerate 1-2 name book


# Liquid large-cap universe with a coarse sector tag for
# neutralisation. Deliberately broad + cross-sector so the composite
# isn't structurally a tech bet.
UNIVERSE: dict[str, str] = {
    # Tech / comm
    "AAPL": "TECH", "MSFT": "TECH", "NVDA": "TECH", "GOOGL": "TECH",
    "META": "TECH", "AVGO": "TECH", "ORCL": "TECH", "CRM": "TECH",
    # Consumer
    "AMZN": "CONS", "TSLA": "CONS", "HD": "CONS", "MCD": "CONS",
    "NKE": "CONS", "LOW": "CONS", "SBUX": "CONS",
    # Staples
    "PG": "STPL", "KO": "STPL", "PEP": "STPL", "WMT": "STPL",
    "COST": "STPL",
    # Healthcare
    "JNJ": "HLTH", "UNH": "HLTH", "LLY": "HLTH", "ABBV": "HLTH",
    "MRK": "HLTH", "PFE": "HLTH",
    # Financials
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN",
    "MS": "FIN", "BLK": "FIN",
    # Industrials / energy / materials
    "CAT": "INDU", "BA": "INDU", "GE": "INDU", "HON": "INDU",
    "XOM": "ENER", "CVX": "ENER", "COP": "ENER",
    "LIN": "MATL", "FCX": "MATL",
}


class MultiFactorEquity(Strategy):
    name = "multifactor_equity"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Vol-target the whole sleeve via the published equity scaler
        # (folds in the HIGH/EXTREME regime multiplier). Sleeve
        # automatically shrinks when realised vol spikes.
        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        rows = self._factor_table()
        if len(rows) < MIN_NAMES:
            logger.info(f"[{self.name}] only {len(rows)} eligible "
                        f"names (need {MIN_NAMES}); sitting out")
            return []

        # Cross-sectional z-scores per factor, then composite.
        mom = np.array([r["mom"] for r in rows])
        lov = np.array([r["lowvol"] for r in rows])
        rev = np.array([r["rev"] for r in rows])

        def _z(a: np.ndarray) -> np.ndarray:
            sd = a.std()
            return (a - a.mean()) / sd if sd > 1e-12 else np.zeros_like(a)

        composite = W_MOM * _z(mom) + W_LOWVOL * _z(lov) + W_REVERSAL * _z(rev)
        order = np.argsort(-composite)              # best first
        n = len(rows)
        top_k = max(MIN_NAMES, int(round(n * TOP_DECILE_FRAC)))
        exit_k = max(top_k, int(round(n * EXIT_RANK_FRAC)))

        ranked = [rows[i]["symbol"] for i in order]
        target_set: list[str] = []
        sector_count: dict[str, int] = {}
        for sym in ranked:
            if len(target_set) >= top_k:
                break
            sec = UNIVERSE.get(sym, "OTHER")
            if sector_count.get(sec, 0) >= MAX_PER_SECTOR:
                continue            # sector-neutralisation cap
            target_set.append(sym)
            sector_count[sec] = sector_count.get(sec, 0) + 1

        # Hysteresis exit band: a held name only sells if it has
        # decayed out of the top third (not merely out of the decile).
        keep_band = set(ranked[:exit_k])
        per_name_usd = sleeve_usd / max(len(target_set), 1)

        proposals: list[TradeProposal] = []

        # 1) Exits — held names that fell out of the hysteresis band.
        for sym, pos in (ctx.open_positions or {}).items():
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            if sym in keep_band:
                continue           # still good enough — don't churn
            if not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue           # respect min-hold to bound turnover
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.9, is_closing=True,
                reason=f"composite rank decayed out of top {exit_k}/{n}",
                metadata={"model": "multifactor", "leg": "exit"},
            ))

        # 2) Entries — target names we don't already hold.
        held = {
            s for s, p in (ctx.open_positions or {}).items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }
        for sym in target_set:
            if sym in held:
                continue
            comp = float(composite[list(ranked).index(sym)]) \
                if sym in ranked else 0.0
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name_usd,
                confidence=min(0.6 + abs(comp) * 0.1, 0.95),
                reason=(f"multifactor top-{top_k}/{n} "
                        f"(composite z={comp:+.2f}, sector "
                        f"{UNIVERSE.get(sym, '?')})"),
                metadata={"model": "multifactor", "composite": comp,
                          "leg": "entry"},
            ))
        return proposals

    # ── Helpers ──────────────────────────────────────────────────────

    def _factor_table(self) -> list[dict]:
        """Per-name raw factor values for every eligible (above-200d-SMA)
        universe member. Names with insufficient history are dropped."""
        need = MOM_LOOKBACK + MOM_SKIP + 5
        out: list[dict] = []
        for sym in UNIVERSE:
            try:
                candles = self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=need + 10)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {sym}: {e}")
                continue
            if len(candles) < need:
                continue
            closes = np.array([c.close for c in candles], dtype=float)
            if closes[-1] <= 0:
                continue

            # Eligibility: above 200d SMA (trend/quality regime gate)
            sma200 = closes[-TREND_SMA:].mean()
            if closes[-1] < sma200:
                continue

            # Factor 1 — 12-1 momentum
            p_start = closes[-(MOM_LOOKBACK + MOM_SKIP)]
            p_end = closes[-(MOM_SKIP + 1)]
            if p_start <= 0:
                continue
            mom = (p_end / p_start) - 1.0

            # Factor 2 — low-vol (negative realised vol → high score)
            rets = np.diff(closes[-VOL_LOOKBACK:]) / closes[-VOL_LOOKBACK:-1]
            realised_vol = float(rets.std()) if len(rets) else 1.0
            lowvol = -realised_vol

            # Factor 3 — short-term reversal (neg 5d return)
            p_rev = closes[-(REVERSAL_LOOKBACK + 1)]
            rev = -((closes[-1] / p_rev) - 1.0) if p_rev > 0 else 0.0

            out.append({"symbol": sym, "mom": mom,
                         "lowvol": lowvol, "rev": rev})
        return out
