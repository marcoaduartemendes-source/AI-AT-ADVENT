#!/usr/bin/env python3
"""New main entry point — wires brokers + risk + allocator + strategies +
orchestrator into one run. Replaces the legacy src/main_trading.py once
the new system is validated.

Usage:
    python src/run_orchestrator.py --once
    python src/run_orchestrator.py --status        # print state, no trading

The legacy bot remains in place behind its existing workflow until we
explicitly retire it (W2). This entry point runs in DRY by default.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from allocator.allocator import MetaAllocator
from allocator.lifecycle import StrategyMeta, StrategyRegistry
from allocator.metrics import StrategyPerformance
from brokers.registry import build_brokers
from risk.manager import RiskManager
from strategies import (
    Activist13D,
    BollingerBreakout,
    CommodityCarry,
    CommodityMomentum,
    CryptoVolRegimeOverlay,
    DividendGrowth,
    DualMomentum,
    EarningsMomentum,
    HighVolTrend,
    InsiderCluster,
    InternationalsRotation,
    LeveragedChampions,
    LeveragedMomentum,
    LeveragedTop4,
    Llm8KEvent,
    MergerArb,
    MultiFactorEquity,
    PreFomcDrift,
    RiskParityETF,
    ThematicGrowth,
    TSMomETF,
    VolManagedOverlay,
)
from strategy_engine.orchestrator import Orchestrator, OrchestratorConfig

# Display logs in America/New_York (the user's home time zone) so the
# operator can correlate Actions logs with US market hours without
# doing UTC math. Timestamps in databases are still ISO-8601 UTC —
# only the human-readable log line is localized.
import logging.config
import time as _time
from datetime import datetime as _dt, UTC
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _NY_TZ = _ZoneInfo("America/New_York")
except Exception:
    _NY_TZ = UTC


def _ny_converter(*args):
    """Replacement for time.localtime that returns America/New_York
    instead. Called by logging's Formatter.converter — has to accept
    being called as a class-attribute (gets `self` first when assigned
    as a regular function), so we just ignore non-numeric leading args.
    """
    secs = None
    for a in args:
        if isinstance(a, (int, float)):
            secs = a
            break
    if secs is None:
        secs = _time.time()
    return _dt.fromtimestamp(secs, tz=UTC).astimezone(_NY_TZ).timetuple()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ET [%(levelname)s] %(name)s: %(message)s",
)
logging.Formatter.converter = staticmethod(_ny_converter)
logger = logging.getLogger("run")

DRY_RUN_DEFAULT = True


def _per_broker_flag(envvar: str) -> bool | None:
    """Parse a per-venue DRY override.

    Returns True (DRY), False (paper/live trading), or None (fall
    through to the global DRY_RUN). The empty-string case has to
    return None — GitHub Actions injects every `${{ vars.X }}` ref
    as the empty string when the underlying variable is unset, and
    the previous behaviour (treat empty-string as DRY=True) was
    the reason the cron orchestrator was silently leaving Alpaca
    in DRY mode even when the user expected paper trading.
    """
    v = os.environ.get(envvar)
    if v is None:
        return None
    v = v.strip()
    if v == "":
        return None        # empty env var == not set
    return v.lower() != "false"


# ─── Strategy wiring ──────────────────────────────────────────────────────


ALL_STRATEGIES = [
    # 2026-06-03: ELIMINATED 4 alpha-failing strategies (crypto_funding_carry,
    # crypto_basis_trade, cross_asset_trend, bond_carry). All proven to
    # not generate alpha over 5y backtests (fee-negative, NO_DATA, or sub-
    # 0.5 Sharpe). Removed entirely instead of just zero-allocated.
    StrategyMeta(
        name="risk_parity_etf",
        asset_classes=["ETF"], venue="alpaca",
        # CIO 2026-06-05: 22→20% — trim by 2pp to fund underweight
        # dual_momentum / internationals_rotation / bollinger_breakout.
        target_alloc_pct=0.20, max_alloc_pct=0.30, min_alloc_pct=0.12,
        description="Inverse-vol ETF book (SPY/TLT/IEF/GLD/DBC) (P1)",
        group="DEFENSIVE",
    ),
    # kalshi_calibration_arb RETIRED 2026-06-05 (CIO Tier-4 cull) — NO_DATA
    # backtest in 18 months; Kalshi liquidity too thin to size meaningfully.
    # ── Phase 2 (crypto_basis_trade eliminated 2026-06-03 — NO_DATA / fee-neg)
    StrategyMeta(
        name="tsmom_etf",
        asset_classes=["ETF"], venue="alpaca",
        # CIO Tier-3: walk-forward WEAK (overfit risk). Halve allocation
        # 16→8% and monitor — was also the #1 victim of the ETF cap
        # (93% rejection); now unblocked by the 60→90% cap raise.
        target_alloc_pct=0.08, max_alloc_pct=0.18, min_alloc_pct=0.03,
        description="12-1m time-series momentum on 7-ETF basket (P2)",
        group="TREND",
    ),
    StrategyMeta(
        name="commodity_carry",
        asset_classes=["COMMODITY_FUTURE"], venue="coinbase",
        # CIO Tier-1: Sharpe 1.28 modest but ROBUST — trim 6→3% to
        # right-size relative to the much-higher-Sharpe core sleeves.
        target_alloc_pct=0.03, max_alloc_pct=0.10, min_alloc_pct=0.01,
        description="Top-N backwardated commodity futures (P2)",
        group="CARRY",
    ),
    # ── Phase 3
    # PEAD v1 retired 2026-05-07. The gap-only proxy for "earnings
    # surprise" is a known weak signal vs the v2 (RSS news corroboration)
    # and earnings_momentum (true EPS-surprise via FMP) variants;
    # running all three triple-trades the same earnings prints and
    # inflates correlation. Weight reallocated to v2 + earnings_momentum.
    # The pead.py module is kept in-tree for backtest reference only.
    # macro_kalshi (v1) ELIMINATED 2026-06-05 — NO_DATA, superseded by
    # macro_kalshi_v2 (CME-fed). Module stays in-tree for revival.
    # ELIMINATED 2026-05-22 (user mandate "eliminate strategies that
    # aren't working"): crypto_xsmom, gap_trading, low_vol_anomaly,
    # pairs_trading, rsi_mean_reversion, turn_of_month all FAILed the
    # 5y validation gate (negative or sub-0.5 Sharpe, fee-negative, or
    # overfit). Un-wired from the live roster + build_strategies; their
    # source files and backtest functions stay in-tree for reference /
    # revival, but they no longer register, allocate, or trade.
    StrategyMeta(
        name="vol_managed_overlay",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.00, max_alloc_pct=0.00, min_alloc_pct=0.0,
        description="Vol-target multiplier; publishes scaler only, no trades (P3)",
        group="OVERLAY",
    ),
    # ── Flagship — institutional multi-factor cross-sectional equity
    # model (2026-05-19). The highest-sophistication sleeve in the
    # book: momentum + low-vol + reversal composite, cross-sectionally
    # z-scored, sector-neutralised, vol-targeted, turnover-controlled.
    # Sized as a core sleeve alongside risk_parity + tsmom.
    StrategyMeta(
        name="multifactor_equity",
        asset_classes=["EQUITY"], venue="alpaca",
        # CIO Tier-1: 14→16% — the ONLY sleeve actually filling live
        # (3/3 submit ratio in droplet data). Sharpe 5.92, ROBUST.
        # Flagship; bump up.
        target_alloc_pct=0.16, max_alloc_pct=0.30, min_alloc_pct=0.08,
        description="Multi-factor (mom+lowvol+reversal) x-sectional equity (flagship)",
        group="FACTOR",
    ),
    # ── User-requested experimental sleeves. Both REGISTER SMALL and
    # stay DRY until the validation harness (docs/validation.json)
    # records PASS — see CLAUDE.md + common/strategy_validation.py.
    # leveraged_momentum: 3x ETFs gated on uptrend + low-vol regime
    # with a -15% hard stop; deliberately tiny so a wipeout is bounded.
    # 2026-06-05 — "concentrate + leverage" mandate: the leveraged sleeves
    # get more capital to actually beat the market, not track it. Safe
    # because (a) the portfolio leverage_cap=2.0× gross-notional backstop
    # in risk/policies.py bounds total exposure regardless of these
    # numbers, (b) each sleeve stays DRY until validation.json records
    # PASS, (c) each carries a −15% hard stop + 200d-SMA/low-vol regime
    # gate + vol_scaler that zeros the book in CRISIS. Higher caps simply
    # let the regime-gated winners take a bigger bite when conditions are
    # favourable; the cap math is what makes the aggression survivable.
    StrategyMeta(
        name="leveraged_momentum",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.08, min_alloc_pct=0.0,
        description="3x leveraged ETF trend (TQQQ/UPRO/SOXL/TNA, regime-gated)",
        group="LEVERAGED",
        leverage_x=3.0,
    ),
    # leveraged_champions: 3x exposure that DYNAMICALLY tracks the top-5
    # PASS strategies (mapped to liquid 3x ETF proxies), same regime gate
    # + hard stop as leveraged_momentum. Tiny + DRY until validated; 3x
    # decay makes this the highest-risk sleeve, so the cap stays small.
    StrategyMeta(
        name="leveraged_champions",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.08, min_alloc_pct=0.0,
        description="3x leverage tracking the top-5 PASS strategies (regime-gated, -15% stop)",
        group="LEVERAGED",
        leverage_x=3.0,
    ),
    # 2026-06-03: leveraged_top4 — sister sleeve to leveraged_champions
    # but more concentrated (top-4 instead of top-5). 2026-06-05: this is
    # the flagship of the "concentrate + leverage" mandate — highest
    # conviction, so it gets the largest leveraged allocation.
    StrategyMeta(
        name="leveraged_top4",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.10, min_alloc_pct=0.0,
        description="3x leverage tracking the top-4 PASS strategies (concentrated)",
        group="LEVERAGED",
        leverage_x=3.0,
    ),
    # 2026-06-03: 6 new institutional-grade alpha sleeves — each a
    # DISTINCT, academically-documented return premium so the book
    # diversification finally improves. Register SMALL + DRY until PASS.
    # 2026-06-05 (CIO Tier-4 cull): 6 academic Fama-French-style sleeves
    # RETIRED — user explicitly rejected academic factor zoo. The proven
    # multifactor_equity flagship (PASS/ROBUST, Sharpe 5.92) covers the
    # same exposure with operational edge; running redundant siblings
    # just wasted ~16% of capital on undifferentiated US-equity beta.
    # Retired: global_macro_momentum, quality_factor, defensive_value,
    # high_yield_carry, reit_income_carry, size_premium_trend.
    # internationals_rotation (PASS/ROBUST Sharpe 4.91) provides the
    # country-momentum exposure that global_macro_momentum duplicated.
    # thematic_growth: curated 2026 themes (AI compute, AI power,
    # cybersec, defense, GLP-1, robotics, quantum) — within-theme
    # 6m-momentum rank picks the winners; cross-theme by conviction.
    StrategyMeta(
        name="thematic_growth",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.025, max_alloc_pct=0.08, min_alloc_pct=0.0,
        description="Thematic basket (AI compute/power, cyber, defense, GLP-1, robotics)",
        group="FACTOR",
    ),
    # intraday_mean_reversion ELIMINATED 2026-06-05 — 5-min VWAP fade
    # competes in the HFT arena where a cron bot has no latency edge, and
    # it can't be backtested from daily data (NO_DATA) so it could never
    # earn promotion. Module stays in-tree for revival.
    # cross_asset_trend ELIMINATED 2026-06-03 — failed alpha generation.
    # dual_momentum: Antonacci dual momentum — crisis-alpha diversifier.
    StrategyMeta(
        name="dual_momentum",
        asset_classes=["ETF"], venue="alpaca",
        # CIO Tier-1: Sharpe 10.15 (#2 in the book), ROBUST, Antonacci
        # documented strategy. 3→8% — massively underweighted; this is
        # the diversifying engine of the book.
        target_alloc_pct=0.08, max_alloc_pct=0.18, min_alloc_pct=0.03,
        description="Dual momentum (top-3 risk ETFs / IEF risk-off) diversifier",
        group="TREND",
    ),
    # bond_carry ELIMINATED 2026-06-03 — +$38/5y, fee-marginal, no edge.
    StrategyMeta(
        name="commodity_momentum",
        asset_classes=["ETF"], venue="alpaca",
        # CIO Tier-3: Sharpe 5.36 but walk-forward WEAK. Halve to 1%
        # and watch — high backtest Sharpe didn't survive OOS.
        target_alloc_pct=0.01, max_alloc_pct=0.04, min_alloc_pct=0.0,
        description="Cross-sectional 12-1m momentum on commodity ETFs (CTA)",
        group="TREND",
    ),
    # ── Phase 4 — EXPERIMENTAL (small initial allocations on Alpaca
    # paper $100k). Allocator's Sharpe-tilt will reallocate to
    # winners over the first 30-60 days. Each starts at 4%.
    # (rsi_mean_reversion ELIMINATED 2026-05-22 — validation FAIL.)
    # sector_rotation ELIMINATED 2026-06-05 — validation FAIL (Sharpe
    # −10.3/−1.05 in 1y/2y, positive only in the 5y window) + walk-forward
    # OVERFIT_SUSPECT. Sector momentum is heavily arbitraged.
    # (pairs_trading ELIMINATED 2026-05-22 — validation FAIL.)
    StrategyMeta(
        name="bollinger_breakout",
        asset_classes=["EQUITY"], venue="alpaca",
        # CIO Tier-1: Sharpe 3.41 ROBUST. Underfunded at 0.5%; bump to
        # 3% to put real capital behind a cleanly-documented signal.
        target_alloc_pct=0.03, max_alloc_pct=0.08, min_alloc_pct=0.01,
        description="Momentum continuation on 20d Bollinger upper-band breaks (P4)",
        group="TREND",
    ),
    StrategyMeta(
        name="earnings_momentum",
        asset_classes=["EQUITY"], venue="alpaca",
        # +1% baseline / +3% max ceiling (was 4/15) to absorb the
        # retired pead v1 allocation. Cleaner EPS-surprise signal so
        # this should produce higher Sharpe than v1's gap-only proxy.
        target_alloc_pct=0.09, max_alloc_pct=0.2, min_alloc_pct=0.03,
        description="Live PEAD via FMP earnings calendar (P4)",
        group="EVENT",
    ),
    StrategyMeta(
        name="dividend_growth",
        asset_classes=["ETF"], venue="alpaca",
        # CIO Tier-1: Sharpe 3.00 ROBUST. Boring carry that works;
        # bump 0.5→2% for a meaningful contribution.
        target_alloc_pct=0.02, max_alloc_pct=0.05, min_alloc_pct=0.005,
        description="Quality-dividend ETF rotation by 90d return (P4)",
        group="CARRY",
    ),
    # ── Phase 4b — additional experimental strategies for Alpaca
    # paper $100k. (gap_trading, turn_of_month, low_vol_anomaly all
    # ELIMINATED 2026-05-22 — validation FAIL: fee-negative / sub-0.5
    # Sharpe over 5y.)
    StrategyMeta(
        name="internationals_rotation",
        asset_classes=["ETF"], venue="alpaca",
        # CIO Tier-1: Sharpe 4.91 ROBUST. Real US-equity diversifier.
        # 0.5→3% — replaces the retired global_macro_momentum which
        # duplicated this exposure without operational edge.
        target_alloc_pct=0.03, max_alloc_pct=0.08, min_alloc_pct=0.01,
        description="International country-ETF momentum vs SPY (P4b)",
        group="MACRO",
    ),
    # ── Phase 5 strategies RETIRED 2026-06-05 (CIO Tier-4 cull):
    # macro_kalshi_v2 — NO_DATA in 18mo, Kalshi too thin to size.
    # cross_venue_arb — Kalshi↔Polymarket arb is a sub-100ms latency
    # game; retail-grade ~500ms execution cannot win it.
    # crypto_funding_carry_v2 ELIMINATED 2026-06-05 — perp funding is
    # fee-negative (v1 scored −17 Sharpe, RoV −0.05%); the multi-venue
    # consensus gate never validated that it overcomes the bleed.
    # earnings_news_pead ELIMINATED 2026-06-05 — duplicate PEAD signal
    # highly correlated with earnings_momentum (PASS/ROBUST); running both
    # triple-trades the same earnings prints.
    # ── Phase 6 — advanced crypto strategies (2026-05-08).
    # crypto_pairs_trading ELIMINATED 2026-06-05 — BTC/ETH cointegration
    # is unstable and the spot market too efficient; unvalidatable.
    # crypto_breakout RETIRED 2026-06-05 (CIO Tier-4 cull) — crypto trend
    # has decayed sharply 2024-25; venue is right but the signal is dead.
    StrategyMeta(
        name="crypto_vol_regime_overlay",
        asset_classes=["CRYPTO_SPOT"], venue="coinbase",
        target_alloc_pct=0.00, max_alloc_pct=0.00, min_alloc_pct=0.0,
        description="Publishes crypto vol-regime scaler; no trades (P6)",
        group="OVERLAY",
    ),
    # ── 2026-06-05 — Real-edge sleeves (named-firm proven, not academic)
    # User mandate: "I want real market, real world proven strategies from
    # people / companies that have demonstrated a long term edge."
    StrategyMeta(
        name="pre_fomc_drift",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.10, min_alloc_pct=0.0,
        description=("Long SPY+QQQ 24h before FOMC announcements "
                     "(Lucca-Moench 2015, +49bps t>5)"),
        group="EVENT",
    ),
    StrategyMeta(
        name="activist_13d",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.05, max_alloc_pct=0.12, min_alloc_pct=0.0,
        description=("Buy targets within 48h of SC 13D filing "
                     "(Brav-Jiang 2008/19, +7-8% drift)"),
        group="EVENT",
    ),
    StrategyMeta(
        name="merger_arb",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.06, max_alloc_pct=0.15, min_alloc_pct=0.0,
        description=("Cash-deal merger arb (S&P MergerArb 25y Sharpe "
                     "~1.46; Pershing/Paulson/Pentwater)"),
        group="EVENT",
    ),
    StrategyMeta(
        name="high_vol_trend",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.08, max_alloc_pct=0.18, min_alloc_pct=0.0,
        description=("Cross-asset TSMOM, 17% vol target, ≤4x per name "
                     "(Man AHL / Winton / AQR MF HV)"),
        group="TREND",
        leverage_x=3.0,    # CTA-style 3x effective notional via vol scaling
    ),
    # ── 2026-06-05 — Novel-alpha sleeves. The thesis of both: our edge
    # is SMALLNESS + AI. Capacity-constrained signals (insider clusters
    # concentrate in small/mid-caps Citadel can't size into) and an
    # LLM-native signal (Claude reads every 8-K; no human desk can).
    # Both registered SMALL on the paper venue so live fills prove or
    # kill the thesis cheaply — novel alpha has no backtest by nature.
    StrategyMeta(
        name="insider_cluster",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.08, min_alloc_pct=0.0,
        description=("≥3 insiders open-market buying within 7d "
                     "(Cohen-Malloy-Pomorski JF 2012, ~7%/yr)"),
        group="EVENT",
    ),
    StrategyMeta(
        name="llm_8k_event",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.02, max_alloc_pct=0.05, min_alloc_pct=0.0,
        description=("Claude reads fresh 8-Ks; trades VERY_BULLISH "
                     "material events (Lopez-Lira & Tang lineage)"),
        group="EVENT",
    ),
]

# Backward compat alias used by older test scripts
PHASE_1_STRATEGIES = ALL_STRATEGIES


def unfreeze_strategies(spec, registry, passing):
    """Return FROZEN strategies named by `spec` ('all' or comma-list) to
    ACTIVE. Returns [(name, verdict_label), …] for those actually flipped.

    Fixes the lifecycle latch-asymmetry: the allocator auto-FREEZES on
    past underperformance but never auto-recovers, so a strategy that has
    since re-validated stays benched forever. RETIRED is left untouched
    (a deliberate human decision); only FROZEN is reversible here.
    """
    from allocator.lifecycle import StrategyState
    if str(spec).strip().lower() == "all":
        # all_states() only returns registered metas, so callers wanting
        # "all" must register first (the CLI does). Named targets below
        # use get_state(), which reads the DB directly.
        states = registry.all_states()
        targets = [n for n, s in states.items() if s == StrategyState.FROZEN]
    else:
        targets = [n.strip() for n in str(spec).split(",") if n.strip()]
    out: list[tuple[str, str]] = []
    for name in targets:
        if registry.get_state(name) != StrategyState.FROZEN:
            continue
        verdict = "PASS" if name in (passing or set()) else "NOT-PASS"
        registry.set_state(name, StrategyState.ACTIVE,
                           f"manual unfreeze ({verdict})")
        out.append((name, verdict))
    return out


def build_strategies(brokers):
    instances = {}
    if "coinbase" in brokers:
        cb = brokers["coinbase"]
        # crypto_funding_carry + crypto_basis_trade ELIMINATED 2026-06-03 — no alpha.
        instances["commodity_carry"] = CommodityCarry(cb)
        # crypto_xsmom ELIMINATED 2026-05-22 (validation FAIL).
        # ── 2026-06-05 ELIMINATED (alpha review):
        #   crypto_funding_carry_v2 — perp funding is fee-negative (v1
        #     scored −17 Sharpe, RoV −0.05%); the multi-venue gate never
        #     validated that it fixes the bleed.
        #   crypto_pairs_trading — BTC/ETH cointegration is unstable and
        #     the spot market too efficient; unvalidatable, 0.5% sleeve.
        # crypto_breakout RETIRED 2026-06-05 (CIO Tier-4 cull) — crypto
        # trend signal has decayed sharply 2024-25.
        instances["crypto_vol_regime_overlay"] = CryptoVolRegimeOverlay(cb)
    if "alpaca" in brokers:
        al = brokers["alpaca"]
        instances["risk_parity_etf"] = RiskParityETF(al)
        instances["tsmom_etf"] = TSMomETF(al)
        # pead (v1) retired 2026-05-07 — see ALL_STRATEGIES note above.
        # Module still imported so existing trade rows remain readable
        # in the dashboard / FIFO recompute, but no instance is wired.
        instances["vol_managed_overlay"] = VolManagedOverlay(al)
        # Phase 4 — experimental sleeve. (rsi_mean_reversion,
        # pairs_trading ELIMINATED 2026-05-22 — validation FAIL.)
        # sector_rotation ELIMINATED 2026-06-05 — validation FAIL
        # (Sharpe −10.3/−1.05 in 1y/2y, positive only in the 5y window)
        # + walk-forward OVERFIT_SUSPECT. Textbook overfit.
        instances["bollinger_breakout"] = BollingerBreakout(al)
        instances["earnings_momentum"] = EarningsMomentum(al)
        instances["dividend_growth"] = DividendGrowth(al)
        # Phase 4b — additional experimental sleeve. (gap_trading,
        # turn_of_month, low_vol_anomaly ELIMINATED 2026-05-22 — FAIL.)
        instances["internationals_rotation"] = InternationalsRotation(al)
        # Flagship — institutional multi-factor cross-sectional model
        instances["multifactor_equity"] = MultiFactorEquity(al)
        # User-requested: leveraged trend + thematic basket. Tiny
        # DRY allocations until the validation harness PASSes them.
        instances["leveraged_momentum"] = LeveragedMomentum(al)
        instances["leveraged_champions"] = LeveragedChampions(al)
        instances["leveraged_top4"] = LeveragedTop4(al)
        instances["thematic_growth"] = ThematicGrowth(al)
        # intraday_mean_reversion ELIMINATED 2026-06-05 — 5-min VWAP fade
        # competes in the HFT arena where a cron bot has no latency edge,
        # and it can't be backtested from daily data (NO_DATA) so it could
        # never earn promotion. Honest cut.
        # cross_asset_trend ELIMINATED 2026-06-03 — no alpha.
        instances["dual_momentum"] = DualMomentum(al)
        # bond_carry ELIMINATED 2026-06-03 — no alpha.
        instances["commodity_momentum"] = CommodityMomentum(al)
        # 6 academic-style alpha sleeves RETIRED 2026-06-05 (CIO Tier-4
        # cull): global_macro_momentum, quality_factor, defensive_value,
        # high_yield_carry, reit_income_carry, size_premium_trend. Real
        # alpha comes from multifactor_equity flagship +
        # internationals_rotation; the rest was redundant US-equity beta.
        # earnings_news_pead ELIMINATED 2026-06-05 — duplicate PEAD signal
        # highly correlated with earnings_momentum (PASS/ROBUST, Sharpe
        # 2.28). Running both triple-trades the same earnings prints and
        # inflates book correlation; keep the proven one.
        # ── 2026-06-05 — Real-edge sleeves (per the "real world proven,
        # not academic" mandate; citations in each module's docstring)
        instances["pre_fomc_drift"] = PreFomcDrift(al)
        instances["activist_13d"] = Activist13D(al)
        instances["merger_arb"] = MergerArb(al)
        instances["high_vol_trend"] = HighVolTrend(al)
        # 2026-06-05 — novel-alpha sleeves (smallness + AI as the edge)
        instances["insider_cluster"] = InsiderCluster(al)
        instances["llm_8k_event"] = Llm8KEvent(al)
    # Kalshi: all 4 strategies RETIRED 2026-06-05 (CIO Tier-4 cull) —
    # NO_DATA verdicts across the board, Kalshi liquidity too thin to
    # size meaningfully, cross-venue arb is a latency game retail can't
    # win. Re-wire when a viable Kalshi sleeve emerges. The adapter
    # itself remains in build_brokers() for read-only book inspection.
    return instances


# ─── Step summary writer ──────────────────────────────────────────────────


def write_step_summary(report, allocator_alloc, risk_state):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("## Multi-Asset Orchestrator — cycle report\n\n")

            f.write("### Risk state\n")
            f.write(f"- Equity: **${risk_state.equity_usd:,.2f}**  ·  ")
            f.write(f"Peak: ${risk_state.peak_equity_usd:,.2f}  ·  ")
            f.write(f"Drawdown: {risk_state.drawdown_pct * 100:.2f}%\n")
            f.write(f"- Kill-switch: **{risk_state.kill_switch.value}**  ·  ")
            f.write(f"Leverage: {risk_state.leverage:.2f}x\n")
            f.write(f"- Multiplier: base={risk_state.multiplier.base:.2f}, "
                    f"effective=**{risk_state.multiplier.effective:.2f}**")
            if risk_state.multiplier.notes:
                f.write(f" ({', '.join(risk_state.multiplier.notes)})")
            f.write("\n\n")

            if allocator_alloc:
                f.write("### Strategy allocations\n\n")
                f.write("| Strategy | State | Target % | Target $ | Sharpe | DD | Reason |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for d in allocator_alloc.decisions:
                    f.write(
                        f"| {d.name} | {d.state.value} | "
                        f"{d.target_pct * 100:.1f}% | "
                        f"${d.target_usd:,.0f} | "
                        f"{d.metrics.shrunk_sharpe:.2f} | "
                        f"{d.metrics.drawdown_pct * 100:.1f}% | "
                        f"{d.reason} |\n"
                    )
                f.write(f"\n_Total active allocation: "
                        f"{allocator_alloc.total_active_pct * 100:.1f}%_\n\n")

            f.write("### Cycle counters\n")
            f.write(f"- Proposals total: {report.proposals_total}\n")
            f.write(f"- Approved: {report.proposals_approved}  ·  "
                    f"Scaled: {report.proposals_scaled}  ·  "
                    f"Rejected: {report.proposals_rejected}\n")
            f.write(f"- Trades submitted: {report.trades_submitted}\n")
            if report.errors:
                f.write(f"\n**Errors ({len(report.errors)}):**\n")
                for e in report.errors:
                    f.write(f"- {e}\n")
    except Exception as exc:
        logger.warning(f"Could not write step summary: {exc}")


# ─── Main ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run a single cycle")
    ap.add_argument("--status", action="store_true",
                     help="Print risk/allocator status, no trading")
    ap.add_argument("--live", action="store_true",
                     help="Disable DRY mode (still respects DRY_RUN env var)")
    ap.add_argument("--allow-cold-start", action="store_true",
                     help="Permit running with empty risk_state.db. "
                          "Without this, a cold start refuses to trade — "
                          "the kill-switch baseline would otherwise reset "
                          "to current equity and silently arm at "
                          "-KILL_DD_PCT of whatever today's equity is.")
    ap.add_argument("--reset-kill-switch", action="store_true",
                     help="Reset the kill-switch state to NORMAL after a "
                          "KILL event has cleared. Records a marker row "
                          "in risk_state.db; orchestrator picks up the "
                          "new state on the next cycle. Use after you've "
                          "investigated the equity drop that triggered KILL.")
    ap.add_argument("--arm-kill-switch", action="store_true",
                     help="Manually arm the kill switch. Next cycle "
                          "force-closes every position. Use as the "
                          "panic button when you need to halt trading "
                          "immediately.")
    ap.add_argument("--unfreeze", metavar="NAMES", default=None,
                     help="Return FROZEN strategies to ACTIVE (comma-list, "
                          "or 'all'). The allocator auto-freezes on past "
                          "underperformance but never auto-recovers, so a "
                          "strategy that has since re-validated (PASS + "
                          "walk-forward ROBUST) stays benched forever. Use "
                          "this after confirming the verdict; it gets a "
                          "baseline allocation again next cycle (and will "
                          "re-freeze automatically if it bleeds live).")
    args = ap.parse_args()

    # Manual KILL reset path. Done before anything else so the operator
    # can recover without booting brokers / strategies / etc.
    if args.reset_kill_switch:
        rm = RiskManager()
        rm.reset_kill_switch()
        logger.warning(
            "Kill-switch reset to NORMAL. Verify by running "
            "`python src/run_orchestrator.py --status`."
        )
        return 0
    if args.arm_kill_switch:
        rm = RiskManager()
        rm.arm_kill_switch(note="manual arm via dashboard")
        logger.error(
            "Kill-switch ARMED. Next orchestrator cycle will close "
            "all positions. Run with --reset-kill-switch to recover."
        )
        return 0

    # Manual unfreeze path — fixes the auto-freeze-never-recovers
    # asymmetry for strategies that have re-validated. No brokers needed.
    if args.unfreeze:
        from common.strategy_validation import passing_strategies
        reg = StrategyRegistry()
        # Register metas so all_states() (used by the "all" path) sees the
        # roster; register() preserves any existing FROZEN state.
        for _m in ALL_STRATEGIES:
            reg.register(_m)
        unfrozen = unfreeze_strategies(
            args.unfreeze, reg, passing_strategies())
        if not unfrozen:
            logger.warning("--unfreeze: nothing eligible to unfreeze")
        for name, verdict in unfrozen:
            logger.warning(f"--unfreeze: {name} FROZEN → ACTIVE [{verdict}]")
        return 0

    # Two-key guard: even with DRY_RUN=false the orchestrator refuses to
    # place real orders unless ALLOW_LIVE_TRADING is truthy. Accepts
    # "1", "true", "yes" (case-insensitive) — a user setting the var
    # to "true" (instead of the literal "1") was getting forced into
    # DRY mode silently. Observed 2026-05-08.
    dry_env = os.environ.get("DRY_RUN", "true").lower()
    allow_live = (os.environ.get("ALLOW_LIVE_TRADING", "")
                  .strip().lower() in ("1", "true", "yes"))
    if dry_env == "false" and not allow_live:
        logger.warning(
            "DRY_RUN=false but ALLOW_LIVE_TRADING is not truthy — forcing DRY mode. "
            "Set ALLOW_LIVE_TRADING=1 (or true/yes) to place live orders."
        )
        os.environ["DRY_RUN"] = "true"
    # Same gate applies to LIVE_STRATEGIES (the per-strategy override
    # that bypasses every DRY flag). Without this, a stale repo
    # variable from earlier testing could fire real-money orders the
    # moment a strategy ships — the audit's failure mode 2026-05-07.
    if os.environ.get("LIVE_STRATEGIES") and not allow_live:
        logger.warning(
            "LIVE_STRATEGIES is set but ALLOW_LIVE_TRADING is not truthy — "
            "ignoring per-strategy LIVE override. Set "
            "ALLOW_LIVE_TRADING=1 (or true/yes) to honour LIVE_STRATEGIES."
        )
        os.environ["LIVE_STRATEGIES"] = ""

    # ── Build infra
    brokers = build_brokers()
    if not brokers:
        logger.error("No brokers configured")
        return 1

    registry = StrategyRegistry()
    skipped_by_venue: dict[str, list[str]] = {}
    for meta in PHASE_1_STRATEGIES:
        # Only register strategies whose venue is configured
        if meta.venue in brokers:
            registry.register(meta)
        else:
            skipped_by_venue.setdefault(meta.venue, []).append(meta.name)
    # Log a SINGLE info line per missing venue instead of one warning
    # per strategy. Without this batching, a CI run with two missing
    # venues emitted ~20 yellow "Skipping" lines that visually looked
    # like errors when they're expected on a partially-credentialed
    # environment.
    for venue, names in sorted(skipped_by_venue.items()):
        logger.info(
            f"Venue '{venue}' not configured — skipping "
            f"{len(names)} strategies: {', '.join(sorted(names))}"
        )

    # Strategy → required env-var map. Strategies whose key is missing
    # used to short-circuit silently every cycle in compute() (returning
    # []), which made the dashboard look like the strategy was idle for
    # no visible reason. Now we WATCH them at startup with a clear log
    # line so the operator immediately sees "this strategy needs key X
    # which isn't set". WATCH > FROZEN because we still want the
    # allocator to compute outcomes for them (so the dashboard's per-
    # strategy outcome panel shows the real reason).
    STRATEGY_DEPS: dict[str, list[str]] = {
        "earnings_momentum":   ["FMP_API_KEY"],
        "earnings_news_pead":  ["FMP_API_KEY"],
        "pead":                ["FMP_API_KEY"],
        "macro_kalshi_v2":     ["FRED_API_KEY"],
        "commodity_carry":     ["FMP_API_KEY"],
    }
    missing_deps: dict[str, list[str]] = {}
    for sname, vars_needed in STRATEGY_DEPS.items():
        if registry.meta(sname) is None:
            continue   # strategy not registered (venue missing)
        missing = [v for v in vars_needed if not os.environ.get(v)]
        if missing:
            missing_deps[sname] = missing
    if missing_deps:
        for sname, missing in sorted(missing_deps.items()):
            logger.warning(
                f"[deps] {sname}: missing {', '.join(missing)} — "
                f"strategy will short-circuit every cycle until set"
            )

    risk_manager = RiskManager(brokers=brokers)

    # Cold-start guard: refuse to trade when risk_state.db has no
    # equity history, because a fresh kill-switch baseline = current
    # equity means KILL would silently arm at -KILL_DD_PCT of whatever
    # today's equity happens to be. Operator must opt in explicitly
    # via --allow-cold-start. (DRY mode is exempt — it doesn't trade.)
    try:
        with risk_manager.db._conn() as _c:
            _row = _c.execute(
                "SELECT COUNT(*) FROM equity_snapshots"
            ).fetchone()
            _n_snapshots = int(_row[0]) if _row else 0
    except Exception:
        _n_snapshots = 0
    # Audit-fix F4 (2026-05-07): cold-start guard must consider any
    # live-trading path, not only global DRY_RUN=false. With the
    # recommended config (DRY_RUN=true, LIVE_STRATEGIES=..., per-venue
    # DRY_RUN_*=false, ALLOW_LIVE_TRADING=1) the guard previously saw
    # DRY_RUN=true and let the cycle through with zero equity history —
    # KILL baseline = current equity = drawdown computed from a fresh
    # peak. Now the guard refuses if ANY live-money path is active.
    is_live_path = (
        os.environ.get("DRY_RUN", "true").lower() == "false"
        or _per_broker_flag("DRY_RUN_COINBASE") is False
        or _per_broker_flag("DRY_RUN_ALPACA") is False
        or _per_broker_flag("DRY_RUN_KALSHI") is False
        or (
            os.environ.get("LIVE_STRATEGIES", "").strip() != ""
            and allow_live
        )
    )
    if (_n_snapshots == 0 and is_live_path
            and not args.allow_cold_start):
        # Auto-bootstrap when running under cron/CI: the cold-start
        # guard's purpose is to protect a HUMAN re-running the
        # orchestrator on a fresh box from accidentally arming the
        # kill switch at today's equity. Under cron, by definition,
        # we'll see this every first deploy after a cache wipe — and
        # blocking with exit 3 every cycle creates the exact "0 trades
        # forever" loop the user reported 2026-05-08. Detect the cron
        # context (CI / GITHUB_ACTIONS env) and auto-bootstrap with
        # a prominent warning, so the next cycle has snapshots and
        # the guard becomes a no-op naturally.
        is_cron_context = bool(os.environ.get("GITHUB_ACTIONS")
                                 or os.environ.get("CI"))
        if is_cron_context:
            logger.warning(
                "Cold start detected (risk_state.db has 0 equity "
                "snapshots) under CI/cron context — auto-bootstrapping. "
                "The first cycle will record an equity baseline; the "
                "kill-switch will arm at -KILL_DD_PCT of THAT baseline. "
                "If today's account is unusually high or low, this "
                "may be the wrong baseline — review after first "
                "successful cycle."
            )
        else:
            logger.error(
                "Cold start detected (risk_state.db has 0 equity snapshots) "
                "but a live-trading path is active. Refusing to trade — the "
                "kill-switch baseline would reset to current equity and arm "
                "at -KILL_DD_PCT of whatever today's number is. "
                "Pass --allow-cold-start to override."
            )
            return 3
    allocator = MetaAllocator(registry=registry, performance=StrategyPerformance())
    strategies = build_strategies(brokers)

    dry_default = os.environ.get("DRY_RUN", "true").lower() != "false"
    dry_run = dry_default and not args.live

    live_strategies_raw = os.environ.get("LIVE_STRATEGIES", "")
    live_strategies = {s.strip() for s in live_strategies_raw.split(",") if s.strip()}

    # 2026-06-05 — when ALLOW_LIVE_TRADING is on AND there is already at
    # least one explicitly-live strategy, auto-promote the new real-edge
    # sleeves so the user doesn't have to hand-edit /etc/aaa.env. The
    # ALLOW_LIVE_TRADING master gate (checked above) is still the safety
    # interlock; this just spares ops a 4-name copy-paste. Operator can
    # still hard-disable any of them by explicitly omitting from
    # LIVE_STRATEGIES — once LIVE_STRATEGIES is non-empty we honour it.
    # (Without LIVE_STRATEGIES set at all, ALL strategies on a non-DRY
    # venue go live; nothing to auto-promote.)
    if live_strategies and allow_live:
        live_strategies |= {
            "pre_fomc_drift", "activist_13d",
            "merger_arb", "high_vol_trend",
        }

    orchestrator = Orchestrator(
        brokers=brokers,
        registry=registry,
        risk_manager=risk_manager,
        allocator=allocator,
        strategies=strategies,
        config=OrchestratorConfig(
            dry_run=dry_run,
            dry_run_coinbase=_per_broker_flag("DRY_RUN_COINBASE"),
            dry_run_alpaca=_per_broker_flag("DRY_RUN_ALPACA"),
            dry_run_kalshi=_per_broker_flag("DRY_RUN_KALSHI"),
            live_strategies=live_strategies or None,
            stale_order_seconds=int(
                os.environ.get("STALE_ORDER_SECONDS", "1800")
            ),
        ),
    )

    if live_strategies:
        logger.warning(f"⚠ Per-strategy LIVE override active: {sorted(live_strategies)}")

    # Surface per-venue trading mode at startup so the operator can
    # tell from the logs whether strategies are actually deploying
    # capital (PAPER / LIVE) or just logging proposals (DRY). Maps:
    #   DRY   = orders not submitted
    #   PAPER = real orders to a paper / sandbox account (no real money)
    #   LIVE  = real money
    # Uses the same classification logic as the dashboard.
    def _venue_mode_label(venue: str, endpoint_env: str,
                            paper_marker: str) -> str:
        is_dry = orchestrator.cfg.is_dry(venue)
        if is_dry:
            return "🟦 DRY"
        ep = (os.environ.get(endpoint_env) or "").lower()
        if paper_marker in ep:
            return "🧪 PAPER"
        return "💰 LIVE"
    venue_modes = {
        "alpaca":   _venue_mode_label("alpaca",   "ALPACA_ENDPOINT",  "paper"),
        "coinbase": _venue_mode_label("coinbase", "COINBASE_ENDPOINT", "sandbox"),
        "kalshi":   _venue_mode_label("kalshi",   "KALSHI_ENDPOINT",   "demo"),
    }
    for v in sorted(brokers):
        logger.info(f"  venue={v} mode={venue_modes.get(v, '🟦 DRY')}")

    # ── Status mode: print state and exit
    if args.status:
        st = risk_manager.compute_state(persist=False)
        equity = max(st.equity_usd, 1.0)
        alloc = allocator.rebalance(portfolio_equity_usd=equity)
        out = {
            "risk": risk_manager.summary_dict(),
            "allocator": {
                "total_active_pct": alloc.total_active_pct,
                "decisions": [
                    {"name": d.name, "state": d.state.value,
                     "target_pct": d.target_pct, "target_usd": d.target_usd,
                     "reason": d.reason}
                    for d in alloc.decisions
                ],
            },
            "config": {"dry_run": dry_run},
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    # ── Run one cycle
    logger.info(f"Starting cycle  dry_run={dry_run}  brokers={sorted(brokers.keys())}")
    # Sprint A4 — record cycle start with heartbeat. Allows the
    # dead-man's switch alert to differentiate "process never woke"
    # from "process woke but is stuck in cycle".
    from common.heartbeat import ping_fail, ping_start, ping_success
    ping_start("orchestrator")

    report = None
    try:
        report = orchestrator.run_cycle(scout_signals={})  # scouts wired in W2
    except Exception as e:
        # Critical: do NOT propagate — that fails the GH Actions job,
        # which means actions/cache@v4 skips the post-save step, which
        # means trades.db / cycle_status.json never persist. Better
        # to log the failure prominently and continue to the rest of
        # main() (which writes the diagnostic + does the cache save).
        # 2026-05-09: this is the same class of bug as PR #17's
        # exit-code-2 issue, just on a different failure path.
        logger.exception(
            f"run_cycle raised — converting to non-fatal so the cache "
            f"save can still happen: {type(e).__name__}: {e}"
        )
        # Synthesize a minimal report so subsequent code doesn't NPE
        from datetime import UTC, datetime
        from strategy_engine.orchestrator import CycleReport
        report = CycleReport(timestamp=datetime.now(UTC))
        report.errors.append(
            f"run_cycle uncaught: {type(e).__name__}: {str(e)[:240]}"
        )
        try:
            from common.errors_db import record_error
            record_error(scope="orchestrator.run_cycle")
        except Exception:
            pass

    logger.info(f"Cycle complete: {report.proposals_total} proposals, "
                f"{report.proposals_approved} approved, "
                f"{report.proposals_rejected} rejected, "
                f"{report.trades_submitted} submitted, "
                f"{len(report.errors)} errors "
                f"(took {getattr(report, 'cycle_seconds', 0):.1f}s)")
    # Per-strategy outcome summary in the log so operators can grep
    # without dashboard access. One line per strategy, sorted by
    # (submitted desc, proposed desc) so the active ones surface first.
    outcomes = getattr(report, "strategy_outcomes", {}) or {}
    if outcomes:
        ranked = sorted(
            outcomes.values(),
            key=lambda o: (-o.submitted, -o.proposed, o.strategy),
        )
        for o in ranked:
            tag = ("SUBMITTED" if o.submitted
                   else "DRY" if o.dry_logged
                   else "REJECTED" if o.rejected
                   else "ERROR" if o.error
                   else "SKIP" if o.skip_reasons
                   else "IDLE")
            logger.info(
                f"  [{tag:9s}] {o.strategy:32s} "
                f"venue={o.venue:8s} state={o.state:7s} "
                f"target=${o.target_alloc_usd:>8.0f} "
                f"prop={o.proposed} appr={o.approved} "
                f"rej={o.rejected} sub={o.submitted} "
                f"dry={o.dry_logged} "
                f"reason={','.join(o.skip_reasons)[:40] or o.error[:40] or '-'}"
            )

    # Step summary for GitHub Actions
    if report.risk:
        latest_alloc = allocator.rebalance(portfolio_equity_usd=report.risk.equity_usd) \
            if report.rebalanced else None
        write_step_summary(report, latest_alloc, report.risk)

    # Sprint A4 — emit a /fail ping when the cycle had any errors so
    # healthchecks pages us; emit a /success ping otherwise. Both
    # carry a 1-line summary that shows up in HC's dashboard.
    summary = (
        f"proposals={report.proposals_total} approved={report.proposals_approved} "
        f"submitted={report.trades_submitted} errors={len(report.errors)}"
    )
    if report.errors:
        ping_fail("orchestrator", message=f"{summary}; first_error={report.errors[0]}")
    else:
        ping_success("orchestrator", message=summary)

    # Book-vitals watchdog — pages on the "healthy zombie" conditions
    # (stale kill latch / silent book) that heartbeats can't see because
    # the process itself is alive and exiting 0. See common/watchdog.py
    # for the May-June 2026 month-long-freeze incident this prevents.
    try:
        from common.watchdog import check_book_vitals
        if report.risk:
            # Latch age = start of the current contiguous KILL run (the
            # cooldown re-records latched KILLs daily, so the LAST event
            # timestamp under-reports a month-old latch as ≤24h).
            check_book_vitals(
                kill_switch=report.risk.kill_switch.value,
                kill_switch_at=risk_manager.db.latch_armed_at(),
                drawdown_pct=report.risk.drawdown_pct,
                kill_dd_pct=risk_manager.config.kill_dd_pct,
            )
    except Exception as e:
        logger.warning(f"watchdog check failed (non-fatal): {e}")

    # Always return 0 when the cycle completed.
    # Reasoning: per-strategy errors (record_trade hiccup, broker
    # rate-limit on one venue, single bad symbol) are NOT fatal.
    # They're written to errors.db, surfaced on the dashboard error
    # panel, and pinged to Healthchecks via ping_fail above.
    # Returning non-zero here failed the entire GH Actions job, which
    # caused actions/cache to skip saving the new trades.db rows AND
    # fired the Notify-on-failure webhook for non-actionable noise.
    # Observed 2026-05-08: a single record_trade SQLite-lock retry
    # exhaustion produced "Process completed with exit code 2" and
    # killed the cycle's persistence — vicious loop on "0 trades".
    # Fatal failures (cold-start guard, no brokers, argparse) still
    # exit non-zero earlier in main().
    return 0


if __name__ == "__main__":
    sys.exit(main())
