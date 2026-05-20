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
    BollingerBreakout,
    CommodityCarry,
    CrossVenueArb,
    CryptoBasisTrade,
    CryptoBreakout,
    CryptoFundingCarry,
    CryptoPairsTrading,
    CryptoVolRegimeOverlay,
    CryptoFundingCarryV2,
    CryptoXSMom,
    DividendGrowth,
    EarningsMomentum,
    EarningsNewsPEAD,
    GapTrading,
    InternationalsRotation,
    KalshiCalibrationArb,
    LeveragedMomentum,
    LowVolAnomaly,
    MacroKalshi,
    MacroKalshiV2,
    MultiFactorEquity,
    PairsTrading,
    RiskParityETF,
    RSIMeanReversion,
    SectorRotation,
    ThematicGrowth,
    TSMomETF,
    TurnOfMonth,
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
    # ── Phase 1 (rebalanced down to make room for P4 experimental sleeve)
    StrategyMeta(
        name="crypto_funding_carry",
        asset_classes=["CRYPTO_PERP"], venue="coinbase",
        target_alloc_pct=0.12, max_alloc_pct=0.25, min_alloc_pct=0.04,
        description="Long spot / short perp; capture funding rate (P1)",
    ),
    StrategyMeta(
        name="risk_parity_etf",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.22, max_alloc_pct=0.32, min_alloc_pct=0.15,
        description="Inverse-vol ETF book (SPY/TLT/IEF/GLD/DBC) (P1)",
    ),
    StrategyMeta(
        name="kalshi_calibration_arb",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.02, max_alloc_pct=0.06, min_alloc_pct=0.005,
        description="Favorite-longshot bias arb on Kalshi (P1)",
    ),
    # ── Phase 2
    StrategyMeta(
        name="crypto_basis_trade",
        asset_classes=["CRYPTO_FUTURE"], venue="coinbase",
        target_alloc_pct=0.1, max_alloc_pct=0.2, min_alloc_pct=0.03,
        description="Long spot / short dated future on Coinbase (P2)",
    ),
    StrategyMeta(
        name="tsmom_etf",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.16, max_alloc_pct=0.28, min_alloc_pct=0.06,
        description="12-1m time-series momentum on 7-ETF basket (P2)",
    ),
    StrategyMeta(
        name="commodity_carry",
        asset_classes=["COMMODITY_FUTURE"], venue="coinbase",
        target_alloc_pct=0.06, max_alloc_pct=0.18, min_alloc_pct=0.02,
        description="Top-N backwardated commodity futures (P2)",
    ),
    # ── Phase 3
    # PEAD v1 retired 2026-05-07. The gap-only proxy for "earnings
    # surprise" is a known weak signal vs the v2 (RSS news corroboration)
    # and earnings_momentum (true EPS-surprise via FMP) variants;
    # running all three triple-trades the same earnings prints and
    # inflates correlation. Weight reallocated to v2 + earnings_momentum.
    # The pead.py module is kept in-tree for backtest reference only.
    StrategyMeta(
        name="macro_kalshi",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Kalshi macro events vs implied probabilities (P3)",
    ),
    StrategyMeta(
        name="crypto_xsmom",
        asset_classes=["CRYPTO_SPOT"], venue="coinbase",
        target_alloc_pct=0.07, max_alloc_pct=0.15, min_alloc_pct=0.03,
        description="Cross-sectional momentum on top-15 alts (P3)",
    ),
    StrategyMeta(
        name="vol_managed_overlay",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.00, max_alloc_pct=0.00, min_alloc_pct=0.0,
        description="Vol-target multiplier; publishes scaler only, no trades (P3)",
    ),
    # ── Flagship — institutional multi-factor cross-sectional equity
    # model (2026-05-19). The highest-sophistication sleeve in the
    # book: momentum + low-vol + reversal composite, cross-sectionally
    # z-scored, sector-neutralised, vol-targeted, turnover-controlled.
    # Sized as a core sleeve alongside risk_parity + tsmom.
    StrategyMeta(
        name="multifactor_equity",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.14, max_alloc_pct=0.28, min_alloc_pct=0.06,
        description="Multi-factor (mom+lowvol+reversal) x-sectional equity (flagship)",
    ),
    # ── User-requested experimental sleeves. Both REGISTER SMALL and
    # stay DRY until the validation harness (docs/validation.json)
    # records PASS — see CLAUDE.md + common/strategy_validation.py.
    # leveraged_momentum: 3x ETFs gated on uptrend + low-vol regime
    # with a -15% hard stop; deliberately tiny so a wipeout is bounded.
    StrategyMeta(
        name="leveraged_momentum",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.02, max_alloc_pct=0.05, min_alloc_pct=0.0,
        description="3x leveraged ETF trend (TQQQ/UPRO/SOXL/TNA, regime-gated)",
    ),
    # thematic_growth: curated 2026 themes (AI compute, AI power,
    # cybersec, defense, GLP-1, robotics, quantum) — within-theme
    # 6m-momentum rank picks the winners; cross-theme by conviction.
    StrategyMeta(
        name="thematic_growth",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.025, max_alloc_pct=0.08, min_alloc_pct=0.0,
        description="Thematic basket (AI compute/power, cyber, defense, GLP-1, robotics)",
    ),
    # ── Phase 4 — EXPERIMENTAL (small initial allocations on Alpaca
    # paper $100k). Allocator's Sharpe-tilt will reallocate to
    # winners over the first 30-60 days. Each starts at 4%.
    StrategyMeta(
        name="rsi_mean_reversion",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Connors-style RSI(2) mean-reversion on 30 large-caps (P4)",
    ),
    StrategyMeta(
        name="sector_rotation",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.02, max_alloc_pct=0.06, min_alloc_pct=0.005,
        description="Top-N SPDR sector ETFs by 90d return (P4)",
    ),
    StrategyMeta(
        name="pairs_trading",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Stat-arb on 6 classic correlated pairs (P4)",
    ),
    StrategyMeta(
        name="bollinger_breakout",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Momentum continuation on 20d Bollinger upper-band breaks (P4)",
    ),
    StrategyMeta(
        name="earnings_momentum",
        asset_classes=["EQUITY"], venue="alpaca",
        # +1% baseline / +3% max ceiling (was 4/15) to absorb the
        # retired pead v1 allocation. Cleaner EPS-surprise signal so
        # this should produce higher Sharpe than v1's gap-only proxy.
        target_alloc_pct=0.09, max_alloc_pct=0.2, min_alloc_pct=0.03,
        description="Live PEAD via FMP earnings calendar (P4)",
    ),
    StrategyMeta(
        name="dividend_growth",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Quality-dividend ETF rotation by 90d return (P4)",
    ),
    # ── Phase 4b — additional experimental strategies for Alpaca
    # paper $100k (audit recommendation: experiment more, identify
    # winners, double down). Each starts at 3% baseline; champion
    # tier kicks in at Sharpe ≥ 1.0 + 10 trades.
    StrategyMeta(
        name="gap_trading",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Overnight-gap reversion on S&P 100 (P4b)",
    ),
    StrategyMeta(
        name="turn_of_month",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Calendar seasonal: SPY around month boundaries (P4b)",
    ),
    StrategyMeta(
        name="low_vol_anomaly",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Lowest-vol ETFs + stocks with positive trend (P4b)",
    ),
    StrategyMeta(
        name="internationals_rotation",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="International country-ETF momentum vs SPY (P4b)",
    ),
    # ── Phase 5 — strategies consuming the new data feeds (Sprint C)
    StrategyMeta(
        name="macro_kalshi_v2",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.04, max_alloc_pct=0.1, min_alloc_pct=0.01,
        description="Kalshi-vs-CME Fed-rate divergence (P5, CME-fed)",
    ),
    StrategyMeta(
        name="cross_venue_arb",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.02, max_alloc_pct=0.06, min_alloc_pct=0.005,
        description="Kalshi vs Polymarket cross-venue arbitrage (P5)",
    ),
    StrategyMeta(
        name="crypto_funding_carry_v2",
        asset_classes=["CRYPTO_PERP"], venue="coinbase",
        # Smaller than v1 (12%) until we have paper-P&L data showing
        # the multi-venue gate adds Sharpe rather than just shrinking
        # opportunity. Champion tier auto-promotes if it earns it.
        target_alloc_pct=0.06, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Funding carry gated on Coinbase+Binance consensus (P5)",
    ),
    StrategyMeta(
        name="earnings_news_pead",
        asset_classes=["EQUITY"], venue="alpaca",
        # +2% baseline (was 3%) to absorb part of retired pead v1.
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="PEAD gated on RSS news corroboration (P5)",
    ),
    # ── Phase 6 — advanced crypto strategies (2026-05-08).
    # All three default to DRY: opt into real money by adding the
    # name to LIVE_STRATEGIES + ensuring ALLOW_LIVE_TRADING=1.
    StrategyMeta(
        name="crypto_pairs_trading",
        asset_classes=["CRYPTO_SPOT", "CRYPTO_PERP"], venue="coinbase",
        target_alloc_pct=0.005, max_alloc_pct=0.015, min_alloc_pct=0.0,
        description="Stat-arb on BTC/ETH and ETH/SOL price ratios (P6)",
    ),
    StrategyMeta(
        name="crypto_breakout",
        asset_classes=["CRYPTO_SPOT"], venue="coinbase",
        target_alloc_pct=0.02, max_alloc_pct=0.06, min_alloc_pct=0.005,
        description="Donchian 30d-high breakout w/ trail-stop (P6)",
    ),
    StrategyMeta(
        name="crypto_vol_regime_overlay",
        asset_classes=["CRYPTO_SPOT"], venue="coinbase",
        target_alloc_pct=0.00, max_alloc_pct=0.00, min_alloc_pct=0.0,
        description="Publishes crypto vol-regime scaler; no trades (P6)",
    ),
]

# Backward compat alias used by older test scripts
PHASE_1_STRATEGIES = ALL_STRATEGIES


def build_strategies(brokers):
    instances = {}
    if "coinbase" in brokers:
        cb = brokers["coinbase"]
        instances["crypto_funding_carry"] = CryptoFundingCarry(cb)
        instances["crypto_basis_trade"] = CryptoBasisTrade(cb)
        instances["commodity_carry"] = CommodityCarry(cb)
        instances["crypto_xsmom"] = CryptoXSMom(cb)
        # Phase 5 — multi-venue consensus version
        instances["crypto_funding_carry_v2"] = CryptoFundingCarryV2(cb)
        # Phase 6 — advanced crypto strategies
        instances["crypto_pairs_trading"] = CryptoPairsTrading(cb)
        instances["crypto_breakout"] = CryptoBreakout(cb)
        instances["crypto_vol_regime_overlay"] = CryptoVolRegimeOverlay(cb)
    if "alpaca" in brokers:
        al = brokers["alpaca"]
        instances["risk_parity_etf"] = RiskParityETF(al)
        instances["tsmom_etf"] = TSMomETF(al)
        # pead (v1) retired 2026-05-07 — see ALL_STRATEGIES note above.
        # Module still imported so existing trade rows remain readable
        # in the dashboard / FIFO recompute, but no instance is wired.
        instances["vol_managed_overlay"] = VolManagedOverlay(al)
        # Phase 4 — experimental sleeve
        instances["rsi_mean_reversion"] = RSIMeanReversion(al)
        instances["sector_rotation"] = SectorRotation(al)
        instances["pairs_trading"] = PairsTrading(al)
        instances["bollinger_breakout"] = BollingerBreakout(al)
        instances["earnings_momentum"] = EarningsMomentum(al)
        instances["dividend_growth"] = DividendGrowth(al)
        # Phase 4b — additional experimental sleeve
        instances["gap_trading"] = GapTrading(al)
        instances["turn_of_month"] = TurnOfMonth(al)
        instances["low_vol_anomaly"] = LowVolAnomaly(al)
        instances["internationals_rotation"] = InternationalsRotation(al)
        # Flagship — institutional multi-factor cross-sectional model
        instances["multifactor_equity"] = MultiFactorEquity(al)
        # User-requested: leveraged trend + thematic basket. Tiny
        # DRY allocations until the validation harness PASSes them.
        instances["leveraged_momentum"] = LeveragedMomentum(al)
        instances["thematic_growth"] = ThematicGrowth(al)
        # Phase 5 — Alpaca-side new-feed strategy
        instances["earnings_news_pead"] = EarningsNewsPEAD(al)
    if "kalshi" in brokers:
        ks = brokers["kalshi"]
        instances["kalshi_calibration_arb"] = KalshiCalibrationArb(ks)
        instances["macro_kalshi"] = MacroKalshi(ks)
        # Phase 5 — strategies consuming the new Sprint-3 data feeds
        instances["macro_kalshi_v2"] = MacroKalshiV2(ks)
        instances["cross_venue_arb"] = CrossVenueArb(ks)
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
