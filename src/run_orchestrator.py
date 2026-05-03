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
    CryptoBasisTrade,
    CryptoFundingCarry,
    CryptoXSMom,
    DividendGrowth,
    EarningsMomentum,
    GapTrading,
    InternationalsRotation,
    KalshiCalibrationArb,
    LowVolAnomaly,
    MacroKalshi,
    PairsTrading,
    PEAD,
    RiskParityETF,
    RSIMeanReversion,
    SectorRotation,
    TSMomETF,
    TurnOfMonth,
    VolManagedOverlay,
)
from strategy_engine.orchestrator import Orchestrator, OrchestratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")

DRY_RUN_DEFAULT = True


# ─── Strategy wiring ──────────────────────────────────────────────────────


ALL_STRATEGIES = [
    # ── Phase 1 (rebalanced down to make room for P4 experimental sleeve)
    StrategyMeta(
        name="crypto_funding_carry",
        asset_classes=["CRYPTO_PERP"], venue="coinbase",
        target_alloc_pct=0.12, max_alloc_pct=0.25, min_alloc_pct=0.03,
        description="Long spot / short perp; capture funding rate (P1)",
    ),
    StrategyMeta(
        name="risk_parity_etf",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.22, max_alloc_pct=0.30, min_alloc_pct=0.15,
        description="Inverse-vol ETF book (SPY/TLT/IEF/GLD/DBC) (P1)",
    ),
    StrategyMeta(
        name="kalshi_calibration_arb",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.04, max_alloc_pct=0.10, min_alloc_pct=0.02,
        description="Favorite-longshot bias arb on Kalshi (P1)",
    ),
    # ── Phase 2
    StrategyMeta(
        name="crypto_basis_trade",
        asset_classes=["CRYPTO_FUTURE"], venue="coinbase",
        target_alloc_pct=0.08, max_alloc_pct=0.20, min_alloc_pct=0.02,
        description="Long spot / short dated future on Coinbase (P2)",
    ),
    StrategyMeta(
        name="tsmom_etf",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.13, max_alloc_pct=0.25, min_alloc_pct=0.05,
        description="12-1m time-series momentum on 7-ETF basket (P2)",
    ),
    StrategyMeta(
        name="commodity_carry",
        asset_classes=["COMMODITY_FUTURE"], venue="coinbase",
        target_alloc_pct=0.07, max_alloc_pct=0.18, min_alloc_pct=0.03,
        description="Top-N backwardated commodity futures (P2)",
    ),
    # ── Phase 3
    StrategyMeta(
        name="pead",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.12, min_alloc_pct=0.02,
        description="Post-earnings announcement drift (P3, scout-fed)",
    ),
    StrategyMeta(
        name="macro_kalshi",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.03, max_alloc_pct=0.10, min_alloc_pct=0.01,
        description="Kalshi macro events vs implied probabilities (P3)",
    ),
    StrategyMeta(
        name="crypto_xsmom",
        asset_classes=["CRYPTO_SPOT"], venue="coinbase",
        target_alloc_pct=0.03, max_alloc_pct=0.10, min_alloc_pct=0.02,
        description="Cross-sectional momentum on top-15 alts (P3)",
    ),
    StrategyMeta(
        name="vol_managed_overlay",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.00, max_alloc_pct=0.00, min_alloc_pct=0.0,
        description="Vol-target multiplier; publishes scaler only, no trades (P3)",
    ),
    # ── Phase 4 — EXPERIMENTAL (small initial allocations on Alpaca
    # paper $100k). Allocator's Sharpe-tilt will reallocate to
    # winners over the first 30-60 days. Each starts at 4%.
    StrategyMeta(
        name="rsi_mean_reversion",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Connors-style RSI(2) mean-reversion on 30 large-caps (P4)",
    ),
    StrategyMeta(
        name="sector_rotation",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Top-N SPDR sector ETFs by 90d return (P4)",
    ),
    StrategyMeta(
        name="pairs_trading",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Stat-arb on 6 classic correlated pairs (P4)",
    ),
    StrategyMeta(
        name="bollinger_breakout",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Momentum continuation on 20d Bollinger upper-band breaks (P4)",
    ),
    StrategyMeta(
        name="earnings_momentum",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Live PEAD via FMP earnings calendar (P4)",
    ),
    StrategyMeta(
        name="dividend_growth",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Quality-dividend ETF rotation by 90d return (P4)",
    ),
    # ── Phase 4b — additional experimental strategies for Alpaca
    # paper $100k (audit recommendation: experiment more, identify
    # winners, double down). Each starts at 3% baseline; champion
    # tier kicks in at Sharpe ≥ 1.0 + 10 trades.
    StrategyMeta(
        name="gap_trading",
        asset_classes=["EQUITY"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.12, min_alloc_pct=0.01,
        description="Overnight-gap reversion on S&P 100 (P4b)",
    ),
    StrategyMeta(
        name="turn_of_month",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.10, min_alloc_pct=0.01,
        description="Calendar seasonal: SPY around month boundaries (P4b)",
    ),
    StrategyMeta(
        name="low_vol_anomaly",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.15, min_alloc_pct=0.01,
        description="Lowest-vol ETFs + stocks with positive trend (P4b)",
    ),
    StrategyMeta(
        name="internationals_rotation",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.03, max_alloc_pct=0.12, min_alloc_pct=0.01,
        description="International country-ETF momentum vs SPY (P4b)",
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
    if "alpaca" in brokers:
        al = brokers["alpaca"]
        instances["risk_parity_etf"] = RiskParityETF(al)
        instances["tsmom_etf"] = TSMomETF(al)
        instances["pead"] = PEAD(al)
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
    if "kalshi" in brokers:
        ks = brokers["kalshi"]
        instances["kalshi_calibration_arb"] = KalshiCalibrationArb(ks)
        instances["macro_kalshi"] = MacroKalshi(ks)
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
    args = ap.parse_args()

    # ── Build infra
    brokers = build_brokers()
    if not brokers:
        logger.error("No brokers configured")
        return 1

    registry = StrategyRegistry()
    for meta in PHASE_1_STRATEGIES:
        # Only register strategies whose venue is configured
        if meta.venue in brokers:
            registry.register(meta)
        else:
            logger.warning(f"Skipping {meta.name}: venue {meta.venue} not configured")

    risk_manager = RiskManager(brokers=brokers)
    allocator = MetaAllocator(registry=registry, performance=StrategyPerformance())
    strategies = build_strategies(brokers)

    dry_default = os.environ.get("DRY_RUN", "true").lower() != "false"
    dry_run = dry_default and not args.live

    def _per_broker_flag(envvar: str):
        v = os.environ.get(envvar)
        if v is None:
            return None
        return v.lower() != "false"

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
        ),
    )

    if live_strategies:
        logger.warning(f"⚠ Per-strategy LIVE override active: {sorted(live_strategies)}")

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

    report = orchestrator.run_cycle(scout_signals={})  # scouts wired in W2

    logger.info(f"Cycle complete: {report.proposals_total} proposals, "
                f"{report.proposals_approved} approved, "
                f"{report.proposals_rejected} rejected, "
                f"{report.trades_submitted} submitted, "
                f"{len(report.errors)} errors")

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

    return 0 if not report.errors else 2


if __name__ == "__main__":
    sys.exit(main())
