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
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from allocator.allocator import AllocatorConfig, MetaAllocator
from allocator.lifecycle import StrategyMeta, StrategyRegistry, StrategyState
from allocator.metrics import StrategyPerformance
from brokers.registry import build_brokers
from risk.manager import RiskManager
from risk.policies import RiskConfig
from strategies import CryptoFundingCarry, KalshiCalibrationArb, RiskParityETF
from strategy_engine.orchestrator import Orchestrator, OrchestratorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")

DRY_RUN_DEFAULT = True


# ─── Strategy wiring ──────────────────────────────────────────────────────


PHASE_1_STRATEGIES = [
    StrategyMeta(
        name="crypto_funding_carry",
        asset_classes=["CRYPTO_PERP"], venue="coinbase",
        target_alloc_pct=0.30, max_alloc_pct=0.40, min_alloc_pct=0.05,
        description="Long spot / short perp; capture funding rate",
    ),
    StrategyMeta(
        name="risk_parity_etf",
        asset_classes=["ETF"], venue="alpaca",
        target_alloc_pct=0.50, max_alloc_pct=0.60, min_alloc_pct=0.30,
        description="Inverse-vol ETF book (SPY/TLT/IEF/GLD/DBC)",
    ),
    StrategyMeta(
        name="kalshi_calibration_arb",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.15, max_alloc_pct=0.20, min_alloc_pct=0.05,
        description="Favorite-longshot bias arb on Kalshi",
    ),
]


def build_strategies(brokers):
    instances = {}
    if "coinbase" in brokers:
        instances["crypto_funding_carry"] = CryptoFundingCarry(brokers["coinbase"])
    if "alpaca" in brokers:
        instances["risk_parity_etf"] = RiskParityETF(brokers["alpaca"])
    if "kalshi" in brokers:
        instances["kalshi_calibration_arb"] = KalshiCalibrationArb(brokers["kalshi"])
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
    orchestrator = Orchestrator(
        brokers=brokers,
        registry=registry,
        risk_manager=risk_manager,
        allocator=allocator,
        strategies=strategies,
        config=OrchestratorConfig(dry_run=dry_run),
    )

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

    return 0 if not report.errors else 2


if __name__ == "__main__":
    sys.exit(main())
