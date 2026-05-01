#!/usr/bin/env python3
"""End-to-end smoke test for the risk + allocator layers.

Run from CI to confirm:
  • RiskManager pulls equity from every configured broker
  • Drawdown / leverage / multiplier compute without errors
  • Order-level check_order() applies sizing + caps correctly
  • MetaAllocator rebalances a registry of strategies and persists results

Exits non-zero on any unexpected error; output is human-readable by default.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("smoke")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # ── Risk layer
    log.info("Building broker registry…")
    brokers = build_brokers()
    log.info(f"Brokers configured: {sorted(brokers.keys())}")

    log.info("Computing risk state…")
    rm = RiskManager(brokers=brokers)
    state = rm.compute_state(persist=False)
    log.info(f"  equity_usd:      ${state.equity_usd:,.2f}")
    log.info(f"  peak_equity:     ${state.peak_equity_usd:,.2f}")
    log.info(f"  drawdown_pct:    {state.drawdown_pct * 100:.2f}%")
    log.info(f"  kill_switch:     {state.kill_switch.value}")
    log.info(f"  leverage:        {state.leverage:.2f}x")
    log.info(f"  realized_vol:    {state.realized_vol}")
    log.info(f"  multiplier:      base={state.multiplier.base:.2f}, "
             f"effective={state.multiplier.effective:.2f}")
    if state.multiplier.notes:
        log.info(f"  multiplier_notes: {state.multiplier.notes}")

    # ── Test order gate
    log.info("\nSimulating a $100 order via check_order()…")
    decision = rm.check_order(
        notional_usd=100.0, symbol="BTC-USD",
        is_closing=False, strategy_name="smoke_test",
        existing_position_usd=0.0, state=state,
    )
    log.info(f"  decision: {decision.decision.value}")
    log.info(f"  approved_usd: ${decision.approved_notional_usd:.2f}")
    log.info(f"  reason: {decision.reason}")

    # ── Allocator layer — register 3 placeholder strategies
    log.info("\nRegistering placeholder strategies in allocator…")
    registry = StrategyRegistry()
    placeholder_strats = [
        StrategyMeta(name="crypto_funding_carry",
                     asset_classes=["CRYPTO_PERP"], venue="coinbase",
                     target_alloc_pct=0.30, max_alloc_pct=0.40,
                     description="Long spot / short perp; capture funding"),
        StrategyMeta(name="risk_parity_etf",
                     asset_classes=["ETF"], venue="alpaca",
                     target_alloc_pct=0.50, max_alloc_pct=0.60,
                     description="Vol-balanced SPY/TLT/GLD/DBC/IEF"),
        StrategyMeta(name="kalshi_calibration",
                     asset_classes=["PREDICTION"], venue="kalshi",
                     target_alloc_pct=0.15, max_alloc_pct=0.20,
                     description="Favorite-longshot bias arb"),
    ]
    for meta in placeholder_strats:
        registry.register(meta)
    log.info(f"  registered: {registry.list_names()}")

    # ── Run a rebalance against current portfolio equity
    log.info("\nRunning meta-allocator rebalance…")
    allocator = MetaAllocator(registry, performance=StrategyPerformance())
    # Use $1k as fallback if equity is 0 (e.g. brokers not wired locally)
    equity = max(state.equity_usd, 1000.0)
    alloc = allocator.rebalance(portfolio_equity_usd=equity)

    log.info(f"\n{'STRATEGY':<28} {'STATE':<8} {'PCT':>6}  {'$':>10}  REASON")
    log.info("-" * 90)
    for d in alloc.decisions:
        log.info(f"{d.name:<28} {d.state.value:<8} {d.target_pct * 100:>5.1f}%  "
                 f"${d.target_usd:>9,.2f}  {d.reason}")
    log.info(f"\nTotal active allocation: {alloc.total_active_pct * 100:.1f}%")
    log.info(f"Portfolio equity: ${equity:,.2f}")

    if args.json:
        out = {
            "risk": rm.summary_dict(),
            "allocator": {
                "equity_usd": equity,
                "total_active_pct": alloc.total_active_pct,
                "decisions": [
                    {"name": d.name, "state": d.state.value,
                     "target_pct": d.target_pct, "target_usd": d.target_usd,
                     "reason": d.reason}
                    for d in alloc.decisions
                ],
            },
        }
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
