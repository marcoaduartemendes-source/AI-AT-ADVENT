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
    CryptoFundingCarry,
    CryptoFundingCarryV2,
    CryptoXSMom,
    DividendGrowth,
    EarningsMomentum,
    EarningsNewsPEAD,
    GapTrading,
    InternationalsRotation,
    KalshiCalibrationArb,
    LowVolAnomaly,
    MacroKalshi,
    MacroKalshiV2,
    PairsTrading,
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
    # PEAD v1 retired 2026-05-07. The gap-only proxy for "earnings
    # surprise" is a known weak signal vs the v2 (RSS news corroboration)
    # and earnings_momentum (true EPS-surprise via FMP) variants;
    # running all three triple-trades the same earnings prints and
    # inflates correlation. Weight reallocated to v2 + earnings_momentum.
    # The pead.py module is kept in-tree for backtest reference only.
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
        # +1% baseline / +3% max ceiling (was 4/15) to absorb the
        # retired pead v1 allocation. Cleaner EPS-surprise signal so
        # this should produce higher Sharpe than v1's gap-only proxy.
        target_alloc_pct=0.06, max_alloc_pct=0.18, min_alloc_pct=0.02,
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
    # ── Phase 5 — strategies consuming the new data feeds (Sprint C)
    StrategyMeta(
        name="macro_kalshi_v2",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.02, max_alloc_pct=0.08, min_alloc_pct=0.01,
        description="Kalshi-vs-CME Fed-rate divergence (P5, CME-fed)",
    ),
    StrategyMeta(
        name="cross_venue_arb",
        asset_classes=["PREDICTION"], venue="kalshi",
        target_alloc_pct=0.02, max_alloc_pct=0.08, min_alloc_pct=0.01,
        description="Kalshi vs Polymarket cross-venue arbitrage (P5)",
    ),
    StrategyMeta(
        name="crypto_funding_carry_v2",
        asset_classes=["CRYPTO_PERP"], venue="coinbase",
        # Smaller than v1 (12%) until we have paper-P&L data showing
        # the multi-venue gate adds Sharpe rather than just shrinking
        # opportunity. Champion tier auto-promotes if it earns it.
        target_alloc_pct=0.04, max_alloc_pct=0.15, min_alloc_pct=0.02,
        description="Funding carry gated on Coinbase+Binance consensus (P5)",
    ),
    StrategyMeta(
        name="earnings_news_pead",
        asset_classes=["EQUITY"], venue="alpaca",
        # +2% baseline (was 3%) to absorb part of retired pead v1.
        target_alloc_pct=0.05, max_alloc_pct=0.15, min_alloc_pct=0.01,
        description="PEAD gated on RSS news corroboration (P5)",
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

    # Two-key guard: even with DRY_RUN=false the orchestrator refuses to
    # place real orders unless ALLOW_LIVE_TRADING=1. Forces an explicit,
    # recent decision rather than one stale toggle going live.
    dry_env = os.environ.get("DRY_RUN", "true").lower()
    if dry_env == "false" and os.environ.get("ALLOW_LIVE_TRADING") != "1":
        logger.warning(
            "DRY_RUN=false but ALLOW_LIVE_TRADING != '1' — forcing DRY mode. "
            "Set ALLOW_LIVE_TRADING=1 to actually place live orders."
        )
        os.environ["DRY_RUN"] = "true"

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
    dry_run_check = os.environ.get("DRY_RUN", "true").lower() != "false"
    if (_n_snapshots == 0 and not dry_run_check
            and not args.allow_cold_start):
        logger.error(
            "Cold start detected (risk_state.db has 0 equity snapshots) "
            "but DRY_RUN=false. Refusing to trade — the kill-switch baseline "
            "would reset to current equity. Pass --allow-cold-start to "
            "override."
        )
        return 3
    allocator = MetaAllocator(registry=registry, performance=StrategyPerformance())
    strategies = build_strategies(brokers)

    dry_default = os.environ.get("DRY_RUN", "true").lower() != "false"
    dry_run = dry_default and not args.live

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
