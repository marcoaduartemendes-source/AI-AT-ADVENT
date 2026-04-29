#!/usr/bin/env python3
"""
Crypto Investment Bot
─────────────────────────────────────────────────────────────────────────────
Three independent strategies running simultaneously on Coinbase Advanced Trade:

  1. Momentum        – EMA crossover + MACD (trend-following)
  2. MeanReversion   – Z-score statistical arbitrage (Renaissance-style)
  3. VolatilityBreakout – Bollinger Band squeeze detection

All orders are capped at $20 USD for testing. Set DRY_RUN=false in .env to
enable live execution. All trades and P&L are tracked in SQLite.

Usage:
  python src/main_trading.py           # run forever on SCAN_INTERVAL_MINUTES
  python src/main_trading.py --once    # run exactly one cycle and exit

Required environment variables (see .env.example):
  COINBASE_API_KEY, COINBASE_API_SECRET
"""

import argparse
import json
import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is in path when run directly
sys.path.insert(0, os.path.dirname(__file__))

from trading.coinbase_client import CoinbaseClient
from trading.market_data import fetch_candles, get_current_price
from trading.performance import PerformanceTracker
from trading.portfolio import PortfolioManager, TradeRecord
from trading.strategies.base import SignalType
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.volatility_breakout import VolatilityBreakoutStrategy

# ── Config ────────────────────────────────────────────────────────────────────

def _placeholder(val: str) -> bool:
    return (
        not val
        or val.startswith("organizations/your-")
        or "REPLACE_WITH_YOUR_KEY" in val
        or val == "your_coinbase_api_key"
    )


SIMULATION = os.environ.get("SIMULATION", "").lower() == "true"
COINBASE_API_KEY = os.environ.get("COINBASE_API_KEY", "")
COINBASE_API_SECRET = os.environ.get("COINBASE_API_SECRET", "")

# Auto-activate simulation mode if keys are missing or still placeholders.
if _placeholder(COINBASE_API_KEY) or _placeholder(COINBASE_API_SECRET):
    SIMULATION = True
    COINBASE_API_KEY = ""
    COINBASE_API_SECRET = ""

# Simulation forces DRY_RUN on (no real orders possible without auth anyway)
DRY_RUN = True if SIMULATION else os.environ.get("DRY_RUN", "true").lower() != "false"
MAX_TRADE_USD = min(float(os.environ.get("MAX_TRADE_USD", "20")), 20.0)
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.6"))
SCAN_INTERVAL_MINUTES = int(os.environ.get("SCAN_INTERVAL_MINUTES", "5"))
PRODUCTS = [p.strip() for p in os.environ.get("TRADING_PRODUCTS", "BTC-USD,ETH-USD,SOL-USD").split(",") if p.strip()]
GRANULARITY = os.environ.get("GRANULARITY", "FIVE_MINUTE")
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.04"))
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "5"))

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs("data", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/trading.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    logger.info("Shutdown signal received — finishing current cycle then exiting.")
    _shutdown = True

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Core loop ─────────────────────────────────────────────────────────────────

def _live_prices(client, products):
    out = {}
    for p in products:
        price = get_current_price(client, p)
        if price is not None:
            out[p] = price
    return out


def _write_step_summary(tracker, portfolio, strategy_names, cycle_trades, mode):
    """Emit a markdown summary to GITHUB_STEP_SUMMARY for the Actions run page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## Crypto Trading Bot — {mode}\n\n")

            # ── This cycle's trades
            if cycle_trades:
                f.write("### Trades this run\n\n")
                f.write("| Strategy | Side | Product | USD | Price | PnL | Mode |\n")
                f.write("|---|---|---|---|---|---|---|\n")
                for t in cycle_trades:
                    pnl = f"${t.pnl_usd:+.2f}" if t.pnl_usd is not None else "—"
                    tag = "DRY" if t.dry_run else "LIVE"
                    f.write(
                        f"| {t.strategy} | {t.side} | {t.product_id} | "
                        f"${t.amount_usd:.2f} | ${t.price:.4f} | {pnl} | {tag} |\n"
                    )
                f.write("\n")
            else:
                f.write("_No trades placed this run._\n\n")

            # ── Open positions
            open_pos = portfolio.get_open_positions()
            if open_pos:
                f.write("### Open positions\n\n")
                f.write("| Strategy | Product | Qty | Cost basis | Entry price | Entry time |\n")
                f.write("|---|---|---|---|---|---|\n")
                for strat, pmap in open_pos.items():
                    for pid, p in pmap.items():
                        f.write(
                            f"| {strat} | {pid} | {p['quantity']:.8f} | "
                            f"${p['cost_basis_usd']:.2f} | ${p['entry_price']:.4f} | "
                            f"{p['entry_time']} |\n"
                        )
                f.write("\n")
            else:
                f.write("_No open positions._\n\n")

            # ── All-time P&L per strategy
            f.write("### All-time P&L\n\n")
            f.write("| Strategy | Closed | Wins | Losses | Win % | Total P&L | Avg/trade |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for name in strategy_names:
                m = tracker.get_metrics(name)
                f.write(
                    f"| {name} | {m['closed_trades']} | {m['wins']} | {m['losses']} | "
                    f"{m['win_rate'] * 100:.1f}% | ${m['total_pnl']:+.2f} | "
                    f"${m['avg_pnl']:+.2f} |\n"
                )
            total = tracker.get_metrics()
            f.write(
                f"| **COMBINED** | **{total['closed_trades']}** | **{total['wins']}** | "
                f"**{total['losses']}** | **{total['win_rate'] * 100:.1f}%** | "
                f"**${total['total_pnl']:+.2f}** | **${total['avg_pnl']:+.2f}** |\n"
            )
    except Exception as exc:
        logger.warning(f"Could not write step summary: {exc}")


def run_cycle(client, strategies, portfolio: PortfolioManager, tracker: PerformanceTracker):
    signals_total, trades_total = 0, 0
    cycle_trades: list = []

    # ── Risk pass: enforce stop-loss / take-profit on open positions BEFORE signals
    open_pos = portfolio.get_open_positions()
    if open_pos:
        prods_held = {pid for pmap in open_pos.values() for pid in pmap}
        prices = _live_prices(client, prods_held)
        forced = portfolio.check_stops(prices)
        for trade in forced:
            tracker.record_trade(trade)
            cycle_trades.append(trade)
            trades_total += 1

    for strategy in strategies:
        for product_id in PRODUCTS:
            try:
                candles = fetch_candles(
                    client,
                    product_id,
                    granularity=strategy.granularity,
                    num_candles=strategy.lookback,
                )

                if len(candles) < 30:
                    logger.warning(f"[{strategy.name}] Only {len(candles)} candles for {product_id}, skipping")
                    continue

                signal = strategy.analyze(product_id, candles)

                if signal.signal != SignalType.HOLD:
                    signals_total += 1
                    logger.info(
                        f"[{strategy.name}] {signal.signal.value} {product_id} "
                        f"conf={signal.confidence:.2f} — {signal.reason}"
                    )
                else:
                    logger.debug(f"[{strategy.name}] HOLD {product_id}: {signal.reason}")

                trade = portfolio.process_signal(signal)
                if trade:
                    trades_total += 1
                    tracker.record_trade(trade)
                    cycle_trades.append(trade)
                    pnl_str = f" | PnL=${trade.pnl_usd:+.2f}" if trade.pnl_usd is not None else ""
                    mode_str = " [DRY RUN]" if trade.dry_run else " [LIVE]"
                    logger.info(
                        f"[{strategy.name}] ✓ {trade.side} {product_id} "
                        f"${trade.amount_usd:.2f} @ ${trade.price:.4f}{pnl_str}{mode_str}"
                    )

            except Exception as exc:
                logger.error(f"[{strategy.name}] Error on {product_id}: {exc}", exc_info=True)

    logger.info(f"Cycle complete — {signals_total} signals, {trades_total} trades")
    return signals_total, trades_total, cycle_trades


def main():
    parser = argparse.ArgumentParser(description="Crypto Investment Bot")
    parser.add_argument("--once", action="store_true",
                        help="Run a single trading cycle and exit (for testing).")
    args = parser.parse_args()

    if SIMULATION:
        mode = "SIMULATION  (public market data · no Coinbase account needed)"
    elif DRY_RUN:
        mode = "PAPER TRADING (DRY RUN)"
    else:
        mode = "⚠  LIVE TRADING — REAL MONEY"
    logger.info("=" * 64)
    logger.info("  Crypto Investment Bot")
    logger.info(f"  Mode      : {mode}")
    logger.info(f"  Products  : {', '.join(PRODUCTS)}")
    logger.info(f"  Max trade : ${MAX_TRADE_USD:.2f} USD")
    logger.info(f"  Min conf  : {MIN_CONFIDENCE}")
    logger.info(f"  Granular. : {GRANULARITY}")
    logger.info(f"  Interval  : {SCAN_INTERVAL_MINUTES} min")
    logger.info(f"  Stop-loss : {STOP_LOSS_PCT * 100:.1f}%   Take-profit: {TAKE_PROFIT_PCT * 100:.1f}%")
    logger.info(f"  Max open  : {MAX_OPEN_POSITIONS} positions")
    logger.info("=" * 64)

    if not DRY_RUN:
        logger.warning("LIVE trading is active — orders will be placed on Coinbase!")
        logger.warning("Ctrl-C within 10 seconds to abort.")
        for i in range(10, 0, -1):
            if _shutdown:
                logger.info("Aborted before first trade.")
                return
            logger.warning(f"  Starting in {i}s…")
            time.sleep(1)

    # ── Initialise client ─────────────────────────────────────────────────────

    client = CoinbaseClient(COINBASE_API_KEY, COINBASE_API_SECRET)

    if SIMULATION:
        logger.info("Simulation mode — using public Coinbase market data (no authentication).")
    else:
        try:
            accounts = client.get_accounts()
            usd_accounts = [a for a in accounts if a.get("currency") == "USD"]
            usd_balance = float(usd_accounts[0]["available_balance"]["value"]) if usd_accounts else 0.0
            logger.info(f"Connected to Coinbase — {len(accounts)} accounts, USD balance: ${usd_balance:.2f}")
        except Exception as exc:
            logger.error(f"Cannot connect to Coinbase: {exc}")
            sys.exit(1)

    # ── Strategies ────────────────────────────────────────────────────────────

    strategies = [
        MomentumStrategy(
            products=PRODUCTS,
            fast_period=10,
            slow_period=30,
            rsi_period=14,
            granularity=GRANULARITY,
        ),
        MeanReversionStrategy(
            products=PRODUCTS,
            window=20,
            z_entry=2.0,
            granularity=GRANULARITY,
        ),
        VolatilityBreakoutStrategy(
            products=PRODUCTS,
            bb_window=20,
            squeeze_threshold=0.5,
            history_window=50,
            granularity=GRANULARITY,
        ),
    ]

    strategy_names = [s.name for s in strategies]
    logger.info(f"Strategies loaded: {strategy_names}")

    portfolio = PortfolioManager(
        client=client,
        max_trade_usd=MAX_TRADE_USD,
        dry_run=DRY_RUN,
        min_confidence=MIN_CONFIDENCE,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT,
        max_open_positions=MAX_OPEN_POSITIONS,
    )
    tracker = PerformanceTracker()
    portfolio.attach_tracker(tracker)   # positions survive between hourly restarts

    # ── Main loop ─────────────────────────────────────────────────────────────

    cycle = 0
    cycle_trades_last: list = []
    while not _shutdown:
        cycle += 1
        logger.info(f"\n{'─' * 30} Cycle #{cycle} {'─' * 30}")

        _, _, cycle_trades_last = run_cycle(client, strategies, portfolio, tracker)

        # Dashboard every 5 cycles (and on cycle 1)
        if cycle == 1 or cycle % 5 == 0:
            tracker.print_dashboard(strategy_names)
            tracker.save_snapshot(strategy_names)

        # Open positions summary
        open_pos = portfolio.get_open_positions()
        if open_pos:
            logger.info(f"Open positions: {json.dumps(open_pos, indent=2)}")

        if args.once:
            logger.info("Single-cycle mode (--once): exiting after one pass.")
            break

        if _shutdown:
            break

        logger.info(f"Sleeping {SCAN_INTERVAL_MINUTES} min until next scan…")
        # Break sleep into 1-minute chunks so SIGTERM is handled quickly
        for _ in range(SCAN_INTERVAL_MINUTES * 60 // 10):
            if _shutdown:
                break
            time.sleep(10)

    logger.info("Bot shut down cleanly.")
    tracker.print_dashboard(strategy_names)
    _write_step_summary(tracker, portfolio, strategy_names, cycle_trades_last, mode)


if __name__ == "__main__":
    main()
