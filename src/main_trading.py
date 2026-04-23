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
from trading.market_data import fetch_candles
from trading.performance import PerformanceTracker
from trading.portfolio import PortfolioManager
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
SCAN_INTERVAL_MINUTES = int(os.environ.get("SCAN_INTERVAL_MINUTES", "60"))
PRODUCTS = [p.strip() for p in os.environ.get("TRADING_PRODUCTS", "BTC-USD,ETH-USD,SOL-USD").split(",") if p.strip()]

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

def run_cycle(client, strategies, portfolio: PortfolioManager, tracker: PerformanceTracker):
    signals_total, trades_total = 0, 0

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
                    pnl_str = f" | PnL=${trade.pnl_usd:+.2f}" if trade.pnl_usd is not None else ""
                    mode_str = " [DRY RUN]" if trade.dry_run else " [LIVE]"
                    logger.info(
                        f"[{strategy.name}] ✓ {trade.side} {product_id} "
                        f"${trade.amount_usd:.2f} @ ${trade.price:.4f}{pnl_str}{mode_str}"
                    )

            except Exception as exc:
                logger.error(f"[{strategy.name}] Error on {product_id}: {exc}", exc_info=True)

    logger.info(f"Cycle complete — {signals_total} signals, {trades_total} trades")
    return signals_total, trades_total


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
    logger.info(f"  Interval  : {SCAN_INTERVAL_MINUTES} min")
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
            granularity="ONE_HOUR",
        ),
        MeanReversionStrategy(
            products=PRODUCTS,
            window=20,
            z_entry=2.0,
            granularity="ONE_HOUR",
        ),
        VolatilityBreakoutStrategy(
            products=PRODUCTS,
            bb_window=20,
            squeeze_threshold=0.5,
            history_window=50,
            granularity="ONE_HOUR",
        ),
    ]

    strategy_names = [s.name for s in strategies]
    logger.info(f"Strategies loaded: {strategy_names}")

    portfolio = PortfolioManager(
        client=client,
        max_trade_usd=MAX_TRADE_USD,
        dry_run=DRY_RUN,
        min_confidence=MIN_CONFIDENCE,
    )
    tracker = PerformanceTracker()
    portfolio.attach_tracker(tracker)   # positions survive between hourly restarts

    # ── Main loop ─────────────────────────────────────────────────────────────

    cycle = 0
    while not _shutdown:
        cycle += 1
        logger.info(f"\n{'─' * 30} Cycle #{cycle} {'─' * 30}")

        run_cycle(client, strategies, portfolio, tracker)

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


if __name__ == "__main__":
    main()
