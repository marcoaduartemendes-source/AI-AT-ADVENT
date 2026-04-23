#!/usr/bin/env python3
"""
Pre-flight check for the crypto trading bot.

Runs a series of validations so you know everything is wired up *before*
you leave the bot running unattended. All checks are safe — NO orders are
placed regardless of DRY_RUN setting.

Usage:
  python scripts/check_setup.py
"""

import os
import sys

# Add src/ to path so we can import the trading package
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from dotenv import load_dotenv
load_dotenv()

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

errors = 0
warnings = 0


def check(label: str, ok: bool, detail: str = "", fatal: bool = True):
    global errors, warnings
    if ok:
        print(f"  {OK} {label}" + (f"  — {detail}" if detail else ""))
    else:
        icon = FAIL if fatal else WARN
        print(f"  {icon} {label}" + (f"  — {detail}" if detail else ""))
        if fatal:
            errors += 1
        else:
            warnings += 1


def main():
    print("\n═══ Crypto Bot Pre-flight Check ═══\n")

    # 1. Python deps
    print("1. Dependencies")
    try:
        import numpy
        check("numpy", True, numpy.__version__)
    except ImportError:
        check("numpy", False, "run: pip install -r requirements.txt")

    try:
        import requests
        check("requests", True, requests.__version__)
    except ImportError:
        check("requests", False, "run: pip install -r requirements.txt")

    try:
        import dotenv
        check("python-dotenv", True)
    except ImportError:
        check("python-dotenv", False, "run: pip install -r requirements.txt")

    # 2. Env vars
    print("\n2. Environment variables (.env)")
    api_key = os.environ.get("COINBASE_API_KEY", "")
    api_secret = os.environ.get("COINBASE_API_SECRET", "")
    check("COINBASE_API_KEY present", bool(api_key) and api_key != "your_coinbase_api_key",
          "missing or still has placeholder value" if not api_key or api_key == "your_coinbase_api_key" else f"{api_key[:8]}…")
    check("COINBASE_API_SECRET present", bool(api_secret) and api_secret != "your_coinbase_api_secret",
          "missing or still has placeholder value" if not api_secret or api_secret == "your_coinbase_api_secret" else "set")

    dry_run = os.environ.get("DRY_RUN", "true").lower() != "false"
    check(f"DRY_RUN = {os.environ.get('DRY_RUN', 'true')}", True,
          "paper trading (safe)" if dry_run else "⚠ LIVE — real money will be spent")

    max_usd = float(os.environ.get("MAX_TRADE_USD", "20"))
    check(f"MAX_TRADE_USD = ${max_usd:.2f}", max_usd <= 20.0,
          "must be ≤ 20" if max_usd > 20 else "within $20 safety cap")

    products = [p.strip() for p in os.environ.get("TRADING_PRODUCTS", "BTC-USD,ETH-USD,SOL-USD").split(",") if p.strip()]
    check(f"TRADING_PRODUCTS ({len(products)})", len(products) > 0, ", ".join(products))

    if errors > 0:
        print(f"\n{FAIL} {errors} fatal error(s). Fix your .env then re-run this script.\n")
        sys.exit(1)

    # 3. Coinbase connectivity
    print("\n3. Coinbase API connectivity")
    try:
        from trading.coinbase_client import CoinbaseClient
        client = CoinbaseClient(api_key, api_secret)
        accounts = client.get_accounts()
        check("Authenticated GET /accounts", True, f"{len(accounts)} account(s)")

        usd = [a for a in accounts if a.get("currency") == "USD"]
        if usd:
            bal = float(usd[0]["available_balance"]["value"])
            if dry_run:
                check(f"USD balance: ${bal:.2f}", True, "dry run — balance not required")
            else:
                check(f"USD balance: ${bal:.2f}", bal >= max_usd,
                      f"need ≥ ${max_usd:.2f} for live trading" if bal < max_usd else "sufficient")
        else:
            check("USD account", False, "no USD wallet found on Coinbase", fatal=False)
    except Exception as exc:
        check("Coinbase auth", False, str(exc)[:120])
        print(f"\n{FAIL} Coinbase connection failed — check your API key/secret and key permissions.\n")
        sys.exit(1)

    # 4. Market data fetch
    print("\n4. Market data")
    try:
        from trading.market_data import fetch_candles
        test_product = products[0]
        candles = fetch_candles(client, test_product, "ONE_HOUR", 50)
        check(f"Fetch {test_product} candles", len(candles) >= 40,
              f"{len(candles)} candles (need ≥ 40)")
        if len(candles) >= 40:
            last_close = candles[-1, 4]
            check(f"Latest {test_product} close", True, f"${last_close:,.2f}")
    except Exception as exc:
        check("Market data fetch", False, str(exc)[:120])

    # 5. Strategy dry-run
    print("\n5. Strategy dry-run (no orders placed)")
    try:
        from trading.strategies.momentum import MomentumStrategy
        from trading.strategies.mean_reversion import MeanReversionStrategy
        from trading.strategies.volatility_breakout import VolatilityBreakoutStrategy

        strategies = [
            MomentumStrategy(products=products),
            MeanReversionStrategy(products=products),
            VolatilityBreakoutStrategy(products=products),
        ]

        for strat in strategies:
            c = fetch_candles(client, products[0], strat.granularity, strat.lookback)
            if len(c) < 30:
                check(strat.name, False, f"only {len(c)} candles", fatal=False)
                continue
            sig = strat.analyze(products[0], c)
            check(f"{strat.name:22}", True,
                  f"{sig.signal.value}  conf={sig.confidence:.2f}")
    except Exception as exc:
        check("Strategy analysis", False, str(exc)[:120])

    # 6. SQLite
    print("\n6. Performance tracking")
    try:
        from trading.performance import PerformanceTracker
        tracker = PerformanceTracker()
        m = tracker.get_metrics()
        check("SQLite DB initialised", True,
              f"{m['closed_trades']} closed trades so far")
    except Exception as exc:
        check("SQLite", False, str(exc)[:120])

    # Summary
    print()
    if errors == 0:
        print(f"{OK} All checks passed. Ready to go.")
        if warnings:
            print(f"{WARN} {warnings} warning(s) — see above.")
        print("\nNext steps:")
        print("  • Single test cycle:   python src/main_trading.py --once")
        print("  • Run continuously:    python src/main_trading.py")
        if not dry_run:
            print(f"\n  {WARN}  DRY_RUN=false — LIVE orders will be placed!")
        print()
    else:
        print(f"{FAIL} {errors} error(s) — fix them before running the bot.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
