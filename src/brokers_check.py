#!/usr/bin/env python3
"""Multi-broker health check.

Run from CI or locally to confirm every configured broker authenticates and
returns sensible account/position data. Used by the daily dashboard build
and as a one-shot verification after credential rotation.

Usage:
    python src/brokers_check.py                # human-readable
    python src/brokers_check.py --json         # JSON output for dashboards
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from brokers.registry import build_brokers


logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def run(as_json: bool = False) -> int:
    brokers = build_brokers()
    if not brokers:
        msg = "No brokers configured."
        print(json.dumps({"error": msg}) if as_json else msg)
        return 1

    results: dict[str, dict] = {}
    overall_ok = True
    for name, adapter in brokers.items():
        h = adapter.healthcheck()
        results[name] = h
        if not h.get("ok"):
            # Kalshi placeholder is allowed to be unconfigured
            if name == "kalshi" and h.get("configured") is False:
                pass
            else:
                overall_ok = False

    if as_json:
        print(json.dumps({"ok": overall_ok, "brokers": results}, indent=2, default=str))
        return 0 if overall_ok else 2

    print(f"\n{'BROKER':<10} {'OK':<5} {'PAPER':<6} {'CASH':>14} {'EQUITY':>14}  NOTE")
    print("-" * 78)
    for name, h in results.items():
        ok = "yes" if h.get("ok") else "NO"
        paper = "yes" if h.get("is_paper") else ("no" if h.get("ok") else "—")
        cash = f"${h['cash_usd']:,.2f}" if h.get("cash_usd") is not None else "—"
        equity = f"${h['equity_usd']:,.2f}" if h.get("equity_usd") is not None else "—"
        note = h.get("error") or h.get("note") or ""
        print(f"{name:<10} {ok:<5} {paper:<6} {cash:>14} {equity:>14}  {note}")
    print()
    return 0 if overall_ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sys.exit(run(as_json=args.json))


if __name__ == "__main__":
    main()
