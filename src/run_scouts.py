#!/usr/bin/env python3
"""Run every scout once, persist signals to the bus, write a step summary.

Designed to be called on its own GitHub Actions schedule (every 30-60 min
or on demand). Scouts run sequentially because each only takes a few
seconds — parallelism would only complicate error reporting.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from scouts.base import ScoutAgent
from scouts.commodities_scout import CommoditiesScout
from scouts.crypto_scout import CryptoScout
from scouts.equities_scout import EquitiesScout
from scouts.macro_scout import MacroScout
from scouts.prediction_scout import PredictionScout
from scouts.signal_bus import SignalBus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_scouts")


SCOUTS: List[ScoutAgent] = []


def init_scouts(bus: SignalBus) -> List[ScoutAgent]:
    """Build every scout. Scouts whose data sources are unreachable will
    surface their own errors at scan() time — they don't fail at construction."""
    return [
        MacroScout(bus=bus),
        CryptoScout(bus=bus),
        PredictionScout(bus=bus),
        EquitiesScout(bus=bus),
        CommoditiesScout(bus=bus),
    ]


def write_step_summary(reports: List[Dict], bus_size: int):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("## Scout sweep\n\n")
            f.write(f"_Total signals on bus: {bus_size}_\n\n")
            f.write("| Scout | Published | Proposed | Errors | Signal types |\n")
            f.write("|---|---|---|---|---|\n")
            for r in reports:
                stypes = ", ".join(r.get("signal_types", [])) or "—"
                err_count = len(r.get("errors", []))
                f.write(f"| {r['scout']} | {r['published']} | "
                        f"{r.get('total_proposed', 0)} | {err_count} | {stypes} |\n")
            for r in reports:
                if r.get("errors"):
                    f.write(f"\n**{r['scout']} errors:**\n")
                    for e in r["errors"]:
                        f.write(f"- {e}\n")
    except Exception as exc:
        logger.warning(f"Could not write step summary: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vacuum", action="store_true",
                     help="Delete expired bus rows before running")
    args = ap.parse_args()

    bus = SignalBus()
    if args.vacuum:
        deleted = bus.vacuum_expired()
        logger.info(f"Vacuum: deleted {deleted} expired rows")

    scouts = init_scouts(bus)
    reports: List[Dict] = []
    for scout in scouts:
        logger.info(f"Running {scout.name}…")
        report = scout.run_once()
        reports.append(report)

    bus_size = len(bus.latest(limit=1000))
    if args.json:
        print(json.dumps({"reports": reports, "bus_size": bus_size},
                          indent=2, default=str))
    else:
        print("\n=== Scout sweep complete ===")
        for r in reports:
            print(f"  {r['scout']:<20} published={r['published']} "
                  f"types={r.get('signal_types', [])}")
        print(f"\nBus size: {bus_size} fresh+stale rows")

    write_step_summary(reports, bus_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
