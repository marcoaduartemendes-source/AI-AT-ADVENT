"""Entrypoint for the daily research workflow.

Runs the HEAVY backtests that must NOT block the 5-minute trading
cycle:
  • strategy_validation.run_validation   (~200s — 3× backtest_all)
  • walk_forward.run_walk_forward        (~200s — backtest_all 1260)

These were originally in the orchestrator's per-cycle finally block
and caused the cycle to exceed its 8-minute workflow timeout, which
killed the process before docs/cycle_status.json could be committed
(observed 2026-05-20→21: cycle frozen 22h). They belong on a daily
cadence — the verdicts barely move intraday.

Usage:
    python src/run_research.py            # run both, write docs JSONs
    python src/run_research.py --force    # ignore the 24h rate-limit
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_research")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="Ignore the 24h rate-limit and re-run now")
    args = ap.parse_args()

    rc = 0
    try:
        from common.strategy_validation import run_validation
        v = run_validation(force=args.force)
        logger.info(f"validation: {v.get('n_pass','?')}/"
                     f"{v.get('n_strategies','?')} PASS")
    except Exception as e:
        logger.error(f"run_validation failed: {e}")
        rc = 1

    try:
        from common.walk_forward import run_walk_forward
        w = run_walk_forward(force=args.force)
        logger.info(f"walk_forward: {w.get('counts','?')}")
    except Exception as e:
        logger.error(f"run_walk_forward failed: {e}")
        rc = 1

    # Refresh the downstream agents now that fresh verdicts exist so
    # the next dashboard build reflects them immediately.
    for mod, fn in (("common.performance_review", "run_performance_review"),
                     ("common.auto_demote", "run_auto_demote"),
                     ("common.data_quality", "run_data_quality"),
                     ("common.self_grade", "run_self_grade")):
        try:
            m = __import__(mod, fromlist=[fn])
            getattr(m, fn)()
        except Exception as e:
            logger.warning(f"{fn} failed: {e}")
    return rc


if __name__ == "__main__":
    # src-rooted imports (mirrors run_orchestrator.py).
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    sys.exit(main())
