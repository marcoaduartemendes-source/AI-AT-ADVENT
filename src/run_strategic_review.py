#!/usr/bin/env python3
"""Run the weekly strategic review.

Calls Claude Opus 4.7 with this week's data, persists structured
recommendations, writes a step summary for the dashboard.

Triggered by .github/workflows/strategic_review.yml on a weekly schedule
or on workflow_dispatch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from review.reviewer import StrategicReviewer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("review")


def write_step_summary(result):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"## Strategic Review — {result.timestamp.isoformat()}\n\n")
            color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(
                result.overall_health, "⚪")
            f.write(f"### {color} Overall health: **{result.overall_health}**\n\n")
            f.write(f"_{result.summary}_\n\n")
            f.write(f"**Risk multiplier recommendation:** "
                    f"{result.risk_multiplier_rec:.2f}x — _{result.risk_multiplier_reason}_\n\n")

            if result.strategy_actions:
                f.write("### Strategy actions\n\n")
                f.write("| Strategy | Action | Target % | Confidence | Reason |\n")
                f.write("|---|---|---|---|---|\n")
                for a in result.strategy_actions:
                    target = a.get("target_alloc_pct")
                    target_str = f"{target * 100:.1f}%" if target is not None else "—"
                    conf = a.get("confidence", 0)
                    f.write(f"| {a.get('strategy', '')} | **{a.get('action', '')}** | "
                            f"{target_str} | {conf:.2f} | {a.get('reason', '')} |\n")
                f.write("\n")

            if result.investigate:
                f.write("### Investigate\n\n")
                for item in result.investigate:
                    f.write(f"- {item}\n")
                f.write("\n")

            if result.cost_usd is not None:
                f.write(f"\n_Model: {result.model_used} · "
                        f"cost: ${result.cost_usd:.4f}_\n")
    except Exception as e:
        logger.warning(f"Could not write step summary: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                     help="Print JSON-formatted result to stdout")
    ap.add_argument("--model", default="claude-opus-4-7")
    args = ap.parse_args()

    reviewer = StrategicReviewer(model=args.model)
    result = reviewer.review()

    if args.json:
        print(json.dumps({
            "timestamp": result.timestamp.isoformat(),
            "overall_health": result.overall_health,
            "summary": result.summary,
            "risk_multiplier_rec": result.risk_multiplier_rec,
            "risk_multiplier_reason": result.risk_multiplier_reason,
            "strategy_actions": result.strategy_actions,
            "investigate": result.investigate,
            "cost_usd": result.cost_usd,
            "model_used": result.model_used,
        }, indent=2, default=str))
    else:
        print(f"\n=== Strategic Review ===")
        print(f"  Health:    {result.overall_health}")
        print(f"  Summary:   {result.summary}")
        print(f"  Risk mult: {result.risk_multiplier_rec:.2f} ({result.risk_multiplier_reason})")
        print(f"  Actions:   {len(result.strategy_actions)}")
        for a in result.strategy_actions:
            print(f"    {a.get('strategy', ''):<28} {a.get('action', ''):<10} "
                  f"conf={a.get('confidence', 0):.2f}  {a.get('reason', '')[:70]}")
        if result.investigate:
            print(f"  Investigate ({len(result.investigate)}):")
            for item in result.investigate:
                print(f"    - {item}")
        if result.cost_usd is not None:
            print(f"\n  Model {result.model_used}, cost ${result.cost_usd:.4f}")

    write_step_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
