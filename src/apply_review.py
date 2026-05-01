#!/usr/bin/env python3
"""Apply the latest strategic review's recommendations.

Reads the most recent review from `data/strategic_review.db`, parses
`strategy_actions`, and edits `src/run_orchestrator.py` to update each
strategy's `target_alloc_pct` based on Opus's recommendation. Also
updates the `RISK_MULTIPLIER` repo variable (caller must do that step
since it's a workflow-level concern).

Usage from CI:
    python src/apply_review.py --commit
        Patches run_orchestrator.py and prints what it changed. Combined
        with the workflow's git config + push it ships the change.

    python src/apply_review.py --dry-run
        Shows what WOULD change but doesn't write.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from review.reviewer import ReviewDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("apply_review")


ORCH_FILE = os.path.join(os.path.dirname(__file__), "run_orchestrator.py")


def _load_latest_actions() -> Tuple[Optional[Dict], List[Dict]]:
    db = ReviewDB()
    latest = db.latest()
    if not latest:
        log.warning("No review found in DB.")
        return None, []
    payload = {}
    try:
        payload = json.loads(latest.get("payload_json") or "{}")
    except Exception as e:
        log.error(f"Could not parse payload_json: {e}")
        return latest, []
    return latest, payload.get("strategy_actions", [])


def _patch_target_alloc(source: str, name: str, new_pct: float) -> Tuple[str, bool]:
    """Find StrategyMeta(name="<name>", ...) and patch target_alloc_pct."""
    # Match the StrategyMeta block for this strategy specifically.
    block_re = re.compile(
        r'(StrategyMeta\(\s*\n\s*name="' + re.escape(name) + r'",\s*\n[\s\S]*?\),)',
        re.MULTILINE,
    )
    m = block_re.search(source)
    if not m:
        return source, False
    block = m.group(1)
    # Replace target_alloc_pct=<num>
    new_block = re.sub(
        r'target_alloc_pct=[0-9.]+',
        f"target_alloc_pct={new_pct:.3f}",
        block, count=1,
    )
    if new_block == block:
        return source, False
    return source[:m.start(1)] + new_block + source[m.end(1):], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                     help="Write changes to run_orchestrator.py")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print what would change without writing")
    args = ap.parse_args()

    latest, actions = _load_latest_actions()
    if not actions:
        print("No actionable recommendations found.")
        return 0
    print(f"Latest review: {latest.get('timestamp')}  health={latest.get('overall_health')}")
    print(f"Reviewing {len(actions)} strategy actions…\n")

    with open(ORCH_FILE, encoding="utf-8") as f:
        source = f.read()
    original_source = source

    applied = []
    skipped = []
    for action in actions:
        name = action.get("strategy")
        act_type = (action.get("action") or "").upper()
        target = action.get("target_alloc_pct")

        if act_type in ("MAINTAIN", "WATCH", "FREEZE", "RETIRE", "ACTIVATE"):
            # These don't change target_alloc_pct directly; they're
            # state actions handled by the lifecycle state machine.
            # FREEZE/RETIRE do force allocation to 0 in the allocator.
            skipped.append((name, act_type, "state action — no target change"))
            continue

        if target is None:
            skipped.append((name, act_type, "no target_alloc_pct provided"))
            continue

        new_source, changed = _patch_target_alloc(source, name, float(target))
        if changed:
            source = new_source
            applied.append((name, act_type, float(target), action.get("reason", "")))
        else:
            skipped.append((name, act_type, "regex did not match"))

    print("Applied:")
    for name, act, target, reason in applied:
        print(f"  ✓ {name:<28} {act:<10} → target_alloc_pct={target:.3f}")
        if reason:
            print(f"      {reason}")
    print()
    print("Skipped:")
    for name, act, why in skipped:
        print(f"  - {name:<28} {act:<10} ({why})")

    if not applied:
        print("\nNo changes to write.")
        return 0

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return 0
    if not args.commit:
        print("\nPass --commit to write the changes.")
        return 0

    with open(ORCH_FILE, "w", encoding="utf-8") as f:
        f.write(source)
    print(f"\nWrote {len(applied)} change(s) to {ORCH_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
