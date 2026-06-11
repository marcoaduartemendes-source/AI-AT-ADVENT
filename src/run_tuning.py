"""Weekly parameter-tuning loop — systematic sweeps over strategy tunables.

THE GAP THIS CLOSES: the daily research run validates strategies at
their CURRENT parameters; nothing ever asks "would a different vol
target / leverage cap / lookback have done better?" This harness sweeps
a registry of (strategy, parameter, candidates) through the existing
backtest functions and writes the comparison to docs/tuning.json.

THE LOOP IT FEEDS: run_research_loop.py includes tuning.json in the
nightly evidence bundle, so Claude (Fable 5) reasons over the sweep
results and emits concrete ALLOCATION/STRATEGY proposals with hard
numbers ("VOL_TARGET 0.22 beat 0.17 by +0.8 Sharpe over 2y — raise
it"). A human (or supervised Claude Code session) applies the change;
next week's sweep then measures from the new baseline. Closed loop,
human-in-the-loop on the apply step — the same safety contract as the
research loop itself.

MECHANICS: each sweep monkeypatches the strategy module's constant and
re-runs the registered backtest (the backtests re-import constants at
call time, so setattr is sufficient). Originals are always restored in
a finally block — a crash mid-sweep can never leave a strategy module
mutated for a later import in the same process. The process is a
oneshot systemd unit, so module state dies with it regardless.

SAFETY: proposals only — this tool NEVER writes strategy code, never
touches the live roster, never changes allocations. Output is one JSON
file the research loop and the operator read.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tuning")

TUNING_PATH = Path("docs/tuning.json")

# Backtest windows for each candidate (days). Two horizons so a
# parameter that only wins recently (regime-fit) is distinguishable
# from one that wins across regimes.
WINDOWS = (252, 504)

# ── Sweep registry ───────────────────────────────────────────────────
# Each entry: which strategy's backtest to run, which module constant to
# vary, and the candidate values. ONLY strategies whose backtest re-reads
# the module constant at call time belong here (function-local constants
# in runner.py won't see the patch — verify before adding).
SWEEPS: list[dict] = [
    {
        "strategy": "high_vol_trend",
        "module": "strategies.high_vol_trend",
        "param": "VOL_TARGET_ANN",
        "candidates": [0.12, 0.17, 0.22],
    },
    {
        "strategy": "high_vol_trend",
        "module": "strategies.high_vol_trend",
        "param": "MAX_LEVERAGE_PER_NAME",
        "candidates": [3.0, 4.0, 5.0],
    },
    {
        "strategy": "high_vol_trend",
        "module": "strategies.high_vol_trend",
        "param": "TREND_LOOKBACK",
        "candidates": [126, 252],
    },
]


def _run_backtest(strategy: str, window_days: int):
    """Indirection point — patched in tests, real backtest on the box."""
    from backtests.runner import backtest_strategy_by_name
    return backtest_strategy_by_name(strategy, window_days)


def run_sweep(sweep: dict) -> dict:
    """Run one parameter sweep; returns the comparison dict.

    The current (production) value is always measured alongside the
    candidates so the recommendation is a like-for-like comparison on
    the same data freshness.
    """
    module = importlib.import_module(sweep["module"])
    param = sweep["param"]
    current = getattr(module, param)
    candidates = list(sweep["candidates"])
    if current not in candidates:
        candidates.insert(0, current)

    results: dict[str, dict] = {}
    try:
        for value in candidates:
            setattr(module, param, value)
            per_window: dict[str, dict] = {}
            for days in WINDOWS:
                try:
                    s = _run_backtest(sweep["strategy"], days)
                    per_window[str(days)] = {
                        "sharpe": s.sharpe,
                        "pnl_usd": round(s.total_pnl_usd, 2),
                        "n_trades": s.n_trades,
                        "note": s.note or "",
                    }
                except Exception as e:  # noqa: BLE001 — one bad window ≠ dead sweep
                    per_window[str(days)] = {"error": str(e)[:120]}
            results[str(value)] = per_window
    finally:
        setattr(module, param, current)   # ALWAYS restore production value

    best_value, best_score = None, None
    for value, per_window in results.items():
        # Score: mean Sharpe across windows (missing/error windows count
        # as 0 so a value that only backtests on one horizon can't win
        # on a single lucky window).
        scores = []
        for days in WINDOWS:
            w = per_window.get(str(days)) or {}
            sh = w.get("sharpe")
            scores.append(float(sh) if isinstance(sh, (int, float)) else 0.0)
        score = sum(scores) / len(scores)
        if best_score is None or score > best_score:
            best_value, best_score = value, score

    recommendation = "KEEP_CURRENT"
    if best_value is not None and str(best_value) != str(current):
        # Require a meaningful margin before recommending a change —
        # parameter churn on noise is how overfitting happens.
        cur_scores = []
        cur_pw = results.get(str(current)) or {}
        for days in WINDOWS:
            sh = (cur_pw.get(str(days)) or {}).get("sharpe")
            cur_scores.append(
                float(sh) if isinstance(sh, (int, float)) else 0.0)
        cur_score = sum(cur_scores) / len(cur_scores)
        if best_score is not None and best_score > cur_score + 0.5:
            recommendation = f"CONSIDER {param}={best_value}"

    return {
        "strategy": sweep["strategy"],
        "param": param,
        "current_value": current,
        "results": results,
        "best_value": best_value,
        "best_mean_sharpe": round(best_score, 3) if best_score else None,
        "recommendation": recommendation,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Weekly strategy-parameter sweep (proposals only)")
    ap.add_argument("--strategy", default=None,
                    help="Limit to one strategy name")
    args = ap.parse_args()

    sweeps = [s for s in SWEEPS
              if args.strategy is None or s["strategy"] == args.strategy]
    out = {"as_of": datetime.now(UTC).isoformat(), "sweeps": []}
    for sweep in sweeps:
        logger.info(f"sweeping {sweep['strategy']}.{sweep['param']} "
                    f"over {sweep['candidates']}")
        try:
            out["sweeps"].append(run_sweep(sweep))
        except Exception as e:  # noqa: BLE001 — one dead sweep ≠ dead run
            logger.warning(f"sweep failed: {e}")
            out["sweeps"].append({
                "strategy": sweep["strategy"], "param": sweep["param"],
                "error": str(e)[:200],
            })

    TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    n_changes = sum(1 for s in out["sweeps"]
                    if str(s.get("recommendation", "")).startswith("CONSIDER"))
    logger.info(f"tuning: {len(out['sweeps'])} sweeps written to "
                f"{TUNING_PATH} ({n_changes} change recommendations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
