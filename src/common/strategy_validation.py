"""Strategy validation gate — turns "hope" into "evidence".

The 2026-05-19 audit + the user's retirement-critical mandate make
this the most important module in the repo: no strategy should
risk capital until it has demonstrated a fee-aware, multi-window
edge on historical data.

WHAT IT DOES
Runs the existing backtest framework (backtests.runner.backtest_all)
across THREE lookback windows — 1y / 2y / 5y — and applies a
hard pass/fail rubric. A strategy that only looks good in one
window is overfit and FAILS; a strategy with too few trades is
UNPROVEN (not enough sample to trust the Sharpe).

PASS rubric (all must hold):
  • Sharpe ≥ MIN_SHARPE in the 5y (longest) window
  • Positive total P&L in ≥ 2 of the 3 windows  (consistency —
    catches single-regime overfit)
  • n_trades ≥ MIN_TRADES in the 5y window      (statistical
    significance — a 4-trade +Sharpe is noise)
  • return_on_volume_pct ≥ MIN_ROV              (the edge must
    survive fees; runner.py already charges 10bps round-trip)

Output: docs/validation.json — { strategy: {verdict, metrics...} }
The dashboard renders it; the orchestrator can gate live
promotion on verdict == "PASS".

This is intentionally STRICT. For retirement-critical capital the
default must be "prove it or stay in DRY", not "trade and hope".
"""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Windows in trading days: ~1y, ~2y, ~5y
_WINDOWS = (252, 504, 1260)

# Pass thresholds. Deliberately conservative — these are the bar a
# strategy must clear before it's allowed near real money.
MIN_SHARPE = 0.5          # net-of-fee annualised; <0.5 isn't worth the risk
MIN_TRADES = 20           # below this the Sharpe is statistically noise
MIN_ROV = 0.0             # return-on-volume must be positive after fees
MIN_POSITIVE_WINDOWS = 2  # of 3 — single-window winners are overfit

# Re-run at most once per this interval (backtests hit external data
# APIs and are slow; the verdict barely moves intraday).
_REVALIDATE_AFTER_SECONDS = 86_400  # 24h


def _verdict(by_window: dict[int, object]) -> tuple[str, dict]:
    """Apply the rubric. Returns (verdict, metrics_dict).

    verdict ∈ {PASS, FAIL, UNPROVEN, NO_DATA}
      PASS      — cleared every bar; eligible for live promotion
      FAIL      — backtested but edge is absent / fee-negative / overfit
      UNPROVEN  — ran but too few trades to conclude (keep in DRY,
                  gather paper data)
      NO_DATA   — no backtest available (data feed missing)
    """
    longest = max(_WINDOWS)
    long_s = by_window.get(longest)
    if long_s is None or getattr(long_s, "note", "").startswith("error"):
        return "NO_DATA", {"reason": getattr(long_s, "note", "no backtest")}

    sharpes = {w: getattr(by_window.get(w), "sharpe", None) for w in _WINDOWS}
    pnls = {w: getattr(by_window.get(w), "total_pnl_usd", 0.0) for w in _WINDOWS}
    n_trades_long = getattr(long_s, "n_trades", 0) or 0
    rov_long = getattr(long_s, "return_on_volume_pct", 0.0) or 0.0
    sharpe_long = sharpes.get(longest)

    metrics = {
        "sharpe_1y": sharpes.get(252),
        "sharpe_2y": sharpes.get(504),
        "sharpe_5y": sharpe_long,
        "pnl_1y": round(pnls.get(252, 0.0), 2),
        "pnl_2y": round(pnls.get(504, 0.0), 2),
        "pnl_5y": round(pnls.get(1260, 0.0), 2),
        "n_trades_5y": n_trades_long,
        "return_on_volume_pct": round(rov_long, 3),
    }

    # No-trade strategies (publish-only overlays) are not FAILs.
    if n_trades_long == 0 and all((pnls.get(w, 0.0) == 0.0)
                                    for w in _WINDOWS):
        return "NO_DATA", {**metrics, "reason": "no trades (overlay?)"}

    positive_windows = sum(1 for w in _WINDOWS if pnls.get(w, 0.0) > 0)

    if n_trades_long < MIN_TRADES:
        metrics["reason"] = (f"only {n_trades_long} trades in 5y "
                              f"(need ≥{MIN_TRADES}) — unproven")
        return "UNPROVEN", metrics

    fails = []
    if sharpe_long is None or sharpe_long < MIN_SHARPE:
        fails.append(f"5y Sharpe {sharpe_long} < {MIN_SHARPE}")
    if positive_windows < MIN_POSITIVE_WINDOWS:
        fails.append(f"positive in only {positive_windows}/3 windows "
                      f"(overfit risk)")
    if rov_long < MIN_ROV:
        fails.append(f"return-on-volume {rov_long:.2f}% ≤ 0 "
                      f"(fee-negative)")
    if fails:
        metrics["reason"] = "; ".join(fails)
        return "FAIL", metrics

    metrics["reason"] = (f"Sharpe {sharpe_long} over 5y, positive "
                          f"{positive_windows}/3 windows, "
                          f"{n_trades_long} trades")
    return "PASS", metrics


def run_validation(out_path: str = "docs/validation.json",
                    force: bool = False) -> dict:
    """Backtest every strategy across 1/2/5y, apply the rubric, write
    docs/validation.json. Rate-limited to once/24h unless force=True.
    Best-effort: never raises (orchestrator calls from finally)."""
    p = Path(out_path)
    # Rate-limit: skip if a fresh verdict already exists.
    if not force and p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(prev.get("as_of", "").replace(
                "Z", "+00:00"))
            age = (datetime.now(UTC) - ts.replace(tzinfo=UTC)
                   ).total_seconds()
            if age < _REVALIDATE_AFTER_SECONDS:
                return prev
        except Exception:
            pass  # corrupt / unparseable → re-run

    try:
        from backtests.runner import backtest_all
    except Exception as e:
        logger.warning(f"validation: backtest import failed: {e}")
        return {}

    t0 = time.time()
    per_window: dict[int, dict] = {}
    for w in _WINDOWS:
        try:
            per_window[w] = backtest_all(w)
        except Exception as e:
            logger.warning(f"validation: backtest_all({w}) failed: {e}")
            per_window[w] = {}

    strategies = set()
    for w in _WINDOWS:
        strategies.update(per_window.get(w, {}).keys())

    results: dict[str, dict] = {}
    n_pass = 0
    for s in sorted(strategies):
        by_w = {w: per_window.get(w, {}).get(s) for w in _WINDOWS}
        verdict, metrics = _verdict(by_w)
        if verdict == "PASS":
            n_pass += 1
        results[s] = {"verdict": verdict, **metrics}

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "windows_days": list(_WINDOWS),
        "rubric": {
            "min_sharpe_5y": MIN_SHARPE,
            "min_trades_5y": MIN_TRADES,
            "min_return_on_volume_pct": MIN_ROV,
            "min_positive_windows": MIN_POSITIVE_WINDOWS,
        },
        "n_strategies": len(results),
        "n_pass": n_pass,
        "elapsed_seconds": round(time.time() - t0, 1),
        "strategies": results,
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"validation: {n_pass}/{len(results)} strategies "
                    f"PASS ({payload['elapsed_seconds']}s)")
    except Exception as e:
        logger.warning(f"validation: write failed: {e}")
    return payload


def passing_strategies(path: str = "docs/validation.json") -> set[str]:
    """The set of strategy names whose verdict == PASS. Used as the
    live-promotion gate. Empty set when validation hasn't run —
    fail-safe: nothing auto-promotes without proof."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return {s for s, v in (d.get("strategies") or {}).items()
                if v.get("verdict") == "PASS"}
    except Exception:
        return set()
