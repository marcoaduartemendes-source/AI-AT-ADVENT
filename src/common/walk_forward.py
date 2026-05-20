"""Walk-forward validation — catches in-sample overfit Sharpes.

The original strategy_validation.py runs each backtest over a single
full window. That's necessary but not sufficient: a strategy with
look-ahead bias or a curve-fitted parameter shines on the full
backtest but collapses when tested on data the model never saw.

WALK-FORWARD METHOD
For each strategy:
  1. Backtest the FIRST 60% of the available history  (IN-SAMPLE)
  2. Backtest the LAST  40%                            (OUT-OF-SAMPLE)
  3. Compare Sharpe / total P&L between the two halves.

  Sharpe ratio:  OOS_Sharpe / IS_Sharpe
    ≥ 0.5   → robust, edge generalises
    < 0.5   → suspect, possibly overfit
    < 0.0   → likely overfit; live capital is risky

The 1260-day full window inside strategy_validation runs the
backtests already; we just split the resulting equity curve.
Avoids re-running every backtest (≈ 200s per pass).

OUTPUT
docs/walk_forward.json — { strategy: {is_sharpe, oos_sharpe, ratio,
                                        verdict, reason} }

The dashboard renders this alongside the main validation panel so
"PASS in single-window but OVERFIT_SUSPECT in walk-forward" stands
out — that's the risk_parity_etf Sharpe-9.43 footgun.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

IS_FRAC = 0.60                   # 60% in-sample / 40% out-of-sample
MIN_OOS_RATIO_PASS = 0.5         # OOS Sharpe must be ≥ 50% of IS
MIN_TRADES_PER_HALF = 5          # below this, can't conclude

_REVALIDATE_AFTER_SECONDS = 86_400


def _split_sharpe(equity_curve: list[dict]) -> tuple[float | None,
                                                       float | None,
                                                       int, int]:
    """Split the BacktestSummary.equity_curve points into IS/OOS halves
    and return (is_sharpe, oos_sharpe, is_trades, oos_trades).

    The equity_curve is a list of {timestamp, equity, pnl} dicts
    in chronological order. We compute Sharpe from the *bar-to-bar*
    P&L deltas in each half, annualised by sqrt(252) — matching
    runner._equity_curve_to_summary.
    """
    if not equity_curve or len(equity_curve) < 10:
        return None, None, 0, 0
    n = len(equity_curve)
    split = max(int(n * IS_FRAC), 1)
    is_pts = equity_curve[:split]
    oos_pts = equity_curve[split:]

    def _curve_sharpe(pts):
        if len(pts) < 3:
            return None, 0
        pnls = []
        prev = pts[0].get("equity", 0.0)
        for p in pts[1:]:
            cur = p.get("equity", 0.0)
            pnls.append(cur - prev)
            prev = cur
        if not pnls:
            return None, 0
        n_trades = sum(1 for x in pnls if x != 0)
        import statistics
        try:
            sd = statistics.pstdev(pnls)
            mu = statistics.fmean(pnls)
            if sd < 1e-9:
                return None, n_trades
            return mu / sd * math.sqrt(252), n_trades
        except statistics.StatisticsError:
            return None, n_trades

    is_s, is_t = _curve_sharpe(is_pts)
    oos_s, oos_t = _curve_sharpe(oos_pts)
    return is_s, oos_s, is_t, oos_t


def _verdict(is_s: float | None, oos_s: float | None,
              is_t: int, oos_t: int) -> tuple[str, str]:
    """Walk-forward verdict + reason."""
    if is_s is None or oos_s is None:
        return "NO_DATA", "insufficient equity-curve points"
    if is_t < MIN_TRADES_PER_HALF or oos_t < MIN_TRADES_PER_HALF:
        return "UNPROVEN", (f"too few trades to compare "
                              f"(IS={is_t}, OOS={oos_t}, "
                              f"need ≥{MIN_TRADES_PER_HALF} each)")
    if is_s <= 0 and oos_s <= 0:
        return "FAIL", (f"both halves negative Sharpe "
                          f"(IS={is_s:+.2f}, OOS={oos_s:+.2f})")
    # Ratio is meaningful when IS_Sharpe > 0
    if is_s > 0.1:
        ratio = oos_s / is_s
        if ratio < 0:
            return "OVERFIT_SUSPECT", (f"OOS Sharpe {oos_s:+.2f} is "
                                         f"NEGATIVE vs IS {is_s:+.2f} — "
                                         f"edge inverted out-of-sample")
        if ratio < MIN_OOS_RATIO_PASS:
            return "OVERFIT_SUSPECT", (f"OOS Sharpe {oos_s:+.2f} only "
                                         f"{ratio:.0%} of IS {is_s:+.2f} "
                                         f"(need ≥ "
                                         f"{MIN_OOS_RATIO_PASS:.0%})")
        return "ROBUST", (f"OOS Sharpe {oos_s:+.2f} is {ratio:.0%} of "
                            f"IS {is_s:+.2f} — edge generalises")
    # IS Sharpe near zero / negative — the strategy was already weak
    return "WEAK", f"IS Sharpe {is_s:+.2f} too low to assess robustness"


def run_walk_forward(longest_window: int = 1260,
                       out_path: str = "docs/walk_forward.json",
                       force: bool = False) -> dict:
    """Re-run backtest_all on the longest validation window, split
    every strategy's equity curve into 60/40, and emit walk-forward
    verdicts. Rate-limited to once/24h (same cadence as the main
    validation harness)."""
    p = Path(out_path)
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
            pass

    try:
        from backtests.runner import backtest_all
    except Exception as e:
        logger.warning(f"walk_forward: backtest import failed: {e}")
        return {}

    summaries = backtest_all(longest_window)
    results: dict[str, dict] = {}
    counts = {"ROBUST": 0, "OVERFIT_SUSPECT": 0, "WEAK": 0,
                "UNPROVEN": 0, "FAIL": 0, "NO_DATA": 0}
    for name, s in summaries.items():
        if s is None:
            results[name] = {"verdict": "NO_DATA",
                              "reason": "no backtest available"}
            counts["NO_DATA"] += 1
            continue
        eq = getattr(s, "equity_curve", None) or []
        is_s, oos_s, is_t, oos_t = _split_sharpe(eq)
        verdict, reason = _verdict(is_s, oos_s, is_t, oos_t)
        counts[verdict] = counts.get(verdict, 0) + 1
        results[name] = {
            "verdict": verdict,
            "is_sharpe": round(is_s, 3) if is_s is not None else None,
            "oos_sharpe": round(oos_s, 3) if oos_s is not None else None,
            "is_trades": is_t, "oos_trades": oos_t,
            "ratio": (round(oos_s / is_s, 3)
                      if (is_s and is_s > 0.1 and oos_s is not None)
                      else None),
            "reason": reason,
        }

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "window_days": longest_window,
        "is_frac": IS_FRAC,
        "min_oos_ratio_pass": MIN_OOS_RATIO_PASS,
        "counts": counts,
        "n_strategies": len(results),
        "strategies": results,
    }
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"walk_forward: {counts.get('ROBUST',0)} ROBUST / "
                     f"{counts.get('OVERFIT_SUSPECT',0)} OVERFIT_SUSPECT "
                     f"/ {len(results)} total")
    except Exception as e:
        logger.warning(f"walk_forward: write failed: {e}")
    return payload
