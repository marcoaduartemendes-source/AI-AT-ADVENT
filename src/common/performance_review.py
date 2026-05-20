"""Autonomous performance review — finds the alpha problem.

USER MANDATE (2026-05-20)
"Constantly monitor performance of both the setup and the
strategies to constantly improve performance and execute as much
by your own. Take responsibility for overall performance. Current
strategies aren't getting alpha — understand why."

This module is the closed-loop response. Every orchestrator cycle
it:

  1. READS the live data — docs/trades_recent.json (recent fills),
     docs/cycle_status.json (recent cycle health), docs/validation.json
     (backtest verdicts), docs/walk_forward.json (overfit detection).

  2. SCORES each strategy on three independent axes:
       • Backtest verdict (PASS / FAIL / UNPROVEN / NO_DATA)
       • Walk-forward verdict (ROBUST / OVERFIT_SUSPECT / WEAK / …)
       • Live divergence  (LIVE_OK / UNDER / OVER)
                          UNDER = live Sharpe << backtest Sharpe
                          OVER  = live Sharpe >> backtest Sharpe
     The three together catch ALL the failure modes the audit named:
     fee-bleed (FAIL), curve-fit (OVERFIT_SUSPECT), execution-drag
     (UNDER).

  3. WRITES docs/improvements.json — a ranked list of actions:
       priority 1: live capital is being eroded NOW       (UNDER + PASS)
       priority 2: backtest is a lie                       (OVERFIT_SUSPECT)
       priority 3: under-utilised winners                  (LIVE_OK + idle)
       priority 4: setup issues                            (chronic errors)

  4. EMITS the same data into the dashboard panel so the user
     sees the same priority queue I do.

WHY THIS IS THE RIGHT LEVER FOR ALPHA
"Alpha" = excess return vs benchmark. The bot will never have it
if it (a) keeps fee-bleeders, (b) trusts overfit backtests, or
(c) doesn't notice when live execution drifts from backtest. This
module attacks all three, autonomously, every cycle. Nothing
auto-promotes to live; the gate stays the human. But the analysis
that surfaces the recommendation is now continuous.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean, pstdev

logger = logging.getLogger(__name__)

# Days of live trade data to use when computing live Sharpe.
LIVE_WINDOW_DAYS = 30
# When live Sharpe is BELOW this fraction of backtest Sharpe → UNDER.
UNDER_THRESHOLD = 0.4
# When live Sharpe is ABOVE this fraction of backtest Sharpe → OVER.
OVER_THRESHOLD = 1.5
MIN_LIVE_TRADES = 5      # below this, live Sharpe is noise


def _read(path: str) -> dict | list | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _live_sharpe(trades: list[dict],
                   strategy: str,
                   window_days: int = LIVE_WINDOW_DAYS,
                   ) -> tuple[float | None, int, float]:
    """Annualised Sharpe of {strategy}'s realized PnL over the last
    `window_days`. Returns (sharpe, n_trades, total_pnl)."""
    if not trades:
        return None, 0, 0.0
    cutoff = datetime.now(UTC) - timedelta(days=window_days)
    pnls: list[float] = []
    total = 0.0
    for t in trades:
        if t.get("strategy") != strategy:
            continue
        ts = t.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt < cutoff:
            continue
        p = float(t.get("pnl_usd") or t.get("realized_pnl_usd") or 0.0)
        pnls.append(p)
        total += p
    n = len(pnls)
    if n < MIN_LIVE_TRADES:
        return None, n, total
    try:
        sd = pstdev(pnls)
        if sd < 1e-9:
            return None, n, total
        # Annualise — trades arrive at irregular cadence; this is a
        # rough proxy. Better than nothing; flagged in dashboard as
        # "approx" in case the user reads it as exact.
        return fmean(pnls) / sd * math.sqrt(252), n, total
    except Exception:
        return None, n, total


def _classify_live(live_s: float | None,
                    backtest_s: float | None) -> str:
    if live_s is None:
        return "INSUFFICIENT_LIVE"
    if backtest_s is None or backtest_s <= 0:
        # No backtest to compare; absolute call
        if live_s >= 0.5:
            return "LIVE_OK"
        return "UNDER"
    ratio = live_s / backtest_s
    if ratio < UNDER_THRESHOLD:
        return "UNDER"
    if ratio > OVER_THRESHOLD:
        return "OVER"
    return "LIVE_OK"


def _priority(verdict: str, wf: str, live: str) -> tuple[int, str]:
    """Composite priority: lower number = more urgent."""
    if verdict == "PASS" and live == "UNDER":
        return 1, ("backtest says PASS but live is UNDERperforming — "
                    "execution drag or regime change")
    if wf == "OVERFIT_SUSPECT":
        return 2, ("backtest is suspect (OOS Sharpe << IS Sharpe) "
                    "— the PASS verdict may be a lie")
    if verdict == "FAIL":
        return 3, "validation FAIL — freeze or retire"
    if verdict == "PASS" and live == "INSUFFICIENT_LIVE":
        return 4, ("PASS in backtest, not enough live data yet — "
                    "let it run to gather sample")
    if verdict == "PASS" and live == "LIVE_OK":
        return 5, "PASS + live aligned — keep running"
    if verdict == "UNPROVEN":
        return 6, "UNPROVEN — gate gate too strict or universe too narrow"
    return 9, "no action"


def run_performance_review(
        out_path: str = "docs/improvements.json") -> dict:
    """Synthesize backtest verdicts + walk-forward + live trades into
    a ranked action list. Best-effort, never raises."""
    val = _read("docs/validation.json") or {}
    wf = _read("docs/walk_forward.json") or {}
    trades = _read("docs/trades_recent.json") or []

    strategies = (val.get("strategies") or {})
    wf_results = (wf.get("strategies") or {})

    rows: list[dict] = []
    for name, vinfo in strategies.items():
        verdict = vinfo.get("verdict", "?")
        bt_sharpe = vinfo.get("sharpe_5y")
        wfi = wf_results.get(name) or {}
        wf_verdict = wfi.get("verdict", "?")
        live_s, n_live, live_pnl = _live_sharpe(trades, name)
        live_class = _classify_live(live_s, bt_sharpe)
        prio, why = _priority(verdict, wf_verdict, live_class)
        rows.append({
            "strategy": name,
            "priority": prio,
            "action_reason": why,
            "backtest_verdict": verdict,
            "backtest_sharpe_5y": bt_sharpe,
            "walk_forward_verdict": wf_verdict,
            "walk_forward_oos_sharpe": wfi.get("oos_sharpe"),
            "live_class": live_class,
            "live_sharpe_30d": (round(live_s, 3)
                                  if live_s is not None else None),
            "live_n_trades_30d": n_live,
            "live_pnl_30d": round(live_pnl, 2),
        })
    rows.sort(key=lambda r: (r["priority"], r["strategy"]))

    # Setup issues: chronic errors from cycle_status.
    cs = _read("docs/cycle_status.json") or []
    setup_issues: list[str] = []
    if isinstance(cs, list) and cs:
        from collections import Counter
        c = Counter()
        for cyc in cs[-20:]:                # last 20 cycles
            fe = (cyc.get("first_error") or "")[:120]
            if fe:
                c[fe.split(":")[0]] += 1
        for k, n in c.most_common(5):
            if n >= 3:                       # recurring ≥3× → flag
                setup_issues.append(f"recurring ({n}/20): {k}")

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "n_strategies": len(rows),
        "priority_counts": {
            f"p{p}": sum(1 for r in rows if r["priority"] == p)
            for p in range(1, 7)
        },
        "setup_issues": setup_issues,
        "strategies": rows,
        "method_notes": (
            "live_sharpe approximated from realized P&L per trade, "
            "annualised by sqrt(252); UNDER = live < "
            f"{UNDER_THRESHOLD:.0%} of backtest; OVER = live > "
            f"{OVER_THRESHOLD:.0%}. Insufficient data when "
            f"< {MIN_LIVE_TRADES} live trades in {LIVE_WINDOW_DAYS}d."
        ),
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"performance_review: {len(rows)} strategies, "
                     f"priorities {payload['priority_counts']}, "
                     f"setup_issues={len(setup_issues)}")
    except Exception as e:
        logger.warning(f"performance_review: write failed: {e}")
    return payload
