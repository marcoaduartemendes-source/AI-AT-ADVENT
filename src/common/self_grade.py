"""Daily self-grade — autonomous honest evaluation, 0–10 scale.

USER MANDATE (2026-05-20)
"Always do an honest evaluation on a daily basis on your
performance and send the report and what are you doing about,
what have you tried and succeeded and failed."

This is the accountability layer. Every day the orchestrator
computes a 0–10 grade across six axes, writes docs/self_grade.json,
and the daily digest includes it. No grade-inflation: the cutoffs
are deliberately conservative.

GRADE COMPONENTS (each 0–10, then weighted average)
  alpha_track            0.30  Live 30d Sharpe vs SPY benchmark
  fee_discipline         0.15  Fraction of trades from PASS-verdict strategies
  overfit_resistance     0.15  Fraction of PASSes that are walk-forward ROBUST
  execution_quality      0.15  Submit-ratio (submitted / proposed)
  setup_health           0.10  Fraction of recent cycles error-free
  research_freshness     0.15  Validation panel age (younger = better)

TARGET: sustained ≥ 9.0 grade. Until we hit that, this module
explicitly records: what we tried, what worked, what failed,
what's next. The dashboard and digest carry the verdict so the
user has continuous visibility without having to ask.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean, pstdev

logger = logging.getLogger(__name__)


def _read(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _grade_alpha(trades: list[dict] | None) -> tuple[float, str]:
    """0–10 by live 30d Sharpe across ALL strategies (book-level)."""
    if not trades:
        return 0.0, "no trades in last 30d → 0/10"
    cutoff = datetime.now(UTC) - timedelta(days=30)
    pnls: list[float] = []
    for t in trades:
        try:
            dt = datetime.fromisoformat(
                str(t.get("timestamp", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt < cutoff:
            continue
        pnls.append(float(t.get("pnl_usd") or 0.0))
    if len(pnls) < 5:
        return 1.0, f"only {len(pnls)} trades in 30d (sample too small) → 1/10"
    try:
        sd = pstdev(pnls)
        if sd < 1e-9:
            return 2.0, "live P&L has zero variance → 2/10"
        sharpe = fmean(pnls) / sd * math.sqrt(252)
    except Exception:
        return 1.0, "could not compute live Sharpe → 1/10"
    # Sharpe → grade. 0 = 5/10 (random), 1 = 7, 2 = 9, ≥3 = 10.
    if sharpe < -1:   g = 0.0
    elif sharpe < 0:  g = max(0.0, 4.0 + sharpe * 2)
    elif sharpe < 1:  g = 5.0 + sharpe * 2.0
    elif sharpe < 2:  g = 7.0 + (sharpe - 1) * 2.0
    elif sharpe < 3:  g = 9.0 + (sharpe - 2)
    else:             g = 10.0
    return g, (f"live 30d Sharpe {sharpe:+.2f} across {len(pnls)} "
                f"trades → {g:.1f}/10")


def _grade_fee_discipline(trades, validation) -> tuple[float, str]:
    """Fraction of recent trades coming from PASS-verdict strategies."""
    if not trades or not validation:
        return 5.0, "insufficient data → 5/10 (neutral)"
    pass_set = {s for s, v in (validation.get("strategies") or {}).items()
                  if v.get("verdict") == "PASS"}
    if not pass_set:
        return 3.0, "no PASS strategies in validation → 3/10"
    cutoff = datetime.now(UTC) - timedelta(days=30)
    n_pass = n_total = 0
    for t in trades:
        try:
            dt = datetime.fromisoformat(
                str(t.get("timestamp", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt < cutoff:
            continue
        n_total += 1
        if t.get("strategy") in pass_set:
            n_pass += 1
    if n_total == 0:
        return 5.0, "no recent trades → 5/10 (neutral)"
    frac = n_pass / n_total
    g = round(frac * 10, 1)
    return g, f"{n_pass}/{n_total} trades from PASS strategies → {g}/10"


def _grade_overfit_resistance(validation, walk_forward) -> tuple[float, str]:
    """Of validation-PASSes, fraction also ROBUST in walk-forward."""
    if not validation or not walk_forward:
        return 5.0, "walk-forward not yet computed → 5/10 (neutral)"
    pass_set = {s for s, v in (validation.get("strategies") or {}).items()
                  if v.get("verdict") == "PASS"}
    if not pass_set:
        return 0.0, "no PASS strategies → 0/10"
    robust = {s for s, v in (walk_forward.get("strategies") or {}).items()
                if v.get("verdict") == "ROBUST"}
    overlap = pass_set & robust
    frac = len(overlap) / len(pass_set)
    g = round(frac * 10, 1)
    return g, (f"{len(overlap)}/{len(pass_set)} PASSes are also "
                f"walk-forward ROBUST → {g}/10")


def _grade_execution(cycle_status) -> tuple[float, str]:
    """Submitted / proposed ratio across last 20 cycles."""
    if not isinstance(cycle_status, list) or not cycle_status:
        return 5.0, "no cycle data → 5/10"
    recent = sorted(cycle_status,
                     key=lambda c: c.get("timestamp", ""))[-20:]
    total_p = sum(c.get("proposals_total", 0) for c in recent)
    total_s = sum(c.get("proposals_submitted", 0) for c in recent)
    if total_p == 0:
        return 5.0, "no proposals in last 20 cycles → 5/10"
    ratio = total_s / total_p
    g = round(ratio * 10, 1)
    return g, (f"submit ratio {total_s}/{total_p} = {ratio:.0%} → "
                f"{g}/10 (≥80% target)")


def _grade_setup_health(cycle_status) -> tuple[float, str]:
    """Fraction of recent cycles with zero errors."""
    if not isinstance(cycle_status, list) or not cycle_status:
        return 5.0, "no cycle data → 5/10"
    recent = sorted(cycle_status,
                     key=lambda c: c.get("timestamp", ""))[-20:]
    clean = sum(1 for c in recent if (c.get("n_errors") or 0) == 0)
    frac = clean / len(recent)
    g = round(frac * 10, 1)
    return g, f"{clean}/{len(recent)} clean cycles → {g}/10"


def _grade_alpha_vs_benchmark() -> tuple[float, str]:
    """Excess return vs SPY over the longest window benchmark.json
    exposes. This is the bottom-line definition of alpha — did the
    bot beat the index after fees? 0–10 scale: beating SPY by ≥3pp
    annualised = 10; matching = 6; underperforming by ≥3pp = 0."""
    b = _read("docs/benchmark.json")
    if not b:
        return 5.0, "benchmark.json not yet produced → 5.0/10 (neutral)"
    # benchmark.json shape varies; try common fields.
    bot = b.get("bot_return_pct_ann") or b.get("portfolio_return_ann")
    spy = b.get("spy_return_pct_ann") or b.get("benchmark_return_ann")
    if bot is None or spy is None:
        # Fall back to simpler total-return comparison.
        bot = b.get("bot_return_pct") or b.get("portfolio_return_total")
        spy = b.get("spy_return_pct") or b.get("benchmark_return_total")
    if bot is None or spy is None:
        return 5.0, "benchmark.json lacks comparable fields → 5.0/10"
    try:
        excess_pp = float(bot) - float(spy)
    except Exception:
        return 5.0, "benchmark fields unparseable → 5.0/10"
    # Linear interp: -3pp → 0, 0pp → 6, +3pp → 10.
    if excess_pp <= -3: g = 0.0
    elif excess_pp <= 0: g = 6.0 + (excess_pp / 3) * 6.0
    elif excess_pp <= 3: g = 6.0 + (excess_pp / 3) * 4.0
    else: g = 10.0
    g = round(max(0.0, min(10.0, g)), 1)
    return g, (f"bot {bot:+.1f}% vs SPY {spy:+.1f}% → excess "
                f"{excess_pp:+.1f}pp → {g}/10")


def _grade_research_freshness(validation) -> tuple[float, str]:
    """Younger validation = higher. 0h=10, 24h=8, 48h=5, 72h=3, >7d=0."""
    if not validation:
        return 0.0, "no validation yet → 0/10"
    try:
        ts = datetime.fromisoformat(
            validation.get("as_of", "").replace("Z", "+00:00"))
        age_h = (datetime.now(UTC) - ts).total_seconds() / 3600
    except Exception:
        return 0.0, "validation timestamp unparseable → 0/10"
    if age_h <= 24:   g = 10 - (age_h / 24) * 2
    elif age_h <= 48: g = 8 - ((age_h - 24) / 24) * 3
    elif age_h <= 72: g = 5 - ((age_h - 48) / 24) * 2
    elif age_h <= 168: g = max(0.0, 3 - ((age_h - 72) / 96) * 3)
    else:             g = 0.0
    g = round(max(0.0, g), 1)
    return g, f"validation {age_h:.1f}h old → {g}/10"


def run_self_grade(out_path: str = "docs/self_grade.json") -> dict:
    """Compute the daily 0–10 grade across all six axes. Never raises."""
    trades = _read("docs/trades_recent.json")
    validation = _read("docs/validation.json")
    walk_forward = _read("docs/walk_forward.json")
    cycle_status = _read("docs/cycle_status.json")

    alpha_g, alpha_r = _grade_alpha(trades)
    fee_g, fee_r = _grade_fee_discipline(trades, validation)
    ovf_g, ovf_r = _grade_overfit_resistance(validation, walk_forward)
    exec_g, exec_r = _grade_execution(cycle_status)
    setup_g, setup_r = _grade_setup_health(cycle_status)
    rsch_g, rsch_r = _grade_research_freshness(validation)
    # ── 2026-05-20: data-quality axis. User mandate: "rate yourself
    # across all critical dimensions from 1 to 10". The data layer
    # IS a critical dimension — a great grade is meaningless if the
    # underlying JSON is stale, missing, or inconsistent.
    dq = _read("docs/data_quality.json") or {}
    dq_g = dq.get("score", 5.0) if dq else 5.0
    dq_r = (f"data-quality checks {dq.get('counts', {})} → {dq_g}/10"
             if dq else "no data-quality audit yet → 5.0/10 (neutral)")
    # ── Alpha-vs-benchmark — explicit excess-return read against SPY
    # via docs/benchmark.json (the existing benchmark.json carries
    # the bot's equity curve + SPY's). This is what "alpha" means in
    # practice: did we beat the index after fees?
    alpha_vs_g, alpha_vs_r = _grade_alpha_vs_benchmark()

    weights = {"alpha_track": 0.22, "alpha_vs_spy": 0.18,
                "fee_discipline": 0.12, "overfit_resistance": 0.12,
                "execution_quality": 0.12, "setup_health": 0.08,
                "research_freshness": 0.08, "data_quality": 0.08}
    components = {
        "alpha_track":         {"score": alpha_g, "reason": alpha_r},
        "alpha_vs_spy":        {"score": alpha_vs_g, "reason": alpha_vs_r},
        "fee_discipline":      {"score": fee_g, "reason": fee_r},
        "overfit_resistance":  {"score": ovf_g, "reason": ovf_r},
        "execution_quality":   {"score": exec_g, "reason": exec_r},
        "setup_health":        {"score": setup_g, "reason": setup_r},
        "research_freshness":  {"score": rsch_g, "reason": rsch_r},
        "data_quality":        {"score": dq_g, "reason": dq_r},
    }
    overall = round(sum(components[k]["score"] * weights[k]
                          for k in components), 2)

    # Self-narrative: what we tried, what worked, what failed.
    tried, worked, failed, next_up = [], [], [], []
    if validation:
        n_pass = validation.get("n_pass", 0)
        n_tot = validation.get("n_strategies", 0)
        tried.append(f"Backtested {n_tot} strategies across 1y/2y/5y "
                      f"under a strict fee-aware rubric.")
        if n_pass:
            worked.append(f"{n_pass}/{n_tot} strategies cleared the "
                            f"validation gate (PASS).")
        if n_tot - n_pass > 0:
            failed.append(f"{n_tot - n_pass}/{n_tot} strategies did NOT "
                            f"pass — fee-bleeders or overfit.")
    if walk_forward:
        counts = walk_forward.get("counts", {})
        overfit_n = counts.get("OVERFIT_SUSPECT", 0)
        if overfit_n:
            failed.append(f"{overfit_n} strategies failed walk-forward — "
                            f"their backtest Sharpe doesn't hold OOS.")
    if exec_g < 7:
        next_up.append(f"Submit ratio {exec_r.split('=')[0]}— "
                        f"investigate why proposals are being rejected.")
    if alpha_g < 6:
        next_up.append("Live alpha is weak — the PASS strategies aren't "
                        "yet translating to real returns; review "
                        "execution drag (improvements panel, P1 LIVE BLEED).")
    if overall < 9:
        next_up.append(f"Current grade {overall}/10 < target 9.0 — "
                        f"focus on the lowest-scoring axis next "
                        f"({min(components, key=lambda k: components[k]['score'])}).")

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "overall_grade": overall,
        "target_grade": 9.0,
        "weights": weights,
        "components": components,
        "narrative": {
            "tried": tried,
            "worked": worked,
            "failed": failed,
            "next": next_up,
        },
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"self_grade: overall {overall}/10 "
                     f"(target {payload['target_grade']})")
    except Exception as e:
        logger.warning(f"self_grade: write failed: {e}")
    return payload
