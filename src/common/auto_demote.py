"""Auto-demote agent — autonomously freezes failing strategies.

The performance review and walk-forward modules surface verdicts;
this module ACTS on them without human approval. It emits a
config-override JSON (docs/auto_overrides.json) that
StrategyMeta lookups respect, zeroing target_alloc for strategies
that meet ANY of:

  • Validation verdict FAIL on the most recent run
  • Walk-forward OVERFIT_SUSPECT
  • Live 30d Sharpe < -0.5 with ≥ 10 trades (real bleed)

(2026-05-21: the "zero proposals for N cycles" rule was REMOVED —
after the 30h orchestrator outage it froze 22/31 strategies,
including PASS ones, because every strategy had zero proposals
during the downtime. Freezing now keys ONLY on evidence of a bad
EDGE, never on operational state.)

This is the "thinks like an obsessed hedge fund manager" layer
the user asked for: nothing keeps a sleeve allocated to a losing
strategy. The freeze is REVERSIBLE — if the strategy recovers
(eg. its backtest moves back to PASS, or a new walk-forward
flips ROBUST), the next cycle removes the override.

The override file is loaded at the top of every strategy's
compute(); a 0% allocation effectively zeros the strategy. The
manual ALL_STRATEGIES list stays as the source of truth for
*registration*; this module overlays *allocation* dynamically.

Money-handling discipline: even this module CAN'T un-freeze a
strategy that the user manually set to 0 in ALL_STRATEGIES — the
override is min(manual, auto). Auto can only shrink, never grow.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Live-bleed threshold: 30d Sharpe below this with enough trades → freeze.
LIVE_BLEED_SHARPE = -0.5
LIVE_BLEED_MIN_TRADES = 10


def _read(path: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _live_sharpe_30d(trades: list[dict], strategy: str
                      ) -> tuple[float | None, int]:
    """Approximated annualised Sharpe of last 30 days' realized P&L."""
    if not trades:
        return None, 0
    import math
    from statistics import fmean, pstdev
    cutoff = datetime.now(UTC) - timedelta(days=30)
    pnls: list[float] = []
    for t in trades:
        if t.get("strategy") != strategy:
            continue
        try:
            dt = datetime.fromisoformat(
                str(t.get("timestamp", "")).replace("Z", "+00:00"))
        except Exception:
            continue
        if dt < cutoff:
            continue
        p = float(t.get("pnl_usd") or 0.0)
        pnls.append(p)
    if len(pnls) < LIVE_BLEED_MIN_TRADES:
        return None, len(pnls)
    try:
        sd = pstdev(pnls)
        if sd < 1e-9:
            return None, len(pnls)
        return fmean(pnls) / sd * math.sqrt(252), len(pnls)
    except Exception:
        return None, len(pnls)


def run_auto_demote(
        out_path: str = "docs/auto_overrides.json") -> dict:
    """Produce docs/auto_overrides.json — a strategy → allocation-
    multiplier map (0.0 = frozen, 1.0 = pass-through). Read by the
    orchestrator at allocation time to override target_alloc_usd
    BEFORE the strategy runs."""
    validation = _read("docs/validation.json") or {}
    walk_forward = _read("docs/walk_forward.json") or {}
    trades = _read("docs/trades_recent.json") or []

    val_strats = validation.get("strategies") or {}
    wf_strats = walk_forward.get("strategies") or {}

    overrides: dict[str, dict] = {}
    for name in val_strats:
        reasons: list[str] = []
        vinfo = val_strats[name]
        wfi = wf_strats.get(name) or {}
        # Rule 1 — validation FAIL.
        if vinfo.get("verdict") == "FAIL":
            reasons.append(
                f"validation FAIL: {vinfo.get('reason','')[:80]}")
        # Rule 2 — walk-forward OVERFIT_SUSPECT.
        if wfi.get("verdict") == "OVERFIT_SUSPECT":
            reasons.append(
                f"walk-forward OVERFIT_SUSPECT: "
                f"{wfi.get('reason','')[:80]}")
        # Rule 3 — live Sharpe deeply negative with sample size.
        live_s, n_live = _live_sharpe_30d(trades, name)
        if live_s is not None and live_s < LIVE_BLEED_SHARPE:
            reasons.append(
                f"live 30d Sharpe {live_s:+.2f} < "
                f"{LIVE_BLEED_SHARPE} ({n_live} trades)")
        # Rule 4 (REMOVED 2026-05-21) — "zero proposals for N cycles"
        # was catastrophically wrong: after the 30h orchestrator
        # outage EVERY strategy had a long zero-proposal streak, so
        # auto-demote froze 22 of 31 strategies — including PASS
        # ones (dividend_growth, bollinger_breakout, earnings_
        # momentum, commodity_carry). Zero proposals is entangled
        # with operational state (outages, market-closed, signal
        # gaps) and is NOT a reliable signal of a broken strategy.
        # Freezing now keys ONLY on evidence of a bad EDGE:
        # validation FAIL, walk-forward OVERFIT, or live bleed.
        if reasons:
            overrides[name] = {
                "multiplier": 0.0,
                "reasons": reasons,
                "frozen_at": datetime.now(UTC).isoformat(),
            }

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "n_frozen": len(overrides),
        "overrides": overrides,
        "rules": {
            "validation_fail": "any 5y verdict FAIL → freeze",
            "walk_forward_overfit_suspect": "OOS Sharpe << IS Sharpe",
            "live_bleed": (f"30d live Sharpe < {LIVE_BLEED_SHARPE} "
                            f"with ≥ {LIVE_BLEED_MIN_TRADES} trades"),
        },
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"auto_demote: {len(overrides)} strategies frozen")
    except Exception as e:
        logger.warning(f"auto_demote: write failed: {e}")
    return payload


def get_auto_multiplier(strategy: str,
                          path: str = "docs/auto_overrides.json"
                          ) -> float:
    """Return the auto-demote multiplier for `strategy` (0.0 = frozen,
    1.0 = pass-through). 1.0 if no override file or strategy not
    present. Used by the allocator to dynamically zero strategies
    that the rules say should not be deployed."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        ov = (d.get("overrides") or {}).get(strategy)
        if ov:
            return float(ov.get("multiplier", 1.0))
    except Exception:
        pass
    return 1.0
