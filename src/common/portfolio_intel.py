"""Portfolio intelligence — correlation, concentration, diversification.

THE GAP THIS CLOSES
The allocator weights strategies by risk-adjusted return (Sharpe tilt),
but it is blind to CORRELATION between them. The 2026-05 audit found the
entire PASS roster is equity-beta (bollinger, multifactor, earnings,
thematic, sector, dual-momentum…). A book of ten 0.9-correlated equity
strategies is not diversified — it's a single levered SPY bet wearing ten
hats, and it draws down all at once. Sharpe-weighting alone can't see
this; you need the cross-strategy correlation structure.

WHAT IT COMPUTES (from the FIFO realized-P&L event stream — the
auditable source, so corrupt stored pnl can't distort it):
  • Per-strategy daily P&L vectors over a window, aligned on dates.
  • Pairwise correlation matrix.
  • Average pairwise correlation (book cohesion — high = concentrated).
  • Effective number of independent bets (ENB) = the participation ratio
    of the correlation matrix's eigenvalues:
        ENB = (Σλ)² / Σλ²   ∈ [1, N]
    N strategies with zero mutual correlation → ENB = N (fully
    diversified); all perfectly correlated → ENB = 1 (one real bet).
  • A plain-English verdict + the most-correlated pair, so the user sees
    "you think you have 11 strategies; you really have 2.3 bets."

This is read-only intelligence (writes docs/portfolio_intel.json, shown
on the dashboard). It does not change allocations — surfacing the risk
is step one; correlation-aware sizing is a deliberate later decision.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_OVERLAP_DAYS = 10          # need this many shared trading days to trust a corr
HIGH_CORR = 0.6                # avg pairwise above this → concentration warning


def _daily_pnl_vectors(db_path: str, window_days: int
                       ) -> dict[str, dict[str, float]]:
    """{strategy: {date_iso: summed_pnl}} from FIFO close events."""
    try:
        from trading.recompute import fifo_realized_events
        events = fifo_realized_events(db_path)
    except Exception as e:
        logger.debug(f"portfolio_intel: fifo events failed: {e}")
        return {}
    cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for strat, evs in events.items():
        for e in evs:
            ts = str(e.get("timestamp", ""))
            if ts < cutoff:
                continue
            day = ts[:10]
            out[strat][day] += float(e.get("pnl_usd") or 0.0)
    return {s: dict(v) for s, v in out.items()}


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < MIN_OVERLAP_DAYS:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 1e-12 or vb <= 1e-12:
        return None
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    return cov / (va ** 0.5 * vb ** 0.5)


def _effective_bets(corr: dict[tuple[str, str], float],
                    names: list[str]) -> float | None:
    """Participation ratio of the correlation matrix eigenvalues:
    ENB = (Σλ)²/Σλ². Uses numpy if available (it is — it's a core dep)."""
    if len(names) < 2:
        return float(len(names))
    try:
        import numpy as np
        n = len(names)
        m = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = corr.get((names[i], names[j]))
                if c is not None:
                    m[i, j] = m[j, i] = c
        eig = np.linalg.eigvalsh(m)
        eig = eig[eig > 0]
        if eig.size == 0:
            return None
        return float((eig.sum() ** 2) / (eig ** 2).sum())
    except Exception as e:
        logger.debug(f"portfolio_intel: ENB failed: {e}")
        return None


def run_portfolio_intel(db_path: str | None = None,
                        window_days: int = 90,
                        out_path: str = "docs/portfolio_intel.json") -> dict:
    """Compute the correlation/diversification snapshot. Never raises."""
    import os
    db_path = db_path or os.environ.get(
        "TRADING_DB_PATH", "data/trading_performance.db")
    vectors = _daily_pnl_vectors(db_path, window_days)
    # Only strategies with enough active days are comparable.
    active = {s: v for s, v in vectors.items() if len(v) >= MIN_OVERLAP_DAYS}
    names = sorted(active)

    corr: dict[tuple[str, str], float] = {}
    pairs_out: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            s1, s2 = names[i], names[j]
            shared = sorted(set(active[s1]) & set(active[s2]))
            if len(shared) < MIN_OVERLAP_DAYS:
                continue
            c = _pearson([active[s1][d] for d in shared],
                         [active[s2][d] for d in shared])
            if c is None:
                continue
            corr[(s1, s2)] = c
            pairs_out.append({"a": s1, "b": s2, "corr": round(c, 3),
                              "days": len(shared)})

    avg_corr = (sum(corr.values()) / len(corr)) if corr else None
    enb = _effective_bets(corr, names) if corr else (
        float(len(names)) if names else None)
    pairs_out.sort(key=lambda p: -p["corr"])

    if avg_corr is None:
        verdict = ("Not enough overlapping live history yet to measure "
                   "cross-strategy correlation — gather more trades.")
    elif avg_corr >= HIGH_CORR:
        verdict = (f"CONCENTRATED: avg pairwise corr {avg_corr:+.2f}. "
                   f"You have {len(names)} strategies but only "
                   f"~{enb:.1f} independent bets — they draw down "
                   f"together. Add uncorrelated sleeves (bonds, "
                   f"commodities, crypto carry, prediction markets) "
                   f"before scaling capital.")
    else:
        verdict = (f"Reasonably diversified: avg pairwise corr "
                   f"{avg_corr:+.2f}, ~{enb:.1f} independent bets across "
                   f"{len(names)} strategies.")

    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "window_days": window_days,
        "n_strategies_compared": len(names),
        "avg_pairwise_corr": round(avg_corr, 3) if avg_corr is not None else None,
        "effective_bets": round(enb, 2) if enb is not None else None,
        "concentrated": bool(avg_corr is not None and avg_corr >= HIGH_CORR),
        "top_correlated_pairs": pairs_out[:8],
        "verdict": verdict,
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info(f"portfolio_intel: {len(names)} strategies, "
                    f"avg_corr={payload['avg_pairwise_corr']}, "
                    f"ENB={payload['effective_bets']}")
    except Exception as e:
        logger.warning(f"portfolio_intel: write failed: {e}")
    return payload
