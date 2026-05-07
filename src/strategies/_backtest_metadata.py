"""Per-strategy backtest priors.

The strategy audit's #3 finding: the meta-allocator's Sharpe-tilt
currently has no real prior — every strategy gets the same baseline
target_alloc_pct from `ALL_STRATEGIES`, then drifts based on the
first ~30 LIVE trades. With ~24 strategies, that's months before
the allocator has anything statistically meaningful.

Wiring real backtest numbers (computed by `src/backtests/`) into
this registry transforms the allocator from "uniform prior →
random-walk seeding" into "evidence-weighted Bayes update from cycle
1." Strategies with known weak edges (the retired pead v1, the
crowded RSI(2), the long-only gap fade) get a smaller prior; the
genuinely uncorrelated edges (kalshi calibration, funding carry) get
a bigger one.

This module is a stub. Each entry has source-of-truth fields:
  - sharpe       : annualized realized Sharpe over the backtest window
  - max_dd_pct   : peak drawdown observed (positive number, 0-1)
  - n_trades     : sample size — important because Bayesian shrinkage
                    (allocator/metrics.py O6) downweights small N
  - period       : human-readable date range, for audit
  - source       : free-text where the number came from (which
                    backtest run, which dataset, etc.)
  - confidence   : LOW / MEDIUM / HIGH — captures whether the
                    backtest is actually trustworthy or just a
                    sanity check on synthetic data

Populating: each strategy that has a runnable backtest in
src/backtests/ should be filled in by running the backtest, copying
the numbers here, and committing alongside a code change. Until then,
entries are absent and the allocator falls back to the uniform prior
(current behaviour) — no regression risk from this module's mere
existence.

DO NOT generate fake numbers. If a strategy doesn't have a real
backtest, leave it out. The empty-dict default is honest; a
fabricated Sharpe is not.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestPrior:
    sharpe: float
    max_dd_pct: float
    n_trades: int
    period: str
    source: str
    confidence: str = "MEDIUM"   # LOW | MEDIUM | HIGH


# Populate as backtests come online. Empty for now (uniform prior is
# the current behaviour).
PRIORS: dict[str, BacktestPrior] = {
    # Example template — uncomment + fill when running backtest:
    # "tsmom_etf": BacktestPrior(
    #     sharpe=0.78, max_dd_pct=0.14, n_trades=42,
    #     period="2024-01-01..2025-12-31",
    #     source="src/backtests/tsmom_etf_backtest.py against Yahoo Finance",
    #     confidence="MEDIUM",
    # ),
}


def prior_for(strategy_name: str) -> BacktestPrior | None:
    """Return the backtest prior for `strategy_name`, or None when
    no real backtest has been wired yet (allocator falls back to
    uniform)."""
    return PRIORS.get(strategy_name)
