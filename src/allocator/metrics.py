"""Per-strategy performance metrics — rolling Sharpe with Bayesian shrinkage.

The allocator weights strategies by recent risk-adjusted return. Naive
Sharpe ratios are noisy with few trades, so we apply two corrections:

  1. Bayesian shrinkage toward zero — a strategy with 5 trades shouldn't
     be allocated heavily even if its raw Sharpe looks great. Effective
     Sharpe = Sharpe × n / (n + tau), where tau = shrinkage prior count.

  2. A negative-Sharpe floor of zero — we never allocate negative weight.
     A losing strategy gets minimum allocation (or freeze, depending on
     lifecycle state), not a short position in itself.

Inputs to the metric calculator are PnL records pulled from the existing
performance.PerformanceTracker SQLite table, scoped to a single strategy
and a rolling window of N days.
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Snapshot of one strategy's recent performance."""

    name: str
    window_days: int
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    total_pnl_usd: float
    mean_pnl_usd: float
    std_pnl_usd: float
    raw_sharpe: float           # mean / std × sqrt(annualization)
    shrunk_sharpe: float        # raw_sharpe × n / (n + tau)
    drawdown_usd: float         # max drawdown from rolling cumulative peak
    drawdown_pct: float         # as % of strategy's peak cumulative pnl


class StrategyPerformance:
    """Reads PnL from the existing performance DB and computes metrics."""

    # Trades are typically intraday; ~252 trading days for annualization
    # is the convention. For ~hourly cadence, scale by sqrt(24*252) — but
    # the dominant noise is small-N, which the shrinkage handles.
    ANNUALIZATION = math.sqrt(252)

    def __init__(self, db_path: str | None = None, *, shrinkage_tau: int = 30):
        self.db_path = os.path.abspath(
            db_path or os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
        )
        self.tau = shrinkage_tau

    @contextmanager
    def _conn(self):
        from common.sqlite_pragmas import apply_pragmas
        conn = sqlite3.connect(self.db_path)
        apply_pragmas(conn, self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # ── Public API -------------------------------------------------------

    def metrics_for(self, strategy_name: str, window_days: int = 60) -> StrategyMetrics:
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT pnl_usd FROM trades "
                    "WHERE strategy=? AND side='SELL' AND timestamp>=? "
                    "ORDER BY timestamp",
                    (strategy_name, cutoff),
                ).fetchall()
            pnls = [float(r["pnl_usd"]) for r in rows if r["pnl_usd"] is not None]
        except sqlite3.OperationalError:
            pnls = []

        return self._compute(strategy_name, window_days, pnls)

    def _compute(self, name: str, window_days: int, pnls: list[float]) -> StrategyMetrics:
        n = len(pnls)
        if n == 0:
            return StrategyMetrics(name=name, window_days=window_days,
                                    n_trades=0, n_wins=0, n_losses=0, win_rate=0.0,
                                    total_pnl_usd=0.0, mean_pnl_usd=0.0,
                                    std_pnl_usd=0.0, raw_sharpe=0.0,
                                    shrunk_sharpe=0.0, drawdown_usd=0.0,
                                    drawdown_pct=0.0)

        wins = sum(1 for p in pnls if p > 0)
        total = sum(pnls)
        mean = total / n
        if n > 1:
            var = sum((p - mean) ** 2 for p in pnls) / (n - 1)
            std = math.sqrt(var)
        else:
            std = 0.0

        raw_sharpe = (mean / std * self.ANNUALIZATION) if std > 0 else 0.0
        shrink = n / (n + self.tau)
        shrunk_sharpe = raw_sharpe * shrink

        cum, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
        dd_pct = (max_dd / peak) if peak > 0 else 0.0

        return StrategyMetrics(
            name=name,
            window_days=window_days,
            n_trades=n,
            n_wins=wins,
            n_losses=n - wins,
            win_rate=wins / n,
            total_pnl_usd=total,
            mean_pnl_usd=mean,
            std_pnl_usd=std,
            raw_sharpe=raw_sharpe,
            shrunk_sharpe=shrunk_sharpe,
            drawdown_usd=max_dd,
            drawdown_pct=dd_pct,
        )

    def metrics_bulk(self, names: list[str], window_days: int = 60) -> dict[str, StrategyMetrics]:
        return {n: self.metrics_for(n, window_days) for n in names}
