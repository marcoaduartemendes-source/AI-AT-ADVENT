"""Meta-Allocator — the rebalancer.

Algorithm (intentionally simple; over-fit risk is the enemy):

  1. Start from each strategy's `target_alloc_pct` baseline.
  2. Pull rolling 60-day metrics; compute shrunk Sharpe for each strategy.
  3. Auto-transition lifecycle states:
       - 30d Sharpe < 0 OR DD > warning_dd  → WATCH
       - 30d Sharpe < freeze OR DD > kill   → FROZEN
       - WATCH with next 14d Sharpe > 0.5   → ACTIVE
  4. Final weight = max(min_alloc_pct, baseline × Sharpe-tilt) for ACTIVE,
     × 0.5 for WATCH, 0 for FROZEN/RETIRED. Cap at max_alloc_pct.
  5. Normalize so weights sum to 100% (across active+watch).
  6. Cap weekly weight delta at ±5% per strategy to avoid whipsaw.
  7. Convert % to USD using current portfolio equity.

Returns an `AllocationDecision` per strategy and persists everything for
the dashboard.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC

from .lifecycle import StrategyRegistry, StrategyState
from .metrics import StrategyMetrics, StrategyPerformance

logger = logging.getLogger(__name__)


# ─── Tunables ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AllocatorConfig:
    primary_window_days: int = 60
    secondary_window_days: int = 14   # for WATCH → ACTIVE recovery check
    warning_sharpe: float = 0.0
    freeze_sharpe: float = -1.0
    recovery_sharpe: float = 0.5
    warning_dd_pct: float = 0.08
    freeze_dd_pct: float = 0.20
    max_weekly_delta_pct: float = 0.05    # ±5% per strategy per rebalance
    sharpe_tilt_strength: float = 0.5
    """How aggressively to tilt weights by Sharpe. 0 = pure baseline,
    1.0 = pure Sharpe weighting. 0.5 is a healthy middle that preserves
    diversification while rewarding winners."""


# ─── Outputs ───────────────────────────────────────────────────────────


@dataclass
class AllocationDecision:
    name: str
    state: StrategyState
    target_pct: float           # 0.0–1.0
    target_usd: float
    delta_pct_vs_prev: float    # change since last allocation
    metrics: StrategyMetrics
    reason: str = ""


@dataclass
class StrategyAllocation:
    """Convenience snapshot of the full allocation table."""

    timestamp: datetime
    portfolio_equity_usd: float
    decisions: list[AllocationDecision] = field(default_factory=list)
    total_active_pct: float = 0.0


# ─── Core ──────────────────────────────────────────────────────────────


class MetaAllocator:
    def __init__(
        self,
        registry: StrategyRegistry,
        performance: StrategyPerformance | None = None,
        config: AllocatorConfig | None = None,
    ):
        self.registry = registry
        self.perf = performance or StrategyPerformance()
        self.cfg = config or AllocatorConfig()

    # ── Public API ------------------------------------------------------

    def rebalance(self, portfolio_equity_usd: float) -> StrategyAllocation:
        """Run a full rebalance and persist results."""
        names = self.registry.list_names()
        primary = self.perf.metrics_bulk(names, window_days=self.cfg.primary_window_days)
        secondary = self.perf.metrics_bulk(names, window_days=self.cfg.secondary_window_days)

        # 1) Lifecycle transitions
        for name in names:
            self._maybe_transition(name, primary[name], secondary[name])

        # 2) Compute target weights
        prev_allocs = self.registry.latest_allocations()
        decisions: list[AllocationDecision] = []

        # Compute Sharpe-tilted weights for ACTIVE/WATCH only
        active_names = [n for n in names
                        if self.registry.get_state(n) in
                        (StrategyState.ACTIVE, StrategyState.WATCH)]

        # Baseline = each strategy's configured target_alloc_pct
        baselines = {n: self.registry.meta(n).target_alloc_pct for n in active_names}

        # Shrunk Sharpe (floored at 0 so losers don't get short)
        sharpes = {n: max(0.0, primary[n].shrunk_sharpe) for n in active_names}
        sharpe_sum = sum(sharpes.values())

        weights: dict[str, float] = {}
        for n in active_names:
            base = baselines[n]
            if sharpe_sum > 0:
                tilt = sharpes[n] / sharpe_sum
            else:
                # No useful Sharpe data — fall back to baseline
                tilt = base / max(sum(baselines.values()), 1e-9)
            blended = (1 - self.cfg.sharpe_tilt_strength) * base + \
                      self.cfg.sharpe_tilt_strength * tilt
            weights[n] = blended

        # WATCH = halve allocation
        for n in active_names:
            if self.registry.get_state(n) == StrategyState.WATCH:
                weights[n] *= 0.5

        # Floor / ceiling per-strategy
        for n in active_names:
            meta = self.registry.meta(n)
            weights[n] = max(meta.min_alloc_pct,
                             min(meta.max_alloc_pct, weights[n]))

        # 3) Apply max-weekly-delta clamp
        # On first allocation for a freshly-registered strategy there is no
        # prior weight; start from the configured baseline so the strategy
        # can begin trading at its intended size instead of ramping over weeks.
        for n in active_names:
            if n in prev_allocs:
                prev_pct = float(prev_allocs[n].get("target_pct") or 0.0)
            else:
                prev_pct = self.registry.meta(n).target_alloc_pct
            delta = weights[n] - prev_pct
            if delta > self.cfg.max_weekly_delta_pct:
                weights[n] = prev_pct + self.cfg.max_weekly_delta_pct
            elif delta < -self.cfg.max_weekly_delta_pct:
                weights[n] = prev_pct - self.cfg.max_weekly_delta_pct
            weights[n] = max(0.0, weights[n])

        # 4) Final normalization — sums to <= 100% no matter what.
        # The floor/ceiling and weekly-delta clamps can push the total above
        # 1.0 when many strategies bump up to their min_alloc_pct floor at
        # once. We renormalize at the end (after all per-strategy bumps) and
        # respect each strategy's max_alloc_pct ceiling on the way down.
        for _ in range(3):  # converge in a couple of passes
            total = sum(weights.values())
            if total <= 1.0 + 1e-9:
                break
            scale = 1.0 / total
            for n in list(weights.keys()):
                meta = self.registry.meta(n)
                weights[n] = max(meta.min_alloc_pct if total > 1.0 else 0.0,
                                  min(meta.max_alloc_pct, weights[n] * scale))
        # Hard cap: if floors still push above 100%, scale floors proportionally
        total = sum(weights.values())
        if total > 1.0:
            scale = 1.0 / total
            weights = {k: v * scale for k, v in weights.items()}

        # 4) Build decisions including FROZEN/RETIRED (target = 0)
        total_active_pct = 0.0
        for n in names:
            state = self.registry.get_state(n)
            if state in (StrategyState.FROZEN, StrategyState.RETIRED):
                pct = 0.0
            else:
                pct = weights.get(n, 0.0)

            prev_pct = float(prev_allocs.get(n, {}).get("target_pct") or 0.0)
            target_usd = pct * portfolio_equity_usd

            reason = self._reason_for(state, primary[n])
            decision = AllocationDecision(
                name=n,
                state=state,
                target_pct=pct,
                target_usd=target_usd,
                delta_pct_vs_prev=pct - prev_pct,
                metrics=primary[n],
                reason=reason,
            )
            decisions.append(decision)
            total_active_pct += pct
            self.registry.record_allocation(
                n, target_pct=pct, target_usd=target_usd, state=state,
                sharpe=primary[n].shrunk_sharpe,
                drawdown_pct=primary[n].drawdown_pct,
                reason=reason,
            )

        return StrategyAllocation(
            timestamp=datetime.now(UTC),
            portfolio_equity_usd=portfolio_equity_usd,
            decisions=decisions,
            total_active_pct=total_active_pct,
        )

    # ── Lifecycle transitions ------------------------------------------

    def _maybe_transition(self, name: str, primary: StrategyMetrics,
                           secondary: StrategyMetrics) -> None:
        state = self.registry.get_state(name)
        cfg = self.cfg

        # Manual states win: don't auto-resurrect FROZEN/RETIRED
        if state in (StrategyState.FROZEN, StrategyState.RETIRED):
            return

        # ACTIVE → FROZEN on severe underperformance
        if state == StrategyState.ACTIVE and primary.n_trades >= 10:
            if primary.shrunk_sharpe < cfg.freeze_sharpe or \
               primary.drawdown_pct >= cfg.freeze_dd_pct:
                self.registry.set_state(
                    name, StrategyState.FROZEN,
                    f"auto-freeze: 60d Sharpe={primary.shrunk_sharpe:.2f}, "
                    f"DD={primary.drawdown_pct * 100:.1f}%"
                )
                return

        # ACTIVE → WATCH on mild underperformance
        if state == StrategyState.ACTIVE and primary.n_trades >= 5:
            if primary.shrunk_sharpe < cfg.warning_sharpe or \
               primary.drawdown_pct >= cfg.warning_dd_pct:
                self.registry.set_state(
                    name, StrategyState.WATCH,
                    f"auto-watch: 60d Sharpe={primary.shrunk_sharpe:.2f}, "
                    f"DD={primary.drawdown_pct * 100:.1f}%"
                )
                return

        # WATCH → ACTIVE on recovery (use shorter 14d window)
        if state == StrategyState.WATCH and secondary.n_trades >= 5:
            if secondary.shrunk_sharpe >= cfg.recovery_sharpe and \
               secondary.drawdown_pct < cfg.warning_dd_pct:
                self.registry.set_state(
                    name, StrategyState.ACTIVE,
                    f"auto-recover: 14d Sharpe={secondary.shrunk_sharpe:.2f}"
                )
                return

        # WATCH → FROZEN if continued underperformance
        if state == StrategyState.WATCH and primary.n_trades >= 15:
            if primary.shrunk_sharpe < cfg.freeze_sharpe or \
               primary.drawdown_pct >= cfg.freeze_dd_pct:
                self.registry.set_state(
                    name, StrategyState.FROZEN,
                    f"auto-freeze (after watch): 60d Sharpe={primary.shrunk_sharpe:.2f}"
                )

    # ── Reason strings -------------------------------------------------

    def _reason_for(self, state: StrategyState, m: StrategyMetrics) -> str:
        if state == StrategyState.FROZEN:
            return "FROZEN"
        if state == StrategyState.RETIRED:
            return "RETIRED"
        if m.n_trades < 5:
            return f"baseline (only {m.n_trades} trades)"
        return (f"Sharpe={m.shrunk_sharpe:.2f}, win={m.win_rate * 100:.0f}%, "
                f"DD={m.drawdown_pct * 100:.1f}%")
