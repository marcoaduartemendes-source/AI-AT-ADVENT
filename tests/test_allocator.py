"""Unit tests for MetaAllocator.

The 122.5%-total-allocation bug shipped in production. These tests
encode the invariants that should never break again:

  1. Total weights NEVER exceed 100% (was 122.5%).
  2. min_alloc_pct floor is respected for ACTIVE strategies (unless
     applying it would push the total over 100% — then it's relaxed
     proportionally).
  3. max_alloc_pct ceiling is always respected.
  4. FROZEN/RETIRED strategies always get target_pct=0.
  5. Per-strategy weekly delta is bounded by max_weekly_delta_pct.
  6. With no Sharpe data, allocator falls back to baseline weights.
"""
from __future__ import annotations

from unittest.mock import MagicMock



def _make_registry(tmp_path, strategies):
    """Build an in-memory registry with the given strategies.
    `strategies` is a list of (name, target_pct, min_pct, max_pct)."""
    from allocator.lifecycle import (
        StrategyMeta, StrategyRegistry
    )
    db = tmp_path / "alloc.db"
    reg = StrategyRegistry(str(db))
    for name, tgt, mn, mx in strategies:
        reg.register(StrategyMeta(
            name=name, asset_classes=["ETF"], venue="alpaca",
            target_alloc_pct=tgt, min_alloc_pct=mn, max_alloc_pct=mx,
        ))
    return reg


def _stub_perf(metrics_by_name):
    """Stub StrategyPerformance with deterministic shrunk_sharpe values."""
    from allocator.metrics import StrategyMetrics
    perf = MagicMock()

    def metrics_bulk(names, window_days=60):
        out = {}
        for n in names:
            m = metrics_by_name.get(n, {})
            sh = m.get("shrunk_sharpe", 1.0)
            out[n] = StrategyMetrics(
                name=n,
                window_days=window_days,
                n_trades=m.get("n_trades", 30),
                n_wins=m.get("wins", 18),
                n_losses=m.get("losses", 12),
                win_rate=m.get("win_rate", 0.6),
                total_pnl_usd=m.get("pnl", 100.0),
                mean_pnl_usd=m.get("pnl", 100.0) / max(m.get("n_trades", 30), 1),
                std_pnl_usd=10.0,
                raw_sharpe=m.get("sharpe", sh),
                shrunk_sharpe=sh,
                drawdown_usd=m.get("dd_usd", 50.0),
                drawdown_pct=m.get("dd", 0.05),
            )
        return out

    perf.metrics_bulk = metrics_bulk
    return perf


class TestAllocatorWeightsDoNotExceed100Percent:
    """The headline invariant. Floors that sum > 100% must be capped."""

    def test_six_strategies_with_high_floors_does_not_blow_past_100pct(self, tmp_path):
        """Reproduces the 122.5% bug: 6 strategies × 20% floor = 120%
        floor sum. After Sharpe tilting + delta clamping, the total
        used to come out to 122.5%. Now it must come out ≤ 100%."""
        from allocator.allocator import MetaAllocator, AllocatorConfig

        strategies = [
            (f"s{i}", 0.166, 0.20, 0.30) for i in range(6)
        ]
        reg = _make_registry(tmp_path, strategies)
        perf = _stub_perf({n: {"shrunk_sharpe": 1.0} for n, *_ in strategies})
        alloc = MetaAllocator(reg, perf, AllocatorConfig())

        result = alloc.rebalance(portfolio_equity_usd=100_000)
        total = sum(d.target_pct for d in result.decisions
                    if d.state.value in ("ACTIVE", "WATCH"))
        assert total <= 1.0 + 1e-9, (
            f"Total weight {total*100:.2f}% exceeds 100%. "
            "122.5% allocator bug regression."
        )

    def test_total_weight_within_tolerance_for_normal_config(self, tmp_path):
        from allocator.allocator import MetaAllocator, AllocatorConfig

        strategies = [
            ("crypto", 0.30, 0.05, 0.40),
            ("etf", 0.40, 0.10, 0.50),
            ("kalshi", 0.20, 0.02, 0.30),
            ("commodity", 0.10, 0.05, 0.20),
        ]
        reg = _make_registry(tmp_path, strategies)
        perf = _stub_perf({})
        alloc = MetaAllocator(reg, perf, AllocatorConfig())
        result = alloc.rebalance(portfolio_equity_usd=100_000)
        total = sum(d.target_pct for d in result.decisions
                    if d.state.value in ("ACTIVE", "WATCH"))
        assert 0.5 <= total <= 1.0 + 1e-9


class TestAllocatorMaxAllocCeiling:
    def test_max_alloc_ceiling_enforced(self, tmp_path):
        """Even with high Sharpe, no strategy should exceed max_alloc_pct."""
        from allocator.allocator import MetaAllocator, AllocatorConfig

        strategies = [
            ("hot", 0.20, 0.05, 0.30),     # max=30%
            ("cold", 0.20, 0.05, 0.30),
            ("dud", 0.20, 0.05, 0.30),
        ]
        reg = _make_registry(tmp_path, strategies)
        # "hot" has very high Sharpe; should still be capped at 30%
        perf = _stub_perf({
            "hot":  {"shrunk_sharpe": 5.0},
            "cold": {"shrunk_sharpe": 0.1},
            "dud":  {"shrunk_sharpe": 0.1},
        })
        alloc = MetaAllocator(reg, perf, AllocatorConfig(
            sharpe_tilt_strength=1.0,    # max tilt
            max_weekly_delta_pct=1.0,    # disable delta clamp for this test
        ))
        result = alloc.rebalance(100_000)
        hot = next(d for d in result.decisions if d.name == "hot")
        assert hot.target_pct <= 0.30 + 1e-9


class TestAllocatorFrozenStrategiesGetZero:
    def test_frozen_strategy_target_pct_is_zero(self, tmp_path):
        from allocator.allocator import MetaAllocator, AllocatorConfig
        from allocator.lifecycle import StrategyState

        strategies = [
            ("active", 0.50, 0.10, 0.60),
            ("frozen", 0.50, 0.10, 0.60),
        ]
        reg = _make_registry(tmp_path, strategies)
        # Manually freeze
        reg.set_state("frozen", StrategyState.FROZEN, reason="manual")
        perf = _stub_perf({})
        alloc = MetaAllocator(reg, perf, AllocatorConfig())
        result = alloc.rebalance(100_000)
        frozen = next(d for d in result.decisions if d.name == "frozen")
        assert frozen.target_pct == 0.0
        assert frozen.target_usd == 0.0


class TestAllocatorWeeklyDelta:
    def test_weekly_delta_clamp_limits_swing(self, tmp_path):
        """If a strategy was previously at 5% and its new target would
        be 30%, the clamp must limit it to 5% + max_weekly_delta_pct."""
        from allocator.allocator import MetaAllocator, AllocatorConfig

        strategies = [
            ("a", 0.50, 0.05, 0.50),
            ("b", 0.50, 0.05, 0.50),
        ]
        reg = _make_registry(tmp_path, strategies)
        # Seed a prior allocation at 5%
        from allocator.lifecycle import StrategyState
        reg.record_allocation(
            "a", target_pct=0.05, target_usd=5000,
            state=StrategyState.ACTIVE,
        )
        reg.record_allocation(
            "b", target_pct=0.95, target_usd=95000,
            state=StrategyState.ACTIVE,
        )
        perf = _stub_perf({
            "a": {"shrunk_sharpe": 5.0},
            "b": {"shrunk_sharpe": 0.0},
        })
        alloc = MetaAllocator(reg, perf, AllocatorConfig(
            sharpe_tilt_strength=1.0,
            max_weekly_delta_pct=0.05,
        ))
        result = alloc.rebalance(100_000)
        a = next(d for d in result.decisions if d.name == "a")
        # Was 5%; max swing 5% → cannot exceed 10% this rebalance.
        assert a.target_pct <= 0.10 + 1e-9


class TestAllocatorBaselineFallback:
    def test_no_sharpe_data_falls_back_to_baseline_proportions(self, tmp_path):
        """When all strategies have shrunk_sharpe=0, the allocator
        must fall back to baseline target_alloc_pct proportions
        rather than dividing by zero."""
        from allocator.allocator import MetaAllocator, AllocatorConfig

        strategies = [
            ("a", 0.30, 0.05, 0.40),
            ("b", 0.50, 0.05, 0.60),
            ("c", 0.20, 0.05, 0.30),
        ]
        reg = _make_registry(tmp_path, strategies)
        perf = _stub_perf({
            "a": {"shrunk_sharpe": 0.0},
            "b": {"shrunk_sharpe": 0.0},
            "c": {"shrunk_sharpe": 0.0},
        })
        alloc = MetaAllocator(reg, perf, AllocatorConfig(
            sharpe_tilt_strength=1.0,
            max_weekly_delta_pct=1.0,
        ))
        result = alloc.rebalance(100_000)
        # No NaN / no exception — and weights are roughly proportional
        # to the baselines (ratio ~ 0.30 : 0.50 : 0.20 = 3:5:2).
        weights = {d.name: d.target_pct for d in result.decisions}
        assert all(w > 0 for w in weights.values())
        assert weights["b"] > weights["a"] > weights["c"]
