"""Meta-Allocator package.

The "fund of strategies" layer. Tracks each strategy as an independent pod;
allocates capital based on rolling risk-adjusted performance; freezes
underperformers; promotes winners. User can manually override every state.
"""
from .allocator import AllocationDecision, MetaAllocator, StrategyAllocation
from .lifecycle import StrategyRegistry, StrategyState
from .metrics import StrategyMetrics, StrategyPerformance

__all__ = [
    "AllocationDecision",
    "MetaAllocator",
    "StrategyAllocation",
    "StrategyMetrics",
    "StrategyPerformance",
    "StrategyRegistry",
    "StrategyState",
]
