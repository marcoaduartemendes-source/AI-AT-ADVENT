"""Strategy engine.

Defines the `Strategy` ABC every concrete strategy implements, plus the
`Orchestrator` that drives a single trading cycle:

    refresh risk state → rebalance allocator (weekly) → for each ACTIVE
    strategy: collect proposals → gate via risk → execute via broker →
    persist trades.

Concrete strategies (carry, risk parity, kalshi calibration, …) live in
src/strategies/ and import from here. The new `main_trading.py` will use
the Orchestrator instead of the legacy per-strategy loop.
"""
from .base import Strategy, StrategyContext, TradeProposal
from .orchestrator import Orchestrator, OrchestratorConfig

__all__ = [
    "Orchestrator",
    "OrchestratorConfig",
    "Strategy",
    "StrategyContext",
    "TradeProposal",
]
