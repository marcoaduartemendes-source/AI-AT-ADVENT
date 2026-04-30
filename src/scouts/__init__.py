"""Scout agents — research layer.

Scouts watch external data sources (funding rates, EIA inventories, FOMC
calendar, prediction-market spreads, news flow) and emit structured
*signals* to a shared SignalBus. Strategies consume those signals via
StrategyContext.scout_signals.

Architectural rule: scouts NEVER place orders. They produce data; the
deterministic strategy logic decides whether to act.

This module exports:
    SignalBus       — the SQLite-backed pub/sub used by all scouts/strategies
    ScoutAgent      — base class implemented once per asset class
    ScoutSignal     — payload dataclass written to the bus
"""
from .base import ScoutAgent, ScoutSignal
from .signal_bus import SignalBus

__all__ = ["ScoutAgent", "ScoutSignal", "SignalBus"]
