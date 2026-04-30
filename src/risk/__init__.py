"""Risk layer.

Sits between strategies and execution. Every order from any strategy must
pass `RiskManager.check_order()` before it reaches a broker. The manager
also owns the **dynamic risk multiplier** — a single 0.5x-2.0x knob that
scales sizing across the whole portfolio, automatically adjusted by
drawdown and volatility regime.
"""
from .manager import RiskDecision, RiskManager, RiskState
from .multiplier import DynamicRiskMultiplier, MultiplierState
from .policies import KillSwitchState, RiskConfig

__all__ = [
    "DynamicRiskMultiplier",
    "KillSwitchState",
    "MultiplierState",
    "RiskConfig",
    "RiskDecision",
    "RiskManager",
    "RiskState",
]
