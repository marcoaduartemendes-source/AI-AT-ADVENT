"""Strategy base class + supporting types.

A `Strategy` is a pure-Python class that consumes market data + scout
signals and emits `TradeProposal` objects. It does NOT execute orders
directly — proposals flow through risk gating and the broker adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from brokers.base import BrokerAdapter, OrderSide, OrderType


@dataclass
class TradeProposal:
    """What a strategy proposes; what risk gates and what execution acts on."""

    strategy: str
    venue: str                          # "coinbase" | "alpaca" | "kalshi"
    symbol: str
    side: OrderSide
    order_type: OrderType
    notional_usd: Optional[float] = None
    quantity: Optional[float] = None
    limit_price: Optional[float] = None
    confidence: float = 0.6             # 0.0–1.0
    expected_return_pct: Optional[float] = None
    reason: str = ""                     # logged + dashboard
    is_closing: bool = False             # True when reducing existing position
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyContext:
    """Read-only snapshot passed to Strategy.compute() each cycle."""

    timestamp: datetime
    portfolio_equity_usd: float
    target_alloc_pct: float              # current target from MetaAllocator
    target_alloc_usd: float
    risk_multiplier: float               # informational; risk layer applies sizing
    open_positions: Dict                 # {symbol: position_dict}
    scout_signals: Dict                  # {scout_name: signal_payload}
    extra: Dict = field(default_factory=dict)


class Strategy(ABC):
    """Implement once per trading idea."""

    name: str                            # e.g. "crypto_funding_carry"
    venue: str                           # primary broker the strategy trades on

    def __init__(self, broker: BrokerAdapter):
        self.broker = broker

    @abstractmethod
    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        """Return proposals for this cycle. Empty list = no signals."""

    def on_fill(self, proposal: TradeProposal, fill: Dict) -> None:
        """Hook after an order fills. Default: no-op."""

    def on_emergency_close(self, ctx: StrategyContext) -> List[TradeProposal]:
        """Called when KILL switch fires. Default: close every open position
        owned by this strategy at market."""
        proposals: List[TradeProposal] = []
        for symbol, pos in ctx.open_positions.items():
            qty = pos.get("quantity", 0)
            if qty <= 0:
                continue
            proposals.append(TradeProposal(
                strategy=self.name,
                venue=self.venue,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=qty,
                confidence=1.0,
                reason="emergency_close (KILL switch)",
                is_closing=True,
            ))
        return proposals
