"""Strategy base class + supporting types.

A `Strategy` is a pure-Python class that consumes market data + scout
signals and emits `TradeProposal` objects. It does NOT execute orders
directly — proposals flow through risk gating and the broker adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from brokers.base import BrokerAdapter, OrderSide, OrderType


@dataclass
class TradeProposal:
    """What a strategy proposes; what risk gates and what execution acts on."""

    strategy: str
    venue: str                          # "coinbase" | "alpaca" | "kalshi"
    symbol: str
    side: OrderSide
    order_type: OrderType
    notional_usd: float | None = None
    quantity: float | None = None
    limit_price: float | None = None
    confidence: float = 0.6             # 0.0–1.0
    expected_return_pct: float | None = None
    reason: str = ""                     # logged + dashboard
    is_closing: bool = False             # True when reducing existing position
    metadata: dict = field(default_factory=dict)


@dataclass
class StrategyContext:
    """Read-only snapshot passed to Strategy.compute() each cycle."""

    timestamp: datetime
    portfolio_equity_usd: float
    target_alloc_pct: float              # current target from MetaAllocator
    target_alloc_usd: float
    risk_multiplier: float               # informational; risk layer applies sizing
    open_positions: dict                 # {symbol: position_dict}
    scout_signals: dict                  # {scout_name: signal_payload}
    # Pending broker-side orders that haven't filled yet — strategies must
    # subtract this from their buying intent or they over-trade across
    # cycles before fills land.
    pending_orders: dict = field(default_factory=dict)
    # ^^ {symbol: {"buy_notional_usd", "sell_qty"}}
    extra: dict = field(default_factory=dict)


class Strategy(ABC):
    """Implement once per trading idea."""

    name: str                            # e.g. "crypto_funding_carry"
    venue: str                           # primary broker the strategy trades on

    def __init__(self, broker: BrokerAdapter):
        self.broker = broker

    @abstractmethod
    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        """Return proposals for this cycle. Empty list = no signals."""

    def on_fill(self, proposal: TradeProposal, fill: dict) -> None:  # noqa: B027
        """Hook after an order fills. Default: no-op (subclasses override)."""

    def on_emergency_close(self, ctx: StrategyContext) -> list[TradeProposal]:
        """Called when KILL switch fires. Default: close every open position
        owned by this strategy at market."""
        proposals: list[TradeProposal] = []
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
