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


# ─── Back-compat dict-shim base class ─────────────────────────────────


class _DictCompat:
    """Mixin that makes a dataclass behave like a dict for read access.

    Old strategy code uses `pos.get("quantity") or 0` and `pos["entry_time"]`
    everywhere. Migrating 24 strategies to attribute access in one go is
    risky; this shim lets both forms work indefinitely so the migration
    is per-file.

    Read-only — `pos["quantity"] = 5` deliberately raises.
    """

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class PositionView(_DictCompat):
    """Read-only view of a broker position passed to strategies.

    Replaces the stringly-typed `dict[str, dict]` of the old
    `StrategyContext.open_positions` map. Strategy code that uses
    `pos.get("quantity") or 0` continues to work via the _DictCompat
    shim; new code can use `pos.quantity` for type-checked attribute
    access.
    """

    venue: str
    symbol: str
    quantity: float
    avg_entry_price: float
    market_price: float
    unrealized_pnl_usd: float
    # Optional fields some venues populate, others don't.
    entry_time: str | None = None
    asset_class: str | None = None


@dataclass
class PendingExposure(_DictCompat):
    """Read-only summary of pending orders for one symbol on one venue.

    Strategies subtract `buy_notional_usd` from buying intent and
    `sell_qty` from selling intent so they don't double-fire across
    cycles before fills land. The wash-trade guard reads
    `n_buy_pending` / `n_sell_pending` to decide opposite-side rejection.
    """

    buy_notional_usd: float = 0.0
    sell_qty: float = 0.0
    n_pending: int = 0
    n_buy_pending: int = 0
    n_sell_pending: int = 0


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
