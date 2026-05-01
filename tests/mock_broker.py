"""Mock BrokerAdapter for unit tests.

Simulates an in-memory broker with configurable account, positions,
and order behavior. The orchestrator can be wired up to one or more
of these without touching real APIs.

Ergonomics:

    broker = MockBroker(
        venue="alpaca",
        cash_usd=100_000,
        positions=[
            MockPosition("SPY", qty=10, entry=720, mark=725),
        ],
    )
    broker.fill_next = "FILLED"     # next place_order returns FILLED
    order = broker.place_order(...)

    # Inject a server-side rejection for the next call:
    broker.reject_next = "wash trade"
    with pytest.raises(BrokerError):
        broker.place_order(...)

The mock honours the BrokerAdapter ABC enough to satisfy
`hasattr(adapter, "method_name")` checks. Methods we don't need yet
raise NotImplementedError — extend as needed.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from brokers.base import (
    Account,
    AssetClass,
    BrokerError,
    Candle,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)


@dataclass
class MockPosition:
    """Compact constructor for tests: MockPosition("SPY", 10, 720, 725)."""
    symbol: str
    qty: float
    entry: float
    mark: float
    asset_class: AssetClass = AssetClass.ETF

    def to_position(self, venue: str) -> Position:
        return Position(
            venue=venue,
            symbol=self.symbol,
            asset_class=self.asset_class,
            quantity=self.qty,
            avg_entry_price=self.entry,
            market_price=self.mark,
            unrealized_pnl_usd=(self.mark - self.entry) * self.qty,
            raw={"qty_available_parsed": self.qty},
        )


class MockBroker:
    """In-memory broker. Subset of BrokerAdapter — implements the
    methods orchestrator + dashboard actually call."""

    def __init__(
        self,
        venue: str = "mock",
        cash_usd: float = 100_000.0,
        equity_usd: Optional[float] = None,
        buying_power_usd: Optional[float] = None,
        positions: Optional[List[MockPosition]] = None,
        is_paper: bool = True,
    ):
        self.venue = venue
        self.is_paper = is_paper
        self._cash = cash_usd
        self._equity = equity_usd if equity_usd is not None else cash_usd
        self._buying_power = (
            buying_power_usd if buying_power_usd is not None else cash_usd
        )
        self._positions: List[MockPosition] = list(positions or [])
        self._orders: Dict[str, Order] = {}
        self._order_seq = 0

        # Test injection points
        self.fill_next: Optional[str] = None      # "FILLED" | "PENDING" | …
        self.reject_next: Optional[str] = None    # raise BrokerError on next place
        self.partial_fill_qty: Optional[float] = None
        self.fill_price: Optional[float] = None   # override fill price

        # Bookkeeping for assertions
        self.placed_orders: List[Order] = []
        self.cancelled_orders: List[str] = []
        self.get_order_calls: List[str] = []

    # ── BrokerAdapter ABI ────────────────────────────────────────────

    def get_account(self) -> Account:
        return Account(
            venue=self.venue,
            cash_usd=self._cash,
            buying_power_usd=self._buying_power,
            equity_usd=self._equity,
            is_paper=self.is_paper,
        )

    def get_positions(self) -> List[Position]:
        return [p.to_position(self.venue) for p in self._positions]

    def get_quote(self, symbol: str) -> Quote:
        # Use the position mark if we have one, else 100.
        for p in self._positions:
            if p.symbol == symbol:
                return Quote(
                    venue=self.venue, symbol=symbol,
                    bid=p.mark - 0.01, ask=p.mark + 0.01,
                    last=p.mark, timestamp=datetime.now(timezone.utc),
                )
        return Quote(
            venue=self.venue, symbol=symbol,
            bid=99.99, ask=100.01, last=100.0,
            timestamp=datetime.now(timezone.utc),
        )

    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> List[Candle]:
        # Flat 100-bar series at $100 — strategies that depend on real
        # candles should be tested with synthesized fixtures, not the
        # mock broker.
        now = datetime.now(timezone.utc)
        return [
            Candle(timestamp=now, open=100, high=100, low=100,
                   close=100, volume=1000)
            for _ in range(num_candles)
        ]

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: Optional[float] = None,
        notional_usd: Optional[float] = None,
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Order:
        if self.reject_next:
            msg = self.reject_next
            self.reject_next = None
            raise BrokerError(msg)

        self._order_seq += 1
        oid = f"mock-{self._order_seq}-{uuid.uuid4().hex[:6]}"

        # Resolve fill price + quantity
        px = self.fill_price if self.fill_price is not None else (
            limit_price or self._mark_for(symbol) or 100.0
        )
        qty = quantity if quantity is not None else (
            (notional_usd / px) if notional_usd and px else 0.0
        )

        # Status defaults to FILLED so tests can assert pnl on a happy path.
        # Set fill_next="PENDING" to simulate the unfilled case.
        status_str = self.fill_next or "FILLED"
        if status_str == "FILLED":
            status = OrderStatus.FILLED
            filled_qty = qty
            filled_px = px
        elif status_str == "PARTIAL":
            status = OrderStatus.PARTIALLY_FILLED
            filled_qty = self.partial_fill_qty or (qty / 2)
            filled_px = px
        elif status_str == "PENDING":
            status = OrderStatus.PENDING
            filled_qty = 0.0
            filled_px = None
        else:
            status = OrderStatus.PENDING
            filled_qty = 0.0
            filled_px = None
        self.fill_next = None

        order = Order(
            venue=self.venue, order_id=oid, symbol=symbol,
            side=side, type=type, quantity=qty,
            notional_usd=notional_usd, limit_price=limit_price,
            status=status,
            filled_quantity=filled_qty, filled_avg_price=filled_px,
            submitted_at=datetime.now(timezone.utc),
        )
        self._orders[oid] = order
        self.placed_orders.append(order)

        # Apply fill to inventory if FILLED
        if status == OrderStatus.FILLED:
            self._apply_fill(symbol, side, filled_qty, filled_px)
        return order

    def get_order(self, order_id: str) -> Order:
        self.get_order_calls.append(order_id)
        if order_id not in self._orders:
            raise BrokerError(f"Order {order_id} not found")
        return self._orders[order_id]

    def get_open_orders(self) -> List[Order]:
        return [
            o for o in self._orders.values()
            if o.status in (OrderStatus.PENDING, OrderStatus.OPEN,
                            OrderStatus.PARTIALLY_FILLED)
        ]

    def cancel_order(self, order_id: str) -> None:
        self.cancelled_orders.append(order_id)
        if order_id in self._orders:
            o = self._orders[order_id]
            self._orders[order_id] = Order(
                **{**o.__dict__, "status": OrderStatus.CANCELED},
            )

    def cancel_stale_orders(self, max_age_seconds: int = 1800) -> int:
        return 0

    # ── Test helpers ─────────────────────────────────────────────────

    def fill_order(self, order_id: str, price: float,
                    quantity: Optional[float] = None) -> None:
        """Manually transition an order from PENDING to FILLED for the
        next get_order() call. Use this to simulate the broker
        eventually filling an order between cycles."""
        if order_id not in self._orders:
            raise KeyError(order_id)
        o = self._orders[order_id]
        qty = quantity if quantity is not None else o.quantity
        self._orders[order_id] = Order(
            venue=o.venue, order_id=o.order_id, symbol=o.symbol,
            side=o.side, type=o.type, quantity=o.quantity,
            notional_usd=o.notional_usd, limit_price=o.limit_price,
            status=OrderStatus.FILLED,
            filled_quantity=qty, filled_avg_price=price,
            submitted_at=o.submitted_at,
        )
        self._apply_fill(o.symbol, o.side, qty, price)

    def _apply_fill(self, symbol: str, side: OrderSide,
                    qty: float, price: float) -> None:
        """Update internal cash + position book on a fill."""
        existing = next((p for p in self._positions if p.symbol == symbol), None)
        if side == OrderSide.BUY:
            self._cash -= qty * price
            if existing:
                # Weighted-average entry
                new_qty = existing.qty + qty
                new_entry = (
                    (existing.qty * existing.entry + qty * price) / new_qty
                    if new_qty > 0 else price
                )
                existing.qty = new_qty
                existing.entry = new_entry
                existing.mark = price
            else:
                self._positions.append(
                    MockPosition(symbol=symbol, qty=qty, entry=price, mark=price)
                )
        else:  # SELL
            self._cash += qty * price
            if existing:
                existing.qty -= qty
                existing.mark = price
                if existing.qty <= 1e-9:
                    self._positions.remove(existing)

    def _mark_for(self, symbol: str) -> Optional[float]:
        for p in self._positions:
            if p.symbol == symbol:
                return p.mark
        return None
