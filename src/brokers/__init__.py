"""Broker adapter package.

A `BrokerAdapter` is a uniform interface over a single venue. Strategies and
the meta-allocator deal with adapters — they don't know whether the underlying
venue is Coinbase, Alpaca, or Kalshi.
"""
from .base import (
    Account,
    AssetClass,
    BrokerAdapter,
    BrokerError,
    Candle,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)

__all__ = [
    "Account",
    "AssetClass",
    "BrokerAdapter",
    "BrokerError",
    "Candle",
    "Order",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
]
