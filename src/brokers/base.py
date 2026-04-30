"""Broker adapter base class and shared dataclasses.

Every supported venue implements the same `BrokerAdapter` ABC. Strategies and
the meta-allocator depend only on this interface — never on a concrete venue.

Naming convention for normalized symbols:
    "<asset_class>:<venue_native_id>"
    e.g.  "crypto:BTC-USD"        (Coinbase spot)
          "crypto_perp:BIP-PERP"  (Coinbase BTC perpetual)
          "future:GOL-27MAY26-CDE"(Coinbase commodity future)
          "equity:SPY"            (Alpaca)
          "kalshi:PRES-2028-DEM"  (Kalshi event contract)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class BrokerError(Exception):
    """Raised by any adapter on broker-side failures (auth, rate limit, etc)."""


class AssetClass(str, Enum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    CRYPTO_FUTURE = "crypto_future"
    COMMODITY_FUTURE = "commodity_future"
    EQUITY_INDEX_FUTURE = "equity_index_future"
    EQUITY = "equity"
    ETF = "etf"
    PREDICTION = "prediction"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass
class Account:
    venue: str
    cash_usd: float
    buying_power_usd: float
    equity_usd: float
    is_paper: bool = False
    raw: Dict = field(default_factory=dict)


@dataclass
class Position:
    venue: str
    symbol: str
    asset_class: AssetClass
    quantity: float
    avg_entry_price: float
    market_price: float
    unrealized_pnl_usd: float
    raw: Dict = field(default_factory=dict)


@dataclass
class Quote:
    venue: str
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    timestamp: datetime


@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    venue: str
    order_id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    notional_usd: Optional[float]
    limit_price: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    raw: Dict = field(default_factory=dict)


class BrokerAdapter(ABC):
    """Abstract base class — implement once per venue."""

    venue: str          # short identifier, e.g. "coinbase", "alpaca", "kalshi"
    is_paper: bool      # paper/sandbox vs live

    # ── Account -----------------------------------------------------------
    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> List[Position]: ...

    # ── Market data -------------------------------------------------------
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> List[Candle]:
        """Recent candles, oldest-first. `granularity` follows Coinbase
        naming (`ONE_MINUTE`, `FIVE_MINUTE`, `ONE_HOUR`, `ONE_DAY`, …).
        Adapters translate to native units."""

    # ── Orders ------------------------------------------------------------
    @abstractmethod
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
        """Submit an order. Provide either `quantity` (units of base asset) OR
        `notional_usd` (dollar amount); adapter picks whichever the venue
        prefers. `limit_price` required for LIMIT orders."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    # ── Capability discovery (used by strategies + meta-allocator) -------
    @abstractmethod
    def list_supported_asset_classes(self) -> List[AssetClass]: ...

    @abstractmethod
    def list_tradable_symbols(
        self, asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        """Return venue-native symbols. Cached by adapter where appropriate."""

    # ── Diagnostics -------------------------------------------------------
    def healthcheck(self) -> Dict:
        """Lightweight check used by the strategy-pod dashboard. Adapters
        may override; default just calls get_account()."""
        try:
            acct = self.get_account()
            return {
                "venue": self.venue,
                "ok": True,
                "is_paper": acct.is_paper,
                "cash_usd": acct.cash_usd,
                "equity_usd": acct.equity_usd,
            }
        except Exception as exc:
            return {"venue": self.venue, "ok": False, "error": str(exc)}
