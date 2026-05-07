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


class BrokerError(Exception):
    """Raised by any adapter on broker-side failures (auth, rate limit, etc)."""


class BrokerCapability(str, Enum):
    """What a broker actually supports.

    Audit-fix 2026-05-07: the orchestrator was deciding whether to call
    `get_open_orders` and `cancel_stale_orders` via `hasattr(...)`.
    That silently treated venues whose ABC defaults return [] / 0 as
    if they had real implementations — Coinbase and Kalshi got no
    wash-trade protection at all. Now adapters declare capabilities
    explicitly; the orchestrator branches on the set.
    """
    GET_OPEN_ORDERS = "get_open_orders"
    CANCEL_STALE_ORDERS = "cancel_stale_orders"
    LIMIT_ORDERS = "limit_orders"
    SHORT_SELLING = "short_selling"


_REDACT_PATTERNS = (
    "APCA-API-SECRET-KEY",
    "APCA-API-KEY-ID",
    "Authorization",
    "Bearer ",
    "KALSHI-ACCESS-KEY",
    "KALSHI-ACCESS-SIGNATURE",
)


def redact_response_text(text: str, max_len: int = 200) -> str:
    """Best-effort scrub of broker error response bodies before they
    end up in BrokerError messages / journalctl. We can't anticipate
    every shape (broker errors sometimes echo headers) but the
    common-case credential leaks (Bearer tokens, Alpaca/Kalshi keys)
    are masked out."""
    if not text:
        return ""
    s = text[:max_len]
    for pat in _REDACT_PATTERNS:
        if pat.lower() in s.lower():
            s = "<redacted: response contained credential-shaped header>"
            break
    return s


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
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    venue: str
    symbol: str
    asset_class: AssetClass
    quantity: float
    avg_entry_price: float
    market_price: float
    unrealized_pnl_usd: float
    raw: dict = field(default_factory=dict)


@dataclass
class Quote:
    venue: str
    symbol: str
    bid: float | None
    ask: float | None
    last: float | None
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
    notional_usd: float | None
    limit_price: float | None
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_avg_price: float | None = None
    submitted_at: datetime | None = None
    raw: dict = field(default_factory=dict)


class BrokerAdapter(ABC):
    """Abstract base class — implement once per venue."""

    venue: str          # short identifier, e.g. "coinbase", "alpaca", "kalshi"
    is_paper: bool      # paper/sandbox vs live

    # Declared capabilities — what the venue actually supports.
    # Subclasses override this set rather than relying on hasattr() at
    # call sites. Default empty == "only the abstract methods work."
    # See BrokerCapability for the full list.
    capabilities: frozenset[BrokerCapability] = frozenset()

    # Per-instance candle cache. Multiple strategies often request the
    # same (symbol, granularity, num_candles) within one orchestrator
    # cycle (e.g. tsmom_etf and risk_parity_etf both want SPY daily
    # bars). Cache TTL is short so we stay fresh across cycles but
    # de-dupe within a cycle.
    _CANDLE_CACHE_TTL_SECONDS = 60.0

    def _get_cached_candles(
        self, symbol: str, granularity: str, num_candles: int
    ) -> list[Candle] | None:
        cache = getattr(self, "_candle_cache", None)
        if cache is None:
            return None
        import time as _t
        entry = cache.get((symbol, granularity, num_candles))
        if entry and entry[0] > _t.time():
            return entry[1]
        return None

    def _put_cached_candles(
        self, symbol: str, granularity: str, num_candles: int,
        candles: list[Candle],
    ) -> None:
        if not hasattr(self, "_candle_cache"):
            self._candle_cache = {}
        import time as _t
        self._candle_cache[(symbol, granularity, num_candles)] = (
            _t.time() + self._CANDLE_CACHE_TTL_SECONDS, candles,
        )

    # ── Account -----------------------------------------------------------
    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    # ── Market data -------------------------------------------------------
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> list[Candle]:
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
        quantity: float | None = None,
        notional_usd: float | None = None,
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit an order. Provide either `quantity` (units of base asset) OR
        `notional_usd` (dollar amount); adapter picks whichever the venue
        prefers. `limit_price` required for LIMIT orders."""

    @abstractmethod
    def get_order(self, order_id: str) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    # Pending-order discovery and stale-order cleanup. Default to a
    # safe no-op so adapters that don't expose these endpoints
    # (Coinbase Spot, Kalshi) remain importable. Strategies that
    # want a venue's wash-trade guard active need that venue to
    # override these — without override, the orchestrator treats
    # the venue as having no pending orders (the wash-trade guard
    # essentially no-ops) and never auto-cancels. Documented gap.
    def get_open_orders(self) -> list[Order]:
        return []

    def cancel_stale_orders(self, max_age_seconds: int) -> int:
        return 0

    # ── Capability discovery (used by strategies + meta-allocator) -------
    @abstractmethod
    def list_supported_asset_classes(self) -> list[AssetClass]: ...

    @abstractmethod
    def list_tradable_symbols(
        self, asset_class: AssetClass | None = None
    ) -> list[str]:
        """Return venue-native symbols. Cached by adapter where appropriate."""

    # ── Diagnostics -------------------------------------------------------
    def healthcheck(self) -> dict:
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
