"""Coinbase adapter — wraps the existing CoinbaseClient.

The lower-level `CoinbaseClient` (in src/trading/coinbase_client.py) was built
before we introduced the BrokerAdapter abstraction. This module re-exposes it
through the shared interface so strategies can be venue-agnostic.

Coinbase covers four asset classes via one API:
    crypto spot, crypto perpetuals, crypto dated futures, commodity futures.

Symbol naming follows Coinbase's own product_id (e.g. "BTC-USD",
"BIP-PERP-INTX", "GOL-27MAY26-CDE").
"""
from __future__ import annotations

import os
from datetime import datetime, UTC

from trading.coinbase_client import CoinbaseClient
from trading.market_data import (
    GRANULARITY_SECONDS,
    fetch_candles,
    get_current_price,
)

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


class CoinbaseAdapter(BrokerAdapter):
    venue = "coinbase"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key or os.environ.get("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("COINBASE_API_SECRET", "")
        self.client = CoinbaseClient(self.api_key, self.api_secret)
        self.is_paper = False  # Coinbase has no paper environment for trading
        self._product_cache: dict[str, dict] = {}

    # ── Account ──────────────────────────────────────────────────────────

    def get_account(self) -> Account:
        try:
            accts = self.client.get_accounts()
        except Exception as e:
            raise BrokerError(f"Coinbase get_accounts: {e}") from e
        usd = [a for a in accts if a.get("currency") == "USD"]
        cash = float(usd[0]["available_balance"]["value"]) if usd else 0.0
        # Equity = cash + sum of crypto holdings * mark price.
        # We approximate with cash here; positions endpoint fills the rest.
        equity = cash
        for a in accts:
            ccy = a.get("currency")
            try:
                bal = float(a.get("available_balance", {}).get("value", 0))
            except Exception:
                bal = 0.0
            if ccy and ccy != "USD" and bal > 0:
                pid = f"{ccy}-USD"
                px = self._safe_price(pid)
                if px:
                    equity += bal * px
        return Account(
            venue=self.venue,
            cash_usd=cash,
            buying_power_usd=cash,  # spot only; futures margin handled separately
            equity_usd=equity,
            is_paper=self.is_paper,
            raw={"accounts": accts},
        )

    def get_positions(self) -> list[Position]:
        accts = self.client.get_accounts()
        out: list[Position] = []
        for a in accts:
            ccy = a.get("currency")
            try:
                qty = float(a.get("available_balance", {}).get("value", 0))
            except Exception:
                qty = 0.0
            if not ccy or ccy == "USD" or qty <= 0:
                continue
            pid = f"{ccy}-USD"
            px = self._safe_price(pid) or 0.0
            out.append(Position(
                venue=self.venue,
                symbol=pid,
                asset_class=AssetClass.CRYPTO_SPOT,
                quantity=qty,
                avg_entry_price=0.0,  # Coinbase API doesn't expose cost basis here
                market_price=px,
                unrealized_pnl_usd=0.0,
                raw=a,
            ))
        return out

    # ── Market data ──────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        try:
            data = self.client.get_best_bid_ask([symbol])
        except Exception as e:
            raise BrokerError(f"Coinbase get_best_bid_ask: {e}") from e
        bid = ask = None
        for pb in data.get("pricebooks", []):
            if pb.get("product_id") != symbol:
                continue
            if pb.get("bids"): bid = float(pb["bids"][0]["price"])
            if pb.get("asks"): ask = float(pb["asks"][0]["price"])
        return Quote(
            venue=self.venue, symbol=symbol,
            bid=bid, ask=ask,
            last=(bid + ask) / 2 if (bid and ask) else None,
            timestamp=datetime.now(UTC),
        )

    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> list[Candle]:
        cached = self._get_cached_candles(symbol, granularity, num_candles)
        if cached is not None:
            return cached
        if granularity not in GRANULARITY_SECONDS:
            raise BrokerError(f"Coinbase granularity not supported: {granularity}")
        arr = fetch_candles(self.client, symbol, granularity, num_candles)
        out: list[Candle] = []
        for row in arr:
            ts, low, high, open_, close, vol = row
            out.append(Candle(
                timestamp=datetime.fromtimestamp(float(ts), tz=UTC),
                open=float(open_), high=float(high),
                low=float(low), close=float(close),
                volume=float(vol),
            ))
        self._put_cached_candles(symbol, granularity, num_candles, out)
        return out

    # ── Orders ───────────────────────────────────────────────────────────

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
        # The existing CoinbaseClient supports MARKET buys (quote_size = USD)
        # and MARKET sells (base_size = quantity). Limit orders not yet wired
        # in the underlying client; flag for now.
        if type == OrderType.LIMIT:
            raise BrokerError("Coinbase LIMIT orders not yet implemented in adapter")

        if side == OrderSide.BUY:
            if notional_usd is None:
                raise BrokerError("Coinbase MARKET BUY requires notional_usd")
            res = self.client.create_market_buy(symbol, f"{notional_usd:.2f}")
        else:
            if quantity is None:
                raise BrokerError("Coinbase MARKET SELL requires quantity")
            res = self.client.create_market_sell(symbol, f"{quantity:.8f}")

        order_id = (
            res.get("order_id")
            or res.get("success_response", {}).get("order_id", "unknown")
        )
        return Order(
            venue=self.venue,
            order_id=order_id,
            symbol=symbol,
            side=side,
            type=type,
            quantity=quantity or 0.0,
            notional_usd=notional_usd,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            filled_avg_price=None,
            submitted_at=datetime.now(UTC),
            raw=res,
        )

    def get_order(self, order_id: str) -> Order:
        d = self.client.get_order(order_id)
        # The existing client returns the raw payload; map fields conservatively
        status_map = {
            "OPEN": OrderStatus.OPEN, "PENDING": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED, "CANCELLED": OrderStatus.CANCELED,
            "EXPIRED": OrderStatus.CANCELED, "FAILED": OrderStatus.REJECTED,
        }
        status = status_map.get(d.get("status", "").upper(), OrderStatus.PENDING)
        return Order(
            venue=self.venue,
            order_id=d.get("order_id", order_id),
            symbol=d.get("product_id", ""),
            side=OrderSide.BUY if d.get("side", "BUY").upper() == "BUY" else OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=float(d.get("filled_size") or 0),
            notional_usd=None,
            limit_price=None,
            status=status,
            filled_quantity=float(d.get("filled_size") or 0),
            filled_avg_price=float(d["average_filled_price"]) if d.get("average_filled_price") else None,
            submitted_at=None,
            raw=d,
        )

    def cancel_order(self, order_id: str) -> None:
        # Not implemented in underlying client; raise so strategies can detect
        raise BrokerError("Coinbase cancel_order not yet implemented in adapter")

    # ── Capabilities ─────────────────────────────────────────────────────

    def list_supported_asset_classes(self) -> list[AssetClass]:
        return [
            AssetClass.CRYPTO_SPOT,
            AssetClass.CRYPTO_PERP,
            AssetClass.CRYPTO_FUTURE,
            AssetClass.COMMODITY_FUTURE,
            AssetClass.EQUITY_INDEX_FUTURE,
        ]

    def list_tradable_symbols(
        self, asset_class: AssetClass | None = None
    ) -> list[str]:
        # Curated default set; full universe is large.
        if asset_class in (None, AssetClass.CRYPTO_SPOT):
            return ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD",
                    "DOT-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD"]
        return []

    # ── Internal ─────────────────────────────────────────────────────────

    def _safe_price(self, symbol: str) -> float | None:
        try:
            return get_current_price(self.client, symbol)
        except Exception:
            return None
