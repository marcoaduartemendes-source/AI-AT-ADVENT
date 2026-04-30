"""Kalshi adapter — placeholder.

Kalshi auth is RSA-signed (private key + key ID), unlike Coinbase/Alpaca
which use HMAC or static headers. We stub the adapter now so the rest of the
framework can be wired up; the signing implementation lands the moment we
have credentials.

API docs: https://trading-api.readme.io/reference/getting-started
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

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


class KalshiAdapter(BrokerAdapter):
    venue = "kalshi"

    def __init__(
        self,
        key_id: str = "",
        private_key_path: str = "",
        endpoint: str = "",
    ):
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID", "")
        self.private_key_path = (
            private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        )
        self.endpoint = (
            endpoint
            or os.environ.get("KALSHI_ENDPOINT")
            or "https://trading-api.kalshi.com/trade-api/v2"
        ).rstrip("/")
        self.is_paper = "demo" in self.endpoint
        self._configured = bool(self.key_id and self.private_key_path)

    # ── Stub methods — raise until credentials wired ─────────────────────

    def _need_creds(self):
        if not self._configured:
            raise BrokerError(
                "Kalshi adapter not yet configured. Provide KALSHI_KEY_ID + "
                "KALSHI_PRIVATE_KEY_PATH env vars (or repo secrets)."
            )

    def get_account(self) -> Account:
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def get_positions(self) -> List[Position]:
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def get_quote(self, symbol: str) -> Quote:
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> List[Candle]:
        # Kalshi doesn't expose OHLCV in the traditional sense; trade history
        # serves as the proxy. Return empty for now — strategies that need
        # candles should not target Kalshi anyway.
        return []

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
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def get_order(self, order_id: str) -> Order:
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def cancel_order(self, order_id: str) -> None:
        self._need_creds()
        raise BrokerError("Kalshi adapter not yet implemented")

    def list_supported_asset_classes(self) -> List[AssetClass]:
        return [AssetClass.PREDICTION]

    def list_tradable_symbols(
        self, asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        return []

    def healthcheck(self) -> Dict:
        return {
            "venue": self.venue,
            "ok": False,
            "configured": self._configured,
            "note": "Kalshi adapter is a placeholder pending KYC + API key.",
        }
