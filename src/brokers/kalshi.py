"""Kalshi adapter — CFTC-regulated US prediction markets.

Auth uses RSA-PSS-SHA256: each request signs `timestamp + METHOD + path`
with the account's private key. Headers:
    KALSHI-ACCESS-KEY:       <key_id>
    KALSHI-ACCESS-SIGNATURE: base64(RSA-PSS-SHA256(msg))
    KALSHI-ACCESS-TIMESTAMP: <ms since epoch>

Docs: https://trading-api.readme.io/reference/getting-started
"""
from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

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

logger = logging.getLogger(__name__)


class KalshiAdapter(BrokerAdapter):
    venue = "kalshi"

    def __init__(
        self,
        key_id: str = "",
        private_key_pem: str = "",
        endpoint: str = "",
    ):
        self.key_id = (key_id or os.environ.get("KALSHI_KEY_ID", "")).strip()
        pem_text = private_key_pem or os.environ.get("KALSHI_PRIVATE_KEY", "")
        # Tolerate JSON-escaped newlines in env values
        pem_text = pem_text.replace("\\n", "\n").strip()
        self._pem_text = pem_text
        ep = (endpoint or os.environ.get("KALSHI_ENDPOINT")
              or "https://api.elections.kalshi.com/trade-api/v2").rstrip("/")
        self.endpoint = ep
        self.is_paper = "demo" in ep
        self._configured = bool(self.key_id and pem_text)

        self._private_key: Optional[RSAPrivateKey] = None
        if self._configured:
            try:
                key = serialization.load_pem_private_key(
                    pem_text.encode("utf-8"), password=None
                )
                if not isinstance(key, RSAPrivateKey):
                    raise BrokerError(
                        f"Kalshi expects RSA private key, got {type(key).__name__}"
                    )
                self._private_key = key
            except Exception as e:
                raise BrokerError(f"Kalshi PEM load failed: {e}") from e

        self._session = requests.Session()

    # ── Signing ──────────────────────────────────────────────────────────

    def _sign(self, method: str, path: str) -> Dict[str, str]:
        if not self._private_key:
            raise BrokerError("Kalshi adapter not configured")
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode("utf-8")
        sig = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs):
        # Path for signing is just the URL path component (incl. /trade-api/v2)
        from urllib.parse import urlparse
        parsed = urlparse(self.endpoint + path)
        sign_path = parsed.path
        headers = self._sign(method, sign_path)
        try:
            r = self._session.request(method, self.endpoint + path,
                                       headers=headers, timeout=15, **kwargs)
        except requests.RequestException as e:
            raise BrokerError(f"Kalshi network error: {e}") from e
        if r.status_code >= 400:
            raise BrokerError(f"Kalshi {method} {path} HTTP {r.status_code}: {r.text[:200]}")
        return r.json() if r.text else {}

    # ── Account ──────────────────────────────────────────────────────────

    def get_account(self) -> Account:
        if not self._configured:
            raise BrokerError("Kalshi adapter not configured")
        # Kalshi exposes /portfolio/balance and /exchange/status
        d = self._request("GET", "/portfolio/balance")
        balance_cents = float(d.get("balance", 0))
        cash = balance_cents / 100.0  # Kalshi reports cents
        return Account(
            venue=self.venue,
            cash_usd=cash,
            buying_power_usd=cash,
            equity_usd=cash,  # plus open position MTM, fetched separately
            is_paper=self.is_paper,
            raw=d,
        )

    def get_positions(self) -> List[Position]:
        if not self._configured:
            return []
        d = self._request("GET", "/portfolio/positions")
        out: List[Position] = []
        for p in d.get("market_positions", []):
            ticker = p.get("ticker") or p.get("market_ticker")
            qty = float(p.get("position", 0))
            if qty == 0:
                continue
            cost = float(p.get("market_exposure", 0)) / 100.0
            out.append(Position(
                venue=self.venue,
                symbol=ticker,
                asset_class=AssetClass.PREDICTION,
                quantity=qty,
                avg_entry_price=(abs(cost) / abs(qty)) if qty else 0.0,
                market_price=0.0,
                unrealized_pnl_usd=float(p.get("realized_pnl", 0)) / 100.0,
                raw=p,
            ))
        return out

    # ── Market data ──────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        if not self._configured:
            raise BrokerError("Kalshi adapter not configured")
        d = self._request("GET", f"/markets/{symbol}")
        m = d.get("market", {})
        # Prices in cents; convert to dollars
        bid = float(m["yes_bid"]) / 100.0 if m.get("yes_bid") is not None else None
        ask = float(m["yes_ask"]) / 100.0 if m.get("yes_ask") is not None else None
        last = float(m["last_price"]) / 100.0 if m.get("last_price") is not None else None
        return Quote(
            venue=self.venue, symbol=symbol,
            bid=bid, ask=ask, last=last,
            timestamp=datetime.now(timezone.utc),
        )

    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> List[Candle]:
        # Prediction markets don't have OHLCV in the equity/crypto sense.
        # Strategies needing time series should hit /markets/{ticker}/history.
        return []

    # ── Orders ───────────────────────────────────────────────────────────

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
        if not self._configured:
            raise BrokerError("Kalshi adapter not configured")
        if quantity is None:
            raise BrokerError("Kalshi orders require integer contract quantity")
        # Kalshi orders are always for YES contracts; SELL on Kalshi means
        # short the YES (== buy NO). Adapter exposes BUY/SELL semantics
        # consistent with rest of system; we map both to the YES side.
        action = "buy" if side == OrderSide.BUY else "sell"
        body: Dict = {
            "ticker": symbol,
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "type": "market" if type == OrderType.MARKET else "limit",
            "action": action,
            "side": "yes",
            "count": int(round(quantity)),
        }
        if type == OrderType.LIMIT:
            if limit_price is None:
                raise BrokerError("LIMIT order requires limit_price")
            # Kalshi limit prices are in cents (1-99 for YES side)
            body["yes_price"] = int(round(limit_price * 100))
        d = self._request("POST", "/portfolio/orders", json=body)
        order = d.get("order", {})
        return _parse_order(order, self.venue, fallback_symbol=symbol,
                            fallback_side=side, fallback_type=type)

    def get_order(self, order_id: str) -> Order:
        if not self._configured:
            raise BrokerError("Kalshi adapter not configured")
        d = self._request("GET", f"/portfolio/orders/{order_id}")
        return _parse_order(d.get("order", d), self.venue)

    def cancel_order(self, order_id: str) -> None:
        if not self._configured:
            raise BrokerError("Kalshi adapter not configured")
        self._request("DELETE", f"/portfolio/orders/{order_id}")

    # ── Capabilities ─────────────────────────────────────────────────────

    def list_supported_asset_classes(self) -> List[AssetClass]:
        return [AssetClass.PREDICTION]

    def list_tradable_symbols(
        self, asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        """Return tickers of currently active markets. Capped at 100 for
        the default surface; strategies that want the full universe should
        page /markets directly."""
        if not self._configured:
            return []
        try:
            d = self._request("GET", "/markets?status=open&limit=100")
            return [m["ticker"] for m in d.get("markets", []) if m.get("ticker")]
        except Exception as e:
            logger.warning(f"Kalshi list_tradable_symbols failed: {e}")
            return []

    def healthcheck(self) -> Dict:
        if not self._configured:
            return {"venue": self.venue, "ok": False, "configured": False,
                    "note": "Kalshi credentials not set"}
        try:
            acct = self.get_account()
            return {
                "venue": self.venue,
                "ok": True,
                "configured": True,
                "is_paper": self.is_paper,
                "cash_usd": acct.cash_usd,
                "equity_usd": acct.equity_usd,
            }
        except Exception as exc:
            return {"venue": self.venue, "ok": False, "configured": True,
                    "error": str(exc)}


# ─── Helpers ──────────────────────────────────────────────────────────────


def _parse_order(d: Dict, venue: str, *,
                 fallback_symbol: str = "",
                 fallback_side: OrderSide = OrderSide.BUY,
                 fallback_type: OrderType = OrderType.MARKET) -> Order:
    status_raw = (d.get("status") or "").lower()
    status_map = {
        "resting": OrderStatus.OPEN,
        "pending": OrderStatus.PENDING,
        "executed": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "expired": OrderStatus.CANCELED,
    }
    status = status_map.get(status_raw, OrderStatus.PENDING)
    yes_price = d.get("yes_price")
    return Order(
        venue=venue,
        order_id=d.get("order_id") or d.get("id") or "unknown",
        symbol=d.get("ticker") or fallback_symbol,
        side=OrderSide.BUY if d.get("action") == "buy" else
              (OrderSide.SELL if d.get("action") == "sell" else fallback_side),
        type=OrderType.LIMIT if d.get("type") == "limit" else
              (OrderType.MARKET if d.get("type") == "market" else fallback_type),
        quantity=float(d.get("count") or 0),
        notional_usd=None,
        limit_price=(float(yes_price) / 100.0) if yes_price is not None else None,
        status=status,
        filled_quantity=float(d.get("filled_count") or 0),
        filled_avg_price=(float(d["yes_price"]) / 100.0) if d.get("yes_price") and status == OrderStatus.FILLED else None,
        submitted_at=None,
        raw=d,
    )
