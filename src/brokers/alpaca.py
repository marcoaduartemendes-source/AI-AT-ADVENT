"""Alpaca adapter — US equities + ETFs.

Uses the v2 trading API directly via `requests`. We deliberately avoid the
official `alpaca-py` SDK so we don't pull a heavy dependency just for a few
endpoints. Auth is two HTTP headers; no JWT, no signing.

Docs: https://docs.alpaca.markets/reference/getauthentication
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

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


# Alpaca timeframe strings — map our Coinbase-style granularity names.
_GRANULARITY_TO_ALPACA = {
    "ONE_MINUTE": "1Min",
    "FIVE_MINUTE": "5Min",
    "FIFTEEN_MINUTE": "15Min",
    "THIRTY_MINUTE": "30Min",
    "ONE_HOUR": "1Hour",
    "TWO_HOUR": "2Hour",
    "FOUR_HOUR": "4Hour",
    "ONE_DAY": "1Day",
    "ONE_WEEK": "1Week",
}


class AlpacaAdapter(BrokerAdapter):
    venue = "alpaca"

    def __init__(
        self,
        key_id: str = "",
        secret_key: str = "",
        endpoint: str = "",
    ):
        self.key_id = (key_id or os.environ.get("ALPACA_API_KEY_ID", "")).strip()
        self.secret = (secret_key or os.environ.get("ALPACA_SECRET_KEY", "")).strip()
        ep = (endpoint or os.environ.get("ALPACA_ENDPOINT") or
              "https://paper-api.alpaca.markets").rstrip("/")
        if not ep.endswith("/v2"):
            ep = ep + "/v2"
        self.endpoint = ep
        self.is_paper = "paper" in ep
        # Market data lives on a different host
        self.data_endpoint = "https://data.alpaca.markets/v2"
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret,
        })

    # ── HTTP helpers ─────────────────────────────────────────────────────

    def _get(self, path: str, *, on_data: bool = False, params: Optional[Dict] = None):
        base = self.data_endpoint if on_data else self.endpoint
        try:
            r = self._session.get(f"{base}{path}", params=params, timeout=15)
        except requests.RequestException as e:
            raise BrokerError(f"Alpaca network error: {e}") from e
        if r.status_code != 200:
            raise BrokerError(f"Alpaca {path} HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _post(self, path: str, body: Dict) -> Dict:
        try:
            r = self._session.post(f"{self.endpoint}{path}", json=body, timeout=15)
        except requests.RequestException as e:
            raise BrokerError(f"Alpaca network error: {e}") from e
        if r.status_code not in (200, 201):
            raise BrokerError(f"Alpaca {path} HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def _delete(self, path: str) -> None:
        r = self._session.delete(f"{self.endpoint}{path}", timeout=15)
        if r.status_code not in (200, 204):
            raise BrokerError(f"Alpaca DELETE {path} HTTP {r.status_code}: {r.text[:160]}")

    # ── Account ──────────────────────────────────────────────────────────

    def get_account(self) -> Account:
        d = self._get("/account")
        return Account(
            venue=self.venue,
            cash_usd=float(d.get("cash", 0)),
            buying_power_usd=float(d.get("buying_power", 0)),
            equity_usd=float(d.get("equity", d.get("portfolio_value", 0))),
            is_paper=self.is_paper,
            raw=d,
        )

    def get_positions(self) -> List[Position]:
        rows = self._get("/positions")
        out: List[Position] = []
        for p in rows:
            sym = p["symbol"]
            qty = float(p.get("qty", 0))
            avg = float(p.get("avg_entry_price", 0))
            mkt = float(p.get("current_price", avg))
            asset_class = AssetClass.ETF if p.get("asset_class") == "us_equity" and \
                _is_likely_etf(sym) else AssetClass.EQUITY
            out.append(Position(
                venue=self.venue,
                symbol=sym,
                asset_class=asset_class,
                quantity=qty,
                avg_entry_price=avg,
                market_price=mkt,
                unrealized_pnl_usd=float(p.get("unrealized_pl", 0)),
                raw=p,
            ))
        return out

    # ── Market data ──────────────────────────────────────────────────────

    def get_quote(self, symbol: str) -> Quote:
        d = self._get(f"/stocks/{symbol}/quotes/latest", on_data=True)
        q = d.get("quote", {})
        return Quote(
            venue=self.venue,
            symbol=symbol,
            bid=float(q.get("bp")) if q.get("bp") else None,
            ask=float(q.get("ap")) if q.get("ap") else None,
            last=None,
            timestamp=datetime.now(timezone.utc),
        )

    def get_candles(
        self, symbol: str, granularity: str, num_candles: int = 100
    ) -> List[Candle]:
        cached = self._get_cached_candles(symbol, granularity, num_candles)
        if cached is not None:
            return cached
        tf = _GRANULARITY_TO_ALPACA.get(granularity)
        if tf is None:
            raise BrokerError(f"Alpaca granularity not supported: {granularity}")
        # Alpaca v2 historical bars endpoint requires a `start` time. Compute
        # one based on granularity × num_candles, with a 2× buffer for
        # non-trading days. Use IEX feed which is free for paper accounts.
        from datetime import timedelta
        seconds_per_bar = {
            "1Min": 60, "5Min": 300, "15Min": 900, "30Min": 1800,
            "1Hour": 3600, "2Hour": 7200, "4Hour": 14400,
            "1Day": 86400, "1Week": 604800,
        }.get(tf, 86400)
        start = (datetime.now(timezone.utc) -
                  timedelta(seconds=seconds_per_bar * num_candles * 2))
        params = {
            "timeframe": tf,
            "limit": num_candles,
            "adjustment": "raw",
            "feed": "iex",
            "start": start.isoformat().replace("+00:00", "Z"),
        }
        d = self._get(f"/stocks/{symbol}/bars", on_data=True, params=params)
        bars = d.get("bars", [])
        out: List[Candle] = []
        for b in bars:
            out.append(Candle(
                timestamp=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                open=float(b["o"]), high=float(b["h"]),
                low=float(b["l"]), close=float(b["c"]),
                volume=float(b.get("v", 0)),
            ))
        self._put_cached_candles(symbol, granularity, num_candles, out)
        return out

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
        body: Dict = {
            "symbol": symbol,
            "side": side.value.lower(),
            "type": "market" if type == OrderType.MARKET else "limit",
            "time_in_force": "day",
            "client_order_id": client_order_id or f"sys-{uuid.uuid4()}",
        }
        if notional_usd is not None and quantity is None:
            body["notional"] = f"{notional_usd:.2f}"
        elif quantity is not None:
            body["qty"] = f"{quantity:.6f}"
        else:
            raise BrokerError("place_order needs quantity or notional_usd")

        if type == OrderType.LIMIT:
            if limit_price is None:
                raise BrokerError("LIMIT order requires limit_price")
            body["limit_price"] = f"{limit_price:.4f}"

        d = self._post("/orders", body)
        return _parse_order(d, self.venue)

    def get_order(self, order_id: str) -> Order:
        d = self._get(f"/orders/{order_id}")
        return _parse_order(d, self.venue)

    def get_open_orders(self) -> List[Order]:
        """Open + pending orders. Strategies subtract pending notional from
        their buying intent so they don't double-fire across cycles."""
        try:
            rows = self._get("/orders", params={"status": "open", "limit": 500})
        except BrokerError:
            return []
        return [_parse_order(r, self.venue) for r in rows]

    def cancel_order(self, order_id: str) -> None:
        self._delete(f"/orders/{order_id}")

    def cancel_stale_orders(self, max_age_seconds: int = 1800) -> int:
        """Cancel any open order older than max_age_seconds. Returns count
        cancelled. Used by the orchestrator at the top of each cycle to
        keep the pending-order queue from growing unbounded."""
        try:
            orders = self.get_open_orders()
        except Exception as e:
            logger.warning(f"cancel_stale_orders: list failed — {e}")
            return 0
        now = datetime.now(timezone.utc)
        cancelled = 0
        for o in orders:
            if not o.submitted_at:
                continue
            age = (now - o.submitted_at).total_seconds()
            if age >= max_age_seconds:
                try:
                    self.cancel_order(o.order_id)
                    cancelled += 1
                except Exception as e:
                    logger.debug(f"cancel {o.order_id} failed: {e}")
        return cancelled

    # ── Capabilities ─────────────────────────────────────────────────────

    def list_supported_asset_classes(self) -> List[AssetClass]:
        return [AssetClass.EQUITY, AssetClass.ETF]

    def list_tradable_symbols(
        self, asset_class: Optional[AssetClass] = None
    ) -> List[str]:
        # Alpaca has ~10k tradable assets; we only return a small curated set
        # by default. Strategies that need the full universe should hit the
        # /assets endpoint directly.
        if asset_class in (None, AssetClass.ETF):
            return ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
                    "DBC", "USMV", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                    "XLI", "XLU", "XLB", "XLRE", "XLC", "USO", "GDX", "VXX",
                    "SVXY", "UVXY"]
        return []


# ─── Helpers ──────────────────────────────────────────────────────────────


def _parse_order(d: Dict, venue: str) -> Order:
    status_map = {
        "new": OrderStatus.OPEN,
        "accepted": OrderStatus.OPEN,
        "pending_new": OrderStatus.PENDING,
        "partially_filled": OrderStatus.PARTIALLY_FILLED,
        "filled": OrderStatus.FILLED,
        "canceled": OrderStatus.CANCELED,
        "expired": OrderStatus.CANCELED,
        "rejected": OrderStatus.REJECTED,
    }
    status = status_map.get(d.get("status", "").lower(), OrderStatus.PENDING)
    submitted_at = None
    if d.get("submitted_at"):
        submitted_at = datetime.fromisoformat(d["submitted_at"].replace("Z", "+00:00"))
    return Order(
        venue=venue,
        order_id=d["id"],
        symbol=d.get("symbol", ""),
        side=OrderSide.BUY if d.get("side") == "buy" else OrderSide.SELL,
        type=OrderType.LIMIT if d.get("type") == "limit" else OrderType.MARKET,
        quantity=float(d.get("qty") or 0),
        notional_usd=float(d["notional"]) if d.get("notional") else None,
        limit_price=float(d["limit_price"]) if d.get("limit_price") else None,
        status=status,
        filled_quantity=float(d.get("filled_qty") or 0),
        filled_avg_price=float(d["filled_avg_price"]) if d.get("filled_avg_price") else None,
        submitted_at=submitted_at,
        raw=d,
    )


# Heuristic — any 3-character all-caps symbol with classic ETF prefix.
# Used only for cosmetic asset_class tagging; not load-bearing.
_ETF_HINT_PREFIXES = ("SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "DBC",
                       "EFA", "EEM", "USO", "VXX", "UVXY", "SVXY", "XL")


def _is_likely_etf(symbol: str) -> bool:
    s = symbol.upper()
    return any(s.startswith(p) for p in _ETF_HINT_PREFIXES)
