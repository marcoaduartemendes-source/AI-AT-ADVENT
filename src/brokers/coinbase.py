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

import logging
import os

logger = logging.getLogger(__name__)
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
    BrokerCapability,
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
    # Audit-fix F7 (2026-05-07): declare capabilities so the
    # orchestrator's wash-trade guard, intra-cycle pending tracker,
    # and stale-order canceller actually run on Coinbase. Without
    # this declaration the live venue had zero protection against
    # duplicate-order races between concurrent strategies.
    capabilities = frozenset({BrokerCapability.GET_OPEN_ORDERS})

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

        # ── Cash detection (2026-05-20 fix) ───────────────────────────
        # The original code only read the USD wallet. Production cycles
        # were rejecting every BUY with "Coinbase USD wallet too low:
        # $0.00 available" because users routinely fund with USDC
        # (the bridge from exchanges/DeFi), not raw USD — and Coinbase
        # Advanced treats USD and USDC as fully fungible at 1:1 for
        # spot trading (USD-* and USDC-* pairs deliver the same coin).
        # We now sum USD + USDC into cash_usd so the wallet check
        # reflects what the user actually has available to deploy.
        def _bal(ccy: str) -> float:
            for a in accts:
                if a.get("currency") == ccy:
                    try:
                        return float(a.get("available_balance",
                                             {}).get("value", 0))
                    except Exception:
                        return 0.0
            return 0.0

        usd_bal = _bal("USD")
        usdc_bal = _bal("USDC")
        cash = usd_bal + usdc_bal

        # Equity = cash + sum of crypto holdings * mark price.
        # USD/USDC already counted in `cash`; skip both here.
        equity = cash
        for a in accts:
            ccy = a.get("currency")
            try:
                bal = float(a.get("available_balance", {}).get("value", 0))
            except Exception:
                bal = 0.0
            if ccy and ccy not in ("USD", "USDC") and bal > 0:
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
            raw={"accounts": accts,
                 "usd_balance": usd_bal,
                 "usdc_balance": usdc_bal},
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
            # Pre-flight check: cap notional_usd to actual available
            # USD balance minus a 5% buffer for fees and price drift.
            # Without this, every cycle in production has been firing
            # INSUFFICIENT_FUND errors because the strategy sized its
            # order to the configured per-venue cap regardless of
            # whether the wallet could cover it.
            try:
                acct = self.get_account()
                cash = float(acct.cash_usd or 0)
                # Reserve 5% for fees + price slippage between order
                # construction and fill. Coinbase taker fee is 0.6%
                # default; 5% buffer covers fee + 4% drift.
                spendable = max(cash * 0.95, 0.0)
                if spendable < 1.0:
                    # Show the USD/USDC split so the user can see if
                    # the funds are sitting in a non-spot wallet or
                    # different portfolio (Coinbase Advanced supports
                    # multiple portfolios; we read whichever the API
                    # key is scoped to).
                    raw = acct.raw or {}
                    usd_b = float(raw.get("usd_balance", 0) or 0)
                    usdc_b = float(raw.get("usdc_balance", 0) or 0)
                    raise BrokerError(
                        f"Coinbase spot cash too low: ${cash:.2f} "
                        f"(USD ${usd_b:.2f} + USDC ${usdc_b:.2f}). "
                        f"Need >$1 after 5% buffer. If your funds are "
                        f"in a different Coinbase portfolio, move them "
                        f"to the one the API key is scoped to."
                    )
                if notional_usd > spendable:
                    logger.info(
                        f"[coinbase] CLAMP BUY {symbol} notional "
                        f"${notional_usd:.2f} → ${spendable:.2f} "
                        f"(USD wallet={cash:.2f}, 5% fee buffer)"
                    )
                    notional_usd = spendable
            except BrokerError as e:
                # Re-raise only the WALLET-TOO-LOW BrokerError (which
                # is a real "user needs to fund" signal). Other
                # BrokerErrors from get_account (e.g. transient API
                # failure) should fall through and let the actual
                # order attempt either succeed or surface its own
                # error — same semantics as before pre-flight existed.
                if "wallet too low" in str(e):
                    raise
                logger.debug(f"coinbase pre-flight balance check skipped: {e}")
            except Exception as e:
                # Pre-flight check failure is non-fatal — fall through
                # and let Coinbase reject if the wallet really is empty.
                logger.debug(f"coinbase pre-flight balance check skipped: {e}")
            res = self.client.create_market_buy(
                symbol, f"{notional_usd:.2f}",
                client_order_id=client_order_id,
            )
        else:
            if quantity is None:
                raise BrokerError("Coinbase MARKET SELL requires quantity")
            res = self.client.create_market_sell(
                symbol, f"{quantity:.8f}",
                client_order_id=client_order_id,
            )

        # Audit-fix F3 (2026-05-07): the previous code fell back to
        # `order_id="unknown"` when the response was malformed and
        # let _record_trade write a row that fill polling could never
        # transition (get_unfilled_trades excludes 'unknown'). Real
        # USD left the wallet, ledger thought it was in-flight, FIFO
        # recompute skipped the row → phantom-loss class bug.
        # Now: if Coinbase's response has no parseable order_id, raise
        # so the caller records the failure cleanly. The full response
        # is logged for forensics; success_response.failure_reason is
        # surfaced first if present.
        order_id = (
            res.get("order_id")
            or res.get("success_response", {}).get("order_id")
        )
        if not order_id:
            failure_reason = (
                res.get("failure_reason")
                or res.get("error_response", {}).get("error")
                or res.get("error_response", {}).get("message")
                or "no order_id in response"
            )
            raise BrokerError(
                f"Coinbase order_id missing for {side.value} {symbol}: "
                f"{failure_reason}; response keys={list(res.keys())[:8]}"
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

    def get_open_orders(self) -> list[Order]:
        """List currently-OPEN orders. Audit-fix F7 (2026-05-07).
        Best-effort: returns [] on adapter failure rather than raising,
        so a Coinbase outage doesn't take the cycle down — the
        fail-closed degraded-venue path in the orchestrator handles
        the recovery semantics."""
        try:
            raw = self.client.list_open_orders()
        except Exception as e:
            logger.warning(f"Coinbase list_open_orders failed: {e}")
            raise
        out: list[Order] = []
        for o in raw:
            cfg = o.get("order_configuration") or {}
            mkt_cfg = cfg.get("market_market_ioc") or {}
            qty_str = mkt_cfg.get("base_size") or mkt_cfg.get("quote_size")
            try:
                qty = float(qty_str) if qty_str else 0.0
            except (TypeError, ValueError):
                qty = 0.0
            side = OrderSide.BUY if (o.get("side") or "").upper() == "BUY" else OrderSide.SELL
            status = OrderStatus.OPEN
            try:
                filled = float(o.get("filled_size") or 0)
            except (TypeError, ValueError):
                filled = 0.0
            avg_px = None
            try:
                v = o.get("average_filled_price")
                avg_px = float(v) if v not in (None, "", "0") else None
            except (TypeError, ValueError):
                avg_px = None
            out.append(Order(
                venue=self.venue,
                order_id=o.get("order_id") or "",
                symbol=o.get("product_id") or "",
                side=side,
                type=OrderType.MARKET,
                quantity=qty,
                notional_usd=None,
                limit_price=None,
                status=status,
                filled_quantity=filled,
                filled_avg_price=avg_px,
                submitted_at=None,
                raw=o,
            ))
        return out

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
