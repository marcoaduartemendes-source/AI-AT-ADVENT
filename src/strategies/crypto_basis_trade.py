"""Crypto basis trade — long spot, short dated future on Coinbase.

Edge: when a dated future trades at premium to spot beyond the carry rate,
buy spot + short the future and capture the convergence at expiry. Pure
arbitrage if the basis stays positive through expiry; risk is mostly
funding/margin during the holding period.

Inputs from scout:
    coinbase.term_structure (commodities scout) — but for crypto we'll
    add a similar funding-vs-basis comparison directly from the broker.

V1 implementation queries the Coinbase futures product list each cycle
(no scout needed for crypto basis — the data is right there in the
public products endpoint). Compares front-month future price against
spot price; if annualized basis > entry threshold, opens the trade.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Crypto futures we trade. Maps spot symbol → futures-root prefix.
PAIRS = {
    "BTC-USD": "BIT",
    "ETH-USD": "ET",
    "SOL-USD": "SOL",
}

# Annualized basis thresholds
ENTRY_BASIS_BPS = 800     # 8% APR — open the carry
EXIT_BASIS_BPS = 200      # 2% APR — close, basis decayed

PUBLIC_PRODUCTS = "https://api.coinbase.com/api/v3/brokerage/market/products"


class CryptoBasisTrade(Strategy):
    name = "crypto_basis_trade"
    venue = "coinbase"

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Fetch crypto futures products (same endpoint we used for the
        # commodities scout, filtered by root)
        try:
            r = requests.get(
                PUBLIC_PRODUCTS,
                params={"product_type": "FUTURE", "limit": 300},
                timeout=15,
            )
            if r.status_code != 200:
                return []
            futures = r.json().get("products", [])
        except Exception as e:
            logger.warning(f"[{self.name}] futures fetch: {e}")
            return []

        proposals: List[TradeProposal] = []
        per_leg = ctx.target_alloc_usd / 2

        for spot_sym, root in PAIRS.items():
            front = self._front_month(futures, root)
            if not front:
                continue
            future_price = front["price"]
            spot_price = self._spot_price(spot_sym)
            if not future_price or not spot_price:
                continue

            # Days to expiry
            expiry = front["expiry"]
            today = datetime.now(timezone.utc).date()
            days_to_exp = max(1, (expiry - today).days)

            # Annualized basis (positive = future trades premium = sell future)
            basis_pct = (future_price - spot_price) / spot_price * 100
            annualized_bps = basis_pct * (365 / days_to_exp) * 100

            # Position state
            spot_pos = ctx.open_positions.get(spot_sym, {})
            fut_pos = ctx.open_positions.get(front["product_id"], {})
            in_pos = (spot_pos.get("quantity", 0) > 0
                       or fut_pos.get("quantity", 0) > 0)

            if not in_pos and abs(annualized_bps) >= ENTRY_BASIS_BPS:
                # Positive basis → long spot, short future
                # Negative basis → short spot, long future
                spot_side = OrderSide.BUY if annualized_bps > 0 else OrderSide.SELL
                fut_side = OrderSide.SELL if annualized_bps > 0 else OrderSide.BUY
                reason = (f"basis {annualized_bps:.0f}bps APR "
                          f"({days_to_exp}d to expiry) — open")
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=spot_sym,
                    side=spot_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg, confidence=0.85, reason=reason,
                    metadata={"leg": "spot", "annualized_bps": annualized_bps},
                ))
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue,
                    symbol=front["product_id"],
                    side=fut_side, order_type=OrderType.MARKET,
                    notional_usd=per_leg, confidence=0.85, reason=reason,
                    metadata={"leg": "future", "annualized_bps": annualized_bps},
                ))
            elif in_pos and abs(annualized_bps) < EXIT_BASIS_BPS:
                # Basis decayed — close
                reason = f"basis {annualized_bps:.0f}bps APR — close"
                if spot_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=spot_sym,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=spot_pos["quantity"], confidence=0.95,
                        reason=reason, is_closing=True,
                    ))
                if fut_pos.get("quantity", 0) > 0:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue,
                        symbol=front["product_id"],
                        side=OrderSide.BUY, order_type=OrderType.MARKET,
                        quantity=fut_pos["quantity"], confidence=0.95,
                        reason=reason, is_closing=True,
                    ))
        return proposals

    # ── Helpers ──────────────────────────────────────────────────────────

    def _front_month(self, futures, root: str):
        """Return the nearest non-expired future for a given root."""
        candidates = []
        for p in futures:
            pid = p.get("product_id", "")
            m = re.match(r"^([A-Z0-9]+)-(\d+[A-Z]+\d+)-([A-Z]+)$", pid)
            if not m:
                continue
            r, expiry, _ = m.groups()
            if r != root:
                continue
            try:
                dt = datetime.strptime(expiry, "%d%b%y").date()
            except ValueError:
                continue
            if dt < datetime.now(timezone.utc).date():
                continue
            try:
                price = float(p.get("price") or 0)
            except (TypeError, ValueError):
                price = 0.0
            if price <= 0:
                continue
            candidates.append({"product_id": pid, "expiry": dt, "price": price})
        candidates.sort(key=lambda c: c["expiry"])
        return candidates[0] if candidates else None

    def _spot_price(self, spot_sym: str) -> Optional[float]:
        try:
            r = requests.get(f"{PUBLIC_PRODUCTS}/{spot_sym}", timeout=10)
            if r.status_code != 200:
                return None
            return float(r.json().get("price") or 0) or None
        except Exception:
            return None
