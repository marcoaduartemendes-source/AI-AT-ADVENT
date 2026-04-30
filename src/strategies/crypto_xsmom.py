"""Crypto cross-sectional momentum — long top-quintile / short bottom-quintile
30-day winners on a curated alt universe (Coinbase spot).

Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency", plus
Starkiller Capital's published cross-sectional crypto momentum results
— this strategy family has held up better than mean-reversion on alts.

V1 long-only variant: hold top-quintile of recent 30-day winners
(equal-weighted), rebalance weekly. Long-only because shorting altcoins
on Coinbase spot is expensive/operationally awkward at retail; the
long-only version captures most of the documented spread.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import requests

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Curated liquid-alt universe (top-15 by Coinbase volume, with a stable-
# enough lookback). Excludes BTC/ETH so we trade the alt-momentum factor
# orthogonal to the underlying market direction.
UNIVERSE = [
    "SOL-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "DOGE-USD",
    "LINK-USD", "LTC-USD", "MATIC-USD", "BCH-USD", "XLM-USD",
    "UNI-USD", "ATOM-USD", "ALGO-USD", "FIL-USD", "ICP-USD",
]

LOOKBACK_DAYS = 30
TOP_QUINTILE_FRAC = 0.20      # top 20% (3 of 15)
PUBLIC_PRODUCTS = "https://api.coinbase.com/api/v3/brokerage/market/products"


class CryptoXSMom(Strategy):
    name = "crypto_xsmom"
    venue = "coinbase"

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Compute 30-day return per coin
        returns = []
        prices = {}
        for sym in UNIVERSE:
            ret = self._fetch_30d_return(sym)
            if ret is None:
                continue
            returns.append((sym, ret))
            prices[sym] = self._latest_price(sym)
        if len(returns) < 5:
            logger.debug(f"[{self.name}] insufficient data ({len(returns)} symbols)")
            return []

        returns.sort(key=lambda r: r[1], reverse=True)
        top_n = max(1, int(len(returns) * TOP_QUINTILE_FRAC))
        winners = {sym for sym, _ in returns[:top_n]}

        per_leg = ctx.target_alloc_usd / max(top_n, 1)
        proposals: List[TradeProposal] = []

        # Open positions for new winners
        for sym in winners:
            if not prices.get(sym):
                continue
            existing_qty = ctx.open_positions.get(sym, {}).get("quantity", 0)
            if existing_qty > 0:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_leg, confidence=0.7,
                reason=(f"top-quintile 30d momentum (n={len(returns)} alts)"),
            ))

        # Close positions for coins that fell out of the winners set
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity", 0) if isinstance(pos, dict) else 0
            if qty <= 0:
                continue
            if sym not in winners and sym in UNIVERSE:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.9, is_closing=True,
                    reason="dropped out of top quintile",
                ))

        return proposals

    # ── Helpers ────────────────────────────────────────────────────────

    def _fetch_30d_return(self, symbol: str):
        """Use the Coinbase public candles endpoint (same as backtest util)
        for a quick 30-day return estimate."""
        try:
            candles = self.broker.get_candles(symbol, "ONE_DAY",
                                                num_candles=LOOKBACK_DAYS + 2)
        except Exception:
            return None
        if len(candles) < 5:
            return None
        closes = [c.close for c in candles]
        if closes[0] <= 0:
            return None
        return (closes[-1] - closes[0]) / closes[0]

    def _latest_price(self, symbol: str):
        try:
            r = requests.get(f"{PUBLIC_PRODUCTS}/{symbol}", timeout=10)
            if r.status_code != 200:
                return None
            return float(r.json().get("price") or 0) or None
        except Exception:
            return None
