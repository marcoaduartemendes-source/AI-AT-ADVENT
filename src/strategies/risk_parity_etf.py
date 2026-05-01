"""Risk Parity ETF — Bridgewater All-Weather flavored ETF book.

Holds 5 broad ETFs at equal *risk* contribution (equal vol-weight, not equal
dollar weight). Rebalances monthly.

Universe (Alpaca):
    SPY  — US equities (growth proxy)
    TLT  — long Treasuries (rising-deflation hedge)
    IEF  — intermediate Treasuries (rates ballast)
    GLD  — gold (inflation / crisis hedge)
    DBC  — broad commodities (inflation hedge)

Algorithm:
    weight_i ∝ 1 / vol_i      (inverse-vol)
    target_qty_i = (target_alloc_usd × weight_i) / price_i

If portfolio is already within 2% of target weights, skip — keeps fee drag
low. The strategy is naturally low-turnover (~12 round-trips/year per leg).
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Annualization for daily-bar realized vol
_ANN = math.sqrt(252)
_MIN_REBALANCE_DELTA_PCT = 0.02  # 2% deviation tolerance


class RiskParityETF(Strategy):
    name = "risk_parity_etf"
    venue = "alpaca"

    def __init__(
        self,
        broker,
        universe: Optional[List[str]] = None,
        vol_lookback_days: int = 60,
    ):
        super().__init__(broker)
        self.universe = universe or ["SPY", "TLT", "IEF", "GLD", "DBC"]
        self.vol_lookback_days = vol_lookback_days

    # ── Core --------------------------------------------------------------

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Inverse-vol weights from daily bars
        weights = self._compute_inverse_vol_weights()
        if not weights:
            logger.warning(f"[{self.name}] insufficient data for vol estimation")
            return []

        # Current prices for sizing
        prices = self._latest_prices()
        if not prices:
            return []

        proposals: List[TradeProposal] = []
        for symbol in self.universe:
            w = weights.get(symbol, 0.0)
            target_usd = ctx.target_alloc_usd * w
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            target_qty = target_usd / price

            current_qty = ctx.open_positions.get(symbol, {}).get("quantity", 0.0)
            current_usd = current_qty * price
            # Pending BUYs are not yet in positions but committed buying
            # power — subtract from intent to avoid double-firing.
            pending = ctx.pending_orders.get(symbol, {})
            pending_buy_usd = pending.get("buy_notional_usd", 0.0)
            committed_usd = current_usd + pending_buy_usd
            delta_usd = target_usd - committed_usd

            # Skip near-target positions
            if abs(delta_usd) < ctx.target_alloc_usd * _MIN_REBALANCE_DELTA_PCT:
                continue
            # Skip dust
            if abs(delta_usd) < 5.0:
                continue
            # Skip if there's already a pending BUY for this symbol
            if pending.get("n_pending", 0) > 0 and delta_usd > 0:
                logger.debug(f"[{self.name}] {symbol} has pending order(s); skip")
                continue

            side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
            is_closing = (current_qty > 0 and side == OrderSide.SELL)
            proposals.append(TradeProposal(
                strategy=self.name,
                venue=self.venue,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                notional_usd=abs(delta_usd),
                confidence=0.9,
                reason=f"rebalance to {w * 100:.1f}% (target ${target_usd:.0f}, "
                       f"current ${current_usd:.0f})",
                is_closing=is_closing,
                metadata={"target_weight": w, "target_qty": target_qty},
            ))

        return proposals

    # ── Helpers ----------------------------------------------------------

    def _compute_inverse_vol_weights(self) -> dict:
        vols: dict = {}
        for sym in self.universe:
            try:
                candles = self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=self.vol_lookback_days)
            except Exception as e:
                logger.warning(f"[{self.name}] candles failed for {sym}: {e}")
                continue
            if len(candles) < 20:
                continue
            closes = np.array([c.close for c in candles])
            rets = np.diff(closes) / closes[:-1]
            sd = float(np.std(rets, ddof=1))
            if sd > 0:
                vols[sym] = sd * _ANN

        if not vols:
            return {}
        inv = {s: 1.0 / v for s, v in vols.items()}
        total = sum(inv.values())
        return {s: w / total for s, w in inv.items()}

    def _latest_prices(self) -> dict:
        out: dict = {}
        for sym in self.universe:
            try:
                q = self.broker.get_quote(sym)
                if q.bid and q.ask:
                    out[sym] = (q.bid + q.ask) / 2
                elif q.last:
                    out[sym] = q.last
            except Exception as e:
                logger.warning(f"[{self.name}] quote failed for {sym}: {e}")
        return out
