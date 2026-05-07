"""TSMOM ETF — time-series momentum on a 7-ETF liquid basket (Alpaca).

Asness-Moskowitz-Pedersen (JFE 2012) "Time Series Momentum" — own the
asset when its own past 12-month return is positive; flat or short
otherwise. We use a robust 12-1m signal: r_{t-12m..t-1m} (skipping the
last month to avoid short-term reversal).

Universe: SPY, QQQ, IWM, EFA, EEM, TLT, GLD — covers US large/mid-cap,
international DM/EM, rates, gold. 7 mostly-uncorrelated risk drivers.

Position sizing: vol-targeted at 8% per leg, equal risk weighting.
Holding: monthly rebalance triggers ~3-6 round-trips/year per leg, fee
math survives.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]

LOOKBACK_DAYS = 252           # 12 months of trading days
SKIP_DAYS = 21                # skip last month per Asness 12-1m
TARGET_VOL_PER_LEG = 0.08     # 8% annualized per leg
_ANN = math.sqrt(252)


class TSMomETF(Strategy):
    name = "tsmom_etf"
    venue = "alpaca"

    def __init__(self, broker):
        super().__init__(broker)

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Vol-managed overlay (Moreira-Muir 2017): scale book exposure
        # inversely to recent SPY realized vol *before* allocating to
        # legs. Composes cleanly with the per-leg vol-target below
        # (overlay sizes the book; per-leg sizes within the book).
        # Default 1.0 when no overlay signal yet — unchanged behaviour.
        from ._helpers import vol_scaler
        overlay = vol_scaler(ctx, "equity_momentum", 1.0)
        book_alloc = ctx.target_alloc_usd * overlay

        # Per-leg dollar budget
        per_leg = book_alloc / len(UNIVERSE)
        proposals: list[TradeProposal] = []

        for symbol in UNIVERSE:
            try:
                candles = self.broker.get_candles(
                    symbol, "ONE_DAY", num_candles=LOOKBACK_DAYS + 30)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {symbol}: {e}")
                continue
            if len(candles) < LOOKBACK_DAYS:
                continue

            closes = np.array([c.close for c in candles])
            # 12-1 momentum: return over [-LOOKBACK, -SKIP]
            start_idx = -LOOKBACK_DAYS
            end_idx = -SKIP_DAYS if SKIP_DAYS > 0 else None
            window = closes[start_idx:end_idx]
            if len(window) < 30:
                continue
            ret_12_1 = (window[-1] - window[0]) / window[0]
            # Realized vol over the same window
            rets = np.diff(window) / window[:-1]
            sd = float(np.std(rets, ddof=1))
            if sd <= 0:
                continue
            ann_vol = sd * _ANN

            current_qty = ctx.open_positions.get(symbol, {}).get("quantity", 0.0)
            current_price = float(candles[-1].close)
            current_usd = current_qty * current_price
            # Subtract pending BUY notional so we don't double-fire
            pending = ctx.pending_orders.get(symbol, {})
            committed_usd = current_usd + pending.get("buy_notional_usd", 0.0)

            # Signal: long if 12-1m momentum > 0, else flat (no shorts in V1)
            target_usd = per_leg if ret_12_1 > 0 else 0.0

            # Vol-target adjustment: scale by (target_vol / realized_vol)
            if target_usd > 0 and ann_vol > 0:
                scaler = TARGET_VOL_PER_LEG / ann_vol
                target_usd = target_usd * min(2.0, max(0.3, scaler))

            delta_usd = target_usd - committed_usd
            if abs(delta_usd) < max(5.0, per_leg * 0.05):
                continue
            # Skip if there's already a pending BUY for this symbol
            if pending.get("n_pending", 0) > 0 and delta_usd > 0:
                continue

            side = OrderSide.BUY if delta_usd > 0 else OrderSide.SELL
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=side, order_type=OrderType.MARKET,
                notional_usd=abs(delta_usd),
                confidence=0.75 if abs(ret_12_1) > 0.05 else 0.55,
                reason=(f"TSMOM signal: 12-1m return={ret_12_1*100:+.1f}%, "
                        f"target=${target_usd:.0f}, current=${current_usd:.0f}"),
                is_closing=(side == OrderSide.SELL and current_qty > 0),
                metadata={"return_12_1m": ret_12_1, "ann_vol": ann_vol},
            ))

        return proposals
