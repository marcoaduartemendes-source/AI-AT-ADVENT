"""Bollinger Band breakout — momentum / continuation on individual stocks.

When a stock breaks above its upper Bollinger Band on rising volume,
it's typically signaling a continuation move (institutional buying
on news, earnings, sector strength). Long the breakout, exit when
price drops back below the middle band (20d SMA).

Asymmetric to the RSI strategy:
  - RSI mean-reversion buys oversold pullbacks (counter-trend)
  - Bollinger breakout buys momentum strength (with-trend)

Holding both is intentionally diversifying — they win in different
regimes (mean-reverting vs trending markets). The allocator's
correlation-aware Sharpe tilt will naturally weight whichever's
working in the current regime.
"""
from __future__ import annotations

import logging
import math

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


BAND_PERIOD = 20            # 20-day SMA + bands
BAND_STDDEV = 2.0           # 2 standard deviations for outer bands
MIN_VOLUME_RATIO = 1.3      # today's volume ≥ 1.3× 20d avg
MAX_CONCURRENT = 5
TRADE_SIZE_USD = 4000.0


UNIVERSE = [
    # Mid+large cap stocks with enough vol to break out cleanly.
    # Skip ultra-low-vol staples (KO, PG) where breakouts rarely
    # extend; favor names with frequent news flow.
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "CRM", "ORCL", "ADBE", "NFLX", "INTC", "QCOM",
    "JPM", "GS", "BAC", "MS", "V", "MA",
    "LLY", "UNH", "ABBV",
    "XOM", "CVX",
    "WMT", "COST", "HD", "MCD",
    "DIS", "NKE", "BA", "CAT",
]


class BollingerBreakout(Strategy):
    name = "bollinger_breakout"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}
        slots_left = max(0, MAX_CONCURRENT - len(held))

        # ── Exits: price closed back below middle band (20d SMA) ─────
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0:
                continue
            stats = self._bband_stats(sym)
            if not stats:
                continue
            last_close, sma, _upper, _lower, _vol_ratio = stats
            if last_close < sma:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.9,
                    reason=f"{sym} closed below 20d SMA "
                           f"(${last_close:.2f} < ${sma:.2f})",
                    is_closing=True,
                ))

        # ── Entries: close > upper band AND volume ≥ 1.3× avg ────────
        if slots_left <= 0:
            return proposals
        candidates = []
        for sym in UNIVERSE:
            if sym in held:
                continue
            stats = self._bband_stats(sym)
            if not stats:
                continue
            last_close, sma, upper, _lower, vol_ratio = stats
            if last_close <= upper:
                continue
            if vol_ratio < MIN_VOLUME_RATIO:
                continue
            # Score: how far above the band × volume confirmation
            score = ((last_close - upper) / upper) * vol_ratio * 100
            candidates.append((sym, last_close, upper, vol_ratio, score))

        candidates.sort(key=lambda c: c[4], reverse=True)
        # Sizing: respect ctx.target_alloc_usd / MAX_CONCURRENT slots,
        # capped by TRADE_SIZE_USD as a per-position max. Vol-managed
        # overlay scaler (Moreira-Muir 2017) multiplies the slot size.
        from ._helpers import vol_scaler
        overlay = vol_scaler(ctx, "equity_momentum", 1.0)
        per_slot_alloc = (ctx.target_alloc_usd * overlay) / max(1, MAX_CONCURRENT)
        per_slot = min(per_slot_alloc, TRADE_SIZE_USD)
        for sym, last_close, upper, vol_ratio, _ in candidates[:slots_left]:
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_slot, confidence=0.75,
                reason=f"{sym} breakout: ${last_close:.2f} > "
                       f"upper band ${upper:.2f}, vol {vol_ratio:.1f}×",
                metadata={"close": last_close, "upper": upper,
                          "vol_ratio": vol_ratio},
            ))
        return proposals

    # ── Indicator helpers ────────────────────────────────────────────

    def _bband_stats(self, symbol: str) -> tuple[float, float, float, float, float] | None:
        """Returns (last_close, sma, upper_band, lower_band, vol_ratio).
        None if not enough bars."""
        try:
            candles = self.broker.get_candles(symbol, "1Day",
                                                num_candles=BAND_PERIOD + 5)
        except Exception as e:
            logger.debug(f"[{self.name}] {symbol} candles failed: {e}")
            return None
        if len(candles) < BAND_PERIOD + 1:
            return None
        recent = candles[-BAND_PERIOD:]
        closes = [c.close for c in recent]
        sma = sum(closes) / BAND_PERIOD
        var = sum((c - sma) ** 2 for c in closes) / (BAND_PERIOD - 1)
        sd = math.sqrt(var)
        upper = sma + BAND_STDDEV * sd
        lower = sma - BAND_STDDEV * sd
        avg_vol = sum(c.volume for c in recent) / BAND_PERIOD
        last = candles[-1]
        vol_ratio = (last.volume / avg_vol) if avg_vol > 0 else 0.0
        return (last.close, sma, upper, lower, vol_ratio)
