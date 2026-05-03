"""RSI mean reversion on liquid US large-cap equities.

Classic edge documented since Connors (2008): equity returns mean-revert
on the 2-5 day horizon. Long oversold stocks, exit on RSI normalization.

Rules:
  - Universe: 30 large-cap stocks with low single-name beta (S&P 100 minus
    the most volatile / news-driven names).
  - Entry:  RSI(2) < 10 AND price > 200d SMA (don't catch falling knives).
  - Exit:   RSI(2) > 65 OR holding > 5 days OR stop-loss -3%.
  - Sizing: equal-weight, max 5 concurrent positions.

Why RSI(2) not RSI(14): two-period RSI is much more responsive — it
spikes to extremes more often, giving more entry signals. The 200d SMA
filter (only buy stocks in long-term uptrends) is what separates this
from a "fall harder" strategy.

Live universe is intentionally small + curated. Backtest can sweep a
larger universe.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, UTC

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


RSI_PERIOD = 2
RSI_OVERSOLD = 10
RSI_OVERBOUGHT = 65
SMA_PERIOD = 200
MAX_HOLD_DAYS = 5
STOP_LOSS_PCT = 0.03
MAX_CONCURRENT_POSITIONS = 5


UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Financials
    "JPM", "BAC", "V", "MA", "GS", "MS",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV",
    # Industrials / Energy
    "CAT", "GE", "XOM", "CVX",
    # Consumer
    "WMT", "COST", "HD", "MCD", "KO", "PEP", "PG",
    # Comms / Utilities
    "DIS", "T", "NEE",
]


class RSIMeanReversion(Strategy):
    name = "rsi_mean_reversion"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        # How many slots are open?
        held_symbols = {
            sym for sym, pos in ctx.open_positions.items()
            if (pos.get("quantity") or 0) > 0
        }
        slots_remaining = MAX_CONCURRENT_POSITIONS - len(held_symbols)
        size_per_slot = ctx.target_alloc_usd / MAX_CONCURRENT_POSITIONS

        # ── Exits: stop-loss FIRST, then holding period, then RSI ─────
        # Sprint B2 audit fix: STOP_LOSS_PCT = 3% was declared but
        # never enforced — a 2-day-RSI-bounce strategy can run -8%
        # against you on a single bad earnings day. The stop-loss
        # check below uses the position's entry_price vs the latest
        # close. We check stop-loss BEFORE the time/RSI exits so a
        # losing position closes via stop-loss reason, not a stale
        # ">5d hold" reason that masks the loss.
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0:
                continue
            should_exit, reason = False, ""

            # Stop-loss check
            entry_price = pos.get("entry_price")
            if entry_price and entry_price > 0:
                last_price = self._last_price(sym)
                if last_price is not None:
                    pct_move = (last_price - entry_price) / entry_price
                    if pct_move <= -STOP_LOSS_PCT:
                        should_exit = True
                        reason = (f"STOP-LOSS {pct_move * 100:+.2f}% "
                                  f"≤ -{STOP_LOSS_PCT * 100:.0f}%")

            # Time-based exit
            if not should_exit:
                entry_time = pos.get("entry_time")
                if entry_time:
                    try:
                        et = (datetime.fromisoformat(entry_time)
                              if isinstance(entry_time, str) else entry_time)
                        if datetime.now(UTC) - et > timedelta(days=MAX_HOLD_DAYS):
                            should_exit, reason = True, f">{MAX_HOLD_DAYS}d hold"
                    except (ValueError, TypeError):
                        pass

            # RSI normalization exit
            if not should_exit:
                rsi = self._rsi(sym, RSI_PERIOD)
                if rsi is not None and rsi >= RSI_OVERBOUGHT:
                    should_exit, reason = True, f"RSI({RSI_PERIOD})={rsi:.0f}≥{RSI_OVERBOUGHT}"

            if should_exit:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty,
                    # Stop-loss exits run at confidence=1.0 to bypass
                    # min_confidence gating in the portfolio manager.
                    confidence=1.0 if "STOP-LOSS" in reason else 0.9,
                    reason=reason, is_closing=True,
                ))

        # ── Entries: RSI < oversold AND price > 200 SMA ──────────────
        if slots_remaining <= 0:
            return proposals

        candidates = []
        for sym in UNIVERSE:
            if sym in held_symbols:
                continue
            rsi = self._rsi(sym, RSI_PERIOD)
            if rsi is None or rsi > RSI_OVERSOLD:
                continue
            sma = self._sma(sym, SMA_PERIOD)
            last_price = self._last_price(sym)
            if sma is None or last_price is None or last_price < sma:
                continue
            # Strength score: more oversold + further above SMA = better
            score = (RSI_OVERSOLD - rsi) + (last_price / sma - 1.0) * 100
            candidates.append((sym, rsi, last_price, score))

        # Pick the top `slots_remaining` candidates
        candidates.sort(key=lambda c: c[3], reverse=True)
        for sym, rsi, last_price, _ in candidates[:slots_remaining]:
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=size_per_slot, confidence=0.7,
                reason=f"RSI({RSI_PERIOD})={rsi:.1f}, ${last_price:.0f}>SMA200",
                metadata={"rsi": rsi, "last_price": last_price},
            ))
        return proposals

    # ── Indicator helpers ────────────────────────────────────────────

    def _rsi(self, symbol: str, period: int = 2) -> float | None:
        """Compute RSI from the broker's daily candle endpoint.

        RSI = 100 - 100/(1 + RS), where RS = avg gain / avg loss
        over `period` bars.
        """
        try:
            candles = self.broker.get_candles(symbol, "1Day", num_candles=period + 50)
        except Exception as e:
            logger.debug(f"[{self.name}] {symbol} candles failed: {e}")
            return None
        if len(candles) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(candles)):
            change = candles[i].close - candles[i - 1].close
            gains.append(max(0, change))
            losses.append(abs(min(0, change)))
        if len(gains) < period:
            return None
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _sma(self, symbol: str, period: int) -> float | None:
        try:
            candles = self.broker.get_candles(symbol, "1Day", num_candles=period)
        except Exception:
            return None
        if len(candles) < period:
            return None
        return sum(c.close for c in candles[-period:]) / period

    def _last_price(self, symbol: str) -> float | None:
        try:
            candles = self.broker.get_candles(symbol, "1Day", num_candles=2)
        except Exception:
            return None
        return candles[-1].close if candles else None
