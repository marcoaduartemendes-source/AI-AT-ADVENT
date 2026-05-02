"""Overnight-gap reversion on US large-caps.

Documented edge (Heston-Sadka 2008, Goldberg 2024 SSRN): stocks that
gap >1.5% on the open tend to mean-revert in the first 30-60 minutes.
The intraday move is dominated by overnight news being slowly priced
in — institutional desks move with VWAP and often "overshoot" at the
open before professionals fade them.

Strategy:
  - Each cycle, scan a curated 25-name S&P 100 watchlist
  - For each stock: today's open vs yesterday's close = gap %
  - If gap_pct > +1.5%: SHORT-bias (gap up tends to fade)
  - If gap_pct < -1.5%: LONG-bias (gap down tends to bounce)
  - Hold for one trading day, exit at next-day open at market

Long-only on Alpaca paper (shorts require margin enabled). Long the
NEGATIVE gaps; ignore positive gaps until shorting is enabled.

Position sizing: $3k per trade, max 4 concurrent positions.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


MIN_GAP_PCT = 1.5           # absolute % gap to qualify
MAX_HOLD_DAYS = 1           # exit at next session open
TRADE_SIZE_USD = 3000.0
MAX_CONCURRENT = 4
LONG_ONLY = True            # set False once Alpaca margin shorting is on


UNIVERSE = [
    # Curated S&P 100 names with frequent overnight news flow
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AMD", "CRM", "ORCL", "NFLX", "ADBE",
    "JPM", "GS", "BAC", "MS", "V", "MA",
    "JNJ", "UNH", "LLY", "ABBV",
    "XOM", "CVX", "WMT",
]


class GapTrading(Strategy):
    name = "gap_trading"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # ── Exits: held > MAX_HOLD_DAYS ───────────────────────────────
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0:
                continue
            entry_time = pos.get("entry_time")
            if not entry_time:
                continue
            try:
                et = (datetime.fromisoformat(entry_time)
                      if isinstance(entry_time, str) else entry_time)
            except (ValueError, TypeError):
                continue
            if datetime.now(UTC) - et > timedelta(days=MAX_HOLD_DAYS):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.95,
                    reason=f"Gap reversion: {MAX_HOLD_DAYS}d hold elapsed",
                    is_closing=True,
                ))

        # ── Entries: gap < -1.5% (long-only fades the gap-down) ──────
        slots_left = max(0, MAX_CONCURRENT - len(held))
        if slots_left <= 0:
            return proposals

        candidates = []
        for sym in UNIVERSE:
            if sym in held:
                continue
            gap_pct = self._overnight_gap_pct(sym)
            if gap_pct is None:
                continue
            if abs(gap_pct) < MIN_GAP_PCT:
                continue
            # Long-only: only fade gap-DOWNS (negative gap → buy bounce)
            if LONG_ONLY and gap_pct >= 0:
                continue
            # Score: more negative gap = stronger bounce signal
            candidates.append((sym, gap_pct, abs(gap_pct)))

        candidates.sort(key=lambda c: c[2], reverse=True)
        for sym, gap_pct, _ in candidates[:slots_left]:
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=TRADE_SIZE_USD, confidence=0.65,
                reason=f"Gap fade: {sym} gapped {gap_pct:+.2f}%, "
                       f"buying the bounce",
                metadata={"gap_pct": gap_pct},
            ))
        return proposals

    def _overnight_gap_pct(self, symbol: str) -> float | None:
        """today_open - yesterday_close as % of yesterday_close."""
        try:
            candles = self.broker.get_candles(symbol, "1Day", num_candles=3)
        except Exception as e:
            logger.debug(f"[{self.name}] {symbol} candles failed: {e}")
            return None
        if len(candles) < 2:
            return None
        yesterday_close = candles[-2].close
        today_open = candles[-1].open
        if yesterday_close <= 0:
            return None
        return (today_open - yesterday_close) / yesterday_close * 100
