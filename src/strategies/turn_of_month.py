"""Turn-of-month seasonal strategy.

Documented anomaly (Lakonishok-Smidt 1988, Etula 2014, multiple CME
papers): equity returns are systematically higher in the last 3 trading
days of the month + first 4 of the next, attributable to:
  - 401(k) inflows hit on payroll dates (last business day)
  - Pension rebalancing concentrated near month-end
  - Window-dressing ahead of monthly statements

The effect is robust out-of-sample for 100+ years on US equities. It
won't be huge in any one trade, but it's UNCORRELATED to the rest of
the book's strategies (momentum, mean-reversion, sector rotation),
making it diversifying capital.

Implementation:
  - Compute today's "month phase" from calendar:
      days_to_month_end = trading days from today to last day of month
      days_from_month_start = trading days since first day of NEXT month
  - LONG SPY when day in window [last 3 of month, first 4 of next]
  - Hold for the remaining days of that window; close on day-after.

This is a long-only equity overlay sleeve. Sizing: full $4k per
window — only one position at a time; ~4-7 days hold; ~12 trades/yr.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


WINDOW_DAYS_BEFORE_MONTH_END = 3   # buy at day -3 from end of month
WINDOW_DAYS_AFTER_MONTH_START = 4  # exit at day +4 of next month
TRADE_SIZE_USD = 10000.0   # per-position cap raised for paper-trading experimentation


# Symbols to use for the seasonal effect. SPY is the highest-coverage
# US equity proxy. We could expand to a basket (SPY/QQQ/IWM) for
# slightly more diversification, but added complexity isn't worth it
# for a calendar effect that has the same direction across all of US
# equities.
SEASONAL_SYMBOL = "SPY"


class TurnOfMonth(Strategy):
    name = "turn_of_month"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        in_window = self._in_seasonal_window()
        held_qty = ctx.open_positions.get(SEASONAL_SYMBOL, {}).get("quantity", 0)

        # Exit: outside the window AND we hold the position
        if not in_window and held_qty > 0:
            return [TradeProposal(
                strategy=self.name, venue=self.venue, symbol=SEASONAL_SYMBOL,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=held_qty, confidence=0.95,
                reason="Outside turn-of-month seasonal window",
                is_closing=True,
            )]

        # Entry: inside window AND we don't already hold.
        # Sizing: a single-position strategy → use the full allocator
        # verdict, capped by TRADE_SIZE_USD as a safety max.
        if in_window and held_qty == 0:
            entry_usd = min(ctx.target_alloc_usd, TRADE_SIZE_USD)
            return [TradeProposal(
                strategy=self.name, venue=self.venue, symbol=SEASONAL_SYMBOL,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=entry_usd, confidence=0.6,
                reason=f"Turn-of-month window ({SEASONAL_SYMBOL})",
                metadata={"phase": self._month_phase_str()},
            )]

        return []

    def _in_seasonal_window(self) -> bool:
        """True if today is within the last N or first M days of a
        calendar month. Uses approximate day-of-month math (doesn't
        skip weekends), which over a year averages out to the same
        4-7 trading day window. Cheap + good enough."""
        today = date.today()
        # Last day of this month
        if today.month == 12:
            next_month = date(today.year + 1, 1, 1)
        else:
            next_month = date(today.year, today.month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day

        # In the last N days of this month?
        days_until_eom = last_day - today.day
        if days_until_eom <= WINDOW_DAYS_BEFORE_MONTH_END:
            return True

        # In the first M days of this month?
        if today.day <= WINDOW_DAYS_AFTER_MONTH_START:
            return True

        return False

    def _month_phase_str(self) -> str:
        today = date.today()
        if today.month == 12:
            next_month = date(today.year + 1, 1, 1)
        else:
            next_month = date(today.year, today.month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        days_until_eom = last_day - today.day
        if days_until_eom <= WINDOW_DAYS_BEFORE_MONTH_END:
            return f"D-{days_until_eom + 1} from EOM"
        return f"D+{today.day} from MOM"
