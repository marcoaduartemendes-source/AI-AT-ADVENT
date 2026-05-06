"""PEAD — Post-Earnings Announcement Drift on Alpaca equities.

Bernard & Thomas (1989, JAR), Brandt et al. (2008): stock prices
underreact to earnings surprises for ~60 days. We don't have access to
real-time earnings surprise estimates here (would need Sharadar /
Estimize / IBES). V1 takes a simpler form:

  • Use the equities scout's earnings_upcoming list to identify names
    reporting in the next 7 days
  • After the report (next cycle following the earnings date), check the
    1-day post-report price action via Alpaca quotes
  • If gap up > 3% on report day, BUY (positive surprise → drift up)
  • If gap down > 3%, SKIP (we don't short equities in V1)
  • Hold for 30 days then mark-to-close

This is a coarse PEAD proxy. Full version with EPS surprises lands when
we license a fundamentals data feed — flagged in research synthesis.
"""
from __future__ import annotations

import logging
from datetime import datetime, UTC

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Gap threshold to qualify as "positive surprise"
GAP_UP_PCT = 3.0
# Hold period in calendar days
HOLD_DAYS = 30
# Per-position sizing as % of strategy alloc
PER_POSITION_PCT = 0.10
MAX_OPEN_POSITIONS = 8


class PEAD(Strategy):
    name = "pead"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []

        # Close positions held > HOLD_DAYS first (drift window expired)
        now = datetime.now(UTC)
        n_open = 0
        for symbol, pos in ctx.open_positions.items():
            entry_iso = pos.get("entry_time") if isinstance(pos, dict) else None
            qty = pos.get("quantity", 0) if isinstance(pos, dict) else 0
            if qty <= 0:
                continue
            n_open += 1
            try:
                if entry_iso:
                    entry_dt = datetime.fromisoformat(entry_iso.replace("Z", "+00:00"))
                    if (now - entry_dt).days >= HOLD_DAYS:
                        proposals.append(TradeProposal(
                            strategy=self.name, venue=self.venue, symbol=symbol,
                            side=OrderSide.SELL, order_type=OrderType.MARKET,
                            quantity=qty, confidence=0.95, is_closing=True,
                            reason=f"PEAD drift window ({HOLD_DAYS}d) elapsed",
                        ))
                        n_open -= 1
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"[pead] entry-date parse failed for "
                             f"{symbol}: {type(e).__name__}: {e}")

        # Earnings-driven entries
        earnings = ctx.scout_signals.get("earnings_upcoming", []) or []
        if not earnings:
            return proposals

        per_position_usd = ctx.target_alloc_usd * PER_POSITION_PCT
        room = max(0, MAX_OPEN_POSITIONS - n_open)

        # For each upcoming-earnings ticker, check if the gap qualifies
        for entry in earnings[:room * 3]:  # examine 3x our slot count
            symbol = entry.get("symbol")
            if not symbol:
                continue
            if symbol in ctx.open_positions and ctx.open_positions[symbol].get("quantity", 0) > 0:
                continue
            try:
                candles = self.broker.get_candles(symbol, "ONE_DAY", num_candles=5)
            except Exception as e:
                logger.debug(f"[pead] get_candles({symbol}) failed: "
                             f"{type(e).__name__}: {e}")
                continue
            if len(candles) < 2:
                continue
            prev = candles[-2].close
            today = candles[-1].close
            if prev <= 0:
                continue
            gap_pct = (today - prev) / prev * 100
            if gap_pct < GAP_UP_PCT:
                continue

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_position_usd,
                confidence=min(0.85, 0.5 + gap_pct / 20),
                reason=(f"PEAD entry: gap +{gap_pct:.1f}% post-earnings, "
                        f"hold {HOLD_DAYS}d"),
                metadata={"gap_pct": gap_pct,
                          "earnings_date": entry.get("date")},
            ))
            room -= 1
            if room <= 0:
                break

        return proposals
