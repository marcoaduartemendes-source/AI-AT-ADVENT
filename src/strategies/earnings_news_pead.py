"""earnings_news_pead — PEAD trades, gated on news corroboration.

Difference from `pead` (v1):
    v1 fires on price gap >3% the day after earnings (proxy for
    positive surprise). It has no way to distinguish:
      • Beat-driven gap (price + headlines all align → real PEAD drift)
      • Macro-driven gap (Fed surprise / CPI day overlapped earnings →
        the gap was systemic, not idiosyncratic, no drift to capture)
      • False-print gap (data error, halted stock, dividend ex-date)

    v2 requires CORROBORATING NEWS HEADLINES on the same ticker
    within the last 24 hours. The new equities_scout publishes
    `ticker_news` signals via the Sprint-3 RSS aggregator (Reuters /
    MarketWatch / Yahoo / SEC EDGAR). When v1's price-gap signal AND
    a same-ticker headline both fire, the gap is much more likely to
    be a real earnings reaction.

Edge:
    The gating cuts false-positive rate by ~30-40% in expectation
    (number is theoretical until paper-trade calibration). Net effect
    on Sharpe: positive — fewer trades, higher hit rate.

Same downstream mechanics as v1: 3% gap-up → BUY, 30-day hold,
target_alloc_usd × PER_POSITION_PCT per name, max MAX_OPEN_POSITIONS.

Failure modes:
    - ticker_news scout signal absent (e.g. RSS feeds rate-limited
      or down) → v2 emits no proposals. v1 keeps running and will
      fire on its own (slightly noisier) signal. We don't want v2
      to fall back to v1's behaviour silently — that defeats the
      gating purpose.
    - Tickers with no news → skip (correctly; means no public
      catalyst confirmed)
    - Same-ticker as v1 → both can fire. Risk gate dedupes via the
      pending-orders cache.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# Same as v1 — only the news gate changes.
GAP_UP_PCT = 3.0
HOLD_DAYS = 30
PER_POSITION_PCT = 0.10
MAX_OPEN_POSITIONS = 6    # smaller than v1 (8) — v2 is more selective

# Minimum headline count to confirm a news catalyst. 1 is enough
# (any mention) — we don't want to require sentiment analysis we
# don't have.
MIN_HEADLINES = 1


class EarningsNewsPEAD(Strategy):
    name = "earnings_news_pead"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        now = datetime.now(UTC)

        # ── Exits: drift window expired
        n_open = 0
        for symbol, pos in ctx.open_positions.items():
            entry_iso = (pos.get("entry_time")
                         if isinstance(pos, dict) else None)
            qty = pos.get("quantity", 0) if isinstance(pos, dict) else 0
            if qty <= 0:
                continue
            n_open += 1
            if entry_iso:
                try:
                    et = datetime.fromisoformat(
                        entry_iso.replace("Z", "+00:00")
                    )
                    if (now - et).days >= HOLD_DAYS:
                        proposals.append(TradeProposal(
                            strategy=self.name, venue=self.venue,
                            symbol=symbol, side=OrderSide.SELL,
                            order_type=OrderType.MARKET, quantity=qty,
                            confidence=0.95, is_closing=True,
                            reason=f"PEAD drift window ({HOLD_DAYS}d) elapsed",
                        ))
                        n_open -= 1
                except (ValueError, TypeError):
                    pass

        # ── Entries: earnings_upcoming + ticker_news intersection
        earnings = ctx.scout_signals.get("earnings_upcoming", []) or []
        if not earnings:
            return proposals

        # Scout publishes ticker_news as list-of-dict
        # [{"symbol": "AAPL", "n_headlines": 3, "headlines": [...]}, …]
        ticker_news_raw = ctx.scout_signals.get("ticker_news", []) or []
        news_by_ticker: dict[str, dict] = {
            row.get("symbol"): row
            for row in ticker_news_raw if isinstance(row, dict)
        }
        if not news_by_ticker:
            logger.debug(f"[{self.name}] no ticker_news — gating closed")
            return proposals

        per_position_usd = ctx.target_alloc_usd * PER_POSITION_PCT
        room = max(0, MAX_OPEN_POSITIONS - n_open)
        if room <= 0:
            return proposals

        for e in earnings:
            if room <= 0:
                break
            symbol = e.get("symbol")
            if symbol is None or symbol in ctx.open_positions:
                continue
            news_row = news_by_ticker.get(symbol)
            if not news_row:
                continue
            n_headlines = news_row.get("n_headlines", 0)
            if n_headlines < MIN_HEADLINES:
                continue

            # Gap check via broker (same as v1). We don't have access
            # to the live price here without a broker call — defer to
            # the broker's recent candles.
            try:
                bars = self.broker.get_candles(symbol, "1Day", num_candles=2)
            except Exception as exc:    # noqa: BLE001
                logger.debug(f"[{self.name}] {symbol} candles failed: {exc}")
                continue
            if len(bars) < 2:
                continue
            prev_close = bars[-2].close
            today_close = bars[-1].close
            if prev_close <= 0:
                continue
            gap_pct = (today_close - prev_close) / prev_close * 100
            if gap_pct < GAP_UP_PCT:
                continue

            # All gates passed: earnings + news + gap up
            qty = per_position_usd / today_close
            if qty <= 0:
                continue

            top_headlines = news_row.get("headlines", [])[:3]
            headline_summary = "; ".join(
                h.get("title", "")[:80] for h in top_headlines
            )

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=symbol,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_position_usd, confidence=0.85,
                reason=(f"PEAD gap +{gap_pct:.1f}% confirmed by "
                        f"{n_headlines} headline(s)"),
                metadata={
                    "gap_pct": gap_pct,
                    "n_headlines": n_headlines,
                    "headlines_preview": headline_summary[:300],
                    "earnings_date": e.get("date"),
                },
            ))
            room -= 1

        return proposals
