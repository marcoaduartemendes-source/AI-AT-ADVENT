"""Live PEAD — earnings momentum on FMP-detected positive surprises.

The PEAD backtest (src/backtests/pead_backtest.py) runs against
historical earnings data from FMP. This is the LIVE counterpart:

  - Each cycle, query FMP's /stable/earnings calendar for any company
    in our watchlist that reported earnings in the last 3 days.
  - If EPS surprise was ≥ +5%, open a long position the next trading
    day at market.
  - Hold for 30 days, then exit at market (matching the backtest rules).

Critical guard: do NOT enter on the same day as the announcement
(after-hours surprises are immediately priced in). We require the
filing_date to be ≥1 trading day in the past so the open price
captures the post-news drift, not the announcement gap.

Position sizing: $4k per trade, max 10 concurrent positions.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, UTC

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


SURPRISE_THRESHOLD_PCT = 5.0
HOLD_DAYS = 30
TRADE_SIZE_USD = 10000.0   # per-position cap raised for paper-trading experimentation
MAX_CONCURRENT_POSITIONS = 10
LOOKBACK_FOR_NEW_FILINGS_DAYS = 3   # look at filings up to 3 days old


UNIVERSE = [
    # Same as the backtest universe — keep them aligned so live
    # numbers are comparable to the backtest projections.
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "ORCL", "CRM", "AMD", "INTC", "QCOM", "ADBE", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO",
    "XOM", "CVX", "BA", "CAT", "GE", "HON", "MMM",
    "WMT", "COST", "HD", "MCD", "NKE", "PG", "KO",
    "DIS", "T", "VZ", "NEE", "DUK",
]


class EarningsMomentum(Strategy):
    name = "earnings_momentum"
    venue = "alpaca"

    def __init__(self, broker):
        super().__init__(broker)
        self._fmp = None
        try:
            from backtests.data.fmp import FMPClient
            client = FMPClient()
            if client.is_configured():
                self._fmp = client
                logger.info(f"[{self.name}] FMP client ready")
        except ImportError:
            pass

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        if self._fmp is None:
            logger.debug(f"[{self.name}] FMP_API_KEY not set — skipping cycle")
            return []

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # ── Exits: holding > HOLD_DAYS ────────────────────────────────
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
            if datetime.now(UTC) - et > timedelta(days=HOLD_DAYS):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.95,
                    reason=f"PEAD {HOLD_DAYS}d hold elapsed",
                    is_closing=True,
                ))

        # ── Entries: earnings surprise ≥ threshold, filed ≥ 1 day ago ─
        slots_left = max(0, MAX_CONCURRENT_POSITIONS - len(held))
        if slots_left <= 0:
            return proposals

        today = date.today()
        # We look at filings from a few days back so we don't miss any
        # late-evening releases on the prior day.
        oldest_eligible = today - timedelta(days=LOOKBACK_FOR_NEW_FILINGS_DAYS)

        candidates = []
        for ticker in UNIVERSE:
            if ticker in held:
                continue
            try:
                earnings = self._fmp.recent_earnings(ticker, limit=4)
            except Exception as e:
                logger.debug(f"[{self.name}] {ticker} earnings fetch: {e}")
                continue
            for er in earnings:
                if er.filing_date is None:
                    continue
                if not (oldest_eligible <= er.filing_date < today):
                    continue
                sp = er.eps_surprise_pct
                if sp is None or sp < SURPRISE_THRESHOLD_PCT:
                    continue
                candidates.append((ticker, sp, er.filing_date))
                break    # only consider the most recent qualifying filing

        # Rank by surprise magnitude — biggest beats first
        candidates.sort(key=lambda c: c[1], reverse=True)
        # Sizing: split allocator's verdict across MAX_CONCURRENT slots,
        # capped by TRADE_SIZE_USD as a per-position max. Avoids
        # blowing through the allocator on a small-account paper run.
        per_slot_alloc = ctx.target_alloc_usd / max(1, MAX_CONCURRENT_POSITIONS)
        per_slot = min(per_slot_alloc, TRADE_SIZE_USD)
        for ticker, surprise, filing in candidates[:slots_left]:
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=ticker,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_slot, confidence=0.8,
                reason=f"PEAD: {ticker} EPS surprise +{surprise:.1f}% "
                       f"on {filing.isoformat()}",
                metadata={"surprise_pct": surprise,
                          "filing_date": filing.isoformat()},
            ))
        return proposals
