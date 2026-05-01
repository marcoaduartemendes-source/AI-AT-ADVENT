"""PEAD (Post-Earnings Announcement Drift) backtest using Polygon data.

Strategy logic (mirrors src/strategies/pead.py):
  1. For each scheduled earnings report in the lookback window, compute
     `eps_surprise = (actual − estimate) / |estimate|`.
  2. Filter to surprises ≥ +5% (positive PEAD only — symmetric is
     possible but the long-only bias keeps things tractable for the
     paper-trading account).
  3. Open a long position at the OPEN of the next trading day after
     `filing_date` (no look-ahead).
  4. Hold for HOLD_DAYS = 30 calendar days, then close at OPEN.
  5. Position size = $5000 per trade (paper sizing; live uses
     allocator).

Universe: a curated mid/large-cap list. Polygon's quarterly financials
work for any US ticker, but to keep API calls reasonable we restrict
to a ~50-name watchlist that covers earnings every quarter.

Data requirement: POLYGON_API_KEY env var must be set. Without it the
backtest returns BacktestSummary(note="POLYGON_API_KEY not set").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .data.polygon import DailyBar, EarningsRecord, PolygonClient, PolygonError
from .runner import BacktestSummary, _FEE_RATE

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


SURPRISE_THRESHOLD_PCT = 5.0      # |EPS surprise| ≥ this to enter
HOLD_DAYS = 30                    # calendar-day holding period
TRADE_SIZE_USD = 5000.0           # per-trade notional (backtest sizing)
LONG_ONLY = True                  # positive surprises only — set False
                                  # to also short negative surprises


# Curated S&P 500 / Nasdaq 100 watchlist. Limited to ~50 names so a
# 90-day backtest hits ≤ 50 financials calls, well within the Polygon
# Starter plan's monthly budget.
PEAD_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    # Other tech / consumer
    "ORCL", "CRM", "AMD", "INTC", "QCOM", "ADBE", "NFLX",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO",
    # Industrials / Energy / Materials
    "XOM", "CVX", "BA", "CAT", "GE", "HON", "MMM",
    # Consumer staples / discretionary
    "WMT", "COST", "HD", "MCD", "NKE", "PG", "KO",
    # Communications / utilities
    "DIS", "T", "VZ", "NEE", "DUK",
]


# ─── Trade representation ─────────────────────────────────────────────


@dataclass
class _PEADTrade:
    ticker: str
    surprise_pct: float
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    notional_usd: float
    quantity: float

    @property
    def gross_return_pct(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def gross_pnl_usd(self) -> float:
        return self.gross_return_pct / 100 * self.notional_usd

    @property
    def net_pnl_usd(self) -> float:
        # Round-trip fee on the entry + exit notional (simplified)
        fee = self.notional_usd * _FEE_RATE * 2
        return self.gross_pnl_usd - fee


# ─── Core backtest ────────────────────────────────────────────────────


def backtest_pead(
    window_days: int,
    *,
    universe: list[str] | None = None,
    polygon: PolygonClient | None = None,
) -> BacktestSummary:
    """Run the PEAD backtest over the last `window_days` calendar days.

    Parameters
    ----------
    window_days : int
        Lookback in calendar days. Earnings reports filed within this
        window become candidate entries.
    universe : list[str] | None
        Tickers to scan. Default: PEAD_UNIVERSE.
    polygon : PolygonClient | None
        Inject a pre-built client (useful for tests). Otherwise creates
        a default client which reads POLYGON_API_KEY from env.

    Returns
    -------
    BacktestSummary
        Same shape the dashboard already consumes. `note` contains the
        skip reason if the API key isn't configured.
    """
    polygon = polygon or PolygonClient()
    if not polygon.is_configured():
        return BacktestSummary(
            strategy="pead",
            window_days=window_days,
            note="POLYGON_API_KEY not set — set the GH secret to enable",
        )

    universe = universe or PEAD_UNIVERSE
    today = date.today()
    window_start = today - timedelta(days=window_days)

    # We need price history to cover entry through exit. The exit is
    # HOLD_DAYS after entry, so we fetch through `today + HOLD_DAYS`
    # in calendar days — but Polygon clamps to the last trading day,
    # so future-dated requests just return what's available.
    price_lookback_start = window_start - timedelta(days=10)
    price_lookback_end = today

    trades: list[_PEADTrade] = []
    skipped: dict[str, int] = {"no_estimate": 0, "small_surprise": 0,
                                "no_prices": 0, "still_open": 0,
                                "fetch_error": 0}

    for ticker in universe:
        try:
            earnings = polygon.recent_earnings(ticker, limit=8)
        except PolygonError as e:
            logger.warning(f"pead: earnings fetch failed for {ticker}: {e}")
            skipped["fetch_error"] += 1
            continue

        # Filter to filings inside our window with usable surprise data
        candidates = [
            e for e in earnings
            if e.filing_date is not None
            and window_start <= e.filing_date <= today
            and e.eps_surprise_pct is not None
        ]
        if not candidates:
            continue

        try:
            bars = polygon.daily_bars(
                ticker, price_lookback_start, price_lookback_end,
            )
        except PolygonError as e:
            logger.warning(f"pead: bars fetch failed for {ticker}: {e}")
            skipped["fetch_error"] += 1
            continue
        if not bars:
            skipped["no_prices"] += 1
            continue

        bars_by_date = {b.date: b for b in bars}
        sorted_dates = sorted(bars_by_date.keys())

        for er in candidates:
            sp = er.eps_surprise_pct
            assert sp is not None      # filtered above

            if abs(sp) < SURPRISE_THRESHOLD_PCT:
                skipped["small_surprise"] += 1
                continue
            if LONG_ONLY and sp < 0:
                continue

            entry_date = _next_trading_day_on_or_after(
                er.filing_date + timedelta(days=1), sorted_dates,
            )
            if entry_date is None:
                skipped["still_open"] += 1
                continue
            target_exit = entry_date + timedelta(days=HOLD_DAYS)
            exit_date = _next_trading_day_on_or_after(target_exit, sorted_dates)
            if exit_date is None:
                skipped["still_open"] += 1
                continue

            entry_bar = bars_by_date[entry_date]
            exit_bar = bars_by_date[exit_date]
            qty = TRADE_SIZE_USD / entry_bar.open
            trades.append(_PEADTrade(
                ticker=ticker,
                surprise_pct=sp,
                entry_date=entry_date,
                entry_price=entry_bar.open,
                exit_date=exit_date,
                exit_price=exit_bar.open,
                notional_usd=TRADE_SIZE_USD,
                quantity=qty,
            ))

    if skipped:
        logger.info(f"pead {window_days}d backtest: {skipped}")

    return _summarize(trades, window_days)


# ─── Helpers ──────────────────────────────────────────────────────────


def _next_trading_day_on_or_after(d: date, available: list[date]) -> date | None:
    """Given a target date and a sorted list of trading days, return
    the earliest available date >= `d`, or None if we ran out of bars."""
    for ad in available:
        if ad >= d:
            return ad
    return None


def _summarize(trades: list[_PEADTrade], window_days: int) -> BacktestSummary:
    """Build the BacktestSummary the dashboard reads."""
    if not trades:
        return BacktestSummary(
            strategy="pead",
            window_days=window_days,
            note="no qualifying earnings surprises in window",
        )

    pnls = [t.net_pnl_usd for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    losses = n - wins
    entry_vol = sum(t.notional_usd for t in trades)

    sharpe = _sharpe(pnls)
    max_dd = _max_drawdown(pnls)

    trade_dicts = [
        {
            "open_time": t.entry_date.isoformat(),
            "close_time": t.exit_date.isoformat(),
            "product_id": t.ticker,
            "side": "BUY",
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "amount_usd": t.notional_usd,
            "pnl_usd": t.net_pnl_usd,
            "reason": f"PEAD surprise +{t.surprise_pct:.1f}%",
            "exit_reason": f"{HOLD_DAYS}d hold elapsed",
        }
        for t in trades
    ]

    equity_curve: list[dict] = []
    cum = 0.0
    for t, pnl in zip(sorted(trades, key=lambda x: x.exit_date), pnls):
        cum += pnl
        equity_curve.append({
            "t": t.exit_date.isoformat(), "pnl_cumulative": cum,
        })

    return BacktestSummary(
        strategy="pead",
        window_days=window_days,
        n_trades=n,
        n_wins=wins,
        n_losses=losses,
        win_rate=wins / n if n else 0.0,
        total_pnl_usd=total_pnl,
        entry_volume_usd=entry_vol,
        return_on_volume_pct=(total_pnl / entry_vol * 100) if entry_vol > 0 else 0.0,
        avg_pnl_usd=total_pnl / n if n else 0.0,
        sharpe=sharpe,
        max_drawdown_usd=max_dd,
        trades=trade_dicts,
        equity_curve=equity_curve,
    )


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    sd = var ** 0.5
    if sd == 0:
        return None
    # Per-trade Sharpe scaled by sqrt(n) (informal — these are
    # event-driven trades, not periodic returns; the dashboard treats
    # this as a relative ranking metric, not an annualized number)
    return mean / sd * (len(pnls) ** 0.5)


def _max_drawdown(pnls: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    return max_dd
