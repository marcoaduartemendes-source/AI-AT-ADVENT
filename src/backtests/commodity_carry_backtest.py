"""Commodity carry / term-structure backtest using ETF price proxies.

True backtest of commodity_carry would require historical futures-curve
snapshots (front + deferred contract prices for each commodity). Those
aren't free — CME data is paid. We use a **proxy backtest** that's
honest about the approximation:

  Long the top-N commodity ETFs ranked by 60-day total return, hold 30
  days, exit on momentum reversal. Returns from individual commodity
  ETFs absorb both spot price changes AND roll yield, so a positive
  60-day return is a good proxy for "this commodity has been in
  backwardation (or had spot rally) and the carry/momentum is working."

This is conservatively similar to the live commodity_carry strategy
which long-onlys the top-N most-backwardated commodities ranked by
annualized roll yield. The main differences:

  - Live strategy reads live futures curve; backtest uses 60-day return
  - Live strategy weights by carry%; backtest equal-weights top-N
  - Live trades futures contracts; backtest uses sector ETFs

The numbers won't match live performance exactly. They give an honest
range-of-outcomes for a momentum-flavored commodity carry approach.

Universe selection (sector ETFs, available on FMP at any tier):
  Energy:    USO (oil), UNG (nat gas), UGA (gasoline)
  Metals:    GLD (gold), SLV (silver), CPER (copper)
  Agri:      DBA (broad ag), CORN, WEAT, SOYB
  Broad:     PDBC (roll-yield-optimized broad commodities — the
                   purest proxy for the live strategy)
             DBC (rules-based broad commodities)

PDBC is intentionally first because its return literally embeds
the roll-yield decision the live strategy is trying to replicate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .data.fmp import FMPError
from .data.polygon import DailyBar, PolygonError
from .runner import BacktestSummary, _FEE_RATE

logger = logging.getLogger(__name__)

_DataError = (PolygonError, FMPError)


# ─── Tunables ─────────────────────────────────────────────────────────


LOOKBACK_DAYS = 60          # ranking window
HOLD_DAYS = 30              # how long to hold each pick
TOP_N = 2                   # long the top 2 by ranking momentum
TRADE_SIZE_USD = 5000.0     # per-leg notional
REBALANCE_EVERY_DAYS = 30   # re-rank + reshuffle this often
MIN_RETURN_PCT = 2.0        # don't enter unless 60-day > 2%


COMMODITY_UNIVERSE = [
    # Roll-yield-optimized broad commodity ETFs (the closest single
    # proxy for the strategy's actual edge):
    "PDBC",   # Invesco Optimum Yield Diversified Commodity
    "DBC",    # Invesco DB Commodity
    # Energy:
    "USO",    # United States Oil Fund
    "UNG",    # United States Natural Gas
    "UGA",    # United States Gasoline
    # Metals:
    "GLD",    # SPDR Gold Trust
    "SLV",    # iShares Silver Trust
    "CPER",   # United States Copper Index Fund
    # Agriculture:
    "DBA",    # Invesco DB Agriculture
    "CORN",   # Teucrium Corn
    "WEAT",   # Teucrium Wheat
    "SOYB",   # Teucrium Soybean
]


# ─── Trade ────────────────────────────────────────────────────────────


@dataclass
class _CarryTrade:
    ticker: str
    rank_return_pct: float        # 60-day return at entry
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
        fee = self.notional_usd * _FEE_RATE * 2
        return self.gross_pnl_usd - fee


# ─── Core backtest ────────────────────────────────────────────────────


def backtest_commodity_carry(
    window_days: int,
    *,
    universe: list[str] | None = None,
    polygon=None,
) -> BacktestSummary:
    """Run the commodity-carry proxy backtest over the last
    `window_days` calendar days.

    Algorithm:
      1. Load LOOKBACK_DAYS+window_days+HOLD_DAYS of price history
         for each ETF.
      2. Walking the time series at REBALANCE_EVERY_DAYS cadence:
         a. For each ETF, compute trailing-LOOKBACK return.
         b. Rank descending. Pick top-N where return ≥ MIN_RETURN_PCT.
         c. Open positions in those (or hold from previous cycle if
            still in top-N).
         d. Close positions that dropped out of top-N.
      3. Returns a BacktestSummary with the round-trip P&L.
    """
    from .data.fmp import get_data_client
    client = polygon if polygon is not None else get_data_client()
    if not client.is_configured():
        return BacktestSummary(
            strategy="commodity_carry",
            window_days=window_days,
            note=("FMP_API_KEY (or POLYGON_API_KEY) not set — "
                  "set the GH secret to enable"),
        )

    universe = universe or COMMODITY_UNIVERSE
    today = date.today()
    history_start = today - timedelta(days=window_days + LOOKBACK_DAYS + 10)

    # Fetch all symbols up-front (FMP handles 12 ETFs in <1 sec each
    # with disk cache; even the cold case is well within Starter plan
    # rate limit at 300 req/min).
    bars_by_ticker: dict[str, list[DailyBar]] = {}
    for ticker in universe:
        try:
            bars = client.daily_bars(ticker, history_start, today)
        except _DataError as e:
            logger.warning(f"commodity_carry: bars fetch {ticker} failed: {e}")
            continue
        if len(bars) < LOOKBACK_DAYS + 5:
            logger.debug(
                f"commodity_carry: {ticker} only {len(bars)} bars — skipping",
            )
            continue
        bars_by_ticker[ticker] = bars

    if not bars_by_ticker:
        return BacktestSummary(
            strategy="commodity_carry",
            window_days=window_days,
            note="no price data fetched (rate limit? check FMP plan)",
        )

    # Build a master sorted list of trading days that ALL tickers have
    common_dates = sorted(set.intersection(
        *[{b.date for b in bars} for bars in bars_by_ticker.values()]
    ))
    if len(common_dates) < LOOKBACK_DAYS + HOLD_DAYS:
        return BacktestSummary(
            strategy="commodity_carry",
            window_days=window_days,
            note=f"insufficient overlapping trading days "
                  f"({len(common_dates)} < {LOOKBACK_DAYS + HOLD_DAYS})",
        )

    # Index bars by (ticker, date) for fast lookup
    bar_idx: dict[tuple[str, date], DailyBar] = {
        (t, b.date): b
        for t, bars in bars_by_ticker.items()
        for b in bars
    }

    # Walk forward: every REBALANCE_EVERY_DAYS days, re-rank + open/close
    trades: list[_CarryTrade] = []
    open_positions: dict[str, _CarryTrade] = {}
    backtest_start = today - timedelta(days=window_days)
    rebal_idx = next(
        (i for i, d in enumerate(common_dates) if d >= backtest_start),
        len(common_dates) - 1,
    )

    while rebal_idx < len(common_dates) - 1:
        ranking_date = common_dates[rebal_idx]

        # Compute trailing-LOOKBACK return for each ETF available on
        # the ranking_date.
        rankings = []
        for ticker, bars in bars_by_ticker.items():
            cur_bar = bar_idx.get((ticker, ranking_date))
            past_idx = rebal_idx - LOOKBACK_DAYS
            if past_idx < 0:
                continue
            past_date = common_dates[past_idx]
            past_bar = bar_idx.get((ticker, past_date))
            if not cur_bar or not past_bar or past_bar.close <= 0:
                continue
            ret_pct = (cur_bar.close - past_bar.close) / past_bar.close * 100
            rankings.append((ticker, ret_pct))

        rankings.sort(key=lambda x: x[1], reverse=True)
        target_set = {t for t, r in rankings[:TOP_N] if r >= MIN_RETURN_PCT}

        # Close positions no longer in target_set
        next_idx = min(rebal_idx + REBALANCE_EVERY_DAYS, len(common_dates) - 1)
        exit_date = common_dates[next_idx]
        for ticker in list(open_positions.keys()):
            if ticker in target_set:
                continue
            entry_trade = open_positions.pop(ticker)
            exit_bar = bar_idx.get((ticker, exit_date))
            if not exit_bar:
                continue
            entry_trade.exit_date = exit_date
            entry_trade.exit_price = exit_bar.close
            trades.append(entry_trade)

        # Open new positions
        for ticker in target_set:
            if ticker in open_positions:
                continue
            entry_bar = bar_idx.get((ticker, ranking_date))
            if not entry_bar:
                continue
            ret_pct = next(r for t, r in rankings if t == ticker)
            qty = TRADE_SIZE_USD / entry_bar.close
            open_positions[ticker] = _CarryTrade(
                ticker=ticker,
                rank_return_pct=ret_pct,
                entry_date=ranking_date,
                entry_price=entry_bar.close,
                exit_date=ranking_date,    # placeholder; updated on close
                exit_price=entry_bar.close,
                notional_usd=TRADE_SIZE_USD,
                quantity=qty,
            )

        rebal_idx = next_idx

    # Close any still-open positions at the last available bar
    final_date = common_dates[-1]
    for ticker, t in open_positions.items():
        exit_bar = bar_idx.get((ticker, final_date))
        if exit_bar:
            t.exit_date = final_date
            t.exit_price = exit_bar.close
            trades.append(t)

    return _summarize(trades, window_days)


# ─── Summary builder ──────────────────────────────────────────────────


def _summarize(trades: list[_CarryTrade], window_days: int) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            strategy="commodity_carry",
            window_days=window_days,
            note="no qualifying commodities passed momentum filter",
        )

    pnls = [t.net_pnl_usd for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
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
            "reason": f"60d momentum +{t.rank_return_pct:.1f}% (top-{TOP_N})",
            "exit_reason": f"{REBALANCE_EVERY_DAYS}d rebalance",
        }
        for t in trades
    ]

    eq_curve: list[dict] = []
    cum = 0.0
    for t, pnl in zip(sorted(trades, key=lambda x: x.exit_date),
                       pnls, strict=False):
        cum += pnl
        eq_curve.append({"t": t.exit_date.isoformat(), "pnl_cumulative": cum})

    return BacktestSummary(
        strategy="commodity_carry",
        window_days=window_days,
        n_trades=n,
        n_wins=wins,
        n_losses=n - wins,
        win_rate=wins / n if n else 0.0,
        total_pnl_usd=total_pnl,
        entry_volume_usd=entry_vol,
        return_on_volume_pct=(total_pnl / entry_vol * 100) if entry_vol > 0 else 0.0,
        avg_pnl_usd=total_pnl / n if n else 0.0,
        sharpe=sharpe,
        max_drawdown_usd=max_dd,
        trades=trade_dicts,
        equity_curve=eq_curve,
        note="proxy backtest using sector-ETF momentum — actual live "
             "strategy uses futures-curve carry which we don't have free data for",
    )


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    sd = var ** 0.5
    if sd == 0:
        return None
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
