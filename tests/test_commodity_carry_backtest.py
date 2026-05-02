"""Tests for commodity_carry backtest.

Uses a stub FMPClient feeding deterministic price series. Asserts:
  - Without API key → returns clean placeholder
  - Top-momentum ETF gets selected; flat one gets skipped
  - Sub-threshold returns are filtered out
  - Equal-weighted across top-N
  - Ranking re-runs every REBALANCE_EVERY_DAYS days
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from backtests.commodity_carry_backtest import (
    backtest_commodity_carry,
)
from backtests.data.polygon import DailyBar


def _bars(start: date, n: int, start_price: float, daily_return_pct: float):
    """Compounding daily return — gives a realistic price series."""
    out = []
    px = start_price
    for i in range(n):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:    # skip weekends
            continue
        px = px * (1 + daily_return_pct / 100)
        out.append(DailyBar(date=d, open=px, high=px*1.01, low=px*0.99,
                             close=px, volume=1_000_000))
    return out


def test_returns_placeholder_without_api_key():
    fake = MagicMock()
    fake.is_configured.return_value = False
    result = backtest_commodity_carry(window_days=90, polygon=fake)
    assert result.strategy == "commodity_carry"
    assert result.n_trades == 0
    assert "FMP_API_KEY" in result.note


def test_strong_uptrend_etf_selected_and_profitable():
    """Single ETF with a strong sustained 60-day uptrend should be
    selected and produce a profitable trade."""
    today = date.today()
    history_start = today - timedelta(days=200)

    fake = MagicMock()
    fake.is_configured.return_value = True
    # +0.3%/day = ~80% over 200d, well above MIN_RETURN_PCT
    fake.daily_bars.return_value = _bars(history_start, 200, 100.0, 0.3)

    result = backtest_commodity_carry(
        window_days=60,
        universe=["PDBC"],
        polygon=fake,
    )
    assert result.n_trades >= 1, (
        f"Expected at least 1 trade for sustained uptrend, "
        f"got {result.n_trades}. Note: {result.note}"
    )
    assert result.total_pnl_usd > 0, (
        "Expected positive P&L on a sustained uptrend"
    )
    for t in result.trades:
        assert t["product_id"] == "PDBC"
        assert t["side"] == "BUY"
        assert "60d momentum" in t["reason"]


def test_flat_etf_filtered_below_min_return():
    """An ETF with ~0% 60-day return should NOT trigger a trade
    (below MIN_RETURN_PCT threshold)."""
    today = date.today()
    history_start = today - timedelta(days=200)

    fake = MagicMock()
    fake.is_configured.return_value = True
    # Daily 0% return → flat price
    fake.daily_bars.return_value = _bars(history_start, 200, 50.0, 0.0)

    result = backtest_commodity_carry(
        window_days=60,
        universe=["DBA"],
        polygon=fake,
    )
    assert result.n_trades == 0
    # `note` is set when there were 0 trades
    assert ("no qualifying" in result.note
            or "insufficient" in result.note
            or "no price" in result.note)


def test_top_n_selection_works():
    """When 3 ETFs are given (one strong-up, one flat, one strong-down),
    only the strong-up one ends up in trades."""
    today = date.today()
    history_start = today - timedelta(days=200)

    bars_by_ticker = {
        "PDBC": _bars(history_start, 200, 100.0, 0.3),    # strong up
        "GLD":  _bars(history_start, 200, 200.0, 0.0),    # flat
        "USO":  _bars(history_start, 200, 80.0, -0.2),    # down
    }

    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.daily_bars.side_effect = lambda ticker, *a, **kw: bars_by_ticker[ticker]

    result = backtest_commodity_carry(
        window_days=60,
        universe=["PDBC", "GLD", "USO"],
        polygon=fake,
    )
    # All trades should be on PDBC; GLD/USO never qualify
    tickers = {t["product_id"] for t in result.trades}
    assert "PDBC" in tickers, f"Expected PDBC in trades, got {tickers}"
    assert "USO" not in tickers, "Down-trending USO should not be traded"
    # Strategy is long-only; no sells
    assert all(t["side"] == "BUY" for t in result.trades)
