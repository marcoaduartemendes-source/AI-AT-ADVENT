"""Tests for the PEAD backtest.

Uses a stub PolygonClient to feed deterministic earnings + price data
without hitting the network. Asserts:
  - Without an API key, returns a clean placeholder summary.
  - With a positive surprise + price uptrend, returns a profitable trade.
  - With a sub-threshold surprise, no trade is opened.
  - Look-ahead is respected: entry is the day AFTER filing.
"""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

from backtests.data.polygon import DailyBar, EarningsRecord
from backtests.pead_backtest import backtest_pead


def _make_bars(start: date, n_days: int, start_price: float, drift: float):
    """Build a sequence of DailyBars with linear drift."""
    out = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        # Skip weekends (Sat=5, Sun=6) to mimic real trading sessions
        if d.weekday() >= 5:
            continue
        px = start_price + drift * i
        out.append(DailyBar(date=d, open=px, high=px*1.01, low=px*0.99,
                             close=px, volume=1_000_000))
    return out


def test_pead_returns_placeholder_when_no_api_key():
    """Without POLYGON_API_KEY, the backtest skips gracefully — must
    not raise, must return a BacktestSummary with the skip reason."""
    fake_client = MagicMock()
    fake_client.is_configured.return_value = False

    result = backtest_pead(window_days=30, polygon=fake_client)
    assert result.strategy == "pead"
    assert result.n_trades == 0
    assert "POLYGON_API_KEY" in result.note


def test_pead_opens_trade_on_positive_surprise():
    """+10% EPS surprise on AAPL → long entry at next day's open,
    exit 30d later. With a simulated +5% price drift, P&L is positive."""
    today = date.today()
    filing = today - timedelta(days=40)
    entry_target = filing + timedelta(days=1)

    fake_client = MagicMock()
    fake_client.is_configured.return_value = True
    fake_client.recent_earnings.return_value = [
        EarningsRecord(
            ticker="AAPL",
            period_end=filing - timedelta(days=10),
            filing_date=filing,
            eps_actual=2.20, eps_estimate=2.00,    # +10% surprise
            revenue_actual=None, revenue_estimate=None,
            raw={},
        ),
    ]
    # 60 days of bars: drift +1% per session for visibility
    fake_client.daily_bars.return_value = _make_bars(
        start=entry_target - timedelta(days=5),
        n_days=60,
        start_price=200.0,
        drift=2.0,    # ~$2/day = +30% over 30 sessions; sharply profitable
    )

    result = backtest_pead(
        window_days=60,
        universe=["AAPL"],
        polygon=fake_client,
    )
    assert result.n_trades == 1, (
        f"Expected 1 trade for +10% AAPL surprise, got {result.n_trades}. "
        f"Note: {result.note}"
    )
    assert result.total_pnl_usd > 0, (
        f"Expected positive P&L on uptrend; got ${result.total_pnl_usd:+.2f}"
    )
    t = result.trades[0]
    # Entry should be on or after filing+1, NOT on/before filing
    assert t["open_time"] >= (filing + timedelta(days=1)).isoformat(), (
        "PEAD must not trade BEFORE filing date — look-ahead bias!"
    )
    assert t["product_id"] == "AAPL"
    assert t["side"] == "BUY"
    assert "PEAD" in t["reason"]


def test_pead_skips_subthreshold_surprise():
    """+2% EPS surprise (below 5% threshold) → no trade."""
    today = date.today()
    filing = today - timedelta(days=40)
    entry_target = filing + timedelta(days=1)

    fake_client = MagicMock()
    fake_client.is_configured.return_value = True
    fake_client.recent_earnings.return_value = [
        EarningsRecord(
            ticker="AAPL",
            period_end=filing - timedelta(days=10),
            filing_date=filing,
            eps_actual=2.04, eps_estimate=2.00,    # +2% — too small
            revenue_actual=None, revenue_estimate=None,
            raw={},
        ),
    ]
    fake_client.daily_bars.return_value = _make_bars(
        start=entry_target - timedelta(days=5),
        n_days=60, start_price=200.0, drift=0.5,
    )
    result = backtest_pead(
        window_days=60, universe=["AAPL"], polygon=fake_client,
    )
    assert result.n_trades == 0
    # `note` is set when there were 0 trades AND no qualifying surprises
    assert "no qualifying" in result.note


def test_pead_long_only_skips_negative_surprise():
    """LONG_ONLY = True (default) → -15% EPS miss does NOT open a short."""
    today = date.today()
    filing = today - timedelta(days=40)
    entry_target = filing + timedelta(days=1)

    fake_client = MagicMock()
    fake_client.is_configured.return_value = True
    fake_client.recent_earnings.return_value = [
        EarningsRecord(
            ticker="MSFT",
            period_end=filing - timedelta(days=10),
            filing_date=filing,
            eps_actual=1.70, eps_estimate=2.00,    # -15% miss
            revenue_actual=None, revenue_estimate=None,
            raw={},
        ),
    ]
    fake_client.daily_bars.return_value = _make_bars(
        start=entry_target - timedelta(days=5),
        n_days=60, start_price=400.0, drift=-1.0,
    )
    result = backtest_pead(
        window_days=60, universe=["MSFT"], polygon=fake_client,
    )
    assert result.n_trades == 0
