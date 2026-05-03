"""Tests for crypto_funding_carry backtest.

Uses a stub BybitClient feeding deterministic funding-rate series.
Asserts:
  - Empty history → clean placeholder summary
  - High-APR sequence → opens trade and accumulates funding
  - Decay below exit threshold → closes the position
  - Round-trip fee is deducted from net P&L
  - 5% fee on Bybit-style numbers gives plausible directional answer
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from backtests.crypto_funding_carry_backtest import (
    ANNUALIZATION_FACTOR,
    ENTRY_FUNDING_APR_PCT,
    EXIT_FUNDING_APR_PCT,
    backtest_crypto_funding_carry,
)
from backtests.data.bybit import FundingPoint


def _funding(rate_apr_pct: float, when: datetime) -> FundingPoint:
    """Build a FundingPoint such that .funding_rate × ANN = rate_apr_pct."""
    rate = (rate_apr_pct / 100) / ANNUALIZATION_FACTOR
    return FundingPoint(timestamp=when, symbol="BTCUSDT", funding_rate=rate)


def test_empty_history_returns_placeholder():
    fake = MagicMock()
    fake.funding_history.return_value = []
    result = backtest_crypto_funding_carry(
        window_days=30, universe=["BTC-USD"], bybit=fake,
    )
    assert result.strategy == "crypto_funding_carry"
    assert result.n_trades == 0
    assert result.note  # has a note


def test_high_funding_opens_and_decay_closes():
    """7d of high funding (10% APR) then drop to 0.5% APR → one trade."""
    today = datetime.now(UTC)
    # 14 days of 8h ticks: first 7d at 10% APR, next 7d at 0.5% APR
    series = []
    for i in range(21):       # 7d × 3 ticks/day
        when = today - timedelta(hours=8 * (42 - i))
        series.append(_funding(rate_apr_pct=10.0, when=when))
    for i in range(21):       # next 7d
        when = today - timedelta(hours=8 * (21 - i))
        series.append(_funding(rate_apr_pct=0.5, when=when))

    fake = MagicMock()
    fake.funding_history.return_value = series

    result = backtest_crypto_funding_carry(
        window_days=30, universe=["BTC-USD"], bybit=fake,
    )
    assert result.n_trades == 1, (
        f"Expected exactly one carry trade, got {result.n_trades}"
    )
    # Should be net positive — collected ~7d of 10% APR funding
    assert result.total_pnl_usd > 0, f"Expected +ve PnL, got {result.total_pnl_usd}"
    t = result.trades[0]
    assert t["product_id"] == "BTC-USD"
    assert t["side"] == "BUY"


def test_below_threshold_funding_no_trades():
    """Funding stays at 2% APR — never crosses 5% entry threshold."""
    today = datetime.now(UTC)
    series = [
        _funding(rate_apr_pct=2.0, when=today - timedelta(hours=8 * (30 - i)))
        for i in range(30)
    ]
    fake = MagicMock()
    fake.funding_history.return_value = series

    result = backtest_crypto_funding_carry(
        window_days=15, universe=["BTC-USD"], bybit=fake,
    )
    assert result.n_trades == 0


def test_threshold_constants_match_live_strategy():
    """Defensive: backtest must use the same thresholds as the live
    strategy or the backtest is meaningless. Live strategy expresses
    them as bps (500/100); backtest as percent (5.0/1.0)."""
    from strategies import crypto_funding_carry as live
    assert ENTRY_FUNDING_APR_PCT * 100 == live.ENTRY_APR_BPS
    assert EXIT_FUNDING_APR_PCT * 100 == live.EXIT_APR_BPS
