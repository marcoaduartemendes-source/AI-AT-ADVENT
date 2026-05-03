"""Tests for kalshi_calibration_arb backtest.

Uses a stub KalshiHistoryClient with deterministic ResolvedMarket
fixtures. Asserts:
  - No Kalshi auth → clean placeholder summary
  - No settled markets → clean note
  - Heavy-favorite YES (90% market) that resolved YES → profitable
    BUY YES trade after recalibration shift
  - Heavy-longshot (8% market) that resolved NO → profitable BUY NO
  - Mid-field (50%) market never triggers (no calibration shift)
  - Trade size obeys the per-trade USD cap
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from backtests.data.kalshi_history import ResolvedMarket
from backtests.kalshi_arb_backtest import (
    ENTRY_EDGE_CENTS,
    KALSHI_FEE_RATE,
    MAX_PER_TRADE_USD,
    backtest_kalshi_calibration_arb,
)


def _market(ticker: str, yes_close: float, settle_yes: bool,
             title: str = "test market") -> ResolvedMarket:
    now = datetime.now(UTC)
    return ResolvedMarket(
        ticker=ticker,
        title=title,
        open_ts=now - timedelta(days=10),
        close_ts=now - timedelta(days=1),
        settlement_value=1.0 if settle_yes else 0.0,
        yes_close_price=yes_close,
        raw={},
    )


def test_unconfigured_adapter_skips_cleanly():
    fake = MagicMock()
    fake.is_configured.return_value = False
    result = backtest_kalshi_calibration_arb(window_days=90, client=fake)
    assert result.strategy == "kalshi_calibration_arb"
    assert result.n_trades == 0
    assert "not configured" in result.note.lower()


def test_no_settled_markets_returns_note():
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.settled_markets.return_value = []
    result = backtest_kalshi_calibration_arb(window_days=90, client=fake)
    assert result.n_trades == 0
    assert result.note


def test_heavy_favorite_resolved_yes_profitable():
    """Market priced at 90% YES, recalibration shifts +1.5¢ → fair=91.5¢
    → edge ~+1.5¢. With ENTRY_EDGE_CENTS=3, this won't trigger.
    Use 88% market: shift +1.5¢ → fair=89.5¢, edge=+1.5¢ STILL won't trigger.
    Use 80% market boundary: shift jumps to +1.5¢ at the 80-90 bucket.
    Use 95% market → fair=97¢, edge=+2¢ — still below 3¢. Use a
    custom recalibration table to force a strong edge for the test."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.settled_markets.return_value = [
        _market("FED-RATE-25BP", yes_close=0.85, settle_yes=True),
    ]
    # Custom table: bucket [.80, .90) gets +5¢ — well over 3¢ entry
    custom_table = [(0.80, 0.90, 0.05)]

    result = backtest_kalshi_calibration_arb(
        window_days=30, client=fake, recalibration=custom_table,
    )
    assert result.n_trades == 1
    t = result.trades[0]
    assert t["side"] == "BUY_YES"
    # Won YES → P&L positive
    assert result.total_pnl_usd > 0
    # 5% Kalshi fee should be deducted from gross winnings
    gross = (1.0 - 0.85) * t["quantity"]
    assert result.total_pnl_usd <= gross * (1 - KALSHI_FEE_RATE) + 0.01


def test_heavy_longshot_resolved_no_profitable():
    """8% YES market with -5¢ shift → fair=3¢ → edge=-5¢ → BUY NO at 92¢."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.settled_markets.return_value = [
        _market("LONGSHOT-X", yes_close=0.08, settle_yes=False),
    ]
    custom_table = [(0.02, 0.10, -0.05)]

    result = backtest_kalshi_calibration_arb(
        window_days=30, client=fake, recalibration=custom_table,
    )
    assert result.n_trades == 1
    t = result.trades[0]
    assert t["side"] == "BUY_NO"
    # Won NO → positive P&L
    assert result.total_pnl_usd > 0


def test_midfield_market_no_trade():
    """Market priced at 50% with default zero shift → no edge → no trade."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.settled_markets.return_value = [
        _market("COIN-FLIP", yes_close=0.50, settle_yes=True),
    ]
    result = backtest_kalshi_calibration_arb(
        window_days=30, client=fake,
    )
    assert result.n_trades == 0


def test_per_trade_usd_cap_enforced():
    """Even on a high-edge market, no trade should exceed MAX_PER_TRADE_USD."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    # 10 markets, all with extreme edge → tests that each individually
    # respects the cap
    fake.settled_markets.return_value = [
        _market(f"MKT-{i}", yes_close=0.5, settle_yes=True)
        for i in range(10)
    ]
    custom_table = [(0.4, 0.6, 0.10)]   # +10¢ shift, huge edge

    result = backtest_kalshi_calibration_arb(
        window_days=30, client=fake, recalibration=custom_table,
    )
    for t in result.trades:
        assert t["amount_usd"] <= MAX_PER_TRADE_USD + 0.5, (
            f"Trade ${t['amount_usd']:.2f} exceeded cap ${MAX_PER_TRADE_USD}"
        )


def test_below_threshold_filtered():
    """1¢ recalibration shift < 3¢ entry edge → skipped."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.settled_markets.return_value = [
        _market("WEAK-EDGE", yes_close=0.85, settle_yes=True),
    ]
    custom_table = [(0.80, 0.90, 0.01)]    # only 1¢ edge — below threshold

    result = backtest_kalshi_calibration_arb(
        window_days=30, client=fake, recalibration=custom_table,
    )
    assert result.n_trades == 0


def test_entry_edge_constant_matches_live():
    """Defensive: backtest must match live strategy's entry threshold."""
    from strategies import kalshi_calibration_arb as live
    assert ENTRY_EDGE_CENTS == live.ENTRY_EDGE_CENTS
