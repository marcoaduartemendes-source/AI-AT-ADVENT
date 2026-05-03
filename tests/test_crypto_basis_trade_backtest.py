"""Tests for crypto_basis_trade backtest.

Uses a stub BybitClient feeding deterministic spot + perp series.
Asserts:
  - No data → clean placeholder
  - High entry basis followed by decay → trade opens & closes
  - Sub-threshold basis never triggers
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from backtests.crypto_basis_trade_backtest import (
    ENTRY_BASIS_BPS,
    EXIT_BASIS_BPS,
    backtest_crypto_basis_trade,
)
from backtests.data.bybit import BybitCandle


def _candle(when: datetime, close: float) -> BybitCandle:
    return BybitCandle(
        timestamp=when, open=close, high=close, low=close,
        close=close, volume=1000.0,
    )


def test_no_data_returns_placeholder():
    fake = MagicMock()
    fake.daily_bars.return_value = []
    result = backtest_crypto_basis_trade(
        window_days=30, universe=["BTC-USD"], bybit=fake,
    )
    assert result.n_trades == 0
    assert result.note


def test_high_basis_then_decay_opens_and_closes():
    """First 7d perp 10% above spot (1000 bps), next 23d perp at parity
    → one trade that opens then closes."""
    today = datetime.now(UTC)
    spot, perp = [], []
    # 60 daily bars: first 30d basis = 0; next 7d basis = 10%; final 23d basis = 0
    for i in range(60):
        when = today - timedelta(days=60 - i)
        spot_px = 100.0
        if 30 <= i < 37:
            perp_px = 110.0      # +10% basis = 1000 bps
        else:
            perp_px = 100.0      # parity
        spot.append(_candle(when, spot_px))
        perp.append(_candle(when, perp_px))

    fake = MagicMock()
    # daily_bars(symbol, kind=...) — return spot vs perp by kind
    def _bars(symbol, kind="spot", days=365):
        return spot if kind == "spot" else perp
    fake.daily_bars.side_effect = _bars

    result = backtest_crypto_basis_trade(
        window_days=45, universe=["BTC-USD"], bybit=fake,
    )
    assert result.n_trades == 1, (
        f"Expected 1 basis trade, got {result.n_trades}. note={result.note}"
    )
    t = result.trades[0]
    assert t["product_id"] == "BTC-USD"
    # Entry basis ~= 1000 bps
    assert t["entry_price"] >= ENTRY_BASIS_BPS - 1


def test_low_basis_no_trades():
    """Perp tracks spot within 10 bps the whole window — no trades."""
    today = datetime.now(UTC)
    spot, perp = [], []
    for i in range(60):
        when = today - timedelta(days=60 - i)
        spot.append(_candle(when, 100.0))
        perp.append(_candle(when, 100.05))   # 5 bps basis
    fake = MagicMock()
    def _bars(symbol, kind="spot", days=365):
        return spot if kind == "spot" else perp
    fake.daily_bars.side_effect = _bars

    result = backtest_crypto_basis_trade(
        window_days=30, universe=["BTC-USD"], bybit=fake,
    )
    assert result.n_trades == 0


def test_threshold_constants_present():
    assert ENTRY_BASIS_BPS > EXIT_BASIS_BPS
    assert ENTRY_BASIS_BPS > 0
