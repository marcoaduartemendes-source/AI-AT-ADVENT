"""Tests for the market-hours gate.

The 770-PENDING / 128-CANCELED state in production was caused by
firing Alpaca SELLs at 23:40 UTC (after-hours). This test pins down
the gate behavior so the bug can't come back."""
from __future__ import annotations

from datetime import UTC, datetime

from common.market_hours import is_market_open


def _ts(year=2026, month=5, day=2, hour=15, minute=0):
    """Helper for fixed UTC timestamps. May 2 2026 is a Saturday;
    May 1 2026 is a Friday — useful for weekend tests."""
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ─── Crypto / Kalshi: always open ────────────────────────────────────


def test_coinbase_always_open():
    # 3 AM Sunday UTC
    assert is_market_open("coinbase", _ts(2026, 5, 3, 3, 0)) is True
    # Christmas Day at midnight
    assert is_market_open("coinbase", _ts(2026, 12, 25, 0, 0)) is True


def test_kalshi_always_open():
    assert is_market_open("kalshi", _ts(2026, 5, 3, 23, 0)) is True


# ─── Alpaca: weekday US session ──────────────────────────────────────


def test_alpaca_open_during_regular_session():
    # Friday May 1 2026 at 15:00 UTC (mid-session)
    assert is_market_open("alpaca", _ts(2026, 5, 1, 15, 0)) is True

    # Right after open
    assert is_market_open("alpaca", _ts(2026, 5, 1, 13, 30)) is True

    # Right before close
    assert is_market_open("alpaca", _ts(2026, 5, 1, 19, 59)) is True


def test_alpaca_closed_after_hours():
    """The bug scenario: 23:40 UTC on a weekday."""
    # 23:40 UTC = 7:40 PM ET — after-hours, day orders rejected
    assert is_market_open("alpaca", _ts(2026, 5, 1, 23, 40)) is False

    # Pre-market 6 AM UTC = 2 AM ET — before pre-market
    assert is_market_open("alpaca", _ts(2026, 5, 1, 6, 0)) is False


def test_alpaca_closed_before_open():
    # 13:00 UTC weekday = 9:00 ET, 30min before regular open
    assert is_market_open("alpaca", _ts(2026, 5, 1, 13, 0)) is False


def test_alpaca_closed_on_weekend():
    # Saturday
    assert is_market_open("alpaca", _ts(2026, 5, 2, 15, 0)) is False
    # Sunday
    assert is_market_open("alpaca", _ts(2026, 5, 3, 15, 0)) is False


def test_unknown_venue_is_permissive():
    """Don't accidentally block trading on a new venue we forgot
    to add to the list."""
    assert is_market_open("ibkr", _ts(2026, 5, 1, 23, 40)) is True
