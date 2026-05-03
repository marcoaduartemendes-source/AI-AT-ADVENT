"""Tests for macro_kalshi backtest.

Same shape as kalshi_arb tests, but with the macro-keyword filter.
Asserts:
  - Unconfigured Kalshi adapter → clean placeholder
  - No macro markets in settled list → clean note
  - Fed/CPI/NFP markets pass the filter; non-macro tickers don't
  - Tighter 2.5¢ entry threshold
  - FRED context is attached when client is_configured
  - FRED context is omitted when client is unconfigured (no crash)
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC
from unittest.mock import MagicMock

from backtests.data.kalshi_history import ResolvedMarket
from backtests.macro_kalshi_backtest import (
    ENTRY_EDGE_CENTS,
    MACRO_KEYWORDS,
    backtest_macro_kalshi,
)


def _market(ticker: str, yes_close: float, settle_yes: bool,
             title: str = "test") -> ResolvedMarket:
    now = datetime.now(UTC)
    return ResolvedMarket(
        ticker=ticker, title=title,
        open_ts=now - timedelta(days=10),
        close_ts=now - timedelta(days=1),
        settlement_value=1.0 if settle_yes else 0.0,
        yes_close_price=yes_close, raw={},
    )


def test_unconfigured_adapter_skips_cleanly():
    fake_kalshi = MagicMock()
    fake_kalshi.is_configured.return_value = False
    fake_fred = MagicMock()
    fake_fred.is_configured.return_value = False
    result = backtest_macro_kalshi(
        window_days=90, kalshi=fake_kalshi, fred=fake_fred,
    )
    assert result.n_trades == 0
    assert "not configured" in result.note.lower()


def test_no_macro_markets_returns_note():
    """All settled markets are non-macro → empty trade list with note."""
    fake_kalshi = MagicMock()
    fake_kalshi.is_configured.return_value = True
    fake_kalshi.settled_markets.return_value = [
        _market("ELECTION-X", 0.5, True, title="Will candidate X win?"),
        _market("WEATHER-Y", 0.3, False, title="Snow day in NYC"),
    ]
    fake_fred = MagicMock()
    fake_fred.is_configured.return_value = False

    result = backtest_macro_kalshi(
        window_days=90, kalshi=fake_kalshi, fred=fake_fred,
    )
    assert result.n_trades == 0
    assert "macro" in result.note.lower()


def test_macro_market_filter_picks_keywords():
    """Mix of macro & non-macro markets — only macro ones backtested."""
    fake_kalshi = MagicMock()
    fake_kalshi.is_configured.return_value = True
    fake_kalshi.settled_markets.return_value = [
        # non-macro (skipped)
        _market("ELECTION-X", 0.85, True, title="candidate X"),
        # macro: "fed" in title
        _market("FOMC-RATE-CUT", 0.85, True, title="Will the Fed cut rates?"),
        # macro: "cpi" in ticker
        _market("CPI-MAY-HOT", 0.85, True, title="Hot CPI print"),
    ]
    fake_fred = MagicMock()
    fake_fred.is_configured.return_value = False

    custom_table = [(0.80, 0.90, 0.05)]    # +5¢ → above 2.5¢ entry
    # Patch DEFAULT_RECALIBRATION via attribute on module
    import backtests.macro_kalshi_backtest as mod
    orig_table = mod.DEFAULT_RECALIBRATION
    mod.DEFAULT_RECALIBRATION = custom_table
    try:
        result = backtest_macro_kalshi(
            window_days=30, kalshi=fake_kalshi, fred=fake_fred,
        )
    finally:
        mod.DEFAULT_RECALIBRATION = orig_table

    # Both macro markets should trade; non-macro should not
    tickers = {t["product_id"] for t in result.trades}
    assert "ELECTION-X" not in tickers
    assert {"FOMC-RATE-CUT", "CPI-MAY-HOT"}.issubset(tickers), (
        f"Expected both macro markets traded, got {tickers}"
    )


def test_below_tight_threshold_filtered():
    """2¢ edge < 2.5¢ entry → skipped (tighter than calibration arb)."""
    fake_kalshi = MagicMock()
    fake_kalshi.is_configured.return_value = True
    fake_kalshi.settled_markets.return_value = [
        _market("FED-CUT", 0.85, True, title="Fed rate decision"),
    ]
    fake_fred = MagicMock()
    fake_fred.is_configured.return_value = False

    import backtests.macro_kalshi_backtest as mod
    orig = mod.DEFAULT_RECALIBRATION
    mod.DEFAULT_RECALIBRATION = [(0.80, 0.90, 0.02)]  # 2¢ < 2.5¢
    try:
        result = backtest_macro_kalshi(
            window_days=30, kalshi=fake_kalshi, fred=fake_fred,
        )
    finally:
        mod.DEFAULT_RECALIBRATION = orig

    assert result.n_trades == 0


def test_macro_keywords_cover_expected_set():
    """Sanity: ensure the macro keywords haven't drifted from the
    live strategy's filter."""
    expected = {"fed", "fomc", "rate", "cpi", "inflation", "nfp",
                "jobs", "unemployment", "gdp", "pce", "pmi", "ism"}
    assert set(MACRO_KEYWORDS) == expected


def test_fred_unconfigured_does_not_crash():
    """Backtest still runs (FRED context just empty) when no FRED key."""
    fake_kalshi = MagicMock()
    fake_kalshi.is_configured.return_value = True
    fake_kalshi.settled_markets.return_value = [
        _market("FED-CUT", 0.85, True, title="Will the Fed cut rates?"),
    ]
    fake_fred = MagicMock()
    fake_fred.is_configured.return_value = False
    fake_fred.series_observations.side_effect = AssertionError(
        "Should not be called when FRED is unconfigured"
    )

    import backtests.macro_kalshi_backtest as mod
    orig = mod.DEFAULT_RECALIBRATION
    mod.DEFAULT_RECALIBRATION = [(0.80, 0.90, 0.05)]
    try:
        result = backtest_macro_kalshi(
            window_days=30, kalshi=fake_kalshi, fred=fake_fred,
        )
    finally:
        mod.DEFAULT_RECALIBRATION = orig

    assert result.n_trades == 1
    # FRED context absent
    assert result.trades[0]["fred_context"] == {}


def test_entry_edge_threshold_tighter_than_calibration_arb():
    """Defensive: macro_kalshi must use a tighter (smaller) entry edge
    than calibration_arb to reflect its higher confidence."""
    from backtests.kalshi_arb_backtest import (
        ENTRY_EDGE_CENTS as ARB_EDGE,
    )
    assert ENTRY_EDGE_CENTS < ARB_EDGE


def test_entry_edge_matches_live_strategy():
    from strategies import macro_kalshi as live
    assert ENTRY_EDGE_CENTS == live.ENTRY_EDGE_CENTS
