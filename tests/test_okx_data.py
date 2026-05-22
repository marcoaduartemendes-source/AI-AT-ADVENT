"""Tests for the OKX crypto-data client (Bybit geo-block replacement).

No network: we patch OKXClient._get with canned OKX-shaped responses and
assert the parser produces the same FundingPoint / candle shape the
backtests consume, plus that the OKX→Bybit fallback chain skips an empty
primary.
"""
from __future__ import annotations

from backtests.data.okx import (
    FallbackCryptoClient,
    OKXClient,
    coinbase_to_okx,
)


def test_symbol_mapping():
    assert coinbase_to_okx("BTC-USD", "spot") == "BTC-USDT"
    assert coinbase_to_okx("BTC-USD", "linear") == "BTC-USDT-SWAP"
    assert coinbase_to_okx("ETH-USD", "perp") == "ETH-USDT-SWAP"


def test_funding_history_parses_okx_shape(tmp_path):
    c = OKXClient(cache_dir=str(tmp_path))
    c._get = lambda path, params, **k: {
        "code": "0",
        "data": [
            {"instId": "BTC-USDT-SWAP", "fundingRate": "0.0002",
             "fundingTime": "1700000000000"},
            {"instId": "BTC-USDT-SWAP", "fundingRate": "-0.0001",
             "fundingTime": "1699971200000"},
        ],
    }
    pts = c.funding_history("BTC-USD", limit=2)
    assert len(pts) == 2
    # oldest-first
    assert pts[0].timestamp < pts[1].timestamp
    assert pts[1].funding_rate == 0.0002


def test_daily_bars_parses_and_dedupes(tmp_path):
    c = OKXClient(cache_dir=str(tmp_path))
    c._get = lambda path, params, **k: {
        "code": "0",
        "data": [
            ["1700000000000", "100", "110", "90", "105", "1000", "x", "y", "1"],
            ["1699913600000", "95", "105", "85", "100", "900", "x", "y", "1"],
        ],
    }
    bars = c.daily_bars("BTC-USD", kind="spot", days=2)
    assert len(bars) == 2
    assert bars[0].close == 100.0 and bars[1].close == 105.0  # oldest-first


class _Empty:
    def funding_history(self, *a, **k):
        return []

    def daily_bars(self, *a, **k):
        return []


class _Good:
    def funding_history(self, *a, **k):
        from backtests.data.bybit import FundingPoint
        from datetime import UTC, datetime
        return [FundingPoint(datetime(2026, 1, 1, tzinfo=UTC), "BTCUSDT", 0.0001)]

    def daily_bars(self, *a, **k):
        from backtests.data.bybit import BybitCandle
        from datetime import UTC, datetime
        return [BybitCandle(datetime(2026, 1, 1, tzinfo=UTC), 1, 1, 1, 1, 1)]


def test_fallback_skips_empty_primary():
    fb = FallbackCryptoClient([_Empty(), _Good()])
    assert len(fb.funding_history("BTC-USD")) == 1
    assert len(fb.daily_bars("BTC-USD")) == 1


def test_fallback_all_empty_returns_empty():
    fb = FallbackCryptoClient([_Empty(), _Empty()])
    assert fb.funding_history("BTC-USD") == []
    assert fb.daily_bars("BTC-USD") == []
