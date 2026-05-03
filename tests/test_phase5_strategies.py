"""Tests for the 3 Phase-5 strategies that consume Sprint-3 data feeds:
  - cross_venue_arb           Kalshi vs Polymarket
  - crypto_funding_carry_v2   Coinbase + Binance consensus
  - earnings_news_pead        PEAD × news corroboration

Each strategy gates on a scout signal that the new data wrappers
publish. Tests stub the broker + scout signals — no network.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brokers.base import OrderSide
from strategies.cross_venue_arb import CrossVenueArb
from strategies.crypto_funding_carry_v2 import CryptoFundingCarryV2
from strategies.earnings_news_pead import EarningsNewsPEAD
from strategy_engine.base import StrategyContext


def _ctx(scout_signals: dict | None = None,
          open_positions: dict | None = None,
          alloc_usd: float = 10000.0) -> StrategyContext:
    return StrategyContext(
        timestamp="2026-05-03T12:00:00+00:00",
        portfolio_equity_usd=alloc_usd,
        target_alloc_pct=0.04, target_alloc_usd=alloc_usd,
        risk_multiplier=1.0,
        open_positions=open_positions or {},
        scout_signals=scout_signals or {},
        pending_orders={},
    )


# ─── cross_venue_arb ─────────────────────────────────────────────────


def test_cross_venue_arb_no_signal_no_proposals():
    s = CrossVenueArb(broker=MagicMock())
    out = s.compute(_ctx({}))
    assert out == []


def test_cross_venue_arb_kalshi_undervalued_buys_yes():
    """Kalshi YES @ 0.40, Polymarket YES @ 0.55, divergence +0.15
    → BUY YES on Kalshi at $0.40 limit."""
    s = CrossVenueArb(broker=MagicMock())
    out = s.compute(_ctx({
        "cross_venue_arb": [{
            "kalshi_ticker": "ELECTION-X",
            "kalshi_yes": 0.40,
            "polymarket_yes": 0.55,
            "polymarket_volume_24h": 50_000,
            "divergence": 0.15,
        }]
    }))
    assert len(out) == 1
    p = out[0]
    assert p.side == OrderSide.BUY
    assert p.symbol == "ELECTION-X"
    assert p.limit_price == 0.40
    assert p.metadata["fair_value"] > 0.40


def test_cross_venue_arb_kalshi_overvalued_buys_no():
    """Kalshi YES @ 0.80, Polymarket YES @ 0.55, divergence -0.25
    → BUY NO (= sell YES) on Kalshi."""
    s = CrossVenueArb(broker=MagicMock())
    out = s.compute(_ctx({
        "cross_venue_arb": [{
            "kalshi_ticker": "ELECTION-X",
            "kalshi_yes": 0.80,
            "polymarket_yes": 0.55,
            "polymarket_volume_24h": 50_000,
            "divergence": -0.25,
        }]
    }))
    assert len(out) == 1
    assert out[0].side == OrderSide.SELL


def test_cross_venue_arb_below_threshold_no_trade():
    s = CrossVenueArb(broker=MagicMock())
    out = s.compute(_ctx({
        "cross_venue_arb": [{
            "kalshi_ticker": "X", "kalshi_yes": 0.40,
            "polymarket_yes": 0.43, "polymarket_volume_24h": 50_000,
            "divergence": 0.03,
        }]
    }))
    assert out == []


def test_cross_venue_arb_low_volume_filtered():
    """Polymarket volume < $5k/24h → skip (stale, not informational)."""
    s = CrossVenueArb(broker=MagicMock())
    out = s.compute(_ctx({
        "cross_venue_arb": [{
            "kalshi_ticker": "X", "kalshi_yes": 0.40,
            "polymarket_yes": 0.55, "polymarket_volume_24h": 100,
            "divergence": 0.15,
        }]
    }))
    assert out == []


# ─── crypto_funding_carry_v2 ─────────────────────────────────────────


def test_v2_no_funding_signal_no_proposals():
    s = CryptoFundingCarryV2(broker=MagicMock())
    out = s.compute(_ctx({}))
    assert out == []


def test_v2_fires_only_when_venues_agree():
    """Funding > 5% APR on Coinbase, but venues_agree=False → skip."""
    s = CryptoFundingCarryV2(broker=MagicMock())
    out = s.compute(_ctx({
        "funding_rates": {
            "BTC-PERP-INTX": {
                "perp_id": "BTC-PERP-INTX", "spot_id": "BTC-USD",
                "apr_bps": 600,    # 6% APR
            },
        },
        "cross_venue_funding": [
            {"symbol": "BTC-USD", "agree": False,
             "coinbase_apr_bps": 600, "binance_apr_bps": 100},
        ],
    }))
    assert out == []


def test_v2_fires_when_both_venues_agree_and_funding_hot():
    s = CryptoFundingCarryV2(broker=MagicMock())
    out = s.compute(_ctx({
        "funding_rates": {
            "BTC-PERP-INTX": {
                "perp_id": "BTC-PERP-INTX", "spot_id": "BTC-USD",
                "apr_bps": 600,
            },
        },
        "cross_venue_funding": [
            {"symbol": "BTC-USD", "agree": True,
             "coinbase_apr_bps": 600, "binance_apr_bps": 580},
        ],
    }, alloc_usd=20000))
    # Two legs (spot + perp)
    assert len(out) == 2
    spot, perp = sorted(out, key=lambda p: p.metadata["leg"])
    assert spot.metadata["leg"] == "perp"   # alphabetical: 'perp' < 'spot'
    # Actually spot < perp alphabetically; but in the strategy spot
    # is appended first.
    legs = {p.metadata["leg"] for p in out}
    assert legs == {"spot", "perp"}


def test_v2_exit_when_venues_stop_agreeing():
    """Already-open position; venues stop agreeing → close."""
    s = CryptoFundingCarryV2(broker=MagicMock())
    out = s.compute(_ctx(
        scout_signals={
            "funding_rates": {
                "BTC-PERP-INTX": {
                    "perp_id": "BTC-PERP-INTX", "spot_id": "BTC-USD",
                    "apr_bps": 400,    # still elevated
                },
            },
            "cross_venue_funding": [
                {"symbol": "BTC-USD", "agree": False,
                 "coinbase_apr_bps": 400, "binance_apr_bps": -50},
            ],
        },
        open_positions={
            "BTC-USD": {"quantity": 0.1},
            "BTC-PERP-INTX": {"quantity": 0.1},
        },
    ))
    # Two close orders (one per leg)
    assert len(out) == 2
    assert all(p.is_closing for p in out)


# ─── earnings_news_pead ──────────────────────────────────────────────


def _bar(close: float):
    """Lightweight Candle-like for broker stub."""
    return MagicMock(close=close)


def test_earnings_news_no_signals_no_proposals():
    s = EarningsNewsPEAD(broker=MagicMock())
    out = s.compute(_ctx({}))
    assert out == []


def test_earnings_news_fires_with_corroboration():
    """earnings_upcoming + ticker_news + gap-up >3% → BUY."""
    broker = MagicMock()
    # Yesterday close 100, today close 105 → +5% gap
    broker.get_candles.return_value = [_bar(100.0), _bar(105.0)]
    s = EarningsNewsPEAD(broker=broker)
    out = s.compute(_ctx({
        "earnings_upcoming": [{"symbol": "AAPL", "date": "2026-05-02"}],
        "ticker_news": [{
            "symbol": "AAPL", "n_headlines": 3,
            "headlines": [{"title": "AAPL beats Q2 EPS", "url": "u",
                            "source": "reuters", "published_at": "..."}],
        }],
    }, alloc_usd=10000))
    assert len(out) == 1
    p = out[0]
    assert p.symbol == "AAPL"
    assert p.side == OrderSide.BUY
    assert p.metadata["n_headlines"] == 3
    assert p.metadata["gap_pct"] >= 3.0


def test_earnings_news_skips_when_no_news_corroborates():
    """earnings + gap but no news → skip (the gating is the whole point)."""
    broker = MagicMock()
    broker.get_candles.return_value = [_bar(100.0), _bar(105.0)]
    s = EarningsNewsPEAD(broker=broker)
    out = s.compute(_ctx({
        "earnings_upcoming": [{"symbol": "AAPL", "date": "2026-05-02"}],
        "ticker_news": [],    # empty — no news corroboration
    }))
    assert out == []


def test_earnings_news_skips_when_gap_below_threshold():
    """Earnings + news but gap is only 1% → skip."""
    broker = MagicMock()
    broker.get_candles.return_value = [_bar(100.0), _bar(101.0)]
    s = EarningsNewsPEAD(broker=broker)
    out = s.compute(_ctx({
        "earnings_upcoming": [{"symbol": "AAPL", "date": "2026-05-02"}],
        "ticker_news": [{"symbol": "AAPL", "n_headlines": 2, "headlines": []}],
    }))
    assert out == []


def test_earnings_news_closes_after_hold_period():
    """Position with entry_time > 30 days ago → close."""
    broker = MagicMock()
    s = EarningsNewsPEAD(broker=broker)
    old = (datetime.now(UTC) - timedelta(days=35)).isoformat()
    out = s.compute(_ctx(
        scout_signals={"earnings_upcoming": [], "ticker_news": []},
        open_positions={
            "AAPL": {"quantity": 100, "entry_time": old},
        },
    ))
    assert len(out) == 1
    assert out[0].side == OrderSide.SELL
    assert out[0].is_closing
