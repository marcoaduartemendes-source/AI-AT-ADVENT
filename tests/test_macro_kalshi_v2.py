"""Tests for macro_kalshi_v2 — Kalshi-vs-CME divergence strategy.

Sprint C deliverable: first strategy to consume the new CME
FedWatch data feed. These tests verify:
  - Ticker parsing handles common Kalshi formats
  - Empty/missing CME data → no proposals (safe failure)
  - Big divergence + matching CME meeting → BUY YES proposal
  - Negative divergence (Kalshi too rich) → BUY NO proposal
  - Sub-threshold divergence → no trade
  - Non-rate markets are filtered out
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from backtests.data.cme_fedwatch import FedMeetingProb
from brokers.base import OrderSide
from strategies.macro_kalshi_v2 import (
    ENTRY_DIVERGENCE,
    MacroKalshiV2,
    _cme_prob_for,
)
from strategy_engine.base import StrategyContext


def _ctx(scout_signals: dict, alloc_usd: float = 1000.0) -> StrategyContext:
    return StrategyContext(
        timestamp="2026-05-03T12:00:00+00:00",
        portfolio_equity_usd=alloc_usd,
        target_alloc_pct=0.02, target_alloc_usd=alloc_usd,
        risk_multiplier=1.0,
        open_positions={}, scout_signals=scout_signals,
        pending_orders={},
    )


def _meeting_jun_2026(probs):
    return FedMeetingProb(
        meeting_date=date(2026, 6, 17),
        target_rate_probs=probs, raw={},
    )


# ─── Ticker parsing ──────────────────────────────────────────────────


def test_parse_ticker_t_format():
    """FED-26JUN-T425 → meeting Jun 2026, target 4.25-4.50%."""
    parsed = MacroKalshiV2._parse_rate_ticker("FED-26JUN-T425")
    assert parsed is not None
    meeting, lo, hi = parsed
    assert meeting.year == 2026 and meeting.month == 6
    assert lo == 425 and hi == 450


def test_parse_ticker_range_format():
    parsed = MacroKalshiV2._parse_rate_ticker("FED-RATE-26JUN-425-450")
    assert parsed is not None
    _, lo, hi = parsed
    assert lo == 425 and hi == 450


def test_parse_ticker_unknown_format_returns_none():
    """Unrecognized ticker should not crash — strategy skips it."""
    assert MacroKalshiV2._parse_rate_ticker("WEATHER-NYC-SNOW") is None


def test_is_rate_decision():
    assert MacroKalshiV2._is_rate_decision("FED-26JUN-T425", "Fed cut")
    assert MacroKalshiV2._is_rate_decision("ANY", "FOMC rate decision Jun 2026")
    assert not MacroKalshiV2._is_rate_decision("ELECTION-2026", "Will X win")


# ─── CME probability lookup ──────────────────────────────────────────


def test_cme_prob_match_exact_range():
    meetings = [_meeting_jun_2026([(425, 450, 0.65), (400, 425, 0.30)])]
    p = _cme_prob_for(date(2026, 6, 17), 425, 450, meetings)
    assert p == 0.65


def test_cme_prob_no_matching_meeting():
    """No CME data for this meeting → return None so the strategy
    falls back to v1 calibration fade."""
    meetings = [_meeting_jun_2026([(425, 450, 0.65)])]
    p = _cme_prob_for(date(2027, 1, 27), 425, 450, meetings)
    assert p is None


def test_cme_prob_within_10_day_window():
    """Strategy approximates Kalshi tickers to mid-month; CME has
    the actual meeting day. Allow ±10 day match window."""
    meetings = [_meeting_jun_2026([(425, 450, 0.65)])]
    # Mid-month proxy date
    p = _cme_prob_for(date(2026, 6, 15), 425, 450, meetings)
    assert p == 0.65


# ─── End-to-end strategy logic ───────────────────────────────────────


def test_no_cme_client_no_proposals():
    s = MacroKalshiV2(broker=MagicMock(), cme_client=None)
    out = s.compute(_ctx({"mispriced": [{"ticker": "FED-26JUN-T425",
                                            "yes_price": 0.5,
                                            "title": "Fed rate decision"}]}))
    assert out == []


def test_no_scout_signal_no_proposals():
    s = MacroKalshiV2(broker=MagicMock(), cme_client=MagicMock())
    out = s.compute(_ctx({}))
    assert out == []


def test_zero_alloc_no_proposals():
    s = MacroKalshiV2(broker=MagicMock(), cme_client=MagicMock())
    out = s.compute(_ctx(scout_signals={}, alloc_usd=0.0))
    assert out == []


def test_kalshi_undervalued_buys_yes():
    """Kalshi YES @ 0.30, CME implies 0.55 → divergence +0.25 ≥ 0.05 → BUY YES."""
    cme = MagicMock()
    cme.upcoming_meetings.return_value = [
        _meeting_jun_2026([(425, 450, 0.55), (400, 425, 0.30)]),
    ]
    s = MacroKalshiV2(broker=MagicMock(), cme_client=cme)
    out = s.compute(_ctx({
        "mispriced": [{
            "ticker": "FED-26JUN-T425",
            "yes_price": 0.30,
            "title": "Fed rate decision Jun 2026",
        }]
    }, alloc_usd=10000))

    assert len(out) == 1
    p = out[0]
    assert p.side == OrderSide.BUY
    assert p.symbol == "FED-26JUN-T425"
    assert p.limit_price == 0.30
    assert p.metadata["cme_prob"] == 0.55
    assert p.metadata["divergence"] > 0


def test_kalshi_overvalued_buys_no():
    """Kalshi YES @ 0.80, CME implies 0.50 → divergence -0.30 → BUY NO (sell YES)."""
    cme = MagicMock()
    cme.upcoming_meetings.return_value = [
        _meeting_jun_2026([(425, 450, 0.50)]),
    ]
    s = MacroKalshiV2(broker=MagicMock(), cme_client=cme)
    out = s.compute(_ctx({
        "mispriced": [{
            "ticker": "FED-26JUN-T425",
            "yes_price": 0.80,
            "title": "Fed rate decision",
        }]
    }, alloc_usd=10000))

    assert len(out) == 1
    p = out[0]
    assert p.side == OrderSide.SELL    # selling YES = buying NO
    assert p.metadata["divergence"] < 0


def test_below_threshold_no_trade():
    """Divergence 3% < 5% threshold → skip."""
    cme = MagicMock()
    cme.upcoming_meetings.return_value = [
        _meeting_jun_2026([(425, 450, 0.53)]),
    ]
    s = MacroKalshiV2(broker=MagicMock(), cme_client=cme)
    out = s.compute(_ctx({
        "mispriced": [{
            "ticker": "FED-26JUN-T425",
            "yes_price": 0.50,
            "title": "Fed",
        }]
    }, alloc_usd=10000))
    assert out == []


def test_non_rate_markets_filtered():
    """Election market in scout signals must NOT trigger v2 even with
    a big divergence — v2 is rate-decision only by construction."""
    cme = MagicMock()
    cme.upcoming_meetings.return_value = [
        _meeting_jun_2026([(425, 450, 0.95)]),
    ]
    s = MacroKalshiV2(broker=MagicMock(), cme_client=cme)
    out = s.compute(_ctx({
        "mispriced": [{
            "ticker": "ELECTION-2026-X",
            "yes_price": 0.10,
            "title": "Will X win the 2026 election",
        }]
    }, alloc_usd=10000))
    assert out == []


def test_cme_failure_no_proposals():
    """CME fetch raising must not propagate — strategy emits nothing,
    risk gate doesn't see proposals to maybe-double-up on stale data."""
    cme = MagicMock()
    cme.upcoming_meetings.side_effect = RuntimeError("CME 503")
    s = MacroKalshiV2(broker=MagicMock(), cme_client=cme)
    out = s.compute(_ctx({
        "mispriced": [{
            "ticker": "FED-26JUN-T425",
            "yes_price": 0.30,
            "title": "Fed",
        }]
    }, alloc_usd=10000))
    assert out == []


def test_entry_threshold_constant():
    """Defensive: 5% entry threshold is the documented edge — bump
    only with audit trail."""
    assert ENTRY_DIVERGENCE == 0.05
