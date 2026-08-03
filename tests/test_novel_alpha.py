"""Tests for the 2026-06-05 novel-alpha sleeves:
  - insider_cluster  (Form 4 cluster buys — Cohen-Malloy-Pomorski)
  - llm_8k_event     (Claude-scored 8-K material events)

All network + LLM surfaces are stubbed. Pins entry conditions, the
conviction ordering, and the safety invariants (caps, stops, age-outs).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brokers.base import OrderSide
from scouts.insider_cluster_scout import _parse_purchases
from scouts.edgar_8k_scout import _parse_feed
from strategies.insider_cluster import InsiderCluster
from strategies.llm_8k_event import Llm8KEvent
from strategy_engine.base import StrategyContext


def _ctx(scout_signals=None, open_positions=None, alloc_usd=10000.0):
    return StrategyContext(
        timestamp="2026-06-05T12:00:00+00:00",
        portfolio_equity_usd=alloc_usd * 10,
        target_alloc_pct=0.03, target_alloc_usd=alloc_usd,
        risk_multiplier=1.0,
        open_positions=open_positions or {},
        scout_signals=scout_signals or {},
        pending_orders={},
    )


class _Candle:
    def __init__(self, close):
        self.close = close


def _candles(closes):
    return [_Candle(c) for c in closes]


# ─── insider_cluster scout parsing ────────────────────────────────


class TestInsiderClusterParsing:
    def _row(self, sym, who, days_ago=1, qty=1000, px=50.0,
             ttype="P-Purchase"):
        when = (datetime.now(UTC) - timedelta(days=days_ago))
        return {"symbol": sym, "reportingName": who,
                "transactionType": ttype,
                "transactionDate": when.strftime("%Y-%m-%d"),
                "securitiesTransacted": qty, "price": px}

    def test_three_insiders_form_cluster(self):
        rows = [self._row("ABC", f"Insider {i}") for i in range(3)]
        out = _parse_purchases(rows)
        assert len(out["ABC"]["insiders"]) == 3
        assert out["ABC"]["total_usd"] == 3 * 1000 * 50.0

    def test_sales_excluded(self):
        rows = [self._row("ABC", f"I{i}", ttype="S-Sale") for i in range(5)]
        assert _parse_purchases(rows) == {}

    def test_stale_purchases_excluded(self):
        rows = [self._row("ABC", f"I{i}", days_ago=30) for i in range(5)]
        assert _parse_purchases(rows) == {}

    def test_same_insider_twice_counts_once(self):
        rows = [self._row("ABC", "Same Guy"), self._row("ABC", "Same Guy")]
        out = _parse_purchases(rows)
        assert len(out["ABC"]["insiders"]) == 1


# ─── insider_cluster strategy ─────────────────────────────────────


class TestInsiderClusterStrategy:
    def test_cluster_signal_triggers_buy(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20, 21])
        s = InsiderCluster(broker=broker)
        out = s.compute(_ctx({"insider_cluster_buy": [
            {"ticker": "ABC", "n_insiders": 4, "total_usd": 250_000}
        ]}))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert len(buys) == 1 and buys[0].symbol == "ABC"

    def test_strongest_cluster_first_under_cap(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20, 21])
        # 4 held already → only ONE slot left (MAX_POSITIONS=5); the
        # 6-insider cluster must win it over the 3-insider one.
        positions = {s: {"quantity": 1.0, "avg_entry_price": 20.0}
                     for s in ("W", "X", "Y", "Z")}
        s = InsiderCluster(broker=broker)
        out = s.compute(_ctx(
            {"insider_cluster_buy": [
                {"ticker": "WEAK", "n_insiders": 3, "total_usd": 60_000},
                {"ticker": "STRONG", "n_insiders": 6, "total_usd": 900_000},
            ]},
            open_positions=positions))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert [p.symbol for p in buys] == ["STRONG"]

    def test_take_profit_exit(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([22, 22.5])  # +12.5%
        positions = {"ABC": {"quantity": 10.0, "avg_entry_price": 20.0}}
        s = InsiderCluster(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        assert any(p.is_closing and p.symbol == "ABC" for p in out)

    def test_stop_loss_exit(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([19, 18.7])  # −6.5%
        positions = {"ABC": {"quantity": 10.0, "avg_entry_price": 20.0}}
        s = InsiderCluster(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        assert any(p.is_closing and p.symbol == "ABC" for p in out)


# ─── edgar_8k scout parsing ───────────────────────────────────────


_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>8-K - ACME ROBOTICS INC (0001234567) (Filer)</title>
    <link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/1234567/000123456726000042-index.htm"/>
    <summary type="html">Item 1.01: Entry into a Material Definitive Agreement Item 9.01</summary>
    <updated>2026-06-05T14:31:02-04:00</updated>
  </entry>
</feed>"""


class TestEdgar8KParsing:
    def test_feed_parse_extracts_cik_and_items(self):
        out = _parse_feed(_ATOM)
        assert len(out) == 1
        assert out[0]["cik"] == "1234567"
        assert "1.01" in out[0]["items"]


# ─── llm_8k_event strategy ────────────────────────────────────────


class TestLlm8KEventStrategy:
    def test_very_bullish_high_confidence_buys(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([50, 51])
        s = Llm8KEvent(broker=broker)
        out = s.compute(_ctx({"llm_8k_event": [
            {"ticker": "ACME", "direction": "VERY_BULLISH",
             "confidence": 0.85, "rationale": "transformative merger"}
        ]}))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert len(buys) == 1 and buys[0].symbol == "ACME"

    def test_low_confidence_rejected(self):
        s = Llm8KEvent(broker=MagicMock())
        out = s.compute(_ctx({"llm_8k_event": [
            {"ticker": "ACME", "direction": "VERY_BULLISH",
             "confidence": 0.55}]}))
        assert [p for p in out if p.side == OrderSide.BUY] == []

    def test_merely_bullish_rejected(self):
        # Only the TOP conviction band trades — BULLISH is not enough.
        s = Llm8KEvent(broker=MagicMock())
        out = s.compute(_ctx({"llm_8k_event": [
            {"ticker": "ACME", "direction": "BULLISH",
             "confidence": 0.95}]}))
        assert [p for p in out if p.side == OrderSide.BUY] == []

    def test_bearish_never_trades_long_only(self):
        s = Llm8KEvent(broker=MagicMock())
        out = s.compute(_ctx({"llm_8k_event": [
            {"ticker": "DOOM", "direction": "VERY_BEARISH",
             "confidence": 0.99}]}))
        assert out == []

    def test_tight_stop_fires(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([48.5, 47.9])  # −4.2%
        positions = {"ACME": {"quantity": 5.0, "avg_entry_price": 50.0}}
        s = Llm8KEvent(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        assert any(p.is_closing and p.symbol == "ACME" for p in out)

    def test_age_out_after_drift_window(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([50, 50.5])
        # 2026-06-11: age-out reads `entry_time` (the real PositionView
        # field the orchestrator now populates from the ledger). The old
        # `opened_at` key never existed, so this exit was dead code.
        opened = (datetime.now(UTC) - timedelta(days=6)).isoformat()
        positions = {"ACME": {"quantity": 5.0, "avg_entry_price": 50.0,
                              "entry_time": opened}}
        s = Llm8KEvent(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        assert any(p.is_closing and p.symbol == "ACME" for p in out)


class TestNewBacktestsRegistered:
    """The 2026-06-09 self-grade work: high_vol_trend + pre_fomc_drift
    get real backtests so they can earn PASS verdicts (fee_discipline
    counts only PASS-strategy trades); the 4 feed-dependent event
    sleeves get honest UNBACKTESTABLE notes instead of the generic
    'no backtest defined'."""

    def test_backtestable_sleeves_in_dispatch(self):
        from backtests.runner import _STRATEGY_BACKTESTS
        assert "high_vol_trend" in _STRATEGY_BACKTESTS
        assert "pre_fomc_drift" in _STRATEGY_BACKTESTS

    def test_event_sleeves_have_honest_notes(self):
        from backtests.runner import UNBACKTESTABLE
        for name in ("activist_13d", "merger_arb",
                     "insider_cluster", "llm_8k_event"):
            assert name in UNBACKTESTABLE
            assert "feed" in UNBACKTESTABLE[name] or \
                   "LLM" in UNBACKTESTABLE[name]

    def test_fomc_calendar_sane(self):
        from backtests.runner import _FOMC_DECISION_DAYS
        from collections import Counter
        years = Counter(d[:4] for d in _FOMC_DECISION_DAYS)
        # The Fed holds exactly 8 scheduled meetings per year.
        assert all(n == 8 for n in years.values()), years
        assert sorted(_FOMC_DECISION_DAYS) == _FOMC_DECISION_DAYS
