"""Tests for the 2026-06-05 real-edge sleeves (Brav-Jiang, Lucca-Moench,
S&P MergerArb Index, AQR MF HV — named-firm proven strategies).

Each test stubs the broker + scout signals; no network. Pins the entry
conditions and the safety invariants (per-name cap, stops, age-outs).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from brokers.base import OrderSide
from strategies.activist_13d import Activist13D
from strategies.high_vol_trend import HighVolTrend
from strategies.merger_arb import MergerArb
from strategies.pre_fomc_drift import PreFomcDrift
from strategy_engine.base import StrategyContext


def _ctx(scout_signals=None, open_positions=None, alloc_usd=10000.0):
    return StrategyContext(
        timestamp="2026-06-05T12:00:00+00:00",
        portfolio_equity_usd=alloc_usd * 10,
        target_alloc_pct=0.04, target_alloc_usd=alloc_usd,
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


# ─── pre_fomc_drift ─────────────────────────────────────────────────


class TestPreFomcDrift:
    # The exit is now time-to-announcement based (2026-06-11 fix), so we
    # pin "now" relative to a fixed meeting date. Announcement is modeled
    # at 18:30 UTC on the meeting day.
    _MEETING = "2026-06-17"

    def _freeze_now(self, monkeypatch, now):
        import strategies.pre_fomc_drift as mod

        class _FrozenDT(mod.datetime):
            @classmethod
            def now(cls, tz=None):
                return now
        monkeypatch.setattr(mod, "datetime", _FrozenDT)

    def test_outside_window_no_proposals(self, monkeypatch):
        # 5 days before the meeting → outside the 24h drift window.
        self._freeze_now(monkeypatch,
                         datetime(2026, 6, 12, 12, 0, tzinfo=UTC))
        s = PreFomcDrift(broker=MagicMock())
        out = s.compute(_ctx({"macro_fomc_window": {
            "next_meeting": self._MEETING,
            "days_to_next": 5, "blackout": False}}))
        assert out == []

    def test_inside_window_proposes_spy_and_qqq(self, monkeypatch):
        # 12h before the 18:30 UTC announcement → in the drift window.
        self._freeze_now(monkeypatch,
                         datetime(2026, 6, 17, 6, 30, tzinfo=UTC))
        broker = MagicMock()
        broker.get_candles.return_value = _candles([500, 502])
        s = PreFomcDrift(broker=broker)
        out = s.compute(_ctx({"macro_fomc_window": {
            "next_meeting": self._MEETING,
            "days_to_next": 0, "blackout": True}}))
        syms = {p.symbol for p in out}
        assert syms == {"SPY", "QQQ"}
        for p in out:
            assert p.side == OrderSide.BUY
            assert p.notional_usd > 0

    def test_holds_through_announcement_is_dead(self, monkeypatch):
        # 5 min BEFORE the announcement (inside the 15min exit buffer) →
        # must flatten, never enter. This is the bug the fix closes.
        self._freeze_now(monkeypatch,
                         datetime(2026, 6, 17, 18, 25, tzinfo=UTC))
        broker = MagicMock()
        broker.get_candles.return_value = _candles([500, 502])
        positions = {"SPY": {"quantity": 10.0, "avg_entry_price": 495.0}}
        s = PreFomcDrift(broker=broker)
        out = s.compute(_ctx(
            {"macro_fomc_window": {"next_meeting": self._MEETING,
                                    "days_to_next": 0, "blackout": True}},
            open_positions=positions))
        assert all(p.side == OrderSide.SELL for p in out), (
            "must not hold or enter through the announcement")

    def test_already_held_no_duplicate_buy(self, monkeypatch):
        self._freeze_now(monkeypatch,
                         datetime(2026, 6, 17, 6, 30, tzinfo=UTC))
        broker = MagicMock()
        broker.get_candles.return_value = _candles([500, 501])
        positions = {
            "SPY": {"quantity": 10.0, "avg_entry_price": 495.0,
                     "current_price": 500.0},
        }
        s = PreFomcDrift(broker=broker)
        out = s.compute(_ctx(
            {"macro_fomc_window": {"next_meeting": self._MEETING,
                                    "days_to_next": 0, "blackout": True}},
            open_positions=positions))
        # Only QQQ should be proposed; SPY already held.
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert {p.symbol for p in buys} == {"QQQ"}

    def test_hard_stop_fires_inside_window(self):
        # Held SPY @ 500, last 489 → 2.2% loss > 2% stop.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([495, 489])
        positions = {"SPY": {"quantity": 10.0,
                              "avg_entry_price": 500.0}}
        s = PreFomcDrift(broker=broker)
        out = s.compute(_ctx(
            {"macro_fomc_window": {"next_meeting": "2026-06-06",
                                    "days_to_next": 1, "blackout": True}},
            open_positions=positions))
        sells = [p for p in out if p.side == OrderSide.SELL]
        assert any(p.symbol == "SPY" and p.is_closing for p in sells)


# ─── activist_13d ─────────────────────────────────────────────────


class TestActivist13D:
    def test_no_signal_no_proposals(self):
        s = Activist13D(broker=MagicMock())
        assert s.compute(_ctx({})) == []

    def test_fresh_filing_triggers_buy(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20, 21])
        s = Activist13D(broker=broker)
        recent = datetime.now(UTC) - timedelta(hours=6)
        out = s.compute(_ctx({"activist_13d_new": [
            {"ticker": "FOO", "filed_date": recent.isoformat()}
        ]}))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert len(buys) == 1
        assert buys[0].symbol == "FOO"

    def test_stale_filing_rejected(self):
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20, 21])
        s = Activist13D(broker=broker)
        stale = datetime.now(UTC) - timedelta(days=7)
        out = s.compute(_ctx({"activist_13d_new": [
            {"ticker": "FOO", "filed_date": stale.isoformat()}
        ]}))
        assert [p for p in out if p.side == OrderSide.BUY] == []

    def test_concurrent_position_cap(self):
        # 4 already held → no new entries even on fresh signal.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20, 21])
        positions = {sym: {"quantity": 10.0, "avg_entry_price": 20.0}
                     for sym in ("A", "B", "C", "D")}
        s = Activist13D(broker=broker)
        recent = datetime.now(UTC) - timedelta(hours=1)
        out = s.compute(_ctx(
            {"activist_13d_new": [
                {"ticker": "NEW", "filed_date": recent.isoformat()}]},
            open_positions=positions))
        assert [p for p in out
                if p.side == OrderSide.BUY and p.symbol == "NEW"] == []

    def test_take_profit_fires(self):
        # Held @ $20, last $21.40 → +7% > 6% take-profit.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([20.50, 21.40])
        positions = {"FOO": {"quantity": 5.0, "avg_entry_price": 20.0}}
        s = Activist13D(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        sells = [p for p in out if p.symbol == "FOO" and p.is_closing]
        assert sells, "expected take-profit exit"


# ─── merger_arb ─────────────────────────────────────────────────


class TestMergerArb:
    def test_no_deals_no_proposals(self):
        s = MergerArb(broker=MagicMock())
        assert s.compute(_ctx({})) == []

    def test_positive_spread_triggers_buy(self):
        # Target trading @ $48 vs deal @ $50 → +4.2% spread, > 1% min.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([47, 48])
        s = MergerArb(broker=broker)
        out = s.compute(_ctx({"merger_arb_deal": [
            {"target": "TGT", "deal_price": 50.0,
             "announced": "2026-05-01", "status": "PENDING"}
        ]}))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert len(buys) == 1
        assert buys[0].symbol == "TGT"
        assert buys[0].metadata["spread_pct"] > 0.01

    def test_negative_spread_rejected(self):
        # Trading $51 vs deal $50 → -2% over, skip.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([50.5, 51])
        s = MergerArb(broker=broker)
        out = s.compute(_ctx({"merger_arb_deal": [
            {"target": "TGT", "deal_price": 50.0,
             "announced": "2026-05-01"}]}))
        assert [p for p in out if p.side == OrderSide.BUY] == []

    def test_deal_break_stop(self):
        # Held @ $50, last $45 → -10% > 8% stop.
        broker = MagicMock()
        broker.get_candles.return_value = _candles([48, 45])
        positions = {"TGT": {"quantity": 100.0, "avg_entry_price": 50.0}}
        s = MergerArb(broker=broker)
        out = s.compute(_ctx({}, open_positions=positions))
        sells = [p for p in out if p.is_closing and p.symbol == "TGT"]
        assert sells, "expected deal-break stop"


# ─── high_vol_trend ─────────────────────────────────────────────────


class TestHighVolTrend:
    def _trending_candles(self, n=280, slope=0.001):
        # Trending series WITH noise so realized-vol is non-zero (vol
        # scaling divides by it). A perfectly smooth geometric series
        # has constant log-returns and zero vol, which legitimately
        # disables the strategy's leverage step.
        import random
        random.seed(7)
        # 0.5% daily noise → ann vol ~8% (< 17% target → strategy levers
        # up). Higher noise made realized vol meet target exactly → 1×
        # leverage and the assertion below saw no proof of vol scaling.
        return _candles([100 * (1 + slope) ** i
                         * (1 + random.gauss(0, 0.005))
                         for i in range(n)])

    def _flat_candles(self, n=280):
        return _candles([100.0 for _ in range(n)])

    def test_uptrend_triggers_buy(self):
        broker = MagicMock()
        broker.get_candles.return_value = self._trending_candles()
        s = HighVolTrend(broker=broker)
        out = s.compute(_ctx(alloc_usd=50_000))
        buys = [p for p in out if p.side == OrderSide.BUY]
        assert buys, "expected at least one trend entry"
        # Leverage scaling should produce notional > equal-weight base.
        equal_weight = 50_000 / 10   # 10 ETFs in UNIVERSE
        any_levered = any(
            (p.notional_usd or 0) > equal_weight for p in buys)
        assert any_levered, "expected vol-targeted leverage on at least one name"
        # Per-name leverage capped at 4x.
        for p in buys:
            assert (p.notional_usd or 0) <= equal_weight * 4 + 1e-6

    def test_flat_trend_no_entries(self):
        broker = MagicMock()
        broker.get_candles.return_value = self._flat_candles()
        s = HighVolTrend(broker=broker)
        out = s.compute(_ctx(alloc_usd=50_000))
        assert [p for p in out if p.side == OrderSide.BUY] == []

    def test_downtrend_exits_position(self):
        # Held in a downtrending name → trend flip exit.
        broker = MagicMock()
        broker.get_candles.return_value = _candles(
            [100 * (1 - 0.001) ** i for i in range(280)])
        positions = {"SPY": {"quantity": 5.0, "avg_entry_price": 100.0,
                              "current_price": 75.0}}
        s = HighVolTrend(broker=broker)
        out = s.compute(_ctx(alloc_usd=50_000,
                              open_positions=positions))
        spy_sells = [p for p in out if p.symbol == "SPY"
                     and p.side == OrderSide.SELL and p.is_closing]
        assert spy_sells, "expected exit on trend flip"
