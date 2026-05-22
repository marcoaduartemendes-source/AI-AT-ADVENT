"""Tests for the bond_carry + commodity_momentum sleeves.

No network: backtests' _yahoo_history is monkeypatched and the live
strategies are fed a fake broker. These verify the SELECTION logic
(trend-gated carry basket; top-K commodity momentum), not the empirical
edge — the validation harness decides that on real data.
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np

from brokers.base import Candle, OrderSide
from strategy_engine.base import StrategyContext
from strategies.bond_carry import BondCarry, CARRY_UNIVERSE, SAFE_ASSET
from strategies.commodity_momentum import CommodityMomentum, UNIVERSE as COMMOD


def _series(start, daily_drift, n):
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    px = start
    rows = []
    for i in range(n):
        px *= (1 + daily_drift)
        rows.append([(t0 + timedelta(days=i)).timestamp(), px, px, px, px, 1e3])
    return np.array(rows)


def _candles(start, daily_drift, n):
    now = datetime.now(UTC)
    px = start
    out = []
    for i in range(n):
        px *= (1 + daily_drift)
        out.append(Candle(timestamp=now - timedelta(days=n - i),
                          open=px, high=px, low=px, close=px, volume=1e3))
    return out


class _FakeBroker:
    def __init__(self, drifts):
        self._d = drifts

    def get_candles(self, symbol, granularity, num_candles=100):
        return _candles(100.0, self._d.get(symbol, 0.0), num_candles)


def _ctx(usd=3000.0):
    return StrategyContext(
        timestamp=datetime.now(UTC), portfolio_equity_usd=100_000.0,
        target_alloc_pct=0.03, target_alloc_usd=usd, risk_multiplier=1.0,
        open_positions={}, scout_signals={}, pending_orders={})


class TestBondCarry:
    def test_uptrend_holds_carry_basket(self):
        broker = _FakeBroker({s: 0.0004 for s in CARRY_UNIVERSE + [SAFE_ASSET]})
        buys = {p.symbol for p in BondCarry(broker).compute(_ctx())
                if p.side == OrderSide.BUY}
        assert set(CARRY_UNIVERSE) <= buys     # all above trend → carry on
        assert SAFE_ASSET not in buys

    def test_downtrend_rotates_to_shy(self):
        # Carry sleeves declining (below their SMA), SHY flat/up.
        drifts = {s: -0.0005 for s in CARRY_UNIVERSE}
        drifts[SAFE_ASSET] = 0.0001
        buys = {p.symbol for p in BondCarry(_FakeBroker(drifts)).compute(_ctx())
                if p.side == OrderSide.BUY}
        assert SAFE_ASSET in buys
        assert not (set(CARRY_UNIVERSE) & buys)

    def test_backtest_runs(self, monkeypatch):
        import backtests.runner as r
        drifts = {s: 0.0004 for s in CARRY_UNIVERSE}
        drifts[SAFE_ASSET] = 0.0001
        monkeypatch.setattr(r, "_yahoo_history",
                            lambda s, d: _series(100, drifts.get(s, 0.0), d + 160))
        summ = r.backtest_bond_carry(252)
        assert summ.strategy == "bond_carry"
        assert summ.entry_volume_usd > 0


class TestCommodityMomentum:
    def test_buys_top3_positive_momentum(self):
        # 3 strong uptrends, rest down → top-3 winners bought, losers not.
        up = {"GLD": 0.0010, "DBC": 0.0009, "CPER": 0.0008}
        drifts = {s: -0.0006 for s in COMMOD}
        drifts.update(up)
        buys = {p.symbol for p in CommodityMomentum(_FakeBroker(drifts)).compute(_ctx())
                if p.side == OrderSide.BUY}
        assert buys == {"GLD", "DBC", "CPER"}

    def test_all_negative_buys_nothing(self):
        drifts = {s: -0.0006 for s in COMMOD}
        props = CommodityMomentum(_FakeBroker(drifts)).compute(_ctx())
        assert [p for p in props if p.side == OrderSide.BUY] == []

    def test_backtest_runs(self, monkeypatch):
        import backtests.runner as r
        # rotating leadership so the top-3 churns → closed trades
        def hist(sym, days):
            phase = COMMOD.index(sym) * 0.7
            t0 = 1_600_000_000
            px = 100.0
            rows = []
            for i in range(days):
                px *= (1 + 0.0005 + 0.0009 * math.sin(i / 50.0 + phase))
                rows.append([t0 + i * 86400, px, px, px, px, 1e3])
            return np.array(rows)
        monkeypatch.setattr(r, "_yahoo_history", hist)
        summ = r.backtest_commodity_momentum(504)
        assert summ.strategy == "commodity_momentum"
        assert summ.entry_volume_usd > 0
