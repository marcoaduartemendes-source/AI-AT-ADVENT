"""Tests for the leveraged_champions 3x sleeve.

No network: live strategy fed a fake broker; backtest's _yahoo_history
monkeypatched. Verifies the dynamic champion→proxy resolution and the
regime gate (3x only deploys in low-vol uptrends), not empirical edge.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np

from brokers.base import Candle, OrderSide
from strategy_engine.base import StrategyContext
from strategies.leveraged_champions import (
    LeveragedChampions, PROXY_UNDERLYING, resolve_champions,
)


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
    """get_candles returns a per-symbol trend with controllable vol."""
    def __init__(self, drifts, noise=0.0):
        self._d = drifts
        self._noise = noise

    def get_candles(self, symbol, granularity, num_candles=100):
        drift = self._d.get(symbol, 0.0)
        now = datetime.now(UTC)
        rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
        px = 100.0
        out = []
        for i in range(num_candles):
            shock = rng.normal(0, self._noise) if self._noise else 0.0
            px *= (1 + drift + shock)
            out.append(Candle(timestamp=now - timedelta(days=num_candles - i),
                              open=px, high=px, low=px, close=px, volume=1e3))
        return out


def _ctx(usd=3000.0, positions=None):
    return StrategyContext(
        timestamp=datetime.now(UTC), portfolio_equity_usd=100_000.0,
        target_alloc_pct=0.02, target_alloc_usd=usd, risk_multiplier=1.0,
        open_positions=positions or {}, scout_signals={}, pending_orders={})


class TestResolveChampions:
    def test_ranks_pass_strategies_by_5y_pnl(self, tmp_path, monkeypatch):
        v = {"strategies": {
            "a": {"verdict": "PASS", "pnl_5y": 100.0},
            "b": {"verdict": "PASS", "pnl_5y": 900.0},
            "c": {"verdict": "FAIL", "pnl_5y": 5000.0},   # excluded
            "d": {"verdict": "PASS", "pnl_5y": 500.0},
        }}
        p = tmp_path / "validation.json"
        p.write_text(json.dumps(v))
        # No FIFO db → falls back to pnl_5y ranking.
        monkeypatch.setenv("TRADING_DB_PATH", str(tmp_path / "nope.db"))
        out = resolve_champions(top_n=2, validation_path=str(p))
        assert out == ["b", "d"]            # top-2 PASS by pnl, FAIL excluded

    def test_no_pass_returns_empty(self, tmp_path):
        p = tmp_path / "v.json"
        p.write_text(json.dumps({"strategies": {"a": {"verdict": "FAIL"}}}))
        assert resolve_champions(validation_path=str(p)) == []

    def test_missing_file_returns_empty(self, tmp_path):
        assert resolve_champions(validation_path=str(tmp_path / "x.json")) == []


class TestLeveragedChampionsLive:
    def test_low_vol_uptrend_goes_long_proxies(self, monkeypatch):
        # All underlyings in a clean low-vol uptrend → buy the proxies.
        drifts = {u: 0.0006 for u in PROXY_UNDERLYING.values()}
        monkeypatch.setattr(
            "strategies.leveraged_champions.resolve_champions",
            lambda *a, **k: ["bollinger_breakout", "earnings_momentum"])
        props = LeveragedChampions(_FakeBroker(drifts)).compute(_ctx())
        buys = {p.symbol for p in props if p.side == OrderSide.BUY}
        # bollinger→UPRO, earnings→TQQQ
        assert "UPRO" in buys and "TQQQ" in buys
        assert all(p.symbol in PROXY_UNDERLYING for p in props)

    def test_downtrend_sits_flat(self, monkeypatch):
        # Underlyings below their 200d SMA → no entries (safe state).
        drifts = {u: -0.0008 for u in PROXY_UNDERLYING.values()}
        monkeypatch.setattr(
            "strategies.leveraged_champions.resolve_champions",
            lambda *a, **k: ["bollinger_breakout"])
        props = LeveragedChampions(_FakeBroker(drifts)).compute(_ctx())
        assert [p for p in props if p.side == OrderSide.BUY] == []

    def test_high_vol_uptrend_blocked(self, monkeypatch):
        # Uptrend but choppy (vol > ceiling) → 3x decay gate blocks entry.
        drifts = {u: 0.0010 for u in PROXY_UNDERLYING.values()}
        monkeypatch.setattr(
            "strategies.leveraged_champions.resolve_champions",
            lambda *a, **k: ["bollinger_breakout"])
        broker = _FakeBroker(drifts, noise=0.05)   # huge daily noise
        props = LeveragedChampions(broker).compute(_ctx())
        assert [p for p in props if p.side == OrderSide.BUY] == []

    def test_no_alloc_no_proposals(self, monkeypatch):
        monkeypatch.setattr(
            "strategies.leveraged_champions.resolve_champions",
            lambda *a, **k: ["bollinger_breakout"])
        ctx = _ctx()
        ctx.target_alloc_usd = 0.0
        assert LeveragedChampions(_FakeBroker({})).compute(ctx) == []


class TestLeveragedChampionsBacktest:
    def test_backtest_runs(self, monkeypatch):
        import backtests.runner as r

        def hist(sym, days):
            t0 = 1_500_000_000
            px = 50.0
            rows = []
            for i in range(days):
                px *= 1.0004           # steady uptrend, low vol
                rows.append([t0 + i * 86400, px, px, px, px, 1e3])
            return np.array(rows)
        monkeypatch.setattr(r, "_yahoo_history", hist)
        from backtests.leveraged_thematic_backtest import (
            backtest_leveraged_champions,
        )
        summ = backtest_leveraged_champions(504)
        assert summ.strategy == "leveraged_champions"
        assert summ.entry_volume_usd > 0   # regime-on → it entered
