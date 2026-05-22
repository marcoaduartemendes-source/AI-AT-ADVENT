"""Tests for the dual_momentum strategy + backtest.

No network in CI, so the backtest's _yahoo_history is monkeypatched with
synthetic price series and the live strategy is fed a fake broker whose
get_candles returns controllable trends. These verify the SELECTION
logic (top-3 by momentum, absolute-momentum gate, Treasury risk-off),
not the empirical edge — the real edge verdict comes from the validation
harness running against live Yahoo data on the box.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from brokers.base import Candle, OrderSide
from strategy_engine.base import StrategyContext
from strategies.dual_momentum import DualMomentum, RISK_UNIVERSE, SAFE_ASSET


def _series(start: float, daily_drift: float, n: int) -> np.ndarray:
    """(n,6) Yahoo-shaped array; col0=ts, col4=close. Deterministic
    geometric drift so 12-1m momentum has a known sign."""
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    rows = []
    px = start
    for i in range(n):
        ts = (t0 + timedelta(days=i)).timestamp()
        px *= (1 + daily_drift)
        rows.append([ts, px, px, px, px, 1000.0])
    return np.array(rows)


def _candles(start: float, daily_drift: float, n: int) -> list[Candle]:
    now = datetime.now(UTC)
    out = []
    px = start
    for i in range(n):
        px *= (1 + daily_drift)
        out.append(Candle(timestamp=now - timedelta(days=n - i),
                          open=px, high=px, low=px, close=px, volume=1000))
    return out


class _FakeBroker:
    """Broker stub that returns a per-symbol candle series."""
    def __init__(self, drifts: dict[str, float]):
        self._drifts = drifts

    def get_candles(self, symbol, granularity, num_candles=100):
        drift = self._drifts.get(symbol, 0.0)
        return _candles(100.0, drift, num_candles)


def _ctx(positions=None):
    return StrategyContext(
        timestamp=datetime.now(UTC), portfolio_equity_usd=100_000.0,
        target_alloc_pct=0.03, target_alloc_usd=3000.0, risk_multiplier=1.0,
        open_positions=positions or {}, scout_signals={}, pending_orders={},
    )


class TestBacktestSelection:
    def test_buys_the_trending_winners(self, monkeypatch):
        import backtests.runner as r
        # SPY/QQQ/EFA up, EEM/VNQ/GLD down, IEF mild up.
        up, dn = 0.0008, -0.0006
        drift = {"SPY": up, "QQQ": up, "EFA": up,
                 "EEM": dn, "VNQ": dn, "GLD": dn, "IEF": 0.0002}
        monkeypatch.setattr(
            r, "_yahoo_history",
            lambda sym, days: _series(100.0, drift.get(sym, 0.0), days + 320))
        summ = r.backtest_dual_momentum(252)
        bought = {t["product_id"] for t in summ.trades if t["side"] == "BUY"}
        # Top-3 positive-momentum names get bought; the down names don't.
        assert {"SPY", "QQQ", "EFA"} <= bought
        assert "EEM" not in bought and "GLD" not in bought
        assert summ.n_trades >= 0  # closed trades may be 0 if never rotated

    def test_all_risk_down_rotates_to_treasuries(self, monkeypatch):
        import backtests.runner as r
        drift = {s: -0.0007 for s in RISK_UNIVERSE}
        drift[SAFE_ASSET] = 0.0003
        monkeypatch.setattr(
            r, "_yahoo_history",
            lambda sym, days: _series(100.0, drift.get(sym, 0.0), days + 320))
        summ = r.backtest_dual_momentum(252)
        bought = {t["product_id"] for t in summ.trades if t["side"] == "BUY"}
        # Every risk slot fails the absolute-momentum gate → all to IEF.
        assert SAFE_ASSET in bought
        assert not ({"SPY", "QQQ", "EFA", "EEM", "VNQ", "GLD"} & bought)


class TestLiveSelection:
    def test_proposes_top3_risk_assets(self):
        up, dn = 0.0009, -0.0007
        broker = _FakeBroker({"SPY": up, "QQQ": up, "EFA": up,
                              "EEM": dn, "VNQ": dn, "GLD": dn, "IEF": 0.0002})
        proposals = DualMomentum(broker).compute(_ctx())
        buys = {p.symbol for p in proposals if p.side == OrderSide.BUY}
        assert {"SPY", "QQQ", "EFA"} <= buys
        assert SAFE_ASSET not in buys   # all 3 slots risk-on

    def test_risk_off_buys_treasuries(self):
        broker = _FakeBroker({**{s: -0.0008 for s in RISK_UNIVERSE},
                              SAFE_ASSET: 0.0003})
        proposals = DualMomentum(broker).compute(_ctx())
        buys = {p.symbol for p in proposals if p.side == OrderSide.BUY}
        assert SAFE_ASSET in buys
        assert not ({s for s in RISK_UNIVERSE} & buys)

    def test_no_alloc_no_proposals(self):
        broker = _FakeBroker({s: 0.001 for s in RISK_UNIVERSE + [SAFE_ASSET]})
        ctx = _ctx()
        ctx.target_alloc_usd = 0.0
        assert DualMomentum(broker).compute(ctx) == []
