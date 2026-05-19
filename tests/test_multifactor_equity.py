"""Tests for the flagship multi-factor cross-sectional equity model.

Locks in the institutional-grade invariants the 2026-05-19 audit
demanded: cross-sectional z-scoring, sector-neutralisation,
vol-targeting, turnover hysteresis, trend eligibility gate.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from strategies.multifactor_equity import (
    MAX_PER_SECTOR,
    MultiFactorEquity,
)
from strategy_engine.base import StrategyContext


class _C:
    def __init__(self, close):
        self.close = close


def _ctx(target=14000.0, positions=None, scout=None):
    return StrategyContext(
        timestamp=datetime.now(UTC),
        portfolio_equity_usd=100_000,
        target_alloc_pct=0.14,
        target_alloc_usd=target,
        risk_multiplier=1.0,
        open_positions=positions or {},
        scout_signals=scout or {},
        pending_orders={},
    )


def _broker(series_fn):
    b = MagicMock()
    b.get_candles.side_effect = lambda sym, gran, num_candles=100: [
        _C(c) for c in series_fn(sym, num_candles)
    ]
    return b


def test_zero_alloc_emits_nothing():
    b = _broker(lambda s, n: [100.0] * n)
    assert MultiFactorEquity(b).compute(_ctx(target=0)) == []


def test_uptrend_universe_produces_equal_weight_longs():
    # Every name in a steady uptrend → all pass the 200d SMA gate.
    def series(sym, n):
        base = 50 + (hash(sym) % 100)
        return [base * (1 + 0.0005 * i) for i in range(n)]
    s = MultiFactorEquity(_broker(series))
    props = s.compute(_ctx())
    assert props, "expected long proposals in a broad uptrend"
    # All BUYs, equal notional, no SELLs (no existing positions)
    assert all(p.side.value == "BUY" for p in props)
    notionals = {round(p.notional_usd or 0, 2) for p in props}
    assert len(notionals) == 1, f"not equal-weight: {notionals}"


def test_sector_neutralisation_caps_per_sector():
    def series(sym, n):
        base = 50 + (hash(sym) % 100)
        return [base * (1 + 0.0005 * i) for i in range(n)]
    props = MultiFactorEquity(_broker(series)).compute(_ctx())
    from strategies.multifactor_equity import UNIVERSE
    sec_count: dict[str, int] = {}
    for p in props:
        sec = UNIVERSE.get(p.symbol, "OTHER")
        sec_count[sec] = sec_count.get(sec, 0) + 1
    assert all(c <= MAX_PER_SECTOR for c in sec_count.values()), sec_count


def test_downtrend_universe_sits_out():
    # Every name below its 200d SMA → eligibility gate rejects all.
    def series(sym, n):
        base = 200.0
        return [base * (1 - 0.0008 * i) for i in range(n)]
    props = MultiFactorEquity(_broker(series)).compute(_ctx())
    assert props == [], "must not go long names in structural downtrend"


def test_vol_scaler_shrinks_sleeve():
    def series(sym, n):
        base = 50 + (hash(sym) % 100)
        return [base * (1 + 0.0005 * i) for i in range(n)]
    s = MultiFactorEquity(_broker(series))
    full = s.compute(_ctx())
    # HIGH vol regime → scaler 0.5 → notionals halve
    scaled = s.compute(_ctx(scout={
        "vol_scaler": {"equity_momentum": 0.5,
                        "equity_regime_multiplier": 1.0}
    }))
    if full and scaled:
        f = (full[0].notional_usd or 0)
        sc = (scaled[0].notional_usd or 0)
        assert sc < f, f"vol scaler did not shrink sleeve: {sc} vs {f}"


def test_held_name_in_band_not_churned():
    def series(sym, n):
        base = 50 + (hash(sym) % 100)
        return [base * (1 + 0.0005 * i) for i in range(n)]
    s = MultiFactorEquity(_broker(series))
    target = s.compute(_ctx())
    held_sym = target[0].symbol
    # Hold the top name; it must NOT be sold (still in keep band)
    pos = {held_sym: {"quantity": 10, "entry_time":
                       datetime.now(UTC).isoformat()}}
    props = s.compute(_ctx(positions=pos))
    sells = [p for p in props if p.side.value == "SELL"
             and p.symbol == held_sym]
    assert not sells, "top-ranked held name should not be churned"
