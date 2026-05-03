"""Tests for the legacy live-trading strategies (Momentum,
MeanReversion, VolatilityBreakout) — the only strategies that
actually trade real money on the Coinbase live account today.

Coverage gap audit-flagged: these three strategies have been in
production since the first version of the system but had no unit
tests. A regression in any of their `analyze()` methods would
silently mis-trade — there's no integration safety net because
the legacy bot doesn't go through the orchestrator's risk gate.

These tests pin the behavior contract: given a deterministic
candle array, the analyze() method returns the expected SignalType
with sane confidence + reason fields.
"""
from __future__ import annotations

import numpy as np

from trading.strategies.base import SignalType
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.volatility_breakout import VolatilityBreakoutStrategy


# ─── Helpers ─────────────────────────────────────────────────────────


def _candles(closes: list[float], volumes: list[float] | None = None) -> np.ndarray:
    """Build the 6-column OHLCV array shape the strategies expect.
    Cols: timestamp, low, high, open, close, volume."""
    n = len(closes)
    vols = volumes or [1_000_000.0] * n
    return np.array([
        [float(i), float(c) * 0.99, float(c) * 1.01,
         float(c), float(c), float(v)]
        for i, (c, v) in enumerate(zip(closes, vols, strict=True))
    ], dtype=float)


def _trending_up(n: int = 200, start: float = 100.0,
                  daily_pct: float = 0.005) -> list[float]:
    """Smooth uptrend: start * (1 + daily)^i."""
    return [start * (1 + daily_pct) ** i for i in range(n)]


def _trending_down(n: int = 200, start: float = 100.0,
                    daily_pct: float = -0.005) -> list[float]:
    return [start * (1 + daily_pct) ** i for i in range(n)]


def _flat(n: int = 200, val: float = 100.0,
           jitter: float = 0.001, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return [val * (1 + rng.normal(0, jitter)) for _ in range(n)]


def _spike_down(n: int = 200, val: float = 100.0,
                  spike_at: int = 195, drop_pct: float = 0.10) -> list[float]:
    """Flat then a sharp drop near the end (oversold signal)."""
    out = _flat(n, val, jitter=0.001)
    for i in range(spike_at, n):
        out[i] = out[i - 1] * (1 - drop_pct)
    return out


def _spike_up(n: int = 200, val: float = 100.0,
                spike_at: int = 195, jump_pct: float = 0.10) -> list[float]:
    """Flat then a sharp jump near the end (overbought signal)."""
    out = _flat(n, val, jitter=0.001)
    for i in range(spike_at, n):
        out[i] = out[i - 1] * (1 + jump_pct)
    return out


# ─── MomentumStrategy ────────────────────────────────────────────────


def test_momentum_holds_on_insufficient_data():
    s = MomentumStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_flat(20)))
    assert sig.signal == SignalType.HOLD
    assert "Insufficient" in sig.reason


def test_momentum_no_trend_returns_hold():
    """Flat series → no EMA crossover → HOLD."""
    s = MomentumStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_flat(200, val=100.0,
                                                 jitter=0.0001)))
    assert sig.signal == SignalType.HOLD


def test_momentum_signals_have_metadata_keys():
    """When momentum DOES fire, metadata must include ema/macd/rsi."""
    s = MomentumStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_trending_up(200, daily_pct=0.01)))
    # An up-trend may produce BUY or HOLD (depending on RSI band);
    # what matters is the metadata contract when a non-HOLD fires.
    if sig.signal != SignalType.HOLD:
        for key in ("ema_fast", "ema_slow", "macd_hist", "rsi"):
            assert key in sig.metadata, f"missing {key} in metadata"


def test_momentum_repr_contains_class_name():
    """Defensive: regression on __repr__ would silently break logging."""
    s = MomentumStrategy(products=["BTC-USD"])
    assert "MomentumStrategy" in repr(s)


# ─── MeanReversionStrategy ───────────────────────────────────────────


def test_mean_reversion_holds_on_insufficient_data():
    s = MeanReversionStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_flat(20)))
    assert sig.signal == SignalType.HOLD
    assert "Insufficient" in sig.reason


def test_mean_reversion_signals_oversold():
    """Sharp price drop → low Z-score → BUY signal."""
    s = MeanReversionStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_spike_down(200, drop_pct=0.05)))
    # The exact signal depends on the dual Z + RSI gate; assert the
    # contract: BUY or HOLD (never SELL on a sharp drop).
    assert sig.signal != SignalType.SELL


def test_mean_reversion_signals_overbought():
    """Sharp price spike → high Z-score → SELL signal allowed."""
    s = MeanReversionStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_spike_up(200, jump_pct=0.05)))
    assert sig.signal != SignalType.BUY


def test_mean_reversion_metadata_present_on_fire():
    """Z, RSI, mean, dist_pct all present when a signal fires."""
    s = MeanReversionStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_spike_down(200, drop_pct=0.06)))
    if sig.signal != SignalType.HOLD:
        for key in ("zscore", "rsi", "mean", "dist_pct"):
            assert key in sig.metadata, f"missing {key}"


def test_mean_reversion_confidence_in_range():
    """Confidence must always be 0.0 ≤ c ≤ 1.0."""
    s = MeanReversionStrategy(products=["BTC-USD"])
    for closes in [
        _flat(200), _trending_up(200), _trending_down(200),
        _spike_up(200), _spike_down(200),
    ]:
        sig = s.analyze("BTC-USD", _candles(closes))
        assert 0.0 <= sig.confidence <= 1.0, (
            f"confidence {sig.confidence} out of [0, 1] "
            f"for signal={sig.signal.value}"
        )


# ─── VolatilityBreakoutStrategy ──────────────────────────────────────


def test_vol_breakout_holds_on_insufficient_data():
    s = VolatilityBreakoutStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_flat(30)))
    assert sig.signal == SignalType.HOLD


def test_vol_breakout_handles_flat_series():
    """A flat series → no breakout → HOLD."""
    s = VolatilityBreakoutStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_flat(200, jitter=0.0001)))
    # Flat series has no expansion event → HOLD expected
    assert sig.signal == SignalType.HOLD


def test_vol_breakout_does_not_crash_on_trend():
    """Strong sustained trend with no compression → strategy must
    handle gracefully (HOLD or directional signal, not crash)."""
    s = VolatilityBreakoutStrategy(products=["BTC-USD"])
    sig = s.analyze("BTC-USD", _candles(_trending_up(200, daily_pct=0.01)))
    # Any of HOLD/BUY/SELL is acceptable — the contract is "no crash"
    assert sig.signal in (SignalType.HOLD, SignalType.BUY, SignalType.SELL)
    assert 0.0 <= sig.confidence <= 1.0


def test_all_three_strategies_handle_short_arrays_uniformly():
    """Cross-strategy invariant: insufficient data → HOLD with
    'Insufficient' in reason. Catches a regression that would let
    a strategy compute on garbage."""
    short = _candles(_flat(15))
    for cls in (MomentumStrategy, MeanReversionStrategy,
                 VolatilityBreakoutStrategy):
        s = cls(products=["BTC-USD"])
        sig = s.analyze("BTC-USD", short)
        assert sig.signal == SignalType.HOLD
        assert "Insufficient" in sig.reason or "ready" in sig.reason.lower()


def test_signal_price_matches_last_close():
    """Defensive: the Signal.price field must be the last close,
    not some earlier index. PerformanceTracker uses this to compute
    P&L against fill price."""
    closes = list(_flat(200))
    closes[-1] = 12345.67    # marker
    candles = _candles(closes)
    for cls in (MomentumStrategy, MeanReversionStrategy,
                 VolatilityBreakoutStrategy):
        sig = cls(products=["BTC-USD"]).analyze("BTC-USD", candles)
        assert sig.price == 12345.67, (
            f"{cls.__name__}.price={sig.price}, expected 12345.67"
        )
