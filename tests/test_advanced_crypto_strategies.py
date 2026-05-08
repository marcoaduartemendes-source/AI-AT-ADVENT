"""Tests for the Phase 6 advanced crypto strategies.

Each test exercises the public contract (compute() -> proposals)
without hitting any real broker. Uses MockBroker with custom candle
fixtures to drive deterministic behavior.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from brokers.base import Candle, OrderSide
from strategies.crypto_breakout import CryptoBreakout
from strategies.crypto_pairs_trading import CryptoPairsTrading
from strategies.crypto_vol_regime_overlay import CryptoVolRegimeOverlay
from strategy_engine.base import StrategyContext


def _ctx(target_alloc_usd: float = 10_000, scout_signals: dict | None = None,
         open_positions: dict | None = None) -> StrategyContext:
    return StrategyContext(
        timestamp=datetime.now(UTC),
        portfolio_equity_usd=100_000,
        target_alloc_pct=0.1,
        target_alloc_usd=target_alloc_usd,
        risk_multiplier=1.0,
        open_positions=open_positions or {},
        scout_signals=scout_signals or {},
    )


def _candles(closes: list[float], volume: float = 1000.0) -> list[Candle]:
    """Make a list of Candles with the given closes (oldest first)."""
    return [
        Candle(
            timestamp=datetime.now(UTC),
            open=c, high=c * 1.01, low=c * 0.99, close=c,
            volume=volume,
        )
        for c in closes
    ]


# ─── crypto_pairs_trading ────────────────────────────────────────────


class TestCryptoPairsTrading:

    def _make_strategy(self, btc_closes: list[float], eth_closes: list[float],
                          sol_closes: list[float] | None = None):
        broker = MagicMock()
        sol_closes = sol_closes or [c * 0.05 for c in eth_closes]
        def fake_candles(symbol, granularity, num_candles=100):
            return {
                "BTC-USD": _candles(btc_closes),
                "ETH-USD": _candles(eth_closes),
                "SOL-USD": _candles(sol_closes),
            }[symbol]
        broker.get_candles.side_effect = fake_candles
        return CryptoPairsTrading(broker)

    def test_no_signals_when_z_in_band(self):
        # Stable ratio → z near 0 → no entry
        s = self._make_strategy(
            btc_closes=[40000.0] * 35,
            eth_closes=[2000.0] * 35,
        )
        proposals = s.compute(_ctx())
        assert proposals == []

    def test_high_z_triggers_pair_entry(self):
        # Last day BTC spikes hugely → ratio z >> 2 → BTC is "rich",
        # ETH is "cheap". Spot-only mode: long the cheap leg only
        # (no perp short until F6 lands).
        btc = [40000.0] * 30 + [40000.0, 40000.0, 40000.0, 40000.0, 60000.0]
        eth = [2000.0] * 35
        s = self._make_strategy(btc_closes=btc, eth_closes=eth)
        proposals = s.compute(_ctx())
        symbols_sides = {(p.symbol, p.side) for p in proposals}
        # Long the cheap leg (ETH) when BTC is rich.
        assert ("ETH-USD", OrderSide.BUY) in symbols_sides
        # No SELLs until perp support lands (would have been rejected
        # by the spot adapter anyway).
        assert not any(side == OrderSide.SELL for _, side in symbols_sides), (
            f"Spot-only mode should not propose SELLs on entry, got {symbols_sides}"
        )

    def test_zero_alloc_emits_nothing(self):
        s = self._make_strategy([1] * 35, [1] * 35)
        assert s.compute(_ctx(target_alloc_usd=0)) == []


# ─── crypto_breakout ─────────────────────────────────────────────────


class TestCryptoBreakout:

    def _make(self, closes_per_symbol: dict[str, list[float]],
              volume: float = 1000.0):
        broker = MagicMock()
        def fake(symbol, granularity, num_candles=100):
            cs = closes_per_symbol.get(symbol)
            if cs is None:
                raise Exception(f"No data for {symbol}")
            return _candles(cs, volume=volume)
        broker.get_candles.side_effect = fake
        return CryptoBreakout(broker)

    def test_breakout_with_volume_triggers_entry(self):
        # 30d high = 40000; today closes at 50000 (clear breakout) on
        # 2× volume → entry expected.
        closes = [40000.0] * 30 + [50000.0]
        broker = MagicMock()
        def fake(symbol, granularity, num_candles=100):
            base = _candles(closes, volume=1000.0)
            base[-1] = Candle(
                timestamp=datetime.now(UTC),
                open=50000, high=50500, low=49500, close=50000,
                volume=2500,    # > 1.5× of 1000 baseline
            )
            return base
        broker.get_candles.side_effect = fake
        s = CryptoBreakout(broker)
        proposals = s.compute(_ctx())
        assert any(p.side == OrderSide.BUY for p in proposals), (
            f"Expected at least one BUY proposal, got {proposals}"
        )

    def test_no_breakout_no_entry(self):
        # Flat: no new high → no entry
        s = self._make({sym: [40000.0] * 31 for sym in
                         ["BTC-USD", "ETH-USD", "SOL-USD",
                          "AVAX-USD", "MATIC-USD"]})
        proposals = s.compute(_ctx())
        assert all(p.side != OrderSide.BUY for p in proposals)

    def test_high_vol_regime_blocks_entry(self):
        # Even with a clean breakout setup, HIGH crypto vol should
        # block entries.
        closes = [40000.0] * 30 + [50000.0]
        broker = MagicMock()
        broker.get_candles.return_value = _candles(closes, volume=2500.0)
        s = CryptoBreakout(broker)
        ctx = _ctx(scout_signals={
            "crypto_vol_scaler": {
                "btc_realized_vol": 0.95,    # 95% — hard floor
                "crypto_regime": "HIGH",
            },
        })
        assert s.compute(ctx) == []


# ─── crypto_vol_regime_overlay ───────────────────────────────────────


class TestCryptoVolRegimeOverlay:

    def _make(self, btc_closes: list[float]):
        broker = MagicMock()
        broker.get_candles.return_value = _candles(btc_closes)
        bus = MagicMock()
        s = CryptoVolRegimeOverlay(broker, bus=bus)
        return s, bus

    def test_low_vol_publishes_low_regime(self):
        # Tiny day-to-day moves → low ann vol → LOW regime
        closes = [40000 + i for i in range(35)]    # +1 USD/day, very low vol
        s, bus = self._make(closes)
        s.compute(_ctx())
        bus.publish.assert_called_once()
        kwargs = bus.publish.call_args.kwargs
        assert kwargs["signal_type"] == "crypto_vol_scaler"
        payload = kwargs["payload"]
        assert payload["crypto_regime"] in ("LOW", "MEDIUM")
        assert payload["crypto_momentum"] >= 0.7

    def test_high_vol_publishes_high_regime_scaler(self):
        # Large alternating moves → high ann vol → HIGH regime → scaler 0.3
        import random
        random.seed(42)
        closes = [40000.0]
        for _ in range(40):
            # ~10% daily moves alternating sign → annualized vol >> 80%
            closes.append(closes[-1] * (1 + random.choice([-0.1, 0.1])))
        s, bus = self._make(closes)
        s.compute(_ctx())
        payload = bus.publish.call_args.kwargs["payload"]
        assert payload["crypto_regime"] == "HIGH"
        assert payload["crypto_momentum"] == pytest.approx(0.3)

    def test_emits_zero_proposals(self):
        s, _ = self._make([40000.0] * 35)
        assert s.compute(_ctx()) == []


# ─── helpers integration ────────────────────────────────────────────


class TestCryptoVolScalerHelper:
    """The crypto_vol_scaler helper must read what the overlay publishes."""

    def test_scaler_reads_overlay_payload(self):
        from strategies._helpers import crypto_regime, crypto_vol_scaler
        ctx = _ctx(scout_signals={
            "crypto_vol_scaler": {
                "crypto_momentum": 0.3,
                "crypto_regime": "HIGH",
                "crypto_regime_multiplier": 0.3,
                "btc_realized_vol": 0.85,
            },
        })
        assert crypto_vol_scaler(ctx) == pytest.approx(0.3)
        assert crypto_regime(ctx) == "HIGH"

    def test_missing_signal_returns_default(self):
        from strategies._helpers import crypto_regime, crypto_vol_scaler
        ctx = _ctx()
        assert crypto_vol_scaler(ctx) == pytest.approx(1.0)
        assert crypto_regime(ctx) == "UNKNOWN"
