"""Crypto vol-regime overlay — publishes a scaler that other crypto
strategies multiply their target_alloc_usd by.

Mirrors `vol_managed_overlay` (the equity version) but tuned for
crypto. Moreira-Muir (2017) on traditional assets shows scaling
inversely to realized vol adds ~0.2-0.4 Sharpe; the same logic
applies in crypto with thresholds that reflect crypto's higher
baseline volatility.

Three regimes (BTC 30d annualized realized vol):
  LOW    < 40%   → 1.0× scaler  (normal)
  MEDIUM 40-80%  → 0.7× scaler  (size down)
  HIGH   > 80%   → 0.3× scaler  (cut to a third)

This strategy doesn't trade — it just publishes a `crypto_vol_scaler`
payload to the signal bus. Consumers (`crypto_xsmom`, `crypto_breakout`,
`crypto_funding_carry`) read it via `_helpers.crypto_vol_scaler` and
multiply `ctx.target_alloc_usd` by the return value.

Why this is more advanced than running each crypto strategy in isolation:
  - Centralizes the "what regime are we in" decision so all crypto
    strategies move together rather than each computing its own
    (possibly disagreeing) vol estimate.
  - Crypto sleeve correlation is high (BTC dominates) — when BTC
    explodes vol-wise, every crypto strategy benefits from the
    same de-leveraging.
  - Makes the regime visible on the dashboard via the published
    `crypto_regime` field (LOW / MEDIUM / HIGH).
"""
from __future__ import annotations

import logging
import math

import numpy as np

from scouts.signal_bus import SignalBus
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


_ANN = math.sqrt(365)        # crypto trades 365 days/yr, not 252
LOOKBACK_DAYS = 30
LOW_VOL_CEILING = 0.40       # BTC ann_vol < 40% → LOW regime
HIGH_VOL_FLOOR = 0.80        # BTC ann_vol > 80% → HIGH regime

REGIME_SCALERS = {
    "LOW":    1.0,
    "MEDIUM": 0.7,
    "HIGH":   0.3,
    "UNKNOWN": 1.0,         # don't penalize when we have no data
}


class CryptoVolRegimeOverlay(Strategy):
    name = "crypto_vol_regime_overlay"
    venue = "coinbase"

    def __init__(self, broker, bus: SignalBus | None = None):
        super().__init__(broker)
        self._bus = bus or SignalBus()

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        # Always emits zero proposals — this is a publisher-only strategy.
        btc_vol = self._btc_realized_vol()
        if btc_vol is None:
            regime = "UNKNOWN"
            scaler = 1.0
            logger.debug(f"[{self.name}] no BTC candles; publishing UNKNOWN")
        else:
            if btc_vol < LOW_VOL_CEILING:
                regime = "LOW"
            elif btc_vol > HIGH_VOL_FLOOR:
                regime = "HIGH"
            else:
                regime = "MEDIUM"
            scaler = REGIME_SCALERS[regime]

        payload = {
            "crypto_momentum": scaler,
            "crypto_regime": regime,
            "crypto_regime_multiplier": scaler,
            "btc_realized_vol": btc_vol if btc_vol is not None else 0.0,
        }
        try:
            self._bus.publish(
                scout=self.name, venue="overlay",
                signal_type="crypto_vol_scaler", payload=payload,
                ttl_seconds=4 * 3600,
            )
        except Exception as e:
            logger.warning(f"[{self.name}] publish crypto_vol_scaler: {e}")
        if regime != "LOW":
            logger.warning(
                f"[{self.name}] BTC ann_vol="
                f"{(btc_vol or 0)*100:.0f}% → regime={regime}, "
                f"crypto sleeve scaled by {scaler:.2f}"
            )
        return []

    def _btc_realized_vol(self) -> float | None:
        """Return BTC's 30d annualized realized vol from daily candles."""
        try:
            candles = self.broker.get_candles(
                "BTC-USD", "ONE_DAY",
                num_candles=LOOKBACK_DAYS + 5,
            )
        except Exception:
            return None
        if len(candles) < 20:
            return None
        closes = np.array([c.close for c in candles])
        rets = np.diff(closes) / closes[:-1]
        sd = float(np.std(rets, ddof=1))
        if sd <= 0:
            return None
        return sd * _ANN
