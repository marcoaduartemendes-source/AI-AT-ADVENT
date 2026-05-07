"""Vol-managed overlay.

Barroso & Santa-Clara (2015) "Momentum has its moments"; Moreira & Muir
(2017) "Volatility-Managed Portfolios": scale momentum exposure inversely
to realized vol. Adds ~0.2-0.4 Sharpe on top of any momentum sleeve at
near-zero implementation cost.

This isn't a standalone strategy that places orders — it's a meta-overlay
that adjusts the *risk multiplier* on momentum-class strategies. We
implement it as a Strategy that publishes a "vol_scaler" signal back to
the bus, which the orchestrator (W3) and momentum strategies can consume.

Concretely: each cycle we compute realized vol of a benchmark portfolio
(SPY for equities, BTC for crypto) and emit scalers in [0.3, 1.5] that
the momentum strategies multiply their target allocations by.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from scouts.signal_bus import SignalBus
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


_ANN = math.sqrt(252)
TARGET_VOL = 0.15            # 15% target — used as the "normal" baseline
LOOKBACK_DAYS = 30

# Realized-correlation regime gate (strategy-audit #1, 2026-05-07).
# 14 of the 24 strategies are functionally long-equity-beta variants
# (RSI, sector, dividend, low_vol, internationals, gap, both PEAD
# variants, earnings_momentum, risk_parity, TSMOM, pairs, Bollinger,
# turn_of_month). When SPY breaks down hard, all of them lose
# correlated money. We scale the equity-momentum sleeve down when
# realized SPY correlation goes structural (vol > 25%) — a poor man's
# proxy for "everything moves together right now, halve the equity
# book."
HIGH_VOL_THRESHOLD = 0.25    # SPY ann_vol above this → halve sleeve
EXTREME_VOL_THRESHOLD = 0.40 # … and quarter it above this


class VolManagedOverlay(Strategy):
    name = "vol_managed_overlay"
    venue = "alpaca"

    def __init__(self, broker, bus: SignalBus | None = None):
        super().__init__(broker)
        self._bus = bus or SignalBus()

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        # Compute scalers and publish to bus — emits zero TradeProposals.
        scalers = {}

        spy_scaler, spy_ann_vol = self._compute_scaler("SPY")
        if spy_scaler is not None:
            scalers["equity_momentum"] = spy_scaler

        # Regime gate: when SPY realized vol is in a high regime, halve
        # / quarter the equity-momentum sleeve regardless of the
        # vol-target scaler above. Strategies look up `equity_regime`
        # via the new helper to decide whether to size down.
        regime = "NORMAL"
        regime_multiplier = 1.0
        if spy_ann_vol is not None:
            if spy_ann_vol >= EXTREME_VOL_THRESHOLD:
                regime = "EXTREME"
                regime_multiplier = 0.25
            elif spy_ann_vol >= HIGH_VOL_THRESHOLD:
                regime = "HIGH"
                regime_multiplier = 0.5

        if scalers:
            scalers["equity_regime"] = regime
            scalers["equity_regime_multiplier"] = regime_multiplier
            scalers["spy_realized_vol"] = (
                spy_ann_vol if spy_ann_vol is not None else 0.0
            )
            try:
                self._bus.publish(
                    scout=self.name, venue="overlay",
                    signal_type="vol_scaler", payload=scalers,
                    ttl_seconds=4 * 3600,
                )
            except Exception as e:
                logger.warning(f"[{self.name}] publish vol_scaler: {e}")
            if regime != "NORMAL":
                logger.warning(
                    f"[{self.name}] SPY ann_vol={spy_ann_vol*100:.1f}% "
                    f"→ regime={regime}, equity sleeve scaled by "
                    f"{regime_multiplier:.2f}"
                )

        return []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_scaler(self, symbol: str):
        """Return (scaler, ann_vol). Both None on failure."""
        try:
            candles = self.broker.get_candles(symbol, "ONE_DAY",
                                                num_candles=LOOKBACK_DAYS + 5)
        except Exception:
            return None, None
        if len(candles) < 20:
            return None, None
        closes = np.array([c.close for c in candles])
        rets = np.diff(closes) / closes[:-1]
        sd = float(np.std(rets, ddof=1))
        if sd <= 0:
            return None, None
        ann_vol = sd * _ANN
        # Scaler = target_vol / realized_vol, clamped
        scaler = TARGET_VOL / ann_vol
        return max(0.3, min(1.5, scaler)), ann_vol
