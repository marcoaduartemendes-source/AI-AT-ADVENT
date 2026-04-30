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
from typing import List

import numpy as np

from brokers.base import OrderSide, OrderType
from scouts.signal_bus import SignalBus
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


_ANN = math.sqrt(252)
TARGET_VOL = 0.15            # 15% target — used as the "normal" baseline
LOOKBACK_DAYS = 30


class VolManagedOverlay(Strategy):
    name = "vol_managed_overlay"
    venue = "alpaca"

    def __init__(self, broker, bus: "SignalBus | None" = None):
        super().__init__(broker)
        self._bus = bus or SignalBus()

    def compute(self, ctx: StrategyContext) -> List[TradeProposal]:
        # Compute scalers and publish to bus — emits zero TradeProposals.
        scalers = {}

        spy_scaler = self._compute_scaler("SPY")
        if spy_scaler is not None:
            scalers["equity_momentum"] = spy_scaler

        # We can't easily fetch BTC vol via Alpaca; just publish what we have
        if scalers:
            try:
                self._bus.publish(
                    scout=self.name, venue="overlay",
                    signal_type="vol_scaler", payload=scalers,
                    ttl_seconds=4 * 3600,
                )
            except Exception as e:
                logger.warning(f"[{self.name}] publish vol_scaler: {e}")

        return []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _compute_scaler(self, symbol: str):
        try:
            candles = self.broker.get_candles(symbol, "ONE_DAY",
                                                num_candles=LOOKBACK_DAYS + 5)
        except Exception:
            return None
        if len(candles) < 20:
            return None
        closes = np.array([c.close for c in candles])
        rets = np.diff(closes) / closes[:-1]
        sd = float(np.std(rets, ddof=1))
        if sd <= 0:
            return None
        ann_vol = sd * _ANN
        # Scaler = target_vol / realized_vol, clamped
        scaler = TARGET_VOL / ann_vol
        return max(0.3, min(1.5, scaler))
