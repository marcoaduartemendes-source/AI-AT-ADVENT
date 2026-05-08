"""Crypto breakout — Donchian-style 30-day high entry, trail-stop exit.

Edge: Trend-following continuation in crypto. When BTC/ETH/SOL breaks
its 30-day high on rising volume, it tends to extend (Faber 2007 GTAA;
Antonacci 2014 dual momentum; Liu-Tsyvinski 2021 documents 1-7d
auto-correlation in crypto returns). Long-only because Coinbase spot
doesn't allow shorts; the matching short side is captured by
`crypto_xsmom_long_short` (separate strategy when we add the perp leg).

Mechanics:
  Entry — daily check:
    - today's close > 30-day high
    - today's volume > 1.5× 20-day average volume
    - VOL filter: BTC realized 30d annualized vol < 90%
      (above this we're in a "everything moves" regime — false breakouts
      dominate; the crypto_vol_regime_overlay scales us down anyway, but
      this is a hard floor.)

  Exit — every cycle:
    - close < highest_close_since_entry × (1 - TRAIL_PCT)  → trailing stop
    - close < entry_price × (1 - HARD_STOP_PCT)            → hard stop
    - close < SMA20                                         → trend break

Position sizing: ctx.target_alloc_usd / MAX_CONCURRENT, capped by
TRADE_SIZE_USD. Coinbase real money mode caps each order at $100 via
MAX_TRADE_USD_COINBASE — so per-position budget is min($100, ~$200/5).

Why this is more advanced than crypto_xsmom:
  - xsmom is a monthly-rebalance cross-section; breakout is an event-
    driven entry tied to a specific price level (Donchian channel).
  - Trailing stop locks in trend moves rather than waiting for the
    monthly rebalance to discover the trend has died.
  - Volume confirmation filters out noise breakouts on thin trading.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "MATIC-USD"]

BREAKOUT_LOOKBACK = 30      # days for the high
VOLUME_AVG_DAYS = 20        # days for vol baseline
MIN_VOLUME_RATIO = 1.5      # today vs 20d avg
MAX_CONCURRENT = 3          # don't overload the book
TRADE_SIZE_USD = 200.0      # per-position cap (Coinbase real-money safety)
TRAIL_PCT = 0.08            # 8% trailing stop from high-since-entry
HARD_STOP_PCT = 0.12        # 12% hard stop from entry
SMA_PERIOD = 20             # close-below SMA20 = trend break exit

# Vol-regime hard floor: above 90% BTC ann vol, breakouts have a long
# history of false positives (2020 March, 2022 May/Nov, etc).
MAX_VOL_FOR_BREAKOUT = 0.90


class CryptoBreakout(Strategy):
    name = "crypto_breakout"
    venue = "coinbase"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Vol-regime check via signal bus (publishes from crypto_vol_regime_overlay)
        crypto_vol = self._crypto_vol(ctx)
        if crypto_vol is not None and crypto_vol > MAX_VOL_FOR_BREAKOUT:
            logger.info(
                f"[{self.name}] BTC ann_vol={crypto_vol*100:.0f}% > "
                f"{MAX_VOL_FOR_BREAKOUT*100:.0f}% — sitting out cycle"
            )
            return []

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}
        slots_left = max(0, MAX_CONCURRENT - len(held))

        # ── Exits: walk current positions ─────────────────────────────
        for sym, pos in ctx.open_positions.items():
            if sym not in UNIVERSE:
                continue
            qty = pos.get("quantity", 0) or 0
            if qty <= 0:
                continue
            stats = self._stats(sym)
            if not stats:
                continue
            last_close, sma, _high30, _vol_ratio = stats

            entry_price = pos.get("avg_entry_price") or 0
            high_since_entry = pos.get("high_since_entry") or entry_price

            exit_reason = None
            if entry_price and last_close < entry_price * (1 - HARD_STOP_PCT):
                exit_reason = (
                    f"hard stop: ${last_close:.2f} < entry ${entry_price:.2f} "
                    f"× (1 - {HARD_STOP_PCT*100:.0f}%)"
                )
            elif high_since_entry and last_close < high_since_entry * (1 - TRAIL_PCT):
                exit_reason = (
                    f"trailing stop: ${last_close:.2f} < high "
                    f"${high_since_entry:.2f} × (1 - {TRAIL_PCT*100:.0f}%)"
                )
            elif last_close < sma:
                exit_reason = (
                    f"trend break: ${last_close:.2f} < SMA{SMA_PERIOD} "
                    f"${sma:.2f}"
                )

            if exit_reason:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.95, is_closing=True,
                    reason=exit_reason,
                ))

        # ── Entries: new breakouts ────────────────────────────────────
        if slots_left <= 0:
            return proposals
        candidates = []
        for sym in UNIVERSE:
            if sym in held:
                continue
            stats = self._stats(sym)
            if not stats:
                continue
            last_close, _sma, high30, vol_ratio = stats
            if last_close <= high30:
                continue
            if vol_ratio < MIN_VOLUME_RATIO:
                continue
            score = (last_close - high30) / high30 * vol_ratio
            candidates.append((sym, last_close, high30, vol_ratio, score))

        candidates.sort(key=lambda c: c[4], reverse=True)
        per_slot = min(
            ctx.target_alloc_usd / max(1, MAX_CONCURRENT),
            TRADE_SIZE_USD,
        )
        for sym, last_close, high30, vol_ratio, _ in candidates[:slots_left]:
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_slot, confidence=0.75,
                reason=(
                    f"{sym} breakout: ${last_close:.2f} > 30d high "
                    f"${high30:.2f}, vol {vol_ratio:.1f}×"
                ),
                metadata={"close": last_close, "high30": high30,
                          "vol_ratio": vol_ratio},
            ))
        return proposals

    # ── Indicator helpers ─────────────────────────────────────────────

    def _stats(self, symbol: str) -> tuple[float, float, float, float] | None:
        """Returns (last_close, SMA20, 30d_high, vol_ratio).
        None if not enough bars."""
        try:
            candles = self.broker.get_candles(
                symbol, "ONE_DAY",
                num_candles=max(BREAKOUT_LOOKBACK, VOLUME_AVG_DAYS, SMA_PERIOD) + 5,
            )
        except Exception as e:
            logger.debug(f"[{self.name}] {symbol} candles failed: {e}")
            return None
        if len(candles) < max(BREAKOUT_LOOKBACK, VOLUME_AVG_DAYS, SMA_PERIOD) + 1:
            return None
        recent_for_high = candles[-(BREAKOUT_LOOKBACK + 1):-1]   # exclude today
        high30 = max(c.high for c in recent_for_high)
        sma_window = candles[-SMA_PERIOD:]
        sma = sum(c.close for c in sma_window) / SMA_PERIOD
        vol_window = candles[-VOLUME_AVG_DAYS:]
        avg_vol = sum(c.volume for c in vol_window) / VOLUME_AVG_DAYS
        last = candles[-1]
        vol_ratio = (last.volume / avg_vol) if avg_vol > 0 else 0.0
        return (last.close, sma, high30, vol_ratio)

    def _crypto_vol(self, ctx) -> float | None:
        """Read BTC realized vol from crypto_vol_regime_overlay's signal."""
        try:
            sig = (ctx.scout_signals or {}).get("crypto_vol_scaler")
        except AttributeError:
            return None
        if not isinstance(sig, dict):
            return None
        try:
            v = sig.get("btc_realized_vol")
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
