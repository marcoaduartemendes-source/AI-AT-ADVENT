"""Low-volatility anomaly — long lowest-vol equities + low-vol ETFs.

Documented edge (Frazzini-Pedersen 2014 "Betting Against Beta",
Blitz-Vliet 2007, Asness 2020): low-volatility stocks deliver
similar long-run returns to high-vol stocks but with much smaller
drawdowns — leveraging or sizing-up low-vol gives a higher Sharpe.

Strategy variant we run (long-only, low-vol bias only — no shorts):
  Universe: 5 low-vol-flavored ETFs + 8 lowest-realized-vol stocks
  from the S&P 100 by 60-day price standard deviation.
  Long the top-3 of each (6 positions total) by smallest 60-day vol,
  re-ranked weekly, with positive-momentum filter (60d return > 0).

Position sizing: $2.5k per position × 6 = $15k peak deployment.
Cooldown: 7 days per slot (avoids churn).
"""
from __future__ import annotations

import logging
import math

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


VOL_LOOKBACK_DAYS = 60      # window for realized-vol calc
RETURN_LOOKBACK_DAYS = 60   # only long names with positive 60d return
TOP_ETF = 3                 # take the top-3 lowest-vol ETFs
TOP_STOCK = 3               # plus top-3 lowest-vol stocks
TRADE_SIZE_USD = 2500.0
COOLDOWN_DAYS = 7


# Low-vol-flavored ETFs (institutional staples)
LOW_VOL_ETFS = [
    "USMV",   # iShares MSCI USA Min Vol Factor
    "SPLV",   # Invesco S&P 500 Low Vol
    "QUAL",   # iShares MSCI USA Quality
    "EFAV",   # iShares MSCI EAFE Min Vol
    "ACWV",   # iShares MSCI ACWI Min Vol
]

# Stable single-stock candidates (utilities, staples, healthcare —
# the low-vol anomaly's natural home)
LOW_VOL_STOCKS = [
    "JNJ", "KO", "PEP", "PG", "WMT", "MCD",
    "VZ", "T", "DUK", "NEE", "MMM", "MRK",
    "ABT", "PFE", "WBA",
]


class LowVolAnomaly(Strategy):
    name = "low_vol_anomaly"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Rank ETFs and stocks separately by realized vol (ascending)
        etf_picks = self._rank_lowest_vol(LOW_VOL_ETFS, TOP_ETF)
        stock_picks = self._rank_lowest_vol(LOW_VOL_STOCKS, TOP_STOCK)
        target_set = {sym for sym, _ in etf_picks + stock_picks}

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # Exits: held but not in current target set + past cooldown
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0 or sym in target_set:
                continue
            if not self._past_cooldown(pos):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85,
                reason=f"{sym} dropped from low-vol top picks",
                is_closing=True,
            ))

        # Entries: target set members not currently held
        for sym, vol in etf_picks + stock_picks:
            if sym in held:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=TRADE_SIZE_USD, confidence=0.7,
                reason=f"Low-vol top pick: {sym} (vol={vol*100:.1f}%)",
                metadata={"realized_vol": vol},
            ))
        return proposals

    def _rank_lowest_vol(
        self, universe: list[str], top_n: int,
    ) -> list[tuple[str, float]]:
        """For each symbol, compute realized vol and 60d return.
        Filter to those with positive return, then take the top-N
        lowest vol. Returns [(symbol, vol_decimal), ...]."""
        candidates = []
        for sym in universe:
            vol = self._realized_vol(sym, VOL_LOOKBACK_DAYS)
            ret = self._return_pct(sym, RETURN_LOOKBACK_DAYS)
            if vol is None or ret is None:
                continue
            if ret <= 0:                # skip negative-momentum names
                continue
            candidates.append((sym, vol))
        candidates.sort(key=lambda c: c[1])
        return candidates[:top_n]

    def _realized_vol(self, symbol: str, days: int) -> float | None:
        """Annualized stdev of daily log returns (standard convention)."""
        try:
            candles = self.broker.get_candles(symbol, "1Day", num_candles=days + 5)
        except Exception as e:
            logger.debug(f"[{self.name}] {symbol} candles failed: {e}")
            return None
        if len(candles) < days + 1:
            return None
        recent = candles[-(days + 1):]
        rets = []
        for i in range(1, len(recent)):
            prev = recent[i - 1].close
            cur = recent[i].close
            if prev <= 0 or cur <= 0:
                continue
            rets.append(math.log(cur / prev))
        if len(rets) < 5:
            return None
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(len(rets) - 1, 1)
        sd = math.sqrt(var)
        return sd * math.sqrt(252)

    def _return_pct(self, symbol: str, days: int) -> float | None:
        from ._helpers import lookback_return_pct
        return lookback_return_pct(self.broker, self.name, symbol, days)

    def _past_cooldown(self, pos: dict) -> bool:
        from ._helpers import past_cooldown
        return past_cooldown(pos, COOLDOWN_DAYS)
