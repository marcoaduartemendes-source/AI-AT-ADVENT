"""Quality-dividend ETF rotation — long the top dividend-growth ETFs.

Documented edge (Asness 2013, Novy-Marx 2013): high-quality dividend-
paying companies outperform the broader market on a risk-adjusted basis,
especially during drawdowns. This strategy rotates between the major
dividend-focused ETFs based on trailing 3-month total return (price +
implicit dividend) so we hold whichever sleeve is currently working.

Universe: the major dividend ETFs covering the strategy variants:
  VYM   - High Dividend Yield
  SCHD  - Dividend Quality (Schwab)
  DVY   - Select Dividend Index
  HDV   - Core High Dividend
  NOBL  - Dividend Aristocrats (S&P 500 25+ year increasers)
  DGRO  - Core Dividend Growth
  SPHD  - High Div + Low Vol

Rules:
  - Compute 90-day return for each ETF
  - Long the top-2 by return; equal-weight
  - Defensive bias: skip if VIX > 30 (let market settle first)
  - Rebalance: rotate when rank changes; cooldown 14d per slot

This is the "set and forget" sleeve — slow turnover, high carry-friendly,
runs alongside the more active strategies.
"""
from __future__ import annotations

import logging

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


LOOKBACK_DAYS = 90
TOP_N = 2
COOLDOWN_DAYS = 14
TRADE_SIZE_USD = 6000.0     # bigger size — slow-turnover sleeve


DIV_ETFS = [
    "VYM",   # Vanguard High Dividend Yield
    "SCHD",  # Schwab US Dividend Equity
    "DVY",   # iShares Select Dividend
    "HDV",   # iShares Core High Dividend
    "NOBL",  # ProShares S&P 500 Dividend Aristocrats
    "DGRO",  # iShares Core Dividend Growth
    "SPHD",  # Invesco S&P 500 High Div Low Vol
]


class DividendGrowth(Strategy):
    name = "dividend_growth"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        # Defensive: skip if VIX > 30 (signal_bus has it from macro_scout)
        vix = self._latest_vix(ctx)
        if vix is not None and vix > 30:
            logger.info(f"[{self.name}] VIX={vix:.1f} > 30 — sitting out cycle")
            return []

        # Rank ETFs by 90d return
        rankings = []
        for sym in DIV_ETFS:
            ret = self._lookback_return_pct(sym, LOOKBACK_DAYS)
            if ret is None:
                continue
            rankings.append((sym, ret))
        rankings.sort(key=lambda r: r[1], reverse=True)
        target_set = {s for s, _ in rankings[:TOP_N]}

        proposals: list[TradeProposal] = []
        held = {sym for sym, p in ctx.open_positions.items()
                if (p.get("quantity") or 0) > 0}

        # Exit holdings not in top-N (with cooldown)
        for sym, pos in ctx.open_positions.items():
            qty = pos.get("quantity") or 0
            if qty <= 0 or sym in target_set:
                continue
            if not self._past_cooldown(pos):
                continue
            ret = next((r for s, r in rankings if s == sym), None)
            ret_str = f"{ret:.1f}%" if ret is not None else "n/a"
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.85,
                reason=f"{sym} ret {ret_str} dropped from top-{TOP_N}",
                is_closing=True,
            ))

        # Enter top-N if not held. Sizing: split allocator's verdict
        # across slots, capped by TRADE_SIZE_USD as a per-position max.
        per_slot_alloc = ctx.target_alloc_usd / max(1, TOP_N)
        per_slot = min(per_slot_alloc, TRADE_SIZE_USD)
        for sym, ret in rankings[:TOP_N]:
            if sym in held:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_slot, confidence=0.8,
                reason=f"{sym} top-{TOP_N} dividend ETF, 90d {ret:+.1f}%",
                metadata={"lookback_return_pct": ret, "vix": vix},
            ))
        return proposals

    # ── Helpers ───────────────────────────────────────────────────────

    def _latest_vix(self, ctx: StrategyContext) -> float | None:
        """Pull most-recent VIX from scout signals or signal_bus."""
        sig = ctx.scout_signals.get("vix") if ctx.scout_signals else None
        if isinstance(sig, dict):
            v = sig.get("value") or sig.get("level") or sig.get("close")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                pass
        return None

    def _lookback_return_pct(self, symbol: str, days: int) -> float | None:
        from ._helpers import lookback_return_pct
        return lookback_return_pct(self.broker, self.name, symbol, days)

    def _past_cooldown(self, pos: dict) -> bool:
        from ._helpers import past_cooldown
        return past_cooldown(pos, COOLDOWN_DAYS)
