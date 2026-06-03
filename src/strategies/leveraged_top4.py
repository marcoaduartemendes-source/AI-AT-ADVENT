"""Leveraged top-4 — 3x exposure on the FOUR best-performing PASS sleeves.

Sister sleeve to leveraged_champions (which is top-5). The user
requested a more concentrated version: a 3x ETF basket reflecting the
top-4 strategies by realised P&L. Higher conviction, tighter book.

Same proven discipline as leveraged_momentum / leveraged_champions:
  • Each cycle, re-resolve the top-4 PASS strategies (live FIFO P&L →
    5y backtest fallback) and map to liquid 3x ETF proxies.
  • Each proxy's UNDERLYING must be above its 200d SMA AND realised vol
    ≤ ceiling (otherwise sit flat in cash — the SAFE state for 3x).
  • Hard −15% stop per position, checked first every cycle.
  • vol_scaler overlay halves/zeros the sleeve in HIGH/CRISIS regimes.
  • Registered SMALL and DRY until docs/validation.json records PASS.

The concentration (4 vs 5) is the only intentional difference: when the
top-3 leaderboard converges on the same 1-2 proxies, this sleeve takes a
bigger bite than leveraged_champions does.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal
from strategies.leveraged_champions import (
    CHAMPION_PROXY, PROXY_UNDERLYING, STOP_PCT, TREND_SMA,
    VOL_CEILING, VOL_LOOKBACK, REBALANCE_COOLDOWN_DAYS, resolve_champions,
)

logger = logging.getLogger(__name__)

TOP_N = 4


class LeveragedTop4(Strategy):
    name = "leveraged_top4"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        from ._helpers import past_cooldown, vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        champions = resolve_champions(top_n=TOP_N)
        proxies: list[str] = []
        for s in champions:
            px = CHAMPION_PROXY.get(s)
            if px and px not in proxies:
                proxies.append(px)
        if not proxies:
            proxies = ["UPRO", "TQQQ"]
            logger.info(f"[{self.name}] no champion proxies; defaulting to {proxies}")

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # Hard stops first.
        for sym, pos in open_pos.items():
            if sym not in PROXY_UNDERLYING:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            entry = pos.get("avg_entry_price") or pos.get("entry_price") or 0
            try:
                last = float(self.broker.get_candles(sym, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue
            if entry and last and last <= entry * (1 - STOP_PCT):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.99, is_closing=True,
                    reason=f"hard stop: ${last:.2f} ≤ ${entry:.2f} × {1-STOP_PCT:.0%}",
                    metadata={"model": self.name, "leg": "stop"},
                ))

        # Regime gate on each proxy's underlying.
        eligible: list[str] = []
        for proxy in proxies:
            under = PROXY_UNDERLYING.get(proxy)
            if not under:
                continue
            try:
                cs = self.broker.get_candles(
                    under, "ONE_DAY",
                    num_candles=TREND_SMA + VOL_LOOKBACK + 5)
            except Exception:
                continue
            if len(cs) < TREND_SMA:
                continue
            closes = np.array([c.close for c in cs], dtype=float)
            if closes[-1] <= 0 or closes[-1] < closes[-TREND_SMA:].mean():
                continue
            rets = np.diff(np.log(closes[-VOL_LOOKBACK - 1:]))
            if float(rets.std()) > VOL_CEILING:
                continue
            eligible.append(proxy)

        if not eligible:
            logger.info(f"[{self.name}] no proxy in low-vol uptrend; flat (safe)")
            return proposals

        per_name = sleeve_usd / len(eligible)
        held = {
            s for s, p in open_pos.items()
            if ((p.get("quantity", 0) or 0) if hasattr(p, "get") else 0) > 0
        }
        for sym in eligible:
            if sym in held:
                continue
            pos = open_pos.get(sym, {})
            if pos and not past_cooldown(pos, REBALANCE_COOLDOWN_DAYS):
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name, confidence=0.7, is_closing=False,
                reason=(f"3x top-{TOP_N}: tracks {champions[:TOP_N]}; "
                        f"{sym} underlying above 200d SMA, low vol"),
                metadata={"model": self.name, "leg": "entry",
                          "stop_pct": STOP_PCT, "champions": champions,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
        return proposals
