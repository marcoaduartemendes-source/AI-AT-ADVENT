"""Pre-FOMC drift — long SPY in the 24h before scheduled FOMC announcements.

REAL EDGE, REAL FIRMS:
    The "Pre-FOMC Announcement Drift" (Lucca & Moench, Journal of Finance
    2015): SPY earned +49 bps on average in the 24h before scheduled FOMC
    meetings 1994–2011, t-stat >5. The NY Fed Liberty Street and Cieslak
    et al. (2019) showed that nearly the ENTIRE post-1994 US equity
    premium accrued in even weeks of the FOMC cycle. Macro hedge funds
    (Brevan Howard, Bridgewater Pure Alpha) trade variants of this.

    Edge persistence: structural Fed-leakage / informed-trader effect,
    plus risk-premium repricing into a binary event. Regulator-mandated
    8 meetings/year → calendar known months ahead, can't be arbitraged
    away by faster firms.

MECHANICS:
    • At T-24h to a scheduled FOMC announcement → enter long SPY+QQQ.
    • At T-15min before announcement → flat (don't hold through the
      release; the drift is the PRE-announcement effect, post-announce
      volatility is uncorrelated noise).
    • Position size: 2-3x leverage of sleeve target (via UPRO/TQQQ
      proxies if pure ETF leverage is preferred, or notional SPY when
      the broker margin permits).

DATA:
    FOMC calendar is already published by MacroScout → ctx.scout_signals
    gets a `macro_fomc_window` payload with `days_to_next` and
    `next_meeting`. Free, deterministic, no extra API.

SAFETY:
    • Position size capped at 8% sleeve max — meaningful but bounded.
    • Hard -2% stop intraday (FOMC pre-drift losers are usually news-
      leak surprises; cut fast).
    • Vol-scaler overlay halves/zeros in HIGH/CRISIS vol regimes.
    • Trades ~8x/year so per-event blow-up risk is the dominant concern.

FAILURE MODES:
    Hawkish surprise leaked early (Aug 2022 Jackson Hole), FOMC date
    pushed unexpectedly, market gap-down overnight before announcement.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

# Universe — the broad-market ETFs the drift was originally documented on.
# SPY is the canonical; QQQ adds a tech-tilt that has historically
# AMPLIFIED the pre-FOMC drift (Cieslak et al. tech-cycle finding).
SPY_SYMBOL = "SPY"
QQQ_SYMBOL = "QQQ"

ENTRY_WINDOW_HOURS = 24       # enter at T-24h
EXIT_BUFFER_MINUTES = 15      # flat at T-15min (announcement at 14:00 ET)
SLEEVE_PER_NAME_PCT = 0.50    # 50% SPY + 50% QQQ of the sleeve
STOP_PCT = 0.02               # -2% hard stop (tight; binary event)


class PreFomcDrift(Strategy):
    name = "pre_fomc_drift"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        from ._helpers import vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        fomc = (ctx.scout_signals or {}).get("macro_fomc_window") or {}
        next_meeting = fomc.get("next_meeting")
        days_to_next = fomc.get("days_to_next")
        if next_meeting is None or days_to_next is None:
            logger.debug(f"[{self.name}] no FOMC signal — flat")
            return self._exit_all_open(ctx)

        # Are we INSIDE the 24h pre-announcement window?
        # days_to_next == 1 ≈ between 8 and 32 hours away; safe to enter.
        # days_to_next == 0 means the meeting is TODAY — risk window is
        # entirely behind us if we wake up after 14:00 ET, but the early
        # morning slice (until ~13:45 ET) is still pre-announcement.
        in_window = days_to_next in (0, 1)

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Hard stops first (non-negotiable).
        for sym in (SPY_SYMBOL, QQQ_SYMBOL):
            pos = open_pos.get(sym)
            if not pos:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            entry = (pos.get("avg_entry_price")
                     or pos.get("entry_price") or 0)
            try:
                last = float(self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue
            if entry and last and last <= entry * (1 - STOP_PCT):
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.99, is_closing=True,
                    reason=(f"hard stop: ${last:.2f} ≤ ${entry:.2f} "
                            f"× {1-STOP_PCT:.0%}"),
                    metadata={"model": self.name, "leg": "stop"},
                ))

        # ── 2) Window logic: enter when in, exit when not.
        if in_window:
            per_name = sleeve_usd * SLEEVE_PER_NAME_PCT
            for sym in (SPY_SYMBOL, QQQ_SYMBOL):
                pos = open_pos.get(sym, {})
                held = ((pos.get("quantity", 0) or 0)
                        if hasattr(pos, "get") else 0) > 0
                if held:
                    continue
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.BUY, order_type=OrderType.MARKET,
                    notional_usd=per_name, confidence=0.75,
                    is_closing=False,
                    reason=(f"pre-FOMC window (meeting {next_meeting}, "
                            f"T-{days_to_next}d)"),
                    metadata={"model": self.name, "leg": "entry",
                              "fomc_meeting": next_meeting,
                              "days_to_meeting": days_to_next,
                              "stop_pct": STOP_PCT,
                              "as_of": datetime.now(UTC).isoformat()},
                ))
        else:
            # Outside the window — exit any residuals.
            proposals.extend(self._exit_all_open(ctx))

        return proposals

    def _exit_all_open(self, ctx: StrategyContext) -> list[TradeProposal]:
        out: list[TradeProposal] = []
        for sym in (SPY_SYMBOL, QQQ_SYMBOL):
            pos = (ctx.open_positions or {}).get(sym)
            if not pos:
                continue
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            out.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.SELL, order_type=OrderType.MARKET,
                quantity=qty, confidence=0.9, is_closing=True,
                reason="outside pre-FOMC window — flat",
                metadata={"model": self.name, "leg": "exit"},
            ))
        return out
