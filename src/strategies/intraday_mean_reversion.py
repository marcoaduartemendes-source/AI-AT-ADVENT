"""Intraday mean reversion — the bot's "high-frequency" strategy.

USER ASK
The user requested "one strategy of high frequency trading". I
pushed back in chat: real HFT (sub-ms, co-located, FPGA) is
architecturally impossible on this stack — we're a cron-scheduled
bot using retail REST APIs with 10-50ms round-trip latency on the
network alone. What IS achievable, and what the user most likely
means, is HIGH-TURNOVER INTRADAY: a strategy that fires every
cycle when minute-bar conditions are met, trading far more often
than the daily/weekly strategies in the book.

That's this module. It is INTRADAY MEAN REVERSION, not HFT.

WHAT IT DOES
On the most liquid US equity ETFs (SPY/QQQ/IWM — wide bid-ask is
death for this style; staying in instruments with sub-1bp spreads
is non-negotiable), measure the z-score of the current close vs
a short rolling VWAP on 5-minute bars. When price spikes far
below VWAP (z ≤ −2), fade the move with a small long. Exit when:

  • Price reverts to VWAP                       (target hit)
  • Price falls another 0.5% from entry         (hard stop)
  • 60 minutes (12 bars) have elapsed           (time stop)

WHY ONLY LONGS, ONLY EXTREMES, ONLY LIQUID ETFs
Mean-reversion has a known asymmetry: long-the-dip on positively-
drifting US equity has an empirical edge (Avellaneda-Lee 2010,
Connors RSI-2 corpus); short-the-rip does not, because the
unconditional drift works against you. Restricting to z ≤ −Z_ENTRY
on broad ETFs targets the most-evidenced corner of the parameter
space and avoids the failure modes that killed similar retail
strategies (chasing reversion on falling-knife single names).

WHY THIS WILL PROBABLY FAIL THE VALIDATION GATE
This is the part the user should hear from me directly. At retail
broker fees + bid-ask spread, the per-trade edge needed to be net-
positive on a 0.3-0.5% reversion target is razor-thin. The
validation harness will likely return UNPROVEN (intraday backtest
not implementable from daily Yahoo bars) or FAIL once a real
backtest is wired. That's the FEATURE, not the bug: if the gate
rejects it, the strategy stays paper and we've learned for free
that retail-grade intraday MR doesn't pay. If the live paper P&L
DOES build a real edge, the gate will eventually flip it to PASS.

LIVE-PROMOTION
DRY at 1.5% target / 4% cap. Stays paper until either:
  (a) The validation harness records PASS, OR
  (b) 90+ days of paper trading show a positive Sharpe at a
      statistically meaningful trade count.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────

UNIVERSE = ["SPY", "QQQ", "IWM"]    # only the most liquid US equity ETFs

BAR_GRANULARITY = "FIVE_MINUTE"     # primary signal timeframe
LOOKBACK_BARS = 30                  # ~2.5h of session for VWAP+stdev
ENTRY_Z = -2.0                      # fade only deep dislocations
EXIT_Z = -0.25                      # take profit as price reverts to VWAP

HARD_STOP_PCT = 0.005               # -0.5% from entry (≈ 1× ETF daily ATR/40)
TIME_STOP_BARS = 12                 # 60 minutes on 5-min bars

PER_NAME_NOTIONAL_FRAC = 0.4        # of sleeve, per concurrent name; 3 names
                                     # × 0.4 = up to 120% gross — capped by
                                     # the allocator's max_alloc_pct anyway.

# Only trade inside the regular US session and leave 30min before
# close to flatten — overnight gaps are not what this style trades.
SESSION_OPEN_UTC = (13, 30)         # 09:30 ET (assumes EDT; close enough
                                     # for a regime filter, not an audit
                                     # trail — the broker enforces the
                                     # actual session boundary on its end).
SESSION_FLATTEN_UTC = (19, 30)      # 15:30 ET — no new entries after this
SESSION_FORCE_CLOSE_UTC = (19, 55)  # 15:55 ET — flat all positions


def _in_entry_window(now: datetime) -> bool:
    h, m = now.hour, now.minute
    after_open = (h, m) >= SESSION_OPEN_UTC
    before_flatten = (h, m) < SESSION_FLATTEN_UTC
    return after_open and before_flatten


def _must_force_close(now: datetime) -> bool:
    return (now.hour, now.minute) >= SESSION_FORCE_CLOSE_UTC


class IntradayMeanReversion(Strategy):
    name = "intraday_mean_reversion"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []

        from ._helpers import vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        now = datetime.now(UTC)
        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Forced flatten near the close (overnight risk is not what
        #       this strategy harvests). Applies regardless of session.
        force_close = _must_force_close(now)
        for sym in UNIVERSE:
            pos = open_pos.get(sym) or {}
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            if qty <= 0:
                continue
            if force_close:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.99, is_closing=True,
                    reason="EOD force-close (15:55 ET)",
                    metadata={"model": "intraday_mr", "leg": "eod"},
                ))

        # If we're past force-close, ONLY flatten — don't open new entries.
        if force_close:
            return proposals

        # ── 2) Per-symbol signal + exit decisions on the 5-min bars.
        for sym in UNIVERSE:
            try:
                candles = self.broker.get_candles(
                    sym, BAR_GRANULARITY, num_candles=LOOKBACK_BARS + 5)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {sym}: {e}")
                continue
            if len(candles) < LOOKBACK_BARS:
                continue
            closes = np.array([c.close for c in candles], dtype=float)
            vols = np.array(
                [getattr(c, "volume", 0) or 0 for c in candles], dtype=float)
            # Volume-weighted average price over the lookback. Falls back
            # to simple mean when volume is missing (data-feed gaps).
            if vols.sum() > 0:
                vwap = float((closes * vols).sum() / vols.sum())
            else:
                vwap = float(closes.mean())
            sd = float(closes.std())
            if sd < 1e-9:
                continue                          # degenerate — skip
            last = float(closes[-1])
            z = (last - vwap) / sd

            pos = open_pos.get(sym) or {}
            qty = (pos.get("quantity", 0) or 0) if hasattr(pos, "get") else 0
            holding = qty > 0

            # ── Exit logic for held position
            if holding:
                entry = float(pos.get("avg_entry_price")
                              or pos.get("entry_price") or 0)
                opened = pos.get("open_time") or pos.get("entry_time")
                bars_held: int | None = None
                if opened:
                    try:
                        opened_dt = datetime.fromisoformat(
                            str(opened).replace("Z", "+00:00"))
                        bars_held = int(
                            (now - opened_dt).total_seconds() // 300)
                    except Exception:
                        bars_held = None

                exit_reason: str | None = None
                if z >= EXIT_Z:
                    exit_reason = f"reverted (z={z:+.2f})"
                elif entry and last <= entry * (1 - HARD_STOP_PCT):
                    exit_reason = (f"hard stop: last ${last:.2f} ≤ "
                                    f"entry ${entry:.2f} × "
                                    f"{1-HARD_STOP_PCT:.1%}")
                elif bars_held is not None and bars_held >= TIME_STOP_BARS:
                    exit_reason = f"time stop ({bars_held} bars)"

                if exit_reason:
                    proposals.append(TradeProposal(
                        strategy=self.name, venue=self.venue, symbol=sym,
                        side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=qty, confidence=0.9, is_closing=True,
                        reason=exit_reason,
                        metadata={"model": "intraday_mr", "leg": "exit"},
                    ))
                continue                          # held: never stack entries

            # ── Entry logic (no position) — gated on session window
            if not _in_entry_window(now):
                continue
            if z > ENTRY_Z:
                continue                          # not dislocated enough
            notional = sleeve_usd * PER_NAME_NOTIONAL_FRAC
            if notional <= 0:
                continue
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=sym,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=notional, confidence=0.6, is_closing=False,
                reason=(f"intraday dip: z={z:+.2f} vs 30-bar VWAP "
                        f"(${vwap:.2f}), last ${last:.2f}"),
                metadata={"model": "intraday_mr", "leg": "entry",
                          "z": round(z, 3),
                          "vwap": round(vwap, 4),
                          "as_of": now.isoformat()},
            ))
        return proposals
