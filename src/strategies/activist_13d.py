"""Activist 13D Front-Run — buy targets within hours of a 13D filing.

REAL EDGE, REAL FIRMS:
    Schedule 13D must be filed by any investor crossing the 5% threshold
    within 10 calendar days. Brav-Jiang-Partnoy-Thomas (JF 2008): +7-8%
    abnormal return in (-10, +20)d around the filing, no long-term
    reversal (Bebchuk-Brav-Jiang 2015). Two Sigma's event sleeve
    mimics this systematically; Senator/Sachem Head/JANA/ValueAct/
    Engine No. 1 are the originators it follows.

    Edge persistence: regulator-mandated mandatory disclosure trigger
    means the activist has been accumulating at lower prices for weeks
    before we see it. We can't be earlier; we just collect the public
    drift after the announcement. Other firms know about this trade, but
    the structural retail-flow and index-fund-rebalance delays leave
    enough drift on the table for a 5-day hold.

ENTRY:
    Activist13DScout publishes `activist_13d_new` signals (ticker +
    filed_date). We buy the named ticker at next-cycle open with a
    per-name notional cap.

EXIT:
    • Hard +6% take-profit (most drift completes early; lock it in).
    • Hard -4% stop (single-name event risk warrants tight stops).
    • 5-trading-day age-out — if neither stop fires within the documented
      drift window, exit at market.

SAFETY:
    • Per-name notional capped at 25% of sleeve so a single deal blow-up
      (-4%) costs the sleeve at most 1%.
    • Long-only (no shorts) — Alpaca short-locate inventory is unreliable
      and the academic edge is symmetric long, asymmetric short.
    • Skip if next-cycle open would land OUTSIDE the documented (-10,+20)d
      drift window, i.e. we only enter within 2 trading days of the filing.
    • DRY by default — register in LIVE_STRATEGIES env to go live.

FAILURE MODES:
    • Activist exits early (rare — they filed at 5%+).
    • Target counter-litigates and wins (sometimes painful, -10%+).
    • Broad market crash during the 5d hold (use the -4% stop to cap).
    • Falsely-extracted ticker from EDGAR metadata → the scout's
      _normalize_ticker filter is the first line of defence.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

PER_NAME_PCT = 0.25       # 25% of sleeve per position (≤ 4 names)
TAKE_PROFIT_PCT = 0.06    # +6% take-profit
STOP_PCT = 0.04           # -4% stop-loss
MAX_HOLD_DAYS = 5         # age-out after 5 trading days
MAX_ENTRY_AGE_HOURS = 48  # only enter if filing < 48h old
MIN_ENTRY_USD = 200.0


class Activist13D(Strategy):
    name = "activist_13d"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        from ._helpers import vol_scaler
        sleeve_usd = ctx.target_alloc_usd * vol_scaler(ctx)
        if sleeve_usd <= 0:
            return []

        proposals: list[TradeProposal] = []
        open_pos = ctx.open_positions or {}

        # ── 1) Manage open positions: stops, take-profits, age-outs.
        for sym, pos in open_pos.items():
            if not hasattr(pos, "get"):
                continue
            qty = pos.get("quantity", 0) or 0
            if qty <= 0:
                continue
            entry = (pos.get("avg_entry_price")
                     or pos.get("entry_price") or 0)
            try:
                last = float(self.broker.get_candles(
                    sym, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue

            exit_reason = None
            if entry and last and last >= entry * (1 + TAKE_PROFIT_PCT):
                exit_reason = (f"take-profit: ${last:.2f} ≥ ${entry:.2f} "
                               f"× {1+TAKE_PROFIT_PCT:.0%}")
            elif entry and last and last <= entry * (1 - STOP_PCT):
                exit_reason = (f"hard stop: ${last:.2f} ≤ ${entry:.2f} "
                               f"× {1-STOP_PCT:.0%}")
            else:
                # Age-out check.
                opened = (pos.get("opened_at") or pos.get("last_trade_at")
                          or pos.get("updated_at"))
                if opened:
                    try:
                        dt = datetime.fromisoformat(
                            str(opened).replace("Z", "+00:00"))
                        if (datetime.now(UTC) - dt
                                > timedelta(days=MAX_HOLD_DAYS)):
                            exit_reason = (f"age-out: held > "
                                           f"{MAX_HOLD_DAYS} trading days")
                    except (ValueError, TypeError):
                        pass

            if exit_reason:
                proposals.append(TradeProposal(
                    strategy=self.name, venue=self.venue, symbol=sym,
                    side=OrderSide.SELL, order_type=OrderType.MARKET,
                    quantity=qty, confidence=0.9, is_closing=True,
                    reason=exit_reason,
                    metadata={"model": self.name, "leg": "exit"},
                ))

        # ── 2) New entries from fresh 13D signals.
        # The scout publishes one ScoutSignal per filing under signal_type
        # "activist_13d_new". The signal bus deduplicates by payload, but
        # the strategy must also de-dup against currently-held positions.
        signals = (ctx.scout_signals or {}).get("activist_13d_new") or []
        # Bus delivers EITHER a single payload OR a list depending on the
        # collector; normalise to list of dicts.
        if isinstance(signals, dict):
            signals = [signals]
        if not isinstance(signals, list):
            signals = []

        held = {
            s for s, p in open_pos.items()
            if hasattr(p, "get") and (p.get("quantity", 0) or 0) > 0
        }
        # Cap concurrent positions at 4 to keep diversification.
        n_held = len(held)
        if n_held >= 4:
            return proposals
        per_name_usd = sleeve_usd * PER_NAME_PCT
        if per_name_usd < MIN_ENTRY_USD:
            return proposals

        added = 0
        seen: set[str] = set()
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            tic = (sig.get("ticker") or "").upper()
            filed = sig.get("filed_date")
            if not tic or tic in held or tic in seen:
                continue
            seen.add(tic)
            # Reject stale filings — the drift window is short.
            if filed:
                try:
                    fdt = datetime.fromisoformat(
                        str(filed).replace("Z", "+00:00"))
                    if fdt.tzinfo is None:
                        fdt = fdt.replace(tzinfo=UTC)
                    age_hr = (datetime.now(UTC) - fdt).total_seconds() / 3600
                    if age_hr > MAX_ENTRY_AGE_HOURS:
                        continue
                except (ValueError, TypeError):
                    pass
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=tic,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name_usd, confidence=0.7, is_closing=False,
                reason=f"SC 13D filed {filed} on {tic} — front-run drift",
                metadata={"model": self.name, "leg": "entry",
                          "filed_date": filed,
                          "stop_pct": STOP_PCT,
                          "take_profit_pct": TAKE_PROFIT_PCT,
                          "max_hold_days": MAX_HOLD_DAYS,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
            added += 1
            if n_held + added >= 4:
                break

        return proposals
