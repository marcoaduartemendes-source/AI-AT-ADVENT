"""LLM 8-K Event — trade Claude's read of fresh material-event filings.

THE THESIS:
    Every 8-K is a regulator-mandated disclosure of a MATERIAL corporate
    event, and post-filing drift runs hours-to-days. The coverage gap —
    ~3,000 filings/week, no human desk reads them all, the drift is too
    slow for HFT to monopolize — is exactly the shape of an LLM analyst.
    Edgar8KScout has Claude Haiku read each fresh filing's headline +
    item codes and publish direction + confidence; this strategy trades
    only the highest-conviction LONG calls.

    Evidence base: LLM-scored headlines predict next-day returns
    (Lopez-Lira & Tang 2023, and the 2024-26 follow-on literature).
    This sleeve is intentionally experimental — registered SMALL so the
    live ledger can prove or kill the thesis cheaply. The validation
    harness can't backtest it (no historical LLM scores), so live paper
    P&L is the test. That's the honest deal with novel alpha: nobody
    has the backtest, which is exactly why the edge can still exist.

MECHANICS:
    Entry  — VERY_BULLISH with confidence ≥ 0.70 only. Long-only (the
             bearish side needs short locates; asymmetric risk on a
             cron cadence).
    Exit   — +8% take-profit, −4% stop, 3-trading-day age-out (8-K
             drift decays fast; don't let an event trade become an
             accidental hold).
    Caps   — ≤4 concurrent names, 25% of sleeve each.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

PER_NAME_PCT = 0.25
MIN_CONFIDENCE = 0.70
TAKE_PROFIT_PCT = 0.08
STOP_PCT = 0.04
MAX_HOLD_DAYS = 5           # ≈ 3 trading days
MAX_POSITIONS = 4
MIN_ENTRY_USD = 200.0


class Llm8KEvent(Strategy):
    name = "llm_8k_event"
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

        # ── 1) Manage open positions: TP / stop / age-out.
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
                opened = (pos.get("opened_at") or pos.get("last_trade_at")
                          or pos.get("updated_at"))
                if opened:
                    try:
                        dt = datetime.fromisoformat(
                            str(opened).replace("Z", "+00:00"))
                        if (datetime.now(UTC) - dt
                                > timedelta(days=MAX_HOLD_DAYS)):
                            exit_reason = (f"age-out: 8-K drift window "
                                           f"closed (> {MAX_HOLD_DAYS}d)")
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

        # ── 2) New entries: highest-conviction VERY_BULLISH only.
        signals = (ctx.scout_signals or {}).get("llm_8k_event") or []
        if isinstance(signals, dict):
            signals = [signals]
        if not isinstance(signals, list):
            signals = []

        held = {
            s for s, p in open_pos.items()
            if hasattr(p, "get") and (p.get("quantity", 0) or 0) > 0
        }
        if len(held) >= MAX_POSITIONS:
            return proposals
        per_name = sleeve_usd * PER_NAME_PCT
        if per_name < MIN_ENTRY_USD:
            return proposals

        seen: set[str] = set()
        added = 0
        candidates = [
            s for s in signals
            if isinstance(s, dict)
            and s.get("direction") == "VERY_BULLISH"
            and (s.get("confidence") or 0) >= MIN_CONFIDENCE
        ]
        for sig in sorted(candidates,
                          key=lambda s: -(s.get("confidence") or 0)):
            tic = (sig.get("ticker") or "").upper()
            if not tic or tic in held or tic in seen:
                continue
            seen.add(tic)
            conf = sig.get("confidence") or 0.0
            rationale = (sig.get("rationale") or "")[:120]
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=tic,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name, confidence=min(conf, 0.9),
                is_closing=False,
                reason=(f"LLM 8-K VERY_BULLISH ({conf:.0%}): "
                        f"{rationale}"),
                metadata={"model": self.name, "leg": "entry",
                          "llm_confidence": conf,
                          "accession": sig.get("accession"),
                          "items": sig.get("items"),
                          "stop_pct": STOP_PCT,
                          "take_profit_pct": TAKE_PROFIT_PCT,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
            added += 1
            if len(held) + added >= MAX_POSITIONS:
                break
        return proposals
