"""Insider Cluster — buy stocks where ≥3 insiders just bought together.

REAL EDGE, REAL EVIDENCE:
    Cohen-Malloy-Pomorski (JF 2012, "Decoding Inside Information"):
    opportunistic insider purchases → ~7%/yr abnormal return; the
    multi-insider cluster variant is the strongest documented flavor.
    Form 4 disclosure is regulator-mandated within 2 business days, so
    the trigger can't be front-run away — the same structural-durability
    family as the 13D and FOMC edges.

    CAPACITY IS THE MOAT: cluster buys concentrate in small/mid-caps
    where a Citadel can't deploy meaningfully but a small book can. This
    is deliberately a capacity-constrained sleeve — our smallness is the
    edge.

MECHANICS:
    Entry — InsiderClusterScout publishes `insider_cluster_buy`
    (ticker, n_insiders, total_usd). Buy at next cycle, per-name cap
    20% of sleeve, max 5 concurrent.
    Exit — +12% take-profit, −6% stop, or 20-trading-day age-out
    (the bulk of the documented drift accrues inside a month).

SAFETY:
    Long-only; per-name cap bounds single-name damage to ~1.2% of the
    sleeve at the stop. Tiny initial allocation until the validation
    harness scores live fills.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

PER_NAME_PCT = 0.20
TAKE_PROFIT_PCT = 0.12
STOP_PCT = 0.06
MAX_HOLD_DAYS = 28          # ≈ 20 trading days
MAX_POSITIONS = 5
MIN_ENTRY_USD = 200.0


class InsiderCluster(Strategy):
    name = "insider_cluster"
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
                # entry_time is the real position age (ledger-derived by
                # the orchestrator); the old opened_at/last_trade_at/
                # updated_at keys never existed on PositionView, so this
                # age-out was dead code before 2026-06-11.
                opened = pos.get("entry_time")
                if opened:
                    try:
                        dt = datetime.fromisoformat(
                            str(opened).replace("Z", "+00:00"))
                        if (datetime.now(UTC) - dt
                                > timedelta(days=MAX_HOLD_DAYS)):
                            exit_reason = f"age-out: > {MAX_HOLD_DAYS}d"
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

        # ── 2) New entries from cluster signals.
        signals = (ctx.scout_signals or {}).get("insider_cluster_buy") or []
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
        # Strongest conviction first: more insiders, then bigger dollars.
        def _conviction(sig: dict) -> tuple:
            return (sig.get("n_insiders") or 0,
                    sig.get("total_usd") or 0.0)
        for sig in sorted(
                (s for s in signals if isinstance(s, dict)),
                key=_conviction, reverse=True):
            tic = (sig.get("ticker") or "").upper()
            if not tic or tic in held or tic in seen:
                continue
            seen.add(tic)
            n_ins = sig.get("n_insiders") or 0
            total = sig.get("total_usd") or 0.0
            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=tic,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name, confidence=0.72, is_closing=False,
                reason=(f"insider cluster: {n_ins} insiders bought "
                        f"${total:,.0f} in {tic} this week"),
                metadata={"model": self.name, "leg": "entry",
                          "n_insiders": n_ins, "cluster_usd": total,
                          "stop_pct": STOP_PCT,
                          "take_profit_pct": TAKE_PROFIT_PCT,
                          "max_hold_days": MAX_HOLD_DAYS,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
            added += 1
            if len(held) + added >= MAX_POSITIONS:
                break
        return proposals
