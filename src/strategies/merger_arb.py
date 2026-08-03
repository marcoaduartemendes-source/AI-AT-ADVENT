"""Merger arbitrage — buy announced cash-deal targets at the spread,
hold to deal close.

REAL EDGE, REAL FIRMS:
    25-year annualized 6.7%, vol 3.2%, Sharpe ~1.46 (S&P Merger Arb
    Index; Accelerate Shares 2023). Pershing Square, Paulson, Water
    Island (MERFX), Millennium, Magnetar, Pentwater all run it.

    Edge source: compensation for deal-break tail risk (regulator
    blocks, financing fall-through). Index funds dump targets on
    announcement → arbs supply the liquidity and harvest the spread.

ENTRY:
    Buy target when (deal_price / current_price - 1) ≥ 1% (the
    minimum spread above frictional costs). Only cash deals — we
    avoid stock-for-stock since short-locate inventory is unreliable
    and the trade is asymmetric without the acquirer hedge.

EXIT:
    • Take-profit if spread compresses to ≤ 0.3% (deal nearly closed).
    • Hard stop if price drops > 8% below entry (deal-break signature).
    • Forced age-out at 180 days (deals dragging > 6 months are
      regulator-risk territory; cut and redeploy).

SAFETY:
    • Per-name cap 20% of sleeve (≤ 5 concurrent deals).
    • Skip targets trading > 2% above deal price (already arb'd out).
    • Long-only.

FAILURE MODES:
    Antitrust regime change (FTC 2021-24 killed ATVI/JBLU-SAVE),
    financing collapse, hostile-bidder withdrawal. The 8% stop is
    designed to cut on these — break-of-deal price drops are usually
    -10% to -25% in minutes.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)

PER_NAME_PCT = 0.20         # 20% of sleeve per deal (≤ 5)
MIN_SPREAD_PCT = 0.01       # 1% spread to enter
EXIT_SPREAD_PCT = 0.003     # close when spread ≤ 0.3%
STOP_PCT = 0.08             # -8% stop (deal-break protection)
SKIP_OVER_DEAL_PCT = 0.02   # skip if trading > 2% over deal price
MAX_HOLD_DAYS = 180
MAX_POSITIONS = 5
MIN_ENTRY_USD = 200.0


class MergerArb(Strategy):
    name = "merger_arb"
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

        # ── 1) Manage open positions.
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
            if entry and last and last <= entry * (1 - STOP_PCT):
                exit_reason = (f"deal-break stop: ${last:.2f} ≤ "
                               f"${entry:.2f} × {1-STOP_PCT:.0%}")
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
                            exit_reason = (f"age-out: > {MAX_HOLD_DAYS}d")
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

        # ── 2) New entries from deal feed.
        signals = (ctx.scout_signals or {}).get("merger_arb_deal") or []
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
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            target = (sig.get("target") or "").upper()
            deal_price = sig.get("deal_price")
            if not target or not deal_price or target in held or target in seen:
                continue
            seen.add(target)
            try:
                deal_price = float(deal_price)
            except (TypeError, ValueError):
                continue
            try:
                last = float(self.broker.get_candles(
                    target, "ONE_DAY", num_candles=2)[-1].close)
            except Exception:
                continue
            if last <= 0:
                continue
            spread = (deal_price / last) - 1.0
            # Need positive spread of MIN_SPREAD_PCT; reject if trading
            # over the deal price (deal failure or arb'd out).
            if spread < MIN_SPREAD_PCT:
                continue
            if spread < -SKIP_OVER_DEAL_PCT:
                continue
            # Close-to-target proposals: skip if already very tight.
            if spread <= EXIT_SPREAD_PCT:
                continue

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=target,
                side=OrderSide.BUY, order_type=OrderType.MARKET,
                notional_usd=per_name, confidence=0.75,
                is_closing=False,
                reason=(f"M&A target {target} @ ${last:.2f} vs deal "
                        f"${deal_price:.2f} (spread {spread:.2%})"),
                metadata={"model": self.name, "leg": "entry",
                          "deal_price": deal_price,
                          "entry_price": last,
                          "spread_pct": spread,
                          "stop_pct": STOP_PCT,
                          "as_of": datetime.now(UTC).isoformat()},
            ))
            added += 1
            if len(held) + added >= MAX_POSITIONS:
                break

        return proposals
