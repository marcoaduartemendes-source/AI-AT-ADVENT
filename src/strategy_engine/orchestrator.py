"""Orchestrator — the new main loop.

One pass:

  1. Compute risk state (equity / drawdown / multiplier / kill-switch).
  2. If KILL: emergency-close every position; return.
  3. If allocator due (configurable cadence): rebalance.
  4. For each ACTIVE/WATCH strategy:
       a. Build StrategyContext (target alloc, open positions, scout signals)
       b. strategy.compute(ctx) → TradeProposals
       c. For each proposal: risk_manager.check_order()
       d. APPROVE/SCALE → submit via broker adapter
  5. Persist trades to existing performance.db.

Designed to be called once per cycle from the workflow (e.g. every 5 min).
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC

from allocator.allocator import MetaAllocator
from allocator.lifecycle import StrategyRegistry, StrategyState
from brokers.base import BrokerAdapter, OrderSide
from risk.manager import Decision, RiskManager, RiskState
from trading.performance import PerformanceTracker
from trading.portfolio import TradeRecord

from .base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    rebalance_cadence_hours: int = 168        # weekly
    cycle_label: str = ""                     # for logging
    dry_run: bool = True                      # global default
    # Per-broker overrides; None means "use global dry_run".
    dry_run_coinbase: bool | None = None
    dry_run_alpaca: bool | None = None
    dry_run_kalshi: bool | None = None
    # Per-strategy LIVE override: even if the broker is DRY, these
    # strategies place real orders. Used for gradual live rollout — set
    # `live_strategies={"crypto_funding_carry"}` to risk only that pod.
    # Strategies in this set ignore both global and per-broker DRY flags.
    live_strategies: set | None = None

    def is_dry(self, venue: str, strategy: str | None = None) -> bool:
        # Per-strategy LIVE override wins over everything else.
        if strategy and self.live_strategies and strategy in self.live_strategies:
            return False
        per_broker = {
            "coinbase": self.dry_run_coinbase,
            "alpaca":   self.dry_run_alpaca,
            "kalshi":   self.dry_run_kalshi,
        }.get(venue)
        return self.dry_run if per_broker is None else per_broker


@dataclass
class CycleReport:
    timestamp: datetime
    risk: RiskState | None = None
    proposals_total: int = 0
    proposals_approved: int = 0
    proposals_rejected: int = 0
    proposals_scaled: int = 0
    trades_submitted: int = 0
    errors: list[str] = field(default_factory=list)
    rebalanced: bool = False


class Orchestrator:
    def __init__(
        self,
        brokers: dict[str, BrokerAdapter],
        registry: StrategyRegistry,
        risk_manager: RiskManager,
        allocator: MetaAllocator,
        strategies: dict[str, Strategy],
        config: OrchestratorConfig | None = None,
    ):
        self.brokers = brokers
        self.registry = registry
        self.risk = risk_manager
        self.allocator = allocator
        self.strategies = strategies          # {name: Strategy instance}
        self.cfg = config or OrchestratorConfig()
        self._last_rebalance_ts: float = 0.0
        # PerformanceTracker writes every executed trade to
        # trading_performance.db so the dashboard can render live P&L.
        try:
            self._tracker = PerformanceTracker()
        except Exception as e:
            logger.warning(f"PerformanceTracker init failed: {e}")
            self._tracker = None

        # ── One-time PnL sanitation ─────────────────────────────────────
        # Trades recorded before commit fc9640f had pnl_usd computed with
        # price=0 (orders captured at submission, before fill). That
        # produced phantom losses like (0 − $85) × 67 = −$5,746. Null
        # those rows out at the DB level so every consumer (dashboard,
        # strategic reviewer, allocator metrics) sees consistent state.
        # Idempotent — only touches rows where price=0 AND pnl_usd != NULL.
        if self._tracker is not None:
            try:
                import sqlite3 as _sql
                conn = _sql.connect(self._tracker.db_path)
                cur = conn.execute(
                    "UPDATE trades SET pnl_usd = NULL "
                    "WHERE pnl_usd IS NOT NULL AND (price IS NULL OR price = 0)"
                )
                if cur.rowcount > 0:
                    logger.info(f"PnL migration: nulled {cur.rowcount} "
                                f"phantom-loss row(s) (price=0)")
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"PnL migration skipped: {e}")

    # ── One cycle ----------------------------------------------------------

    def run_cycle(self, scout_signals: dict | None = None) -> CycleReport:
        report = CycleReport(timestamp=datetime.now(UTC))
        # Cache pending orders per venue for the whole cycle so we don't
        # re-query the broker dozens of times.
        self._pending_cache: dict[str, dict] = {}

        # Cancel pending orders older than the configured threshold.
        # Prevents a backlog of stale orders from blocking new ones.
        stale_threshold = int(os.environ.get("STALE_ORDER_SECONDS", "1800"))
        for vname, adapter in self.brokers.items():
            if not hasattr(adapter, "cancel_stale_orders"):
                continue
            try:
                n = adapter.cancel_stale_orders(stale_threshold)
                if n:
                    logger.info(f"[{vname}] cancelled {n} stale order(s) "
                                f"(>{stale_threshold}s old)")
            except Exception as e:
                logger.debug(f"[{vname}] cancel_stale_orders: {e}")

        # Phase-0: poll previously-PENDING orders for fills and backfill
        # the trade ledger with real fill prices. Without this every
        # SELL pnl_usd stays NULL because record_trade ran at submit
        # time. With it, the dashboard's "Realized P&L" finally
        # reflects the truth.
        try:
            self._poll_pending_fills(report)
        except Exception as e:
            logger.warning(f"poll_pending_fills failed: {e}")
        # If no scout_signals dict was passed in, hydrate from the bus
        if scout_signals is None:
            try:
                from scouts.signal_bus import SignalBus
                bus = SignalBus()
                scout_signals = {
                    venue: bus.get_fresh_for_strategy(venue)
                    for venue in {s.venue for s in self.strategies.values()}
                }
            except Exception as e:
                logger.debug(f"No signal bus available: {e}")
                scout_signals = {}

        # 1) Risk state
        try:
            state = self.risk.compute_state(persist=True)
        except Exception as e:
            report.errors.append(f"risk.compute_state failed: {e}")
            logger.exception("Risk state computation failed")
            return report
        report.risk = state

        # 2) Kill switch
        from risk.policies import KillSwitchState
        if state.kill_switch == KillSwitchState.KILL:
            logger.warning("KILL switch active — emergency closing all positions")
            try:
                from common.alerts import alert
                alert(
                    f"🚨 KILL switch fired — drawdown {state.drawdown_pct*100:.1f}%, "
                    f"equity ${state.equity_usd:,.2f}. "
                    f"Closing all positions.",
                    severity="critical",
                )
            except Exception as e:
                logger.debug(f"alert dispatch failed: {e}")
            self._emergency_close_all(report, state)
            return report

        # 3) Allocator (weekly)
        if self._is_rebalance_due():
            try:
                alloc = self.allocator.rebalance(portfolio_equity_usd=state.equity_usd)
                report.rebalanced = True
                self._last_rebalance_ts = time.time()
                logger.info(f"Allocator rebalance: {len(alloc.decisions)} strategies, "
                            f"total active={alloc.total_active_pct * 100:.1f}%")
            except Exception as e:
                report.errors.append(f"allocator.rebalance failed: {e}")
                logger.exception("Allocator rebalance failed")

        # 4) Per-strategy compute → gate → execute
        latest_alloc = self.registry.latest_allocations()
        for name, strategy in self.strategies.items():
            strategy_state = self.registry.get_state(name)
            if strategy_state in (StrategyState.FROZEN, StrategyState.RETIRED):
                logger.debug(f"[{name}] skipped — state={strategy_state.value}")
                continue

            target_pct = float(latest_alloc.get(name, {}).get("target_pct") or 0.0)
            target_usd = float(latest_alloc.get(name, {}).get("target_usd") or 0.0)
            if target_pct <= 0 and strategy_state == StrategyState.ACTIVE:
                # No allocation yet (first run) — give it the configured baseline
                meta = self.registry.meta(name)
                if meta:
                    target_pct = meta.target_alloc_pct
                    target_usd = state.equity_usd * target_pct

            ctx = StrategyContext(
                timestamp=report.timestamp,
                portfolio_equity_usd=state.equity_usd,
                target_alloc_pct=target_pct,
                target_alloc_usd=target_usd,
                risk_multiplier=state.multiplier.effective,
                open_positions=self._positions_for(strategy.venue),
                scout_signals=scout_signals.get(strategy.venue, {}),
                pending_orders=self._pending_orders_for(strategy.venue),
            )

            try:
                proposals = strategy.compute(ctx)
            except Exception as e:
                report.errors.append(f"[{name}] compute failed: {e}")
                logger.exception(f"[{name}] strategy.compute raised")
                continue

            report.proposals_total += len(proposals)
            for p in proposals:
                self._handle_proposal(p, state, report, strategy)

        return report

    # ── Helpers ----------------------------------------------------------

    def _is_rebalance_due(self) -> bool:
        if self._last_rebalance_ts == 0:
            return True
        elapsed_hours = (time.time() - self._last_rebalance_ts) / 3600
        return elapsed_hours >= self.cfg.rebalance_cadence_hours

    def _pending_orders_for(self, venue: str) -> dict:
        """Return aggregated pending-order notional per symbol.

        Strategies subtract pending BUY notional from their buying intent
        and pending SELL qty from their selling intent — prevents
        double-firing across consecutive 5-min cycles before fills land.

        Result is cached for the duration of one run_cycle().
        """
        cache = getattr(self, "_pending_cache", None)
        if cache is not None and venue in cache:
            return cache[venue]
        adapter = self.brokers.get(venue)
        if adapter is None or not hasattr(adapter, "get_open_orders"):
            if cache is not None: cache[venue] = {}
            return {}
        try:
            orders = adapter.get_open_orders()
        except Exception as e:
            logger.debug(f"[{venue}] get_open_orders failed: {e}")
            if cache is not None: cache[venue] = {}
            return {}
        out: dict[str, dict] = {}
        for o in orders:
            sym = o.symbol
            entry = out.setdefault(sym, {"buy_notional_usd": 0.0,
                                         "sell_qty": 0.0,
                                         "n_pending": 0})
            entry["n_pending"] += 1
            if o.side and o.side.value == "BUY":
                # Best-effort notional: limit price × qty, or notional_usd
                if o.notional_usd is not None:
                    entry["buy_notional_usd"] += o.notional_usd
                elif o.limit_price and o.quantity:
                    entry["buy_notional_usd"] += o.limit_price * o.quantity
                elif o.quantity and o.filled_avg_price:
                    entry["buy_notional_usd"] += o.quantity * o.filled_avg_price
            else:
                entry["sell_qty"] += float(o.quantity or 0)
        if cache is not None:
            cache[venue] = out
        return out

    def _positions_for(self, venue: str) -> dict:
        # Reuse the risk manager's per-cycle position cache; avoids hitting
        # the broker API again for each strategy on the same venue.
        cached = []
        try:
            cached = self.risk.cached_positions(venue)
        except Exception:
            cached = []

        if not cached:
            adapter = self.brokers.get(venue)
            if adapter is None:
                return {}
            try:
                cached = adapter.get_positions()
            except Exception as e:
                logger.warning(f"[{venue}] get_positions failed: {e}")
                return {}

        out = {}
        for p in cached:
            out[p.symbol] = {
                "venue": p.venue,
                "quantity": p.quantity,
                "avg_entry_price": p.avg_entry_price,
                "market_price": p.market_price,
                "unrealized_pnl_usd": p.unrealized_pnl_usd,
            }
        return out

    def _handle_proposal(
        self,
        proposal: TradeProposal,
        state: RiskState,
        report: CycleReport,
        strategy: Strategy,
    ) -> None:
        notional = proposal.notional_usd
        if notional is None and proposal.quantity is not None and proposal.limit_price:
            notional = proposal.quantity * proposal.limit_price
        if notional is None:
            # Best-effort fallback: use target alloc as size
            notional = state.equity_usd * 0.01

        existing = self._positions_for(proposal.venue).get(proposal.symbol, {})
        existing_usd = abs(existing.get("quantity", 0) *
                            existing.get("market_price", 0))

        # Pull asset_class from the registry so the per-asset-class
        # exposure cap (audit fix #5) can fire. Each strategy is
        # registered with a list of asset classes — use the first
        # since they're effectively single-class in practice.
        meta = self.registry.meta(proposal.strategy)
        asset_class = (meta.asset_classes[0]
                       if meta and meta.asset_classes else None)

        decision = self.risk.check_order(
            notional_usd=notional,
            symbol=proposal.symbol,
            is_closing=proposal.is_closing,
            strategy_name=proposal.strategy,
            existing_position_usd=existing_usd,
            state=state,
            venue=proposal.venue,
            asset_class=asset_class,
        )

        if decision.decision == Decision.REJECT:
            report.proposals_rejected += 1
            logger.info(f"[{proposal.strategy}] REJECTED {proposal.side.value} "
                        f"{proposal.symbol}: {decision.reason}")
            return

        if decision.decision == Decision.SCALE:
            report.proposals_scaled += 1
            proposal.notional_usd = decision.approved_notional_usd
            logger.info(f"[{proposal.strategy}] SCALED to "
                        f"${decision.approved_notional_usd:.2f}: {decision.reason}")

        # Wash-trade guard: if there's any pending order on this symbol
        # from a previous cycle, skip — let it resolve before we submit
        # something on the opposite side.
        pending_for_symbol = self._pending_orders_for(proposal.venue).get(
            proposal.symbol, {})
        if pending_for_symbol.get("n_pending", 0) > 0:
            logger.info(f"[{proposal.strategy}] SKIP {proposal.symbol}: "
                        f"{pending_for_symbol['n_pending']} pending order(s)")
            report.proposals_rejected += 1
            return

        # SELL-quantity guard: clamp SELL orders to qty_available so we
        # don't request more than the broker permits (HTTP 403
        # "insufficient qty available for order").
        #
        # Why we always convert to a fixed QTY (never let notional pass
        # through): when a proposal is notional-based, Alpaca's server
        # converts notional → qty using its CURRENT inside price, which
        # can differ from the cached price we used to compute the clamp.
        # A 1-2% intra-cycle price drop can push the server-derived qty
        # above qty_available even though our notional-clamp looked safe.
        # By forcing the qty path on the broker, the clamp becomes
        # authoritative. 10% buffer (was 5%) absorbs partial fills from
        # earlier in the cycle and any remaining snapshot staleness.
        if proposal.side == OrderSide.SELL:
            cached = self.risk.cached_positions(proposal.venue)
            BUFFER = 0.90  # keep 10% safety margin under qty_available
            for pos in cached:
                if pos.symbol != proposal.symbol:
                    continue
                avail = float(pos.raw.get("qty_available_parsed", pos.quantity)
                                if pos.raw else pos.quantity)
                if avail <= 0:
                    logger.info(f"[{proposal.strategy}] SKIP SELL "
                                f"{proposal.symbol}: 0 qty available")
                    report.proposals_rejected += 1
                    return

                # Resolve the requested qty (explicit or derived from
                # notional via cached market price).
                requested_qty = proposal.quantity
                if not requested_qty and proposal.notional_usd and pos.market_price > 0:
                    requested_qty = proposal.notional_usd / pos.market_price
                if not requested_qty:
                    break

                max_qty = avail * BUFFER
                if requested_qty > max_qty:
                    logger.info(f"[{proposal.strategy}] CLAMP SELL "
                                f"{proposal.symbol} qty {requested_qty:.4f} "
                                f"→ {max_qty:.4f} "
                                f"(qty_available={avail:.4f}, buffer=10%)")
                    proposal.quantity = max_qty
                else:
                    proposal.quantity = requested_qty
                # Force qty path on the broker so its server-side
                # notional→qty conversion can't drift back over.
                proposal.notional_usd = None
                break

        report.proposals_approved += 1

        # 5) Execute
        venue_dry = self.cfg.is_dry(proposal.venue, proposal.strategy)
        if venue_dry:
            logger.info(f"[{proposal.strategy}] DRY[{proposal.venue}] "
                        f"{proposal.side.value} {proposal.symbol} "
                        f"${decision.approved_notional_usd:.2f} ({proposal.reason})")
            return

        adapter = self.brokers.get(proposal.venue)
        if adapter is None:
            report.errors.append(f"venue {proposal.venue} not configured")
            return

        try:
            order = adapter.place_order(
                symbol=proposal.symbol,
                side=proposal.side,
                type=proposal.order_type,
                quantity=proposal.quantity,
                notional_usd=decision.approved_notional_usd,
                limit_price=proposal.limit_price,
                client_order_id=f"{proposal.strategy}-{uuid.uuid4().hex[:8]}",
            )
            report.trades_submitted += 1
            logger.info(f"[{proposal.strategy}] LIVE {proposal.side.value} "
                        f"{proposal.symbol} order_id={order.order_id} "
                        f"status={order.status.value}")
            strategy.on_fill(proposal, {"order_id": order.order_id,
                                          "status": order.status.value})
            # Register the just-placed order in the per-cycle pending
            # cache so subsequent proposals in the SAME cycle see it.
            # Without this, two strategies firing opposite-side orders
            # on the same symbol within a single cycle (e.g.
            # risk_parity_etf BUY SPY then tsmom_etf SELL SPY) trip
            # Alpaca's wash-trade rule because the broker sees two
            # opposite-side market orders for one symbol back-to-back.
            self._mark_pending_intracycle(proposal, order, decision)
            # Record the trade so the dashboard can render it.
            self._record_trade(proposal, order, decision)
        except Exception as e:
            report.errors.append(f"[{proposal.strategy}] execution failed: {e}")
            logger.exception(f"[{proposal.strategy}] place_order raised")

    def _poll_pending_fills(self, report: CycleReport) -> None:
        """Walk the trade ledger looking for orders we recorded at
        submit-time that the broker has since filled, and backfill the
        row with real fill data.

        The ledger row is created with `price=0, pnl_usd=NULL` because
        the broker hasn't reported a fill yet (record_trade runs
        immediately after the API call returns the order_id). Every
        cycle we revisit those rows, ask the broker for the current
        order status, and:
          - For FILLED  → write filled_avg_price, filled_quantity,
                          and (for closing SELLs) realized PnL.
          - For PARTIAL → write what's filled so far; row stays in
                          the unfilled set so the next cycle picks
                          up the rest.
          - For CANCELED / REJECTED → leave price=0; the dashboard
                          treats it as an un-realized fake.
          - For PENDING  → ignore; try again next cycle.

        Bounded at 48h via get_unfilled_trades — anything older is
        effectively dead and not worth polling.

        Independent FIFO recompute is done via
        recompute_pnl_independent() and the result is logged whenever
        it disagrees with the broker-attributed PnL by > $1, so the
        next phantom-loss class of bug surfaces as an alert instead
        of a user complaint.
        """
        if self._tracker is None:
            return
        try:
            unfilled = self._tracker.get_unfilled_trades(max_age_hours=48)
        except Exception as e:
            logger.warning(f"get_unfilled_trades failed: {e}")
            return
        if not unfilled:
            return

        # Map strategy name → venue (every strategy is mounted on
        # exactly one broker; this lookup is O(n_strategies) once
        # per cycle).
        strat_to_venue: dict[str, str] = {
            name: getattr(s, "venue", "")
            for name, s in self.strategies.items()
        }

        n_filled = 0
        n_partial = 0
        n_lost = 0
        for row in unfilled:
            order_id = row.get("order_id")
            strategy_name = row.get("strategy", "")
            venue = strat_to_venue.get(strategy_name)
            if not venue or venue not in self.brokers:
                continue
            adapter = self.brokers[venue]
            if not hasattr(adapter, "get_order"):
                continue
            try:
                ord_obj = adapter.get_order(order_id)
            except Exception as e:
                logger.debug(f"poll get_order({order_id}) failed: {e}")
                continue

            from brokers.base import OrderStatus as _OS
            status = ord_obj.status
            fill_qty = float(ord_obj.filled_quantity or 0)
            fill_px = float(ord_obj.filled_avg_price or 0)

            if status in (_OS.CANCELED, _OS.REJECTED) and fill_qty == 0:
                # Mark with sentinel price=-1 so we don't keep polling.
                # The dashboard treats negative price the same as 0
                # (no PnL contribution), so this is a no-op visually
                # but stops the polling loop from scanning it.
                self._tracker.update_trade_fill(
                    trade_id=row["id"],
                    price=-1.0,
                    quantity=0.0,
                    amount_usd=0.0,
                    pnl_usd=None,
                )
                n_lost += 1
                continue
            if fill_px <= 0 or fill_qty <= 0:
                continue  # still PENDING / OPEN

            # We have at least a partial fill. Compute realized PnL
            # only for closing SELLs we can attribute to a known entry.
            pnl_usd: float | None = None
            if row.get("side") == "SELL":
                # Look up entry price from the cached broker position.
                cached = self.risk.cached_positions(venue)
                for pos in cached:
                    if (pos.symbol == row.get("product_id")
                            and pos.avg_entry_price > 0):
                        pnl_usd = (fill_px - pos.avg_entry_price) * fill_qty
                        break

            self._tracker.update_trade_fill(
                trade_id=row["id"],
                price=fill_px,
                quantity=fill_qty,
                amount_usd=fill_qty * fill_px,
                pnl_usd=pnl_usd,
            )
            if status == _OS.PARTIALLY_FILLED:
                n_partial += 1
            else:
                n_filled += 1

        if n_filled or n_partial or n_lost:
            logger.info(
                f"poll_pending_fills: {n_filled} filled, "
                f"{n_partial} partial, {n_lost} canceled/rejected "
                f"(of {len(unfilled)} candidates)"
            )

        # Independent sanity check — recompute realized PnL from the
        # ledger via FIFO matching and alert if the broker-attributed
        # number drifts. This is the "second brain" that catches the
        # next phantom-loss bug pre-deployment.
        try:
            self._sanity_check_realized_pnl()
        except Exception as e:
            logger.debug(f"sanity_check_realized_pnl failed: {e}")

    def _sanity_check_realized_pnl(self) -> None:
        """Compare DB-stored realized PnL (sum of pnl_usd) against an
        independent FIFO recompute over the trade ledger. Logs a
        WARNING if they differ by > $1.

        The FIFO recompute uses ONLY trade rows (timestamp, side, qty,
        price); it doesn't trust avg_entry_price from the broker. Two
        independent computations of the same number — disagreement
        means somebody's wrong, and the dashboard should be considered
        suspect until reconciled.
        """
        if self._tracker is None:
            return
        try:
            from trading.recompute import recompute_realized_pnl_fifo
            db_total, recomputed_total, per_strategy_drift = (
                recompute_realized_pnl_fifo(self._tracker.db_path)
            )
        except ImportError:
            return  # module not present yet — first deploy
        drift = abs(db_total - recomputed_total)
        if drift > 1.0:
            logger.warning(
                f"PnL DRIFT: DB total ${db_total:+.2f} vs "
                f"FIFO recompute ${recomputed_total:+.2f} "
                f"(drift ${drift:.2f}) — strategies: {per_strategy_drift}"
            )
            # Drift > $1 means our P&L is wrong somewhere — page the
            # human. Severity escalates by magnitude so a 5-cent drift
            # doesn't ping the same channel as a $1k drift.
            try:
                from common.alerts import alert
                sev = ("critical" if drift > 100 else
                       "warning" if drift > 10 else "info")
                alert(
                    f"PnL drift ${drift:.2f}: DB ${db_total:+.2f} "
                    f"vs FIFO ${recomputed_total:+.2f}. "
                    f"Per-strategy: {per_strategy_drift}",
                    severity=sev,
                )
            except Exception as e:
                logger.debug(f"alert dispatch failed: {e}")
        else:
            logger.debug(
                f"PnL sanity OK: DB ${db_total:+.2f} == "
                f"FIFO ${recomputed_total:+.2f} (drift ${drift:.2f})"
            )

    def _mark_pending_intracycle(self, proposal: TradeProposal, order, decision) -> None:
        """After a successful place_order, push the order into
        _pending_cache[venue][symbol] so the wash-trade guard fires
        for any later proposal in the SAME cycle on the same symbol.

        Alpaca rejects opposite-side market orders for the same symbol
        within seconds of each other (`reject_reason: "opposite side
        market/stop order exists"`). The per-cycle broker snapshot
        was taken before the cycle started, so it doesn't know about
        orders we just placed — without this hook, two strategies
        with opposing views (risk_parity buying / tsmom selling)
        will trip Alpaca every cycle."""
        cache = getattr(self, "_pending_cache", None)
        if cache is None:
            return
        venue = proposal.venue
        sym = proposal.symbol
        venue_map = cache.setdefault(venue, {})
        entry = venue_map.setdefault(sym, {
            "buy_notional_usd": 0.0,
            "sell_qty": 0.0,
            "n_pending": 0,
        })
        entry["n_pending"] += 1
        if proposal.side == OrderSide.BUY:
            notional = decision.approved_notional_usd or proposal.notional_usd or 0.0
            if not notional and proposal.quantity and proposal.limit_price:
                notional = proposal.quantity * proposal.limit_price
            entry["buy_notional_usd"] += float(notional or 0)
        else:
            entry["sell_qty"] += float(proposal.quantity or 0)

    def _record_trade(self, proposal: TradeProposal, order, decision) -> None:
        """Persist an executed trade to trading_performance.db so the
        dashboard can render it.

        IMPORTANT: orders are recorded at submission time, before the
        broker reports a fill. order.filled_avg_price is None and
        proposal.limit_price is None for MARKET orders. We MUST NOT
        compute realized PnL with price=0 — that produces gigantic
        false losses like `(0 − entry_$85) × 67 qty = −$5,746`.

        Fall-back price for the trade record (cosmetic only, used to
        display a sensible price column on the dashboard) is the
        cached market_price of the position. Realized PnL is left as
        None until the order fills and a follow-up cycle attributes
        it — the dashboard treats None correctly (excluded from
        win/loss totals).
        """
        if self._tracker is None:
            return
        try:
            # Best-effort price for the cosmetic price column. Cached
            # market_price >> 0 when the order is still PENDING.
            cached = self.risk.cached_positions(proposal.venue)
            cached_pos = next(
                (p for p in cached if p.symbol == proposal.symbol),
                None,
            )
            cached_mark = (
                cached_pos.market_price
                if (cached_pos and cached_pos.market_price > 0)
                else 0.0
            )
            price = (
                order.filled_avg_price
                or proposal.limit_price
                or cached_mark
                or 0.0
            )
            qty = order.filled_quantity or order.quantity or proposal.quantity or 0.0
            amount_usd = decision.approved_notional_usd
            if not amount_usd and qty and price:
                amount_usd = qty * price

            # Realized PnL: ONLY when we have a genuine fill price
            # (not 0, not the cached mark — those don't reflect what
            # actually executed). For now leave None; a future
            # follow-up that polls the order for fills can backfill.
            pnl_usd = None
            real_fill = order.filled_avg_price
            if (real_fill and real_fill > 0
                    and proposal.is_closing
                    and proposal.side == OrderSide.SELL
                    and cached_pos and cached_pos.avg_entry_price > 0):
                pnl_usd = (real_fill - cached_pos.avg_entry_price) * (
                    qty or cached_pos.quantity
                )

            record = TradeRecord(
                timestamp=datetime.now(UTC),
                strategy=proposal.strategy,
                product_id=proposal.symbol,
                side=proposal.side.value,
                amount_usd=float(amount_usd or 0),
                quantity=float(qty or 0),
                price=float(price or 0),
                order_id=order.order_id or "",
                pnl_usd=pnl_usd,
                dry_run=self.cfg.is_dry(proposal.venue, proposal.strategy),
            )
            self._tracker.record_trade(record)
        except Exception as e:
            logger.warning(f"[{proposal.strategy}] record_trade failed: {e}")

    def _emergency_close_all(self, report: CycleReport, state: RiskState) -> None:
        """KILL switch fired — every strategy closes its own positions."""
        for strategy in self.strategies.values():
            ctx = StrategyContext(
                timestamp=report.timestamp,
                portfolio_equity_usd=state.equity_usd,
                target_alloc_pct=0.0,
                target_alloc_usd=0.0,
                risk_multiplier=0.0,
                open_positions=self._positions_for(strategy.venue),
                scout_signals={},
            )
            for p in strategy.on_emergency_close(ctx):
                self._handle_proposal(p, state, report, strategy)
