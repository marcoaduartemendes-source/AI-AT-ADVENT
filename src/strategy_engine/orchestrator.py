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
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC

from allocator.allocator import MetaAllocator
from allocator.lifecycle import StrategyRegistry, StrategyState
from brokers.base import (
    BrokerAdapter,
    BrokerCapability,
    OrderSide,
    OrderStatus,
)
from common.market_hours import is_market_open, venue_window_str
from risk.manager import Decision, RiskManager, RiskState
from risk.policies import KillSwitchState
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
    # Cancel pending orders older than this many seconds at the top of
    # each cycle. Was a bare env-var read inside run_cycle; now
    # configurable via OrchestratorConfig and overridable via env.
    stale_order_seconds: int = 1800

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
        except (OSError, sqlite3.Error) as e:
            logger.warning(f"PerformanceTracker init failed: {e}")
            self._tracker = None

        # ── DB migrations (audit fix: previously this was an inline
        # UPDATE running on every init at debug-level on failure).
        # Now: marker-row-gated, runs at most once per migration name,
        # WARNING-level on per-migration errors so journalctl surfaces
        # schema drift instead of silently swallowing it.
        if self._tracker is not None:
            from trading.migrations import apply_pending
            apply_pending(self._tracker.db_path)

    # ── One cycle ----------------------------------------------------------

    def run_cycle(self, scout_signals: dict | None = None) -> CycleReport:
        report = CycleReport(timestamp=datetime.now(UTC))
        # Cache pending orders per venue for the whole cycle so we don't
        # re-query the broker dozens of times.
        self._pending_cache: dict[str, dict] = {}
        # Venues whose pending-orders read failed this cycle. Strategies
        # are blocked from opening NEW positions on a degraded venue
        # (closing positions still allowed) — see _pending_orders_for
        # and _handle_proposal.
        self._degraded_venues: set[str] = set()
        # Defensive: clear the risk manager's broker snapshot cache so
        # an aborted previous cycle can't leak stale positions/account
        # data into this one. compute_state would clear it on success
        # anyway, but if it raises we still want the next cycle to start
        # clean.
        try:
            self.risk._broker_snapshots = {}
        except Exception:
            pass

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

        # Pager when one or more brokers fail their account read for
        # 2+ consecutive cycles. Without this, an expired Alpaca
        # API key / IP-allowlist drift / Coinbase auth issue degrades
        # silently (we just stop trading that venue) until the user
        # notices via the dashboard. Threshold of 2 cycles avoids
        # paging on a single transient blip.
        if not state.venues_ok:
            self._venues_ok_consecutive_failures = (
                getattr(self, "_venues_ok_consecutive_failures", 0) + 1
            )
            if self._venues_ok_consecutive_failures == 2:
                try:
                    from common.alerts import alert
                    alert(
                        f"One or more broker accounts unreachable for "
                        f"2 consecutive cycles. Trading is degraded; "
                        f"check broker auth / API status. "
                        f"equity_usd=${state.equity_usd:,.2f}",
                        severity="warning",
                    )
                except Exception as e:
                    logger.warning(f"alert dispatch failed: {e}")
        else:
            self._venues_ok_consecutive_failures = 0

        # Cancel pending orders older than the configured threshold.
        # Runs AFTER compute_state so we have an up-to-date view of
        # broker reachability (state.venues_ok). When the global flag
        # is False at least one broker is unreachable — we still try
        # the others; the per-venue try/except guards us against firing
        # cancels at the broken one.
        stale_threshold = self.cfg.stale_order_seconds
        for vname, adapter in self.brokers.items():
            if BrokerCapability.CANCEL_STALE_ORDERS not in adapter.capabilities:
                continue
            try:
                n = adapter.cancel_stale_orders(stale_threshold)
                if n:
                    logger.info(f"[{vname}] cancelled {n} stale order(s) "
                                f"(>{stale_threshold}s old)")
            except Exception as e:
                logger.debug(f"[{vname}] cancel_stale_orders: {e}")

        # 2) Kill switch
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
                # Audit fix: was debug — alert pipeline failure is
                # itself an alertable condition; we just can't alert
                # about it. Log loud so journalctl shows it.
                logger.warning(f"alert dispatch failed: {e}")
            self._emergency_close_all(report, state)
            return report

        # 3) Allocator (weekly)
        if self._is_rebalance_due():
            try:
                alloc = self.allocator.rebalance(portfolio_equity_usd=state.equity_usd)
                report.rebalanced = True
                self._last_rebalance_ts = time.time()
                self._allocator_consecutive_failures = 0
                logger.info(f"Allocator rebalance: {len(alloc.decisions)} strategies, "
                            f"total active={alloc.total_active_pct * 100:.1f}%")
            except Exception as e:
                report.errors.append(f"allocator.rebalance failed: {e}")
                logger.exception("Allocator rebalance failed")
                # Track consecutive failures so a permanent allocator
                # outage (corrupt metrics.db, schema drift) surfaces
                # via alert rather than silently freezing allocations.
                self._allocator_consecutive_failures = (
                    getattr(self, "_allocator_consecutive_failures", 0) + 1
                )
                if self._allocator_consecutive_failures in (3, 10):
                    try:
                        from common.alerts import alert
                        alert(
                            f"Allocator rebalance failing for "
                            f"{self._allocator_consecutive_failures} "
                            f"consecutive cycles: {type(e).__name__}: "
                            f"{str(e)[:160]}",
                            severity="warning",
                        )
                    except Exception as alert_e:
                        logger.warning(
                            f"alert dispatch failed: {alert_e}"
                        )

        # 4) Per-strategy compute → gate → execute
        # Early exit when no active strategy's venue is open. Saves
        # ~15-25s of useless work on overnight / weekend cycles where
        # _handle_proposal would reject everything at the market-hours
        # gate anyway. Crypto venues (coinbase, kalshi) are always
        # open, so this only short-circuits cycles where the active
        # set is purely Alpaca-equity and Alpaca is closed.
        active_venues = {s.venue for s in self.strategies.values()}
        if active_venues and not any(
                is_market_open(v) for v in active_venues):
            logger.info(
                f"All active venues closed ({sorted(active_venues)}); "
                f"skipping per-strategy compute."
            )
            return report

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
                # Sprint E1 wiring — record this strategy as failed for
                # the consecutive-error tracker. Best-effort: tracking
                # itself raising must not break the cycle.
                err_msg = f"compute: {type(e).__name__}: {str(e)[:160]}"
                self._record_strategy_outcome(name, had_error=True,
                                                error_text=err_msg)
                continue

            # Strategy compute succeeded → record clean cycle for the
            # consecutive-error tracker. Resets the count if it was
            # accumulating.
            self._record_strategy_outcome(name, had_error=False)

            report.proposals_total += len(proposals)
            for p in proposals:
                self._handle_proposal(p, state, report, strategy)

        return report

    @staticmethod
    def _record_strategy_outcome(name: str, *, had_error: bool,
                                   error_text: str | None = None) -> None:
        """Wrapper around common.strategy_alerts.record_cycle_outcome
        that swallows all exceptions — alerting must never break the
        cycle. Lazy-imported so test fixtures that don't need the
        alert DB don't pay the import cost."""
        try:
            from common.strategy_alerts import record_cycle_outcome
            record_cycle_outcome(
                name, had_error=had_error, error_text=error_text,
            )
        except Exception as e:    # noqa: BLE001
            logger.debug(f"strategy_alerts hook failed for {name}: {e}")

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
        if (adapter is None
                or BrokerCapability.GET_OPEN_ORDERS not in adapter.capabilities):
            if cache is not None: cache[venue] = {}
            return {}
        try:
            orders = adapter.get_open_orders()
        except Exception as e:
            # Broker outage: fail-CLOSED. Mark the venue as degraded so
            # _handle_proposal blocks NEW opens (BUYs and non-closing
            # SELLs) for this cycle. Closing SELLs are still allowed —
            # we want to be able to reduce exposure even when the
            # exchange's read API is flaky. Without this flag, returning
            # {} would let strategies double-fire orders that already
            # exist (a previous WARNING-level log was the only signal).
            logger.warning(f"[{venue}] get_open_orders failed: {e} "
                           f"— marking venue degraded for this cycle")
            self._degraded_venues.add(venue)
            if cache is not None: cache[venue] = {}
            return {}
        out: dict[str, dict] = {}
        for o in orders:
            sym = o.symbol
            entry = out.setdefault(sym, {"buy_notional_usd": 0.0,
                                         "sell_qty": 0.0,
                                         "n_pending": 0,
                                         "n_buy_pending": 0,
                                         "n_sell_pending": 0})
            entry["n_pending"] += 1
            if o.side and o.side.value == "BUY":
                entry["n_buy_pending"] += 1
                # Best-effort notional: limit price × qty, or notional_usd
                if o.notional_usd is not None:
                    entry["buy_notional_usd"] += o.notional_usd
                elif o.limit_price and o.quantity:
                    entry["buy_notional_usd"] += o.limit_price * o.quantity
                elif o.quantity and o.filled_avg_price:
                    entry["buy_notional_usd"] += o.quantity * o.filled_avg_price
            else:
                entry["n_sell_pending"] += 1
                entry["sell_qty"] += float(o.quantity or 0)
        if cache is not None:
            cache[venue] = out
        return out

    def _positions_for(self, venue: str) -> dict[str, "PositionView"]:
        # Reuse the risk manager's per-cycle position cache; avoids hitting
        # the broker API again for each strategy on the same venue.
        # Returns dict[symbol, PositionView] — the dataclass shim keeps
        # `pos.get("quantity")` and `pos["quantity"]` working in legacy
        # strategy code while typed `pos.quantity` access is also valid.
        from .base import PositionView
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

        out: dict[str, PositionView] = {}
        for p in cached:
            entry_time = None
            if p.raw and isinstance(p.raw, dict):
                entry_time = p.raw.get("entry_time")
            out[p.symbol] = PositionView(
                venue=p.venue,
                symbol=p.symbol,
                quantity=float(p.quantity or 0),
                avg_entry_price=float(p.avg_entry_price or 0),
                market_price=float(p.market_price or 0),
                unrealized_pnl_usd=float(p.unrealized_pnl_usd or 0),
                entry_time=entry_time,
                asset_class=p.asset_class.value if p.asset_class else None,
            )
        return out

    def _handle_proposal(
        self,
        proposal: TradeProposal,
        state: RiskState,
        report: CycleReport,
        strategy: Strategy,
    ) -> None:
        """Orchestrate one proposal through the risk → guards →
        execute pipeline. Audit fix #4: decomposed into focused
        helpers; market-hours guard added later as the FIRST gate
        because there's no point running risk + wash-trade logic
        for an order that the broker is going to cancel for being
        out-of-session.
        """
        # Market-hours gate (fixes the 770-PENDING / 128-CANCELED
        # state on Alpaca caused by orders fired at 23:40 UTC). The
        # broker silently cancels day orders submitted outside the
        # US regular session; we now skip those proposals up front.
        if not is_market_open(proposal.venue):
            logger.debug(
                f"[{proposal.strategy}] SKIP {proposal.venue} closed "
                f"({venue_window_str(proposal.venue)})"
            )
            report.proposals_rejected += 1
            return

        # Degraded-venue gate: if get_open_orders failed this cycle the
        # wash-trade guard has no visibility, so block NEW opens. Closing
        # SELLs are allowed — better to reduce exposure than be stuck.
        if (proposal.venue in getattr(self, "_degraded_venues", set())
                and not proposal.is_closing):
            logger.info(
                f"[{proposal.strategy}] SKIP {proposal.symbol}: "
                f"venue {proposal.venue} degraded "
                f"(get_open_orders failed); opens blocked this cycle"
            )
            report.proposals_rejected += 1
            return

        notional, asset_class, existing_usd = self._resolve_proposal_size(
            proposal, state,
        )

        decision = self._gate_through_risk(
            proposal, notional, existing_usd, asset_class, state, report,
        )
        if decision is None:
            return    # rejected by risk

        if self._check_intracycle_wash(proposal, report):
            return    # pending conflict — skip

        if not self._clamp_sell_quantity(proposal, report):
            return    # 0 qty available — skip

        report.proposals_approved += 1
        self._execute_proposal(proposal, decision, report, strategy)

    # ── Helpers extracted from _handle_proposal ───────────────────────

    def _resolve_proposal_size(
        self, proposal: TradeProposal, state: RiskState,
    ) -> tuple[float, str | None, float]:
        """Return (notional_usd, asset_class, existing_position_usd) for
        the risk gate. notional falls back to 1% of equity if the
        proposal didn't specify; asset_class comes from the strategy
        registry."""
        notional = proposal.notional_usd
        if notional is None and proposal.quantity is not None and proposal.limit_price:
            notional = proposal.quantity * proposal.limit_price
        if notional is None:
            notional = state.equity_usd * 0.01

        existing = self._positions_for(proposal.venue).get(proposal.symbol, {})
        existing_usd = abs(existing.get("quantity", 0) *
                            existing.get("market_price", 0))

        meta = self.registry.meta(proposal.strategy)
        asset_class = (meta.asset_classes[0]
                       if meta and meta.asset_classes else None)
        return notional, asset_class, existing_usd

    def _gate_through_risk(
        self,
        proposal: TradeProposal,
        notional: float,
        existing_usd: float,
        asset_class: str | None,
        state: RiskState,
        report: CycleReport,
    ):
        """Call risk.check_order and apply REJECT / SCALE side effects.
        Returns the RiskDecision when the proposal should continue, or
        None if it was rejected."""
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
            return None

        if decision.decision == Decision.SCALE:
            report.proposals_scaled += 1
            proposal.notional_usd = decision.approved_notional_usd
            logger.info(f"[{proposal.strategy}] SCALED to "
                        f"${decision.approved_notional_usd:.2f}: {decision.reason}")
        return decision

    def _check_intracycle_wash(
        self, proposal: TradeProposal, report: CycleReport,
    ) -> bool:
        """Return True if the proposal should be skipped due to an
        OPPOSITE-side pending order on the same symbol (carryover from
        previous cycle OR the just-placed earlier proposal in THIS
        cycle). Same-side proposals are allowed through — Alpaca only
        rejects opposite-side concurrency, and rejecting same-side
        proposals stops legitimate strategy aggregation (e.g. risk_parity
        + tsmom both buying SPY).
        Side-effect: increments report.proposals_rejected on skip."""
        pending = self._pending_orders_for(proposal.venue).get(
            proposal.symbol, {})
        if proposal.side == OrderSide.BUY:
            opposite = pending.get("n_sell_pending", 0)
            opposite_label = "SELL"
        else:
            opposite = pending.get("n_buy_pending", 0)
            opposite_label = "BUY"
        if opposite > 0:
            logger.info(f"[{proposal.strategy}] SKIP {proposal.symbol}: "
                        f"{opposite} opposite-side pending {opposite_label} "
                        f"order(s)")
            report.proposals_rejected += 1
            return True
        return False

    def _clamp_sell_quantity(
        self, proposal: TradeProposal, report: CycleReport,
    ) -> bool:
        """For SELL proposals, clamp `proposal.quantity` to
        qty_available × 0.90 (10% safety buffer) and force the qty
        path on the broker so server-side notional→qty conversion
        can't drift over the limit. Returns False (and increments
        report.proposals_rejected) only if 0 qty is available — caller
        should skip in that case."""
        if proposal.side != OrderSide.SELL:
            return True

        BUFFER = 0.90  # keep 10% safety margin under qty_available
        cached = self.risk.cached_positions(proposal.venue)
        for pos in cached:
            if pos.symbol != proposal.symbol:
                continue
            avail = float(pos.raw.get("qty_available_parsed", pos.quantity)
                            if pos.raw else pos.quantity)
            if avail <= 0:
                logger.info(f"[{proposal.strategy}] SKIP SELL "
                            f"{proposal.symbol}: 0 qty available")
                report.proposals_rejected += 1
                return False

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
        return True

    def _execute_proposal(
        self,
        proposal: TradeProposal,
        decision,
        report: CycleReport,
        strategy: Strategy,
    ) -> None:
        """Final step: place the order with the broker (or log DRY),
        record to ledger, mark intra-cycle pending. Catches
        broker-side errors so a bad call doesn't kill the cycle."""
        if self.cfg.is_dry(proposal.venue, proposal.strategy):
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
            # Register the just-placed order so subsequent proposals
            # in the same cycle see it (audit/regression: prevents
            # the wash-trade rejection from Alpaca on opposing-side
            # orders fired by different strategies in one cycle).
            self._mark_pending_intracycle(proposal, order, decision)
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
        # per cycle). Used as fallback when the trade row predates
        # migration 002 (no `venue` column).
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
            # Prefer the venue stored on the trade row (migration 002):
            # it survives the strategy being retired between submit and
            # fill, which strat_to_venue does not.
            venue = row.get("venue") or strat_to_venue.get(strategy_name)
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

            status = ord_obj.status
            fill_qty = float(ord_obj.filled_quantity or 0)
            fill_px = float(ord_obj.filled_avg_price or 0)

            if status in (OrderStatus.CANCELED, OrderStatus.REJECTED) and fill_qty == 0:
                # Audit fix #2: explicit fill_status='CANCELED' rather
                # than the price=-1 sentinel. The query in
                # get_unfilled_trades now keys on fill_status, so
                # canceled rows are excluded automatically.
                cancel_status = ("REJECTED" if status == OrderStatus.REJECTED
                                  else "CANCELED")
                self._tracker.update_trade_fill(
                    trade_id=row["id"],
                    price=0.0,
                    quantity=0.0,
                    amount_usd=0.0,
                    pnl_usd=None,
                    fill_status=cancel_status,
                )
                n_lost += 1
                continue
            if fill_px <= 0 or fill_qty <= 0:
                continue  # still PENDING / OPEN

            # We have at least a partial fill. Compute realized PnL
            # only for closing SELLs we can attribute to a known entry.
            # PnL is ONLY recorded when fill_status=='FILLED' (the
            # invariant that audit fix #2 enforces in update_trade_fill).
            pnl_usd: float | None = None
            new_status = ("PARTIALLY_FILLED"
                          if status == OrderStatus.PARTIALLY_FILLED else "FILLED")
            if new_status == "FILLED" and row.get("side") == "SELL":
                # Prefer the entry_price persisted at submit time
                # (migration 002): it works even when the position
                # has been fully closed and is no longer in the
                # broker's positions list. Fall back to cached
                # positions for legacy rows that predate the column.
                row_entry = row.get("entry_price")
                if row_entry and row_entry > 0:
                    pnl_usd = (fill_px - float(row_entry)) * fill_qty
                else:
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
                fill_status=new_status,
            )
            if new_status == "PARTIALLY_FILLED":
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
            # Audit fix: was debug — this is the "second brain" that
            # catches phantom-loss-style consistency bugs. Silent
            # failure here is exactly what the previous regression hid.
            logger.warning(f"sanity_check_realized_pnl failed: {e}")

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
        # Use find_spec rather than catching ImportError around the
        # actual import: a real bug in trading.recompute (syntax error,
        # transitively missing dep) would otherwise be caught and
        # silently disable the consistency check forever.
        import importlib.util
        if importlib.util.find_spec("trading.recompute") is None:
            return    # module not deployed yet — first deploy
        from trading.recompute import recompute_realized_pnl_fifo
        db_total, recomputed_total, per_strategy_drift = (
            recompute_realized_pnl_fifo(self._tracker.db_path)
        )
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
            #
            # Audit fix (alert spam): the alert text now ROUNDS the
            # drift dollars to the nearest $5 bucket so persistent
            # ~$5 drift collapses to one Pushover via the dedup
            # cache. Only NEW drift magnitude (or sign flip) bypasses
            # dedup. Without this, a stable phantom-loss row caused
            # 12 identical alerts/hour for the user.
            try:
                from common.alerts import alert
                sev = ("critical" if drift > 100 else
                       "warning" if drift > 10 else "info")
                # Bucket the drift into $5 increments so the message
                # text is stable across cycles when the underlying
                # mismatch is the same row.
                bucket = round(drift / 5.0) * 5.0
                alert(
                    f"PnL drift ~${bucket:.0f} (actual ${drift:.2f}): "
                    f"DB ${db_total:+.2f} vs FIFO ${recomputed_total:+.2f}",
                    severity=sev,
                )
            except Exception as e:
                # Audit fix: was debug — alert pipeline failure is
                # itself an alertable condition; we just can't alert
                # about it. Log loud so journalctl shows it.
                logger.warning(f"alert dispatch failed: {e}")
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
            "n_buy_pending": 0,
            "n_sell_pending": 0,
        })
        entry["n_pending"] += 1
        if proposal.side == OrderSide.BUY:
            entry["n_buy_pending"] += 1
            notional = decision.approved_notional_usd or proposal.notional_usd or 0.0
            if not notional and proposal.quantity and proposal.limit_price:
                notional = proposal.quantity * proposal.limit_price
            entry["buy_notional_usd"] += float(notional or 0)
        else:
            entry["n_sell_pending"] += 1
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

            # Map the broker's order status into our fill_status enum.
            # FILLED / PARTIALLY_FILLED → use as-is.
            # OPEN / PENDING / NEW → 'PENDING' (the poller will transition).
            # CANCELED / REJECTED → use as-is.
            broker_status = order.status
            if broker_status == OrderStatus.FILLED:
                fill_status_str: str | None = "FILLED"
            elif broker_status == OrderStatus.PARTIALLY_FILLED:
                fill_status_str = "PARTIALLY_FILLED"
            elif broker_status in (OrderStatus.CANCELED, OrderStatus.REJECTED):
                fill_status_str = broker_status.value
            else:
                # PENDING / OPEN / NEW etc. — let the polling loop
                # transition once the broker reports a real fill.
                fill_status_str = "PENDING"

            # Capture entry_price + venue at submit time so the
            # fill-polling loop can attribute PnL even if the position
            # has been fully closed by the time it runs (cached_positions
            # would no longer carry the symbol) or if the strategy was
            # RETIRED in the meantime.
            entry_price_at_submit: float | None = None
            if cached_pos and cached_pos.avg_entry_price > 0:
                entry_price_at_submit = float(cached_pos.avg_entry_price)
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
                fill_status=fill_status_str,
                entry_price=entry_price_at_submit,
                venue=proposal.venue,
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
        # Once the closes are submitted, give the broker a chance to
        # report fills before we return. Without this the KILL path
        # leaves orders stuck in PENDING and the next cycle re-fires
        # KILL (no cooldown) without re-entering the polling code,
        # so realized PnL on the close is never written.
        try:
            self._poll_pending_fills(report)
        except Exception as e:
            logger.warning(f"poll_pending_fills (post-KILL) failed: {e}")
