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
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from allocator.allocator import AllocationDecision, MetaAllocator
from allocator.lifecycle import StrategyRegistry, StrategyState
from brokers.base import BrokerAdapter, OrderSide, OrderType
from risk.manager import Decision, RiskManager, RiskState

from .base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    rebalance_cadence_hours: int = 168        # weekly
    cycle_label: str = ""                     # for logging
    dry_run: bool = True                      # global default
    # Per-broker overrides; None means "use global dry_run".
    dry_run_coinbase: Optional[bool] = None
    dry_run_alpaca: Optional[bool] = None
    dry_run_kalshi: Optional[bool] = None
    # Per-strategy LIVE override: even if the broker is DRY, these
    # strategies place real orders. Used for gradual live rollout — set
    # `live_strategies={"crypto_funding_carry"}` to risk only that pod.
    # Strategies in this set ignore both global and per-broker DRY flags.
    live_strategies: Optional[set] = None

    def is_dry(self, venue: str, strategy: Optional[str] = None) -> bool:
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
    risk: Optional[RiskState] = None
    proposals_total: int = 0
    proposals_approved: int = 0
    proposals_rejected: int = 0
    proposals_scaled: int = 0
    trades_submitted: int = 0
    errors: List[str] = field(default_factory=list)
    rebalanced: bool = False


class Orchestrator:
    def __init__(
        self,
        brokers: Dict[str, BrokerAdapter],
        registry: StrategyRegistry,
        risk_manager: RiskManager,
        allocator: MetaAllocator,
        strategies: Dict[str, Strategy],
        config: Optional[OrchestratorConfig] = None,
    ):
        self.brokers = brokers
        self.registry = registry
        self.risk = risk_manager
        self.allocator = allocator
        self.strategies = strategies          # {name: Strategy instance}
        self.cfg = config or OrchestratorConfig()
        self._last_rebalance_ts: float = 0.0

    # ── One cycle ----------------------------------------------------------

    def run_cycle(self, scout_signals: Optional[Dict] = None) -> CycleReport:
        report = CycleReport(timestamp=datetime.now(timezone.utc))
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

    def _positions_for(self, venue: str) -> Dict:
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

        decision = self.risk.check_order(
            notional_usd=notional,
            symbol=proposal.symbol,
            is_closing=proposal.is_closing,
            strategy_name=proposal.strategy,
            existing_position_usd=existing_usd,
            state=state,
            venue=proposal.venue,
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
        except Exception as e:
            report.errors.append(f"[{proposal.strategy}] execution failed: {e}")
            logger.exception(f"[{proposal.strategy}] place_order raised")

    def _emergency_close_all(self, report: CycleReport, state: RiskState) -> None:
        """KILL switch fired — every strategy closes its own positions."""
        for name, strategy in self.strategies.items():
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
