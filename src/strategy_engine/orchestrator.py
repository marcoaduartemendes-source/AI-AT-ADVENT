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

from .base import PositionView, Strategy, StrategyContext, TradeProposal

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
class StrategyOutcome:
    """Per-strategy outcome for a single cycle. Persisted so the
    dashboard can answer 'why didn't strategy X trade?' without
    requiring the user to read GH Actions logs.
    """
    strategy: str
    venue: str
    state: str = "ACTIVE"               # registry state
    target_alloc_pct: float = 0.0       # what allocator gave
    target_alloc_usd: float = 0.0
    proposed: int = 0                   # strategy.compute() returned N
    approved: int = 0                   # passed risk + clamp
    rejected: int = 0
    submitted: int = 0                  # actually placed at broker
    dry_logged: int = 0                 # logged DRY, not placed
    skip_reasons: list[str] = field(default_factory=list)
    # Why-rejected reasons (one per rejected proposal). Lets the
    # dashboard show "rej=5: kill_cooldown, max_strategy_daily_orders,
    # …" instead of just a count. Capped at last 10 to keep the JSON
    # bounded.
    reject_reasons: list[str] = field(default_factory=list)
    # Why-failed-to-submit reasons. Same idea for execution errors.
    execute_errors: list[str] = field(default_factory=list)
    error: str = ""                     # populated if compute() raised


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
    cycle_seconds: float = 0.0
    venue_health: dict[str, str] = field(default_factory=dict)
    strategy_outcomes: dict[str, StrategyOutcome] = field(default_factory=dict)


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
        cycle_start = time.time()
        # Heartbeat write at the very start of the cycle, before
        # ANYTHING else can fail. Disambiguates "orchestrator not
        # running" (heartbeat absent) from "orchestrator running but
        # diagnostics broken" (heartbeat present, cycle_diagnostics
        # empty). The dashboard reads this to show "Orchestrator alive
        # — last heartbeat 2m ago" even when nothing else has worked.
        try:
            self._write_heartbeat(report.timestamp)
        except Exception as e:
            logger.debug(f"heartbeat write failed: {e}")
        try:
            return self._run_cycle_body(report, cycle_start, scout_signals)
        finally:
            # ALWAYS write diagnostics — even on KILL early-return,
            # all-venues-closed early-return, or compute_state failure.
            # Without this, the dashboard's Cycle activity panel showed
            # "Bootstrapping" forever because every cycle hit an early
            # return path before reaching the persist call.
            # Observed 2026-05-09: PRs 17/18/19/20 all merged but
            # the panel stayed empty.
            try:
                report.cycle_seconds = round(time.time() - cycle_start, 2)
                report.venue_health = self._venue_health_snapshot()
                self._persist_cycle_diagnostics(report)
            except Exception as e:
                logger.warning(f"persist_cycle_diagnostics failed: {e}")
            # Belt-and-suspenders: also dump the cycle summary to a
            # JSON file in docs/ that gets committed alongside
            # index.html. This bypasses the actions/cache flow
            # entirely. If SQLite persistence is somehow being lost
            # between cycles (cache races, cache eviction, runner
            # crash before save), the dashboard can still render
            # truth from this file. The file accumulates the last
            # 50 cycles in a ring buffer.
            try:
                self._dump_cycle_status_json(report)
            except Exception as e:
                logger.warning(f"dump_cycle_status_json failed: {e}")
            # And the same for recent trades — read the cache-backed
            # trades.db, write the latest rows to docs/trades_recent.json
            # so the dashboard's Recent trades panel can show the
            # truth even if cache restore on the dashboard side is
            # off.
            try:
                self._dump_recent_trades_json()
            except Exception as e:
                logger.warning(f"dump_recent_trades_json failed: {e}")
            # Benchmark comparison: portfolio trailing return vs
            # SPY/QQQ/BTC over 7/14/30d. Runs here because this job
            # has both the FMP key and risk_state.db's equity curve.
            try:
                from common.benchmark import write_benchmark_json
                write_benchmark_json()
            except Exception as e:
                logger.warning(f"write_benchmark_json failed: {e}")
            # NOTE 2026-05-21 — the heavy backtests (run_validation +
            # run_walk_forward, ~200s EACH) used to run HERE every
            # cycle. That blew past the workflow's 8-min timeout and
            # KILLED the orchestrator process before the git-commit
            # step ran — so cycle_status froze at 2026-05-20T15:58 for
            # ~22h while the dashboard kept rebuilding stale data.
            # They now run in a SEPARATE daily workflow
            # (.github/workflows/research.yml → src/run_research.py).
            # The light agents below only READ the JSON those produce,
            # so they stay in the cycle (each is milliseconds).
            #
            # Autonomous performance review — synthesises backtest +
            # walk-forward + live trades into a ranked action queue.
            try:
                from common.performance_review import run_performance_review
                run_performance_review()
            except Exception as e:
                logger.warning(f"run_performance_review failed: {e}")
            # Data-quality agent — audits every JSON the dashboard
            # reads (staleness, validity, internal consistency).
            # Must run BEFORE run_self_grade so the data_quality
            # score is available to fold into the self-grade.
            try:
                from common.data_quality import run_data_quality
                run_data_quality()
            except Exception as e:
                logger.warning(f"run_data_quality failed: {e}")
            # Auto-demote agent — autonomously freezes strategies that
            # FAIL validation, OVERFIT in walk-forward, bleed live, or
            # produce zero proposals 5+ cycles. Reads docs/validation,
            # walk_forward, trades_recent, cycle_status; writes docs/
            # auto_overrides.json. The allocator reads that override
            # next cycle and applies the freeze multiplier — no human.
            try:
                from common.auto_demote import run_auto_demote
                run_auto_demote()
            except Exception as e:
                logger.warning(f"run_auto_demote failed: {e}")
            # Daily 0–10 self-grade — accountability layer.
            try:
                from common.self_grade import run_self_grade
                run_self_grade()
            except Exception as e:
                logger.warning(f"run_self_grade failed: {e}")

    def _write_heartbeat(self, timestamp) -> None:
        """Tiny single-row table the dashboard polls to confirm the
        orchestrator is at least starting cycles. Idempotent."""
        if self._tracker is None:
            return
        with self._tracker._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS cycle_heartbeat (
                    id        INTEGER PRIMARY KEY CHECK (id = 1),
                    timestamp TEXT NOT NULL,
                    git_sha   TEXT
                )
            """)
            git_sha = os.environ.get("GITHUB_SHA", "")[:7] or "local"
            c.execute(
                "INSERT INTO cycle_heartbeat (id, timestamp, git_sha) "
                "VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "  timestamp = excluded.timestamp, "
                "  git_sha = excluded.git_sha",
                (timestamp.isoformat(), git_sha),
            )

    def _dump_cycle_status_json(self, report: CycleReport) -> None:
        """Belt-and-suspenders: write the cycle's diagnostics to a
        JSON file in docs/ that gets committed to the repo, bypassing
        the actions/cache flow that has historically been the source
        of "0 trades for 5 days" symptoms.

        Maintains a ring buffer of the last 50 cycles. Writes are
        atomic (temp + rename) so a partial write doesn't corrupt
        the file. Reads are best-effort — corrupted JSON is treated
        as "start over".
        """
        import json as _json
        from pathlib import Path as _Path
        out_dir = _Path("docs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "cycle_status.json"

        # Load existing buffer (if any), append this cycle, trim.
        buffer: list[dict] = []
        if out_path.exists():
            try:
                buffer = _json.loads(out_path.read_text(encoding="utf-8"))
                if not isinstance(buffer, list):
                    buffer = []
            except Exception:
                buffer = []

        entry = {
            "timestamp": report.timestamp.isoformat(),
            "cycle_seconds": report.cycle_seconds,
            "git_sha": os.environ.get("GITHUB_SHA", "")[:7] or "local",
            "proposals_total": report.proposals_total,
            "proposals_submitted": report.trades_submitted,
            "n_errors": len(report.errors),
            "first_error": (report.errors[0][:240] if report.errors else None),
            "venue_health": report.venue_health,
            "strategy_outcomes": {
                n: {
                    "venue": o.venue,
                    "state": o.state,
                    "target_alloc_usd": round(o.target_alloc_usd, 2),
                    "target_alloc_pct": round(o.target_alloc_pct, 4),
                    "proposed": o.proposed,
                    "approved": o.approved,
                    "rejected": o.rejected,
                    "submitted": o.submitted,
                    "dry_logged": o.dry_logged,
                    "skip_reasons": o.skip_reasons,
                    "reject_reasons": o.reject_reasons,
                    "execute_errors": o.execute_errors,
                    "error": o.error,
                } for n, o in (report.strategy_outcomes or {}).items()
            },
        }
        buffer.append(entry)
        # Ring buffer: keep last 50 cycles (~4 hours at 5-min cron)
        if len(buffer) > 50:
            buffer = buffer[-50:]

        # Atomic write
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(_json.dumps(buffer, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

    def _dump_recent_trades_json(self, limit: int = 50) -> None:
        """Snapshot the last N trades from trading_performance.db
        into docs/trades_recent.json. Same belt-and-suspenders
        rationale as _dump_cycle_status_json — bypasses the cache
        flow on the dashboard side."""
        if self._tracker is None:
            return
        import json as _json
        from pathlib import Path as _Path
        rows: list[dict] = []
        try:
            with self._tracker._conn() as c:
                c.row_factory = sqlite3.Row
                cursor = c.execute(
                    "SELECT timestamp, strategy, product_id, side, "
                    "       amount_usd, quantity, price, order_id, "
                    "       pnl_usd, dry_run, fill_status, venue "
                    "  FROM trades "
                    " ORDER BY id DESC "
                    f" LIMIT {int(limit)}"
                )
                for r in cursor.fetchall():
                    rows.append({
                        "timestamp": r["timestamp"],
                        "strategy": r["strategy"],
                        "symbol": r["product_id"],
                        "side": r["side"],
                        "amount_usd": float(r["amount_usd"] or 0),
                        "quantity": float(r["quantity"] or 0),
                        "price": float(r["price"] or 0),
                        "order_id": r["order_id"],
                        "pnl_usd": (float(r["pnl_usd"])
                                     if r["pnl_usd"] is not None else None),
                        "dry_run": bool(r["dry_run"]),
                        "fill_status": r["fill_status"] or "UNKNOWN",
                        "venue": r["venue"] or "",
                    })
        except Exception as e:
            logger.debug(f"trades read for json dump failed: {e}")
            return
        out_dir = _Path("docs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trades_recent.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(_json.dumps(rows, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

    def _run_cycle_body(
        self, report: CycleReport, cycle_start: float,
        scout_signals: dict | None = None,
    ) -> CycleReport:
        # Try to flush any dead-letter records from prior cycles
        # before doing anything else. Best-effort; silent failure
        # here is fine — the rows stay in the queue for next cycle.
        try:
            self._retry_dead_letters()
        except Exception as e:
            logger.debug(f"dead-letter retry sweep failed: {e}")
        # Cache pending orders per venue for the whole cycle so we don't
        # re-query the broker dozens of times.
        self._pending_cache: dict[str, dict] = {}
        # Cache of materialized PositionView dicts per venue. Populated
        # lazily by _positions_for and invalidated at the start of
        # every cycle.
        self._position_view_cache: dict[str, dict] = {}
        # Per-cycle running counter of qty SOLD across strategies on
        # the same (venue, symbol). Read+updated by _clamp_sell_quantity
        # so strategy N+1 sees the pool diminished by strategies 1..N.
        # Without this, multi-strategy SELL contention triggers HTTP
        # 403 "insufficient qty available" — observed 2026-05-08.
        self._sold_this_cycle: dict[tuple[str, str], float] = {}
        # Per-strategy position cache (ledger-attributed) — see
        # _positions_for_strategy. Reset every cycle so closed-out
        # positions don't linger.
        self._per_strategy_pos_cache: dict[tuple[str, str], dict] = {}
        # Per-strategy rejection / execution-error reasons, populated
        # by the various reject + execute sites and attributed to
        # outcome.reject_reasons / outcome.execute_errors after the
        # per-strategy loop. Lets the dashboard show "REJ x3 because
        # market_closed" instead of just "rej=3".
        self._cycle_reject_reasons: dict[str, list[str]] = {}
        self._cycle_execute_errors: dict[str, list[str]] = {}
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
        except Exception as e:
            # Promoted from silent pass → log so a future bug in
            # risk-snapshot init surfaces instead of vanishing
            # (audit P1, 2026-05-08).
            logger.warning(f"risk snapshot reset failed: {e}")

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
            try:
                from common.errors_db import record_error
                record_error(scope="orchestrator.compute_state")
            except Exception:
                pass
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
            # Audit-fix F9 (2026-05-07): when live-trading is active,
            # alert on the FIRST failed cycle instead of waiting for 2.
            # 5 minutes of unmonitored crypto exposure with stuck
            # open orders is too long.
            is_live = bool(self.cfg.live_strategies) or any([
                self.cfg.dry_run_coinbase is False,
                self.cfg.dry_run_alpaca is False and "paper" not in
                    (os.environ.get("ALPACA_ENDPOINT") or "").lower(),
                self.cfg.dry_run_kalshi is False,
            ])
            alert_threshold = 1 if is_live else 2
            if self._venues_ok_consecutive_failures == alert_threshold:
                try:
                    from common.alerts import alert
                    severity = "critical" if is_live else "warning"
                    alert(
                        f"Broker unreachable "
                        f"({self._venues_ok_consecutive_failures} cycle"
                        f"{'s' if self._venues_ok_consecutive_failures != 1 else ''}) — "
                        f"{'LIVE TRADING' if is_live else 'paper'} degraded. "
                        f"equity_usd=${state.equity_usd:,.2f}",
                        severity=severity,
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
            outcome = StrategyOutcome(
                strategy=name, venue=strategy.venue,
                state=self.registry.get_state(name).value,
            )
            report.strategy_outcomes[name] = outcome
            strategy_state = self.registry.get_state(name)
            if strategy_state in (StrategyState.FROZEN, StrategyState.RETIRED):
                outcome.skip_reasons.append(f"state={strategy_state.value}")
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
            # Apply auto-demote multiplier (0.0 freezes the strategy
            # this cycle without any source-code change). Source of
            # truth: docs/auto_overrides.json written by run_auto_demote.
            try:
                from common.auto_demote import get_auto_multiplier
                mult = get_auto_multiplier(name)
                if mult < 1.0:
                    target_pct *= mult
                    target_usd *= mult
                    if mult == 0.0:
                        outcome.skip_reasons.append(
                            "auto-demoted (see docs/auto_overrides.json)")
            except Exception as e:
                logger.debug(f"auto_demote multiplier read failed: {e}")
            outcome.target_alloc_pct = target_pct
            outcome.target_alloc_usd = target_usd
            if target_usd <= 0:
                outcome.skip_reasons.append("target_alloc_usd=0")

            # Skip compute when the venue's market is closed — paper AND
            # live. The earlier "let paper queue 24/7" carve-out
            # backfired: place_order refuses off-RTH (day orders
            # auto-cancel at the open), so every off-hours paper proposal
            # raised "market closed", was logged as an EXECUTION ERROR,
            # and dragged setup_health to 0/0 clean cycles (observed
            # 2026-05-22: earnings_momentum erroring every night). Gating
            # before compute means no off-hours churn, no spurious errors,
            # and the strategy still trades in-session (where it fills).
            # 24/7 venues (coinbase/kalshi) are always "open" here.
            if not is_market_open(strategy.venue):
                outcome.skip_reasons.append(
                    f"venue_closed ({venue_window_str(strategy.venue)})"
                )
                continue

            ctx = StrategyContext(
                timestamp=report.timestamp,
                portfolio_equity_usd=state.equity_usd,
                target_alloc_pct=target_pct,
                target_alloc_usd=target_usd,
                risk_multiplier=state.multiplier.effective,
                # Per-strategy position attribution: a strategy only sees
                # positions IT opened (via the trades ledger). Without this,
                # 6 different strategies all see the same GLD position and
                # all propose to SELL it simultaneously → 5 of 6 hit Alpaca
                # HTTP 403 "insufficient qty available" (observed 2026-05-08
                # — 70+ errors per cycle). Falls back to the full venue
                # snapshot when the ledger reports nothing for the strategy
                # (e.g. cold start, position opened before this fix).
                open_positions=self._positions_for_strategy(
                    name, strategy.venue,
                ),
                scout_signals=scout_signals.get(strategy.venue, {}),
                pending_orders=self._pending_orders_for(strategy.venue),
            )

            try:
                proposals = strategy.compute(ctx)
            except Exception as e:
                report.errors.append(f"[{name}] compute failed: {e}")
                logger.exception(f"[{name}] strategy.compute raised")
                # Persist full traceback to errors.db so the dashboard
                # can show debug context without an SSH session.
                try:
                    from common.errors_db import record_error
                    record_error(
                        scope="strategy.compute",
                        strategy=name, venue=strategy.venue,
                    )
                except Exception:
                    pass
                # Sprint E1 wiring — record this strategy as failed for
                # the consecutive-error tracker. Best-effort: tracking
                # itself raising must not break the cycle.
                err_msg = f"compute: {type(e).__name__}: {str(e)[:160]}"
                self._record_strategy_outcome(name, had_error=True,
                                                error_text=err_msg)
                outcome.error = err_msg
                continue

            # Strategy compute succeeded → record clean cycle for the
            # consecutive-error tracker. Resets the count if it was
            # accumulating.
            self._record_strategy_outcome(name, had_error=False)

            outcome.proposed = len(proposals)
            if not proposals:
                outcome.skip_reasons.append("no proposals")
            report.proposals_total += len(proposals)
            for p in proposals:
                # Snapshot counters so we can attribute the delta to
                # this specific proposal (and thus this strategy).
                pre_approved = report.proposals_approved
                pre_rejected = report.proposals_rejected
                pre_submitted = report.trades_submitted
                self._handle_proposal(p, state, report, strategy)
                if report.proposals_approved > pre_approved:
                    outcome.approved += 1
                if report.proposals_rejected > pre_rejected:
                    outcome.rejected += 1
                if report.trades_submitted > pre_submitted:
                    outcome.submitted += 1
                else:
                    # Approved but not submitted = either DRY-logged
                    # OR execution raised. We can disambiguate by
                    # checking if a new execute_error was recorded
                    # for this strategy mid-call.
                    if report.proposals_approved > pre_approved:
                        n_errs = len(self._cycle_execute_errors.get(name, []))
                        # Did the count grow during _handle_proposal? If
                        # yes, this was an execution failure not a DRY.
                        # We approximate by always classifying as
                        # dry_logged here; the dashboard can use
                        # execute_errors to disambiguate.
                        outcome.dry_logged += 1
                        del n_errs  # tracking only used by dashboard
            # After the loop, copy per-strategy reject + error reasons
            # into the outcome object so they persist into JSON.
            outcome.reject_reasons = list(
                self._cycle_reject_reasons.get(name, [])
            )[-10:]
            outcome.execute_errors = list(
                self._cycle_execute_errors.get(name, [])
            )[-10:]

        # Telemetry persistence is in the run_cycle() finally block
        # so it ALWAYS runs (including early-returns above).
        return report

    def _venue_health_snapshot(self) -> dict[str, str]:
        """Per-venue reachability status for the dashboard. 'ok' if the
        per-cycle account fetch succeeded, 'unreachable' otherwise.
        Reads from the risk manager's snapshot cache (already populated
        by compute_state)."""
        out: dict[str, str] = {}
        snap = getattr(self.risk, "_broker_snapshots", {}) or {}
        for venue in self.brokers:
            v = snap.get(venue) or {}
            if "account" in v:
                out[venue] = "ok"
            else:
                out[venue] = "unreachable"
        return out

    def _persist_cycle_diagnostics(self, report: CycleReport) -> None:
        """Write the cycle's per-strategy outcomes to a SQLite table
        the dashboard reads. Idempotent — schema-on-first-write."""
        if self._tracker is None:
            return
        import json as _json
        with self._tracker._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS cycle_diagnostics (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    cycle_seconds   REAL    NOT NULL,
                    proposals_total INTEGER NOT NULL,
                    proposals_submitted INTEGER NOT NULL,
                    n_errors        INTEGER NOT NULL,
                    venue_health    TEXT    NOT NULL,
                    strategy_outcomes TEXT  NOT NULL
                )
            """)
            c.execute(
                "INSERT INTO cycle_diagnostics "
                "(timestamp, cycle_seconds, proposals_total, "
                " proposals_submitted, n_errors, venue_health, "
                " strategy_outcomes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    report.timestamp.isoformat(),
                    report.cycle_seconds,
                    report.proposals_total,
                    report.trades_submitted,
                    len(report.errors),
                    _json.dumps(report.venue_health),
                    _json.dumps({
                        n: {
                            "venue": o.venue,
                            "state": o.state,
                            "target_alloc_pct": o.target_alloc_pct,
                            "target_alloc_usd": o.target_alloc_usd,
                            "proposed": o.proposed,
                            "approved": o.approved,
                            "rejected": o.rejected,
                            "submitted": o.submitted,
                            "dry_logged": o.dry_logged,
                            "skip_reasons": o.skip_reasons,
                            "reject_reasons": o.reject_reasons,
                            "execute_errors": o.execute_errors,
                            "error": o.error,
                        } for n, o in report.strategy_outcomes.items()
                    }),
                ),
            )
            # Keep only the most recent 200 cycles to bound DB size.
            c.execute(
                "DELETE FROM cycle_diagnostics WHERE id NOT IN "
                "(SELECT id FROM cycle_diagnostics "
                " ORDER BY id DESC LIMIT 200)"
            )

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

    def _positions_for_strategy(
        self, strategy_name: str, venue: str,
    ) -> dict[str, PositionView]:
        """Per-strategy position view, derived from the trades ledger.

        Without this, every strategy sees the full account positions
        and 6 strategies all decide to SELL GLD because GLD appears
        in everyone's `open_positions` — collectively trying to sell
        more GLD than exists. Per-strategy attribution: a strategy
        only sees the symbols where its OWN net buys exceed its
        OWN net sells.

        Builds on top of the venue snapshot (so market_price /
        unrealized_pnl_usd come from the broker), but masks down to
        the symbols + qty owned by THIS strategy. Symbols not in the
        broker snapshot are skipped — the broker is the source of
        truth for whether the position still exists at all.

        Cached per-cycle so repeated calls within a strategy's compute
        don't re-hit the SQLite ledger.
        """
        cache = getattr(self, "_per_strategy_pos_cache", None)
        if cache is None:
            cache = {}
            self._per_strategy_pos_cache = cache
        key = (strategy_name, venue)
        if key in cache:
            return cache[key]

        from strategies._helpers import net_qty_from_ledger
        ledger_failed = False
        try:
            ledger_qty = net_qty_from_ledger(strategy_name, venue)
        except Exception as e:
            # Promoted from debug→warning so a SQLite lock / schema
            # mismatch surfaces in the orchestrator log instead of
            # silently blocking every SELL the strategy would have
            # made (audit P1, 2026-05-08).
            logger.warning(f"[{strategy_name}] ledger fetch failed: {e}")
            ledger_qty = {}
            ledger_failed = True

        venue_positions = self._positions_for(venue)

        # Ledger failure: fall back to the FULL venue snapshot so
        # closing trades still work. The previous behaviour ({} →
        # strategy "owns nothing") silently blocked every SELL
        # whenever the ledger DB was momentarily unreachable.
        if ledger_failed:
            cache[key] = venue_positions
            return venue_positions

        # Strategy legitimately holds nothing per the ledger — empty
        # view is the right answer.
        if not ledger_qty:
            cache[key] = {}
            return {}

        out: dict[str, PositionView] = {}
        for symbol, net_qty in ledger_qty.items():
            if net_qty <= 0:
                # Short legs (handled by on_emergency_close ledger
                # fallback for KILL — not surfaced as longable spot
                # positions).
                continue
            full = venue_positions.get(symbol)
            if full is None:
                # Strategy's ledger shows ownership but the broker
                # doesn't see it — could be a partial fill in flight,
                # a manual-close outside the orchestrator, or a stale
                # ledger row. Don't pretend we still own it.
                continue
            # Cap the per-strategy qty at the broker's actual qty so
            # we never propose a SELL larger than what exists, even
            # if the ledger has rounding drift.
            attributed_qty = min(net_qty, float(full.quantity or 0))
            if attributed_qty <= 0:
                continue
            out[symbol] = PositionView(
                venue=full.venue,
                symbol=full.symbol,
                quantity=attributed_qty,
                avg_entry_price=full.avg_entry_price,
                market_price=full.market_price,
                unrealized_pnl_usd=full.unrealized_pnl_usd
                    * (attributed_qty / max(float(full.quantity or 1), 1e-9)),
                entry_time=full.entry_time,
                asset_class=full.asset_class,
            )
        cache[key] = out
        return out

    def _positions_for(self, venue: str) -> dict[str, PositionView]:
        # Reuse the risk manager's per-cycle position cache; avoids hitting
        # the broker API again for each strategy on the same venue.
        # Returns dict[symbol, PositionView] — the dataclass shim keeps
        # `pos.get("quantity")` and `pos["quantity"]` working in legacy
        # strategy code while typed `pos.quantity` access is also valid.
        # Also caches per-cycle inside the orchestrator so a broker
        # outage doesn't produce N identical "get_positions failed"
        # warnings (one per strategy on that venue).
        view_cache = getattr(self, "_position_view_cache", None)
        if view_cache is None:
            view_cache = {}
            self._position_view_cache = view_cache
        if venue in view_cache:
            return view_cache[venue]

        cached = []
        try:
            cached = self.risk.cached_positions(venue)
        except Exception:
            cached = []

        if not cached:
            adapter = self.brokers.get(venue)
            if adapter is None:
                view_cache[venue] = {}
                return {}
            try:
                cached = adapter.get_positions()
            except Exception as e:
                logger.warning(f"[{venue}] get_positions failed: {e}")
                view_cache[venue] = {}
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
        view_cache[venue] = out
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
        # US regular session.
        # EXCEPTION: paper-mode adapters queue orders 24/7 and fill
        # them at next open. Letting paper through keeps strategies
        # actively trading on the weekend → user gets visible
        # round-trips without waiting for Monday.
        _adapter = self.brokers.get(proposal.venue)
        _is_paper = bool(_adapter and getattr(_adapter, "is_paper", False))
        if not is_market_open(proposal.venue) and not _is_paper:
            logger.debug(
                f"[{proposal.strategy}] SKIP {proposal.venue} closed "
                f"({venue_window_str(proposal.venue)})"
            )
            report.proposals_rejected += 1
            self._cycle_reject_reasons.setdefault(
                proposal.strategy, []).append(
                f"market_closed ({venue_window_str(proposal.venue)})")
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
            self._cycle_reject_reasons.setdefault(
                proposal.strategy, []).append(
                f"venue_degraded ({proposal.venue})")
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
            self._cycle_reject_reasons.setdefault(
                proposal.strategy, []).append(
                f"risk: {decision.reason}"[:200])
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
        should skip in that case.

        Multi-strategy contention: when N strategies all want to SELL
        the same symbol in one cycle, the cached qty_available is
        shared. Without intra-cycle bookkeeping, strategy 2..N each
        try to sell ~90% of the *original* qty_available — but
        strategy 1 already consumed most of it, so 2..N hit Alpaca
        HTTP 403 "insufficient qty available" (observed 2026-05-08).
        Fix: deduct each strategy's clamped qty from a per-cycle
        running counter so subsequent strategies see the diminished
        pool. Counter is reset at cycle start in run_cycle().
        """
        if proposal.side != OrderSide.SELL:
            return True

        BUFFER = 0.90  # keep 10% safety margin under qty_available
        # Tiny-qty cutoff: Alpaca formats `f"{q:.6f}"` rounds 5e-7 → "0.000000"
        # which the API rejects as "qty must be > 0". And anything below ~1e-4
        # share is below typical fractional minimums anyway.
        MIN_SELL_QTY = 1e-4

        consumed = getattr(self, "_sold_this_cycle", None)
        if consumed is None:
            consumed = {}
            self._sold_this_cycle = consumed
        key = (proposal.venue, proposal.symbol)
        already_consumed = consumed.get(key, 0.0)

        cached = self.risk.cached_positions(proposal.venue)
        matched = False
        for pos in cached:
            if pos.symbol != proposal.symbol:
                continue
            matched = True
            avail_raw = float(pos.raw.get("qty_available_parsed", pos.quantity)
                            if pos.raw else pos.quantity)
            avail = max(avail_raw - already_consumed, 0.0)
            if avail <= MIN_SELL_QTY:
                logger.info(f"[{proposal.strategy}] SKIP SELL "
                            f"{proposal.symbol}: 0 qty available "
                            f"(raw={avail_raw:.4f}, "
                            f"consumed_this_cycle={already_consumed:.4f})")
                report.proposals_rejected += 1
                return False

            requested_qty = proposal.quantity
            if not requested_qty and proposal.notional_usd and pos.market_price > 0:
                requested_qty = proposal.notional_usd / pos.market_price
            if not requested_qty:
                break

            max_qty = avail * BUFFER
            final_qty = min(requested_qty, max_qty)
            if final_qty < MIN_SELL_QTY:
                logger.info(f"[{proposal.strategy}] SKIP SELL "
                            f"{proposal.symbol}: clamped qty "
                            f"{final_qty:.6f} < min {MIN_SELL_QTY}")
                report.proposals_rejected += 1
                return False

            if requested_qty > max_qty:
                logger.info(f"[{proposal.strategy}] CLAMP SELL "
                            f"{proposal.symbol} qty {requested_qty:.4f} "
                            f"→ {max_qty:.4f} "
                            f"(qty_available={avail:.4f}, buffer=10%)")
            proposal.quantity = final_qty
            consumed[key] = already_consumed + final_qty
            # Force qty path on the broker so its server-side
            # notional→qty conversion can't drift back over.
            proposal.notional_usd = None
            break
        if not matched:
            # Closing orders with an explicit quantity bypass the
            # "must be held in cache" check — KILL emergency-close +
            # ledger-derived shorts (perp/futures) legitimately reach
            # this path with positions the cached_positions() snapshot
            # doesn't surface. Trust the caller's quantity in that case.
            if proposal.is_closing and proposal.quantity:
                return True
            # The broker has no position for this symbol — we don't
            # own it, so we can't sell it. Without this guard the
            # SELL went through with notional_usd, the adapter
            # computed a qty, and Alpaca rejected with HTTP 422
            # "qty must be > 0" or HTTP 403. Observed 2026-05-08.
            logger.info(f"[{proposal.strategy}] SKIP SELL "
                        f"{proposal.symbol}: not held on {proposal.venue}")
            report.proposals_rejected += 1
            return False
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
            self._record_trade(proposal, order, decision, report)
        except Exception as e:
            # Market-closed is an EXPECTED defer, not a failure. The
            # pre-compute venue-closed gate normally prevents reaching
            # here, but a cycle straddling the 09:30/16:00 ET boundary
            # (clock skew between our check and Alpaca's) can still trip
            # it. Recording it in report.errors would tank setup_health
            # for a non-event, so surface it as a soft defer instead.
            if "market closed" in str(e).lower():
                logger.info(f"[{proposal.strategy}] deferred {proposal.symbol}"
                            f": market closed (will retry in-session)")
                self._cycle_reject_reasons.setdefault(
                    proposal.strategy, []).append("market closed — deferred")
                return
            report.errors.append(f"[{proposal.strategy}] execution failed: {e}")
            logger.exception(f"[{proposal.strategy}] place_order raised")
            self._cycle_execute_errors.setdefault(
                proposal.strategy, []).append(
                f"{type(e).__name__}: {str(e)[:200]}")

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

    def _record_trade(self, proposal: TradeProposal, order, decision,
                       report: CycleReport | None = None) -> None:
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
            # Retry with exponential backoff: SQLite locks under
            # concurrent dashboard reads sometimes need a moment.
            # Three attempts with 0.2s, 0.4s, 0.8s sleeps.
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    self._tracker.record_trade(record)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(0.2 * (2 ** attempt))
            if last_err is not None:
                raise last_err
        except Exception as e:
            # Surfaced via report.errors so the dashboard's error
            # panel shows it — silently warn-logging meant the user
            # saw "0 trades" without ever knowing the recording side
            # was broken (audit P1, 2026-05-08).
            msg = (f"[{proposal.strategy}] record_trade failed after "
                   f"3 retries: {e} (order placed, ledger out-of-sync)")
            logger.warning(msg)
            if report is not None:
                report.errors.append(msg)
            # Dead-letter queue: persist the unrecorded order so a
            # later cycle can retry. Without this, a transient SQLite
            # lock at the worst moment loses the trade record forever
            # while the order is real on the broker. Phantom-position
            # class bug — same family as the original "770 stuck
            # PENDING orders" failure mode.
            try:
                self._record_dead_letter(record, str(e))
            except Exception as dl_err:
                logger.warning(
                    f"[{proposal.strategy}] dead-letter persist also "
                    f"failed: {dl_err} — order is now invisible to the "
                    f"orchestrator. Manual reconcile required."
                )

    def _retry_dead_letters(self) -> None:
        """Retry any record_trade rows that previously failed.

        Walks the dead-letter table and re-attempts the insert. On
        success, removes the row. On failure, increments retry_count
        (and if >5, leaves a warning log; the row stays in the queue
        but the operator should manual-reconcile).
        """
        if self._tracker is None:
            return
        try:
            with self._tracker._conn() as c:
                rows = c.execute(
                    "SELECT id, timestamp, strategy, product_id, side, "
                    "       venue, quantity, price, amount_usd, "
                    "       order_id, fill_status, dry_run, retry_count "
                    "  FROM record_trade_dead_letter "
                    " WHERE retry_count < 5 "
                    " ORDER BY id ASC LIMIT 50"
                ).fetchall()
        except Exception:
            # Table doesn't exist yet on first deploy — nothing to do.
            return
        from trading.performance import TradeRecord
        from datetime import UTC, datetime
        for r in rows or []:
            (row_id, ts, strat, pid, side, venue, qty, px, amount,
             order_id, fill_status, dry, retry_count) = r
            try:
                # ts is an ISO string from the dead-letter row; the
                # TradeRecord dataclass expects a datetime.
                try:
                    ts_dt = datetime.fromisoformat(
                        ts.replace("Z", "+00:00") if isinstance(ts, str) else ts
                    )
                except Exception:
                    ts_dt = datetime.now(UTC)
                rec = TradeRecord(
                    timestamp=ts_dt,
                    strategy=strat,
                    product_id=pid,
                    side=side,
                    quantity=qty,
                    price=px,
                    amount_usd=amount,
                    order_id=order_id,
                    pnl_usd=None,
                    dry_run=bool(dry),
                    fill_status=fill_status,
                    venue=venue,
                )
                self._tracker.record_trade(rec)
                # Successful retry → drop the dead-letter row
                with self._tracker._conn() as c:
                    c.execute(
                        "DELETE FROM record_trade_dead_letter WHERE id = ?",
                        (row_id,),
                    )
                logger.info(
                    f"[deadletter] recovered {strat} {side} {pid} "
                    f"(retry #{retry_count + 1})"
                )
            except Exception as e:
                # Bump retry_count, leave the row for next cycle
                try:
                    with self._tracker._conn() as c:
                        c.execute(
                            "UPDATE record_trade_dead_letter SET "
                            "  retry_count = retry_count + 1, "
                            "  last_retry_at = ? "
                            "WHERE id = ?",
                            (datetime.now(UTC).isoformat(), row_id),
                        )
                except Exception:
                    pass
                if retry_count + 1 >= 5:
                    logger.warning(
                        f"[deadletter] {strat} {side} {pid} has "
                        f"failed 5 retries — manual reconcile needed: {e}"
                    )

    def _record_dead_letter(self, record, original_error: str) -> None:
        """Persist a failed-to-record trade to a dead-letter table.
        On the next cycle, _retry_dead_letters() picks these up and
        attempts the insert again. Keeps the broker side and ledger
        side eventually consistent."""
        if self._tracker is None:
            return
        with self._tracker._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS record_trade_dead_letter (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT NOT NULL,
                    strategy     TEXT NOT NULL,
                    product_id   TEXT NOT NULL,
                    side         TEXT NOT NULL,
                    venue        TEXT,
                    quantity     REAL,
                    price        REAL,
                    amount_usd   REAL,
                    order_id     TEXT,
                    fill_status  TEXT,
                    dry_run      INTEGER NOT NULL DEFAULT 0,
                    original_error TEXT NOT NULL,
                    retry_count  INTEGER NOT NULL DEFAULT 0,
                    last_retry_at TEXT
                )
            """)
            from datetime import UTC, datetime
            c.execute(
                "INSERT INTO record_trade_dead_letter "
                "(timestamp, strategy, product_id, side, venue, "
                " quantity, price, amount_usd, order_id, fill_status, "
                " dry_run, original_error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    record.strategy,
                    record.product_id,
                    record.side,
                    getattr(record, "venue", None),
                    float(record.quantity or 0),
                    float(record.price or 0),
                    float(record.amount_usd or 0),
                    record.order_id or "",
                    getattr(record, "fill_status", None) or "UNKNOWN",
                    1 if getattr(record, "dry_run", False) else 0,
                    original_error[:500],
                ),
            )

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
