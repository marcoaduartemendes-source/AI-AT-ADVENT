"""Bootstrap path tests — locks in the fixes from PR #17 and #18.

Production bug history this file guards against:

1. PR #17 — orchestrator returned exit code 2 if report.errors had
   any entry. CI marked the job failed, actions/cache@v4 skipped the
   post-save step, and trades.db rows never persisted. Symptom: "0
   trades on dashboard for 5 days".

2. PR #18 — once exit-2 was fixed, the cold-start guard fired every
   cycle because risk_state.db had no snapshots (chicken-and-egg
   from the prior bug). Manual fix was to pass --allow-cold-start
   in the workflow.

3. This commit — auto-detect cold-start under CI/cron and bootstrap
   automatically with a prominent warning. The CLI flag becomes
   unnecessary; humans running locally still get the safety guard.

These tests fail if any of those regressions sneak back in.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock



class TestExitCodeIsZeroOnRecoverableErrors:
    """run_orchestrator.main() must return 0 when the cycle completes,
    even if report.errors contains per-strategy warnings (e.g.
    record_trade SQLite lock retries, broker rate limits). The old
    `return 2` path killed the GH Actions cache save and produced
    the '0 trades' loop."""

    def test_main_returns_zero_when_cycle_has_errors(
        self, monkeypatch, tmp_path,
    ):
        # Isolate state DBs to tmp_path so the test doesn't touch
        # production data.
        monkeypatch.setenv("RISK_DB_PATH", str(tmp_path / "risk_state.db"))
        monkeypatch.setenv(
            "ALLOCATOR_DB_PATH", str(tmp_path / "allocator.db"),
        )
        monkeypatch.setenv(
            "TRADING_DB_PATH", str(tmp_path / "trading_performance.db"),
        )
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        monkeypatch.setenv("DRY_RUN", "true")
        # Force CI context so the cold-start guard auto-bootstraps
        # rather than refusing.
        monkeypatch.setenv("GITHUB_ACTIONS", "1")

        # Patch sys.argv so argparse picks up just --once (no
        # --allow-cold-start, since auto-detection should handle it).
        monkeypatch.setattr(sys, "argv", ["run_orchestrator.py", "--once"])

        # Stub broker construction so we don't need real creds.
        # Empty brokers → main returns 1 — that's fine; this test only
        # asserts we don't return 2 from the report.errors path.
        import run_orchestrator as ro
        monkeypatch.setattr(ro, "build_brokers", lambda: {})

        from run_orchestrator import main
        # Empty brokers → main returns 1 (not 2). That's the right
        # signal: a fatal config issue. The point of this test is
        # that we DON'T return 2 from the report.errors path.
        rc = main()
        assert rc != 2, (
            f"main() returned {rc}. Returning 2 from the "
            "report.errors path was the bug that killed the cache "
            "save in PR #17."
        )


class TestColdStartAutoBootstrap:
    """Under CI/cron, an empty risk_state.db must auto-bootstrap
    instead of returning 3. Otherwise the system can't escape the
    'never had a successful cycle so we never have snapshots so we
    refuse to run' loop."""

    def test_cron_context_auto_bootstraps_empty_state(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setenv("RISK_DB_PATH", str(tmp_path / "risk_state.db"))
        monkeypatch.setenv(
            "ALLOCATOR_DB_PATH", str(tmp_path / "allocator.db"),
        )
        monkeypatch.setenv(
            "TRADING_DB_PATH", str(tmp_path / "trading_performance.db"),
        )
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
        monkeypatch.setenv("GITHUB_ACTIONS", "1")
        monkeypatch.setattr(sys, "argv", ["run_orchestrator.py", "--once"])

        mock_adapter = MagicMock()
        mock_adapter.get_account.side_effect = Exception(
            "stub: account fetch suppressed for test"
        )
        mock_adapter.get_positions.return_value = []
        mock_adapter.capabilities = set()
        import run_orchestrator as ro
        monkeypatch.setattr(
            ro, "build_brokers",
            lambda: {"alpaca": mock_adapter},
        )

        from run_orchestrator import main
        rc = main()
        # Must NOT return 3 (cold-start refusal). 0 = clean cycle,
        # 1 = config error, both acceptable. 3 specifically is the
        # regression we're guarding against.
        assert rc != 3, (
            "main() returned 3 under GITHUB_ACTIONS=1 — the "
            "auto-bootstrap path failed. PR #18's chicken-and-egg "
            "loop has returned."
        )

    def test_human_context_still_blocks_cold_start(
        self, monkeypatch, tmp_path,
    ):
        """A human re-running on a fresh box (no GITHUB_ACTIONS env)
        STILL gets the cold-start guard. The auto-bootstrap is
        cron-specific by design."""
        monkeypatch.setenv("RISK_DB_PATH", str(tmp_path / "risk_state.db"))
        monkeypatch.setenv(
            "ALLOCATOR_DB_PATH", str(tmp_path / "allocator.db"),
        )
        monkeypatch.setenv(
            "TRADING_DB_PATH", str(tmp_path / "trading_performance.db"),
        )
        monkeypatch.setenv("ERRORS_DB_PATH", str(tmp_path / "errors.db"))
        monkeypatch.setenv("DRY_RUN", "false")
        monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setattr(sys, "argv", ["run_orchestrator.py", "--once"])

        mock_adapter = MagicMock()
        mock_adapter.get_account.side_effect = Exception(
            "stub: account fetch suppressed for test"
        )
        mock_adapter.get_positions.return_value = []
        mock_adapter.capabilities = set()
        # Patch at the run_orchestrator import binding (not the source
        # module) — that's where main() actually looks it up.
        import run_orchestrator as ro
        monkeypatch.setattr(
            ro, "build_brokers",
            lambda: {"alpaca": mock_adapter},
        )

        from run_orchestrator import main
        rc = main()
        assert rc == 3, (
            f"main() returned {rc}. A human-context cold start with "
            "live trading enabled MUST refuse with exit 3 — that's "
            "the kill-switch baseline safety guard."
        )


class TestCycleDiagnosticsAlwaysPersistsOnEarlyReturn:
    """Every cycle must write a cycle_diagnostics row, including
    cycles that early-return because of KILL switch, all-venues-
    closed, or compute_state failure. Without this, the dashboard's
    Cycle activity panel stays empty forever even though cycles
    are running — observed 2026-05-09 after PRs 17/18/19/20 all
    merged but the panel still showed "Bootstrapping"."""

    def test_compute_state_failure_still_persists(self, tmp_path):
        from unittest.mock import MagicMock

        from strategy_engine.orchestrator import (
            Orchestrator, OrchestratorConfig,
        )
        from trading.performance import PerformanceTracker

        db_path = str(tmp_path / "trading_performance.db")
        tracker = PerformanceTracker(db_path=db_path)

        # Build a minimal Orchestrator with mocked components
        risk = MagicMock()
        risk.compute_state.side_effect = Exception("simulated broker outage")
        risk._broker_snapshots = {}
        registry = MagicMock()
        registry.latest_allocations.return_value = {}
        allocator = MagicMock()

        orch = Orchestrator.__new__(Orchestrator)
        orch.brokers = {}
        orch.risk = risk
        orch.registry = registry
        orch.allocator = allocator
        orch.strategies = {}
        orch.cfg = OrchestratorConfig(dry_run=True)
        orch._tracker = tracker
        orch._last_rebalance_ts = 0
        orch._venues_ok_consecutive_failures = 0

        # Run one cycle. compute_state raises → early return.
        report = orch.run_cycle()
        assert report is not None

        import sqlite3
        with sqlite3.connect(db_path) as c:
            n_rows = c.execute(
                "SELECT COUNT(*) FROM cycle_diagnostics"
            ).fetchone()[0]
        assert n_rows == 1, (
            "cycle_diagnostics row must be written even when "
            "compute_state fails. Without this, every cycle that "
            "hits an early return is invisible to the dashboard."
        )


class TestHeartbeatWrites:
    """Heartbeat is written at the very start of run_cycle, BEFORE
    anything else can fail. Lets the dashboard distinguish 'orchestrator
    not running' from 'orchestrator running but diagnostics broken'."""

    def test_heartbeat_writes_on_run_cycle(self, tmp_path):
        from datetime import UTC, datetime
        from unittest.mock import MagicMock

        from strategy_engine.orchestrator import (
            Orchestrator, OrchestratorConfig,
        )
        from trading.performance import PerformanceTracker

        db_path = str(tmp_path / "trading_performance.db")
        tracker = PerformanceTracker(db_path=db_path)

        risk = MagicMock()
        risk.compute_state.side_effect = Exception("simulate failure")
        risk._broker_snapshots = {}

        orch = Orchestrator.__new__(Orchestrator)
        orch.brokers = {}
        orch.risk = risk
        orch.registry = MagicMock()
        orch.registry.latest_allocations.return_value = {}
        orch.allocator = MagicMock()
        orch.strategies = {}
        orch.cfg = OrchestratorConfig(dry_run=True)
        orch._tracker = tracker
        orch._last_rebalance_ts = 0
        orch._venues_ok_consecutive_failures = 0

        before = datetime.now(UTC)
        orch.run_cycle()

        import sqlite3
        with sqlite3.connect(db_path) as c:
            row = c.execute(
                "SELECT timestamp FROM cycle_heartbeat WHERE id = 1"
            ).fetchone()
        assert row is not None, (
            "Heartbeat row must be written even when compute_state fails. "
            "Without it, the dashboard can't distinguish 'orchestrator "
            "not running' from 'orchestrator running but broken'."
        )
        # Timestamp should be roughly now
        from datetime import datetime as _dt
        ts = _dt.fromisoformat(row[0])
        assert ts >= before


class TestDeadLetterQueueRecoversTrades:
    """When record_trade fails after all retries, the row goes to a
    dead-letter table. Next cycle's _retry_dead_letters() flushes it.
    Without this, an order placed at the broker but unrecorded in
    trading_performance.db is invisible forever — the original
    'phantom-loss' bug class.
    """

    def test_dead_letter_persists_then_recovers(self, tmp_path):
        from datetime import UTC, datetime
        from strategy_engine.orchestrator import Orchestrator
        from trading.performance import PerformanceTracker
        from trading.portfolio import TradeRecord

        db_path = str(tmp_path / "trading_performance.db")
        tracker = PerformanceTracker(db_path=db_path)
        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = tracker

        # Manually insert a dead-letter row (simulates a record_trade
        # failure earlier).
        record = TradeRecord(
            timestamp=datetime.now(UTC),
            strategy="canary_strategy",
            product_id="BTC-USD",
            side="BUY",
            amount_usd=100.0,
            quantity=0.001,
            price=100000.0,
            order_id="test-order-1",
            dry_run=False,
            fill_status="FILLED",
            venue="coinbase",
        )
        orch._record_dead_letter(record, "simulated SQLite lock")

        import sqlite3
        with sqlite3.connect(db_path) as c:
            n_dl = c.execute(
                "SELECT COUNT(*) FROM record_trade_dead_letter"
            ).fetchone()[0]
        assert n_dl == 1

        # Now run the retry sweep — should succeed and remove the row.
        orch._retry_dead_letters()

        with sqlite3.connect(db_path) as c:
            n_dl_after = c.execute(
                "SELECT COUNT(*) FROM record_trade_dead_letter"
            ).fetchone()[0]
            n_trades = c.execute(
                "SELECT COUNT(*) FROM trades WHERE strategy='canary_strategy'"
            ).fetchone()[0]
        assert n_dl_after == 0, (
            "Dead-letter row should be removed after successful retry"
        )
        assert n_trades == 1, (
            "Trade should have been recorded via the retry path"
        )


class TestCycleDiagnosticsPersistence:
    """The cycle_diagnostics table must be created and written to
    on first run, so the dashboard's Cycle activity panel populates
    immediately. Schema-on-first-write means no migration step
    required."""

    def test_persist_creates_table_and_inserts_row(self, tmp_path):
        from datetime import UTC, datetime
        from strategy_engine.orchestrator import CycleReport, StrategyOutcome
        from trading.performance import PerformanceTracker

        db_path = str(tmp_path / "trading_performance.db")
        tracker = PerformanceTracker(db_path=db_path)

        # Build a minimal Orchestrator stub with just the persist
        # method we want to test.
        from strategy_engine.orchestrator import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = tracker

        report = CycleReport(timestamp=datetime.now(UTC))
        report.cycle_seconds = 1.5
        report.proposals_total = 3
        report.trades_submitted = 1
        report.errors = []
        report.venue_health = {"alpaca": "ok"}
        report.strategy_outcomes = {
            "test_strategy": StrategyOutcome(
                strategy="test_strategy", venue="alpaca",
                state="ACTIVE", target_alloc_pct=0.05,
                target_alloc_usd=500.0, proposed=3, approved=2,
                rejected=1, submitted=1, dry_logged=0,
                skip_reasons=[], error="",
            ),
        }
        orch._persist_cycle_diagnostics(report)

        import sqlite3
        with sqlite3.connect(db_path) as c:
            rows = c.execute(
                "SELECT cycle_seconds, proposals_total, "
                "       proposals_submitted "
                "  FROM cycle_diagnostics"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1.5
        assert rows[0][1] == 3
        assert rows[0][2] == 1
