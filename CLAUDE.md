# Notes for Claude (claude.ai/code, agent SDK, etc.)

This file is read at session start. It's the orientation a fresh Claude session needs to be useful in <5 minutes.

## What this project is

A live multi-asset systematic trading bot. **Real money is at stake** when `ALLOW_LIVE_TRADING=1` in production. Default everywhere else is DRY mode. See `README.md` for architecture; this file covers Claude-specific operating norms.

## Norms for this codebase

1. **Tests must pass before push.** `pytest tests/` runs in ~11s and is hermetic. CI re-runs the same suite + ruff + an import-smoke before the VPS deploys, but treat the local run as the gate.
2. **Never push directly to the deploy branch unless asked.** Bot-authored commits go to `claude/strategic-review-pending` for human PR review. The user's named branch (`claude/<feature>-<id>`) is fine for in-flight work.
3. **No new dependencies without confirming.** `requirements.txt` is intentionally tiny. Adding a heavy dep (numpy/pandas notwithstanding — those are in) requires user sign-off because every cron tick reinstalls it.
4. **Don't touch real broker accounts.** All tests use `tests/mock_broker.py`. If a test would hit a real API, it's wrong — patch `requests.get` / `requests.post` instead.
5. **Comments explain WHY, not WHAT.** Many comments in this repo reference specific historical bugs (the "phantom-loss" PnL, the "770 stuck PENDING orders", the "wash-trade rejection storm"). When you fix a bug, leave a comment naming the failure mode.
6. **Audit-fix annotations are load-bearing.** Lines tagged `# audit fix:` document an invariant. Don't remove the comment when refactoring; refactor the code to keep the invariant.

## Where the dragons are

- **`src/strategy_engine/orchestrator.py`** is 1000+ lines. Touch with care. The order of operations in `run_cycle()` matters: poll fills → risk → cancel stale → allocate → per-strategy.
- **`StrategyContext.open_positions` and `scout_signals` are `dict[str, dict]`** — stringly-typed. There's an open improvement to introduce dataclasses; until then, defensive `.get(...)` is required.
- **The `_pending_cache` and `_broker_snapshots` lifecycles are implicit.** Both are set on the orchestrator instance and cleared at cycle start. Adding parallelism here will break things subtly.
- **SQLite migrations live in `src/trading/migrations.py`** with a `schema_migrations` marker table. **Never mutate an existing migration's name or body** — the marker prevents replay and you'll get split-brain across deployments.
- **Two runtimes (GH Actions cron + VPS systemd timer) currently both exist.** A single source-of-truth move is on the operations TODO list. Until then, don't assume one or the other.
- **Healthchecks.io ping URLs (`HEALTHCHECKS_PING_URL_*`)** are referenced but the user must create the checks externally. The repo can't verify they exist; if `heartbeat.ping_*` returns False silently, that's why.

## Quick recipes

### Add a strategy

```python
# src/strategies/my_strategy.py
from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

class MyStrategy(Strategy):
    name = "my_strategy"
    venue = "alpaca"

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        # ... your logic ...
        return [TradeProposal(strategy=self.name, venue=self.venue, ...)]
```

Then add a `StrategyMeta` to `ALL_STRATEGIES` in `src/run_orchestrator.py`, instantiate in `build_strategies()`, and re-export from `src/strategies/__init__.py`. The dashboard, allocator, and risk layer pick it up automatically.

### Add a regression test for a bug you just fixed

```python
# tests/test_regression_bugs.py — append a class
class TestYourBugName:
    """Reproduces the bug where <one-line description>.

    Failing the test should communicate the bug; passing should
    communicate the fix's invariant.
    """
    def test_invariant_holds(self):
        # arrange: minimum viable setup that reproduces the bug
        # act: trigger the code path
        # assert: the invariant the bug violated
```

### Run a single test

```bash
pytest tests/test_orchestrator_e2e.py::TestKillSwitchEmergencyClose -v
```

### Smoke-test the orchestrator end-to-end (no trades)

```bash
DRY_RUN=true python src/run_orchestrator.py --status   # JSON dump
DRY_RUN=true python src/run_orchestrator.py --once     # full cycle
```

## House style

- Type hints on public functions, especially anything in `brokers/base.py`, `strategy_engine/base.py`, `risk/policies.py`, `allocator/lifecycle.py`. mypy is not yet enforced — don't trust silence.
- Prefer `from datetime import UTC, datetime` over `timezone.utc`. The repo uses 3.12 native UTC.
- SQLite reads/writes go through context-managed connections (`with self._conn() as c:`). Don't keep connections open across calls.
- Dataclasses for outputs of pure functions (e.g. `RiskDecision`, `RiskState`, `AllocationDecision`). Free dicts only at IO boundaries.
- One concept per file in `src/strategies/` — the file is the unit of git history.

## What's intentionally not done (yet)

These are known and on the TODO list — don't "fix" them as a surprise:

- **mypy strict mode** — will be added incrementally; until then type hints are documentation.
- **Async / concurrent strategies** — strategies run serially on purpose so risk decisions see updated state.
- **Hash-pinned `requirements.lock`** — Dependabot is configured but the lockfile generation is a user step (`pip-compile`).
- **SHA-pinned actions** — Dependabot will start opening PRs once it runs.
- **GH Actions cron retired in favor of VPS-only** — both currently run; consolidation is on the ops list.
- **Per-strategy daily notional governor** — a request from the ops audit; not yet implemented.

## When in doubt

Read `AUDIT.md` for the performance retrospective and the existing audit findings. Read `README.md` for setup. Read `tests/test_orchestrator_e2e.py` for end-to-end behavior contracts. The 1000-line orchestrator is the heartbeat; everything else is called from `run_cycle()`.
