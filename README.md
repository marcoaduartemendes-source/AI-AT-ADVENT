# AI-AT-ADVENT

Multi-asset systematic trading bot. 24 strategies across equities (Alpaca), crypto (Coinbase), and prediction markets (Kalshi). Runs on GitHub Actions cron + a DigitalOcean VPS.

**Status by default:** Alpaca and Kalshi run in **PAPER mode** (real orders to simulated accounts — strategies actually deploy capital and produce fill data, no real money at risk). Coinbase stays **DRY** (no orders submitted). Going LIVE on Coinbase requires both `DRY_RUN_COINBASE=false` AND `ALLOW_LIVE_TRADING=1` in repo Variables — single-key flips are intentionally not enough.

## Architecture (60-second tour)

```
┌────────────────────────────────────────────────────────────────────┐
│  Schedulers                                                        │
│   • orchestrator.yml  (every 5 min)   src/run_orchestrator.py      │
│   • scouts.yml        (every 30 min)  src/run_scouts.py            │
│   • dashboard.yml     (every 15 min)  src/build_dashboard.py       │
│   • strategic_review  (Mon 13:00 UTC) src/run_strategic_review.py  │
│   • daily-digest      (09:00 UTC)     src/run_daily_digest.py      │
└──────────────────┬─────────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  Orchestrator cycle (src/strategy_engine/orchestrator.py)          │
│   1. Poll pending fills          ──→ trades ledger                 │
│   2. Risk state + KILL switch    ──→ data/risk_state.db            │
│   3. Cancel stale orders                                           │
│   4. Allocator rebalance (weekly cadence)                          │
│   5. Per-strategy compute → risk gate → wash-trade guard           │
│        → SELL clamp → broker submit                                │
│   6. KILL path: emergency-close all + poll resulting fills         │
└──────────────────┬─────────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────────┐
│  Persistence                                                       │
│   • data/trading_performance.db   trades + fill polling            │
│   • data/risk_state.db            equity snapshots, KILL events    │
│   • data/allocator.db             strategy state, weekly weights   │
│   • data/signal_bus.db            scout signals (TTL'd)            │
│   • data/strategic_review.db      Opus weekly recommendations      │
│   • data/alert_dedup.db           recent-alert hash dedup          │
│                                                                    │
│  Failover: every write that matters dual-writes to Supabase if     │
│  SUPABASE_URL is set. Daily Spaces snapshot of all *.db (14d).     │
└────────────────────────────────────────────────────────────────────┘
```

## Where to start reading

If you have 30 minutes, read in this order:

1. [`src/strategy_engine/orchestrator.py`](src/strategy_engine/orchestrator.py) — `Orchestrator.run_cycle()` is the heartbeat. Every other module is called from here.
2. [`src/risk/manager.py`](src/risk/manager.py) — `compute_state()` decides equity, drawdown, KILL state, multiplier. `check_order()` decides whether each proposal can execute.
3. [`src/strategies/_helpers.py`](src/strategies/_helpers.py) + any one strategy under `src/strategies/` — strategies are stateless functions of `StrategyContext`.
4. [`src/brokers/base.py`](src/brokers/base.py) — every venue implements this ABC. Strategies only see this interface.
5. [`AUDIT.md`](AUDIT.md) — performance retrospective (caching, indexes, allocator math).

## Adding things

### A new strategy

1. Subclass `Strategy` in `src/strategies/your_strategy.py`. Override `compute(ctx) -> list[TradeProposal]`.
2. Add a `StrategyMeta(...)` entry to `ALL_STRATEGIES` in `src/run_orchestrator.py`.
3. Wire the instance in `build_strategies()` in the same file.
4. Re-export from `src/strategies/__init__.py`.
5. Add a backtest in `src/backtests/` if practical.
6. Add a test under `tests/` that calls `compute()` with a stubbed context.

The dashboard, allocator, and risk layer auto-pick up new strategies — no other edits needed.

### A new venue

This requires more files and needs care.
1. Implement `BrokerAdapter` in `src/brokers/your_venue.py`.
2. Register in `src/brokers/registry.py::build_brokers()`.
3. Add per-venue caps in `src/risk/policies.py` (`RiskConfig.max_trade_usd_<venue>`).
4. Add credentials to `.env.example`, `deploy/aaa.env.example`, and `.github/workflows/deploy_vps.yml`'s env render.
5. Add a parametrized test in `tests/test_venue_contract.py` (TODO).

### A new scout

1. Subclass `Scout` in `src/scouts/your_scout.py`. Implement `sweep() -> list[Signal]`.
2. Register in `src/run_scouts.py`.
3. Define the signal payload schema as a `TypedDict` (TODO: `src/scouts/signal_types.py`).
4. Strategy code consumes via `ctx.scout_signals.get("your_signal_type")`.

## Local dev

```bash
git clone <repo>
cd AI-AT-ADVENT
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]            # editable install (pyproject.toml)

cp .env.example .env             # fill in only what you need
pytest tests/ -v                 # 294 tests, ~11s, fully hermetic
ruff check src/ tests/

DRY_RUN=true python src/run_orchestrator.py --status   # JSON dump, no trading
DRY_RUN=true python src/run_orchestrator.py --once     # one full cycle, no trading
python src/build_dashboard.py    # writes docs/index.html
```

## Live trading

The default ships paper trading on Alpaca + Kalshi (orders submitted to
the simulated accounts, no real money). Coinbase stays DRY by default.
The orchestrator logs `venue=alpaca mode=🧪 PAPER` etc. at the top of
every cycle so you can confirm at a glance.

To go LIVE on Coinbase (real money):

1. Confirm `data/risk_state.db` has equity history. The cold-start
   guard refuses live trading otherwise (a fresh kill-switch baseline =
   current equity arms KILL at -KILL_DD_PCT of whatever today's equity is).
2. Set repo Variables (Settings → Secrets and variables → Actions → Variables):
   - `MAX_TRADE_USD_GLOBAL=50` (or higher, deliberately)
   - `DRY_RUN_COINBASE=false`
   - `LIVE_STRATEGIES=crypto_funding_carry,...` (allowlist of strategies that can fire on Coinbase)
   - `ORCHESTRATOR_DRY_RUN=false`
3. Set `ALLOW_LIVE_TRADING=1` in repo Variables. **Without this, the
   orchestrator forces DRY mode no matter what else is set** — this is a
   deliberate two-key guard.
4. Watch [`docs/index.html`](docs/index.html) for the first cycle. The
   mode badge per strategy will flip from 🟦 DRY → 🧪 PAPER → 💰 LIVE
   depending on venue and overrides.

To turn OFF paper trading on a venue (rarely needed):
- Set `DRY_RUN_ALPACA=true` (or `DRY_RUN_KALSHI=true`) in repo Variables.

## Operations cheatsheet

| Symptom | Where to look | Action |
|---|---|---|
| Dashboard shows old timestamp | Healthchecks.io check status | If RED → bot is silently down. Check Actions tab + VPS journalctl. |
| KILL switch fired | Pushover alert + dashboard banner | Investigate the equity drop, then `python -c "from risk.manager import RiskManager; RiskManager().reset_kill_switch()"` on VPS. |
| One strategy emits no proposals | Strategy alert after 3 consecutive errors | Check journalctl for the strategy name; usually a scout signal went stale. |
| PnL on dashboard disagrees with broker | `data/trading_performance.db` | Run `recompute_realized_pnl_fifo` (the orchestrator already does this every cycle and alerts on >$1 drift). |
| Need to roll back a deploy | VPS | `bash /opt/ai-at-advent/deploy/rollback.sh` (resets to `deploy/last-good` tag). |

## Required external setup

These cannot be done from inside the repo. Confirm each before going live:

- [ ] **Healthchecks.io** — 4 checks created (orchestrator/scouts/dashboard/db_backup) with Pushover integration. Without this, MTTD for a silent stop is "until the user notices."
- [ ] **Branch protection** on the active deploy branch — require `Tests` workflow green before merge.
- [ ] **DigitalOcean Spaces** bucket for backups + `SPACES_*` secrets configured.
- [ ] **Supabase** (optional but recommended) — gives Postgres-grade durability for risk state.
- [ ] **Pushover** — confirm a priority-2 emergency alert reaches your phone end-to-end.

## Layout

- `src/strategy_engine/` — orchestrator + base classes
- `src/strategies/` — 24 strategy implementations
- `src/scouts/` — data feeds (macro, equities, crypto, commodities, predictions)
- `src/brokers/` — venue adapters (Alpaca, Coinbase, Kalshi)
- `src/risk/` — policies, manager, multiplier, kill-switch
- `src/allocator/` — meta-allocator + lifecycle (FROZEN/ACTIVE/RETIRED)
- `src/trading/` — performance ledger, FIFO recompute, Supabase failover
- `src/common/` — alerts, heartbeat, market_hours, http_cache
- `src/backtests/` — out-of-sample evaluation per strategy
- `tests/` — 294 tests (regression + unit + e2e)
- `deploy/` — VPS install/update/rollback + systemd units + Spaces backup
- `.github/workflows/` — every cron + the test gate
- `docs/index.html` — auto-built dashboard (15-min cadence)

## Contributing

- Run `pytest tests/` and `ruff check src/ tests/` before pushing.
- A push to the deploy branch deploys to the VPS within ~5 min if the test gate passes.
- Bot-authored commits go to `claude/strategic-review-pending` (Claude's weekly review) — those are PR-gated, not auto-deployed.
