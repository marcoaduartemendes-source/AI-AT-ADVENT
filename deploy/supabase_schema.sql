-- AI-AT-ADVENT Supabase schema.
--
-- Mirrors the SQLite tables we have today so the dual-write migration
-- doesn't have to translate column names. Run once against your
-- Supabase project:
--
--   1. Open https://supabase.com/dashboard → your project → "SQL Editor"
--   2. Paste this whole file
--   3. Click "Run"
--
-- Or via psql with the connection string from
-- Project Settings → Database → Connection string:
--
--   psql "postgresql://postgres:[YOUR-DB-PASS]@db.xxxxxxx.supabase.co:5432/postgres" \
--        -f deploy/supabase_schema.sql
--
-- Idempotent — uses CREATE TABLE IF NOT EXISTS, safe to re-run.
--
-- Row Level Security: ENABLED on every table (2026-05-23 audit fix).
-- The bot writes with the service_role key, which BYPASSES RLS, so this
-- doesn't change the bot's behaviour — but it slams the door on the
-- anon/public API key, which by default has read access to any
-- RLS-disabled table. Without RLS, anyone who guesses or scrapes the
-- anon key (often grep-able from frontend code) could read the full
-- trade ledger, equity history, and kill-switch events. Supabase's
-- dashboard correctly flags RLS-off as a security warning.
--
-- The orchestrator continues to work unchanged because service_role
-- ignores RLS. No policies are added below; that produces a deny-all
-- result for every non-service_role identity, which is exactly what
-- this internal trading bot needs.

-- ─── trades — every order we placed ─────────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    strategy      TEXT NOT NULL,
    product_id    TEXT NOT NULL,
    side          TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    amount_usd    DOUBLE PRECISION NOT NULL,
    quantity      DOUBLE PRECISION NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    order_id      TEXT,
    pnl_usd       DOUBLE PRECISION,
    dry_run       BOOLEAN NOT NULL DEFAULT TRUE,
    -- Mirror columns we don't have in SQLite yet but Postgres lets us
    -- index efficiently. fill_status helps the polling loop find
    -- unfilled orders. recorded_via tags whether the row originated
    -- on the GH cron or the Hetzner VPS so we can spot dupes during
    -- the cutover window.
    fill_status   TEXT,
    recorded_via  TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_strategy_side
    ON trades(strategy, side, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp
    ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_unfilled
    ON trades(timestamp, order_id)
    WHERE price = 0 AND order_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_product
    ON trades(product_id, timestamp);


-- ─── equity_snapshots — portfolio value timeseries ─────────────────
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    equity_usd  DOUBLE PRECISION NOT NULL,
    note        TEXT
);
CREATE INDEX IF NOT EXISTS idx_equity_snapshots_ts
    ON equity_snapshots(timestamp DESC);


-- ─── kill_switch_events — every trip of the kill switch ────────────
CREATE TABLE IF NOT EXISTS kill_switch_events (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    state         TEXT NOT NULL,
    drawdown_pct  DOUBLE PRECISION NOT NULL,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_kill_switch_ts
    ON kill_switch_events(timestamp DESC);


-- ─── allocations — allocator's per-strategy weight history ─────────
CREATE TABLE IF NOT EXISTS allocations (
    id            BIGSERIAL PRIMARY KEY,
    timestamp     TIMESTAMPTZ NOT NULL,
    name          TEXT NOT NULL,
    target_pct    DOUBLE PRECISION NOT NULL,
    target_usd    DOUBLE PRECISION NOT NULL,
    state         TEXT NOT NULL,
    sharpe        DOUBLE PRECISION,
    drawdown_pct  DOUBLE PRECISION,
    reason        TEXT
);
CREATE INDEX IF NOT EXISTS idx_allocations_name_ts
    ON allocations(name, timestamp DESC);


-- ─── lifecycle_events — strategy state transitions ─────────────────
CREATE TABLE IF NOT EXISTS lifecycle_events (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    name        TEXT NOT NULL,
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_ts
    ON lifecycle_events(timestamp DESC);


-- ─── strategy_state — current state per strategy ───────────────────
CREATE TABLE IF NOT EXISTS strategy_state (
    name        TEXT PRIMARY KEY,
    state       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    note        TEXT
);


-- ─── signals — scout signal bus ────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    venue       TEXT NOT NULL,
    name        TEXT NOT NULL,
    payload     JSONB NOT NULL,
    expires_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_signals_venue_name_ts
    ON signals(venue, name, timestamp DESC);
-- (Removed an `idx_signals_active` partial index that used NOW() in
-- the predicate — Postgres requires index predicates to be IMMUTABLE,
-- and NOW() isn't. The full (venue, name, timestamp) index above
-- covers the active-signal lookup path well enough — the signals
-- table is small and TTL-cleaned, so a tiny seq scan is fine.)


-- ─── strategic_review — Opus reviews + recommendations ─────────────
CREATE TABLE IF NOT EXISTS strategic_review (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    overall_health  TEXT,
    summary         TEXT,
    payload_json    JSONB,
    risk_mult_rec   DOUBLE PRECISION,
    risk_mult_reason TEXT,
    model_used      TEXT,
    cost_usd        DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_strategic_review_ts
    ON strategic_review(timestamp DESC);


-- ─── Row Level Security (2026-05-23 security audit fix) ───────────
-- ENABLE on every table. Idempotent — re-running is a no-op. Service
-- role bypasses RLS so the orchestrator is unaffected; anon/public
-- access becomes deny-all (no policies attached = deny-by-default).
ALTER TABLE trades             ENABLE ROW LEVEL SECURITY;
ALTER TABLE equity_snapshots   ENABLE ROW LEVEL SECURITY;
ALTER TABLE kill_switch_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE allocations        ENABLE ROW LEVEL SECURITY;
ALTER TABLE lifecycle_events   ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_state     ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals            ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategic_review   ENABLE ROW LEVEL SECURITY;
-- Belt-and-suspenders: explicitly revoke the default PUBLIC grants
-- (Postgres ships SELECT to PUBLIC on new tables in some configs).
-- service_role keeps full access via its role grant.
REVOKE ALL ON trades, equity_snapshots, kill_switch_events,
              allocations, lifecycle_events, strategy_state,
              signals, strategic_review
       FROM PUBLIC, anon, authenticated;


-- ─── Reset/cleanup helpers (for dev) ───────────────────────────────
-- Comment IN if you ever need to wipe a table:
--   TRUNCATE trades, equity_snapshots, kill_switch_events,
--            allocations, lifecycle_events, strategy_state, signals,
--            strategic_review RESTART IDENTITY;
