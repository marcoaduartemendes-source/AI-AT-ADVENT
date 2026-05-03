"""Sprint A audit fixes — Supabase failover for the kill-switch
baseline + venues_ok gate on equity persistence.

Background: the senior-review audit's #1 finding was that the
drawdown kill-switch's `peak_equity` lived in a single SQLite file
on a single VPS — corruption or accidental wipe would reset the
baseline and false-fire the kill switch. Supabase was already
written to (shadow-write) but never read from, so it gave no
disaster-recovery value.

These tests verify:
  - Reads come from SQLite under normal conditions (no behaviour
    change for the happy path)
  - When SQLite has lost data (peak materially lower than Supabase)
    Supabase wins
  - When Supabase is unconfigured/unreachable, behaviour matches
    the pre-fix code (no crash, no false data)
  - venues_ok=False suppresses the equity_snapshot write, preventing
    a transient broker outage from ratcheting the rolling peak down
"""
from __future__ import annotations

from unittest.mock import MagicMock

from risk.manager import EquitySnapshotDB


# ─── Sprint A1: SQLite-primary, Supabase-failover read path ─────────


def test_peak_equity_uses_sqlite_when_supabase_unconfigured(tmp_path):
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    db.record_snapshot(equity_usd=100.0)
    db.record_snapshot(equity_usd=150.0)
    db.record_snapshot(equity_usd=120.0)
    assert db.peak_equity() == 150.0


def test_peak_equity_uses_sqlite_when_consistent_with_supabase(tmp_path):
    """Happy path: both stores agree → SQLite wins (it's the primary)."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.peak_equity_since.return_value = 150.0    # same as SQLite

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=100.0)
    db.record_snapshot(equity_usd=150.0)
    assert db.peak_equity() == 150.0


def test_peak_equity_prefers_supabase_when_sqlite_corrupted(tmp_path):
    """SQLite peak materially lower than Supabase peak → SQLite has
    lost data → use Supabase to keep the kill-switch baseline intact."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.peak_equity_since.return_value = 105_000.0    # truth from Supabase

    # SQLite only has a single low row (simulates a fresh-start /
    # restored-from-backup scenario where the local DB lost history)
    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=98_000.0)
    # Supabase peak is 7% higher → above the 1% disaster-recovery threshold
    assert db.peak_equity() == 105_000.0


def test_peak_equity_ignores_minor_supabase_drift(tmp_path):
    """Supabase 0.3% higher than SQLite (timing race between writes)
    should NOT swap — only material divergence triggers failover.
    Otherwise we'd flip-flop on every cycle."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.peak_equity_since.return_value = 100_300.0    # +0.3%

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=100_000.0)
    # Below the 1% threshold → SQLite wins
    assert db.peak_equity() == 100_000.0


def test_peak_equity_handles_supabase_failure(tmp_path):
    """Supabase raising/returning None must not crash — fall back
    to SQLite cleanly so the cycle doesn't abort."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.peak_equity_since.side_effect = RuntimeError("postgrest 503")

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=42_000.0)
    assert db.peak_equity() == 42_000.0


def test_record_snapshot_dual_writes(tmp_path):
    """Every record_snapshot call must mirror to Supabase. Mirror
    failures must NOT block the SQLite write."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.insert_equity_snapshot.return_value = True
    # Don't trigger the failover path — keep peak_equity returns
    # consistent with SQLite so we test the write path in isolation.
    fake.peak_equity_since.return_value = 12345.67

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=12345.67, note="test")

    # SQLite has the row
    assert db.peak_equity() == 12345.67
    # Supabase was called
    assert fake.insert_equity_snapshot.call_count == 1
    call = fake.insert_equity_snapshot.call_args
    assert call.kwargs["equity_usd"] == 12345.67
    assert call.kwargs["note"] == "test"
    assert "T" in call.kwargs["timestamp"]    # ISO format


def test_record_snapshot_continues_on_supabase_failure(tmp_path):
    """Supabase write raising must not break the SQLite path."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    fake.insert_equity_snapshot.side_effect = RuntimeError("network")
    fake.peak_equity_since.return_value = 5000.0    # consistent

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    db.record_snapshot(equity_usd=5000.0)
    # SQLite write succeeded
    assert db.peak_equity() == 5000.0


def test_recent_returns_uses_supabase_when_sqlite_sparse(tmp_path):
    """If SQLite has fewer than n/2 rows but Supabase has more,
    prefer Supabase. Disaster-recovery scenario: local DB was
    truncated but Supabase preserved the history."""
    fake = MagicMock()
    fake.is_configured.return_value = True
    # Supabase has 60 ascending equity values (1% growth each step)
    sb_eq = [10000.0 * (1.01 ** i) for i in range(60)]
    fake.recent_equity_snapshots.return_value = sb_eq

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=fake)
    # Only 1 SQLite row → way below n/2
    db.record_snapshot(equity_usd=20000.0)

    rets = db.recent_returns(n=60)
    # Should reflect the 60 Supabase rows, not the lone SQLite one
    assert len(rets) == 59
    # Each return ~1%
    assert all(0.005 < r < 0.015 for r in rets)


# ─── Sprint A5: venues_ok gate on equity persistence ────────────────


def test_compute_state_skips_persistence_when_venues_unreachable(tmp_path):
    """When one or more brokers fail their account fetch, the equity
    sum drops by the missing broker's contribution. Persisting that
    truncated equity would falsely depress the rolling peak. The fix
    skips the snapshot insert on venues_ok=False."""
    from risk.manager import RiskManager, RiskConfig

    # Two brokers — one healthy, one raising
    healthy = MagicMock()
    healthy.get_account.return_value = MagicMock(
        cash_usd=80000.0, buying_power_usd=80000.0, equity_usd=85000.0,
    )
    healthy.get_positions.return_value = []
    failing = MagicMock()
    failing.get_account.side_effect = RuntimeError("API 503")
    failing.get_positions.return_value = []

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    # Pre-seed SQLite with a higher prior peak
    db.record_snapshot(equity_usd=100000.0, note="prior cycle (all ok)")

    rm = RiskManager(
        brokers={"alpaca": healthy, "coinbase": failing},
        config=RiskConfig.from_env(), db=db,
    )
    # No new snapshot should land — venues_ok=False
    state = rm.compute_state()

    assert state.venues_ok is False
    # Still only 1 row (the pre-seeded one)
    rets = db.recent_returns(n=10)
    assert len(rets) == 0    # 1 row → 0 returns
    # Peak unchanged
    assert db.peak_equity() == 100000.0


def test_compute_state_persists_when_all_venues_ok(tmp_path):
    """Sanity: when all brokers are healthy, equity DOES get persisted."""
    from risk.manager import RiskManager, RiskConfig

    ok = MagicMock()
    ok.get_account.return_value = MagicMock(
        cash_usd=50000.0, buying_power_usd=50000.0, equity_usd=50000.0,
    )
    ok.get_positions.return_value = []

    db = EquitySnapshotDB(db_path=str(tmp_path / "risk.db"), supabase=None)
    rm = RiskManager(
        brokers={"alpaca": ok}, config=RiskConfig.from_env(), db=db,
    )
    state = rm.compute_state()
    assert state.venues_ok is True
    # Snapshot was recorded
    assert db.peak_equity() == 50000.0
