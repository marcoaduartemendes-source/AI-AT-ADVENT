"""Tests for src/scouts/signal_bus.py — the SQLite pub/sub layer
between scouts and strategies.

Coverage gap: this module mediates EVERY signal in the system but
had no direct tests. Strategies break if a scout publishes the
wrong shape; without unit tests on the bus itself, regressions in
publish/query semantics surface only through integration failures.
"""
from __future__ import annotations

import time as _time
from datetime import UTC, datetime, timedelta

from scouts.signal_bus import SignalBus, SignalRow


def test_publish_returns_id_and_persists(tmp_path):
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    rid = bus.publish(
        scout="t", venue="alpaca", signal_type="ticker_news",
        payload={"symbol": "AAPL"},
    )
    assert rid > 0
    rows = bus.latest(venue="alpaca")
    assert len(rows) == 1
    assert rows[0].signal_type == "ticker_news"
    assert rows[0].payload == {"symbol": "AAPL"}


def test_latest_filters_by_venue_and_type(tmp_path):
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    bus.publish(scout="m", venue="macro", signal_type="vix_regime",
                  payload={"vix": 18})
    bus.publish(scout="c", venue="coinbase", signal_type="funding_rates",
                  payload={"rates": []})
    bus.publish(scout="m", venue="macro", signal_type="fomc_window",
                  payload={"days_to_next": 5})

    macro = bus.latest(venue="macro")
    assert len(macro) == 2
    assert {r.signal_type for r in macro} == {"vix_regime", "fomc_window"}

    rates_only = bus.latest(signal_type="funding_rates")
    assert len(rates_only) == 1
    assert rates_only[0].venue == "coinbase"


def test_get_fresh_for_strategy_groups_latest_per_type(tmp_path):
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    # Two publishes of the same signal_type — only the newest wins
    bus.publish(scout="c", venue="coinbase", signal_type="funding_rates",
                  payload={"rates": "old"})
    bus.publish(scout="c", venue="coinbase", signal_type="funding_rates",
                  payload={"rates": "new"})
    bus.publish(scout="m", venue="macro", signal_type="vix_regime",
                  payload={"vix": 22})
    out = bus.get_fresh_for_strategy("coinbase")
    assert out["funding_rates"] == {"rates": "new"}
    # Macro signals merged in via macro_<type> namespace
    assert out["macro_vix_regime"] == {"vix": 22}


def test_get_fresh_for_strategy_excludes_macro_when_called_for_macro(tmp_path):
    """A macro-venue strategy doesn't double-up its own signals."""
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    bus.publish(scout="m", venue="macro", signal_type="vix_regime",
                  payload={"vix": 18})
    out = bus.get_fresh_for_strategy("macro")
    assert "vix_regime" in out
    # No macro_vix_regime key — that prefix is for cross-venue
    # consumption, not the macro venue itself
    assert "macro_vix_regime" not in out


def test_stale_signals_excluded_from_get_fresh(tmp_path):
    """Signals past their TTL must not appear in get_fresh_for_strategy."""
    bus = SignalBus(db_path=str(tmp_path / "bus.db"),
                     default_ttl_seconds=1)
    bus.publish(scout="c", venue="coinbase", signal_type="x",
                  payload={"v": 1}, ttl_seconds=1)
    # Wait past TTL
    _time.sleep(1.1)
    out = bus.get_fresh_for_strategy("coinbase")
    assert "x" not in out


def test_signal_row_is_fresh():
    now = datetime.now(UTC)
    fresh = SignalRow(id=1, scout="x", venue="v", signal_type="t",
                       payload={}, ttl_seconds=3600,
                       created_at=now)
    assert fresh.is_fresh()
    stale = SignalRow(id=2, scout="x", venue="v", signal_type="t",
                       payload={}, ttl_seconds=10,
                       created_at=now - timedelta(seconds=60))
    assert not stale.is_fresh()


def test_vacuum_expired_deletes_old_rows(tmp_path):
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    bus.publish(scout="x", venue="v", signal_type="t",
                  payload={}, ttl_seconds=1)
    bus.publish(scout="x", venue="v", signal_type="t2",
                  payload={}, ttl_seconds=3600)
    # 2.0s sleep gives plenty of slop above the 1s TTL — strftime
    # truncates created_at to whole seconds, so on a sub-second-boundary
    # publish a 1.1s sleep can fall just short of expiration.
    _time.sleep(2.0)
    n_deleted = bus.vacuum_expired()
    assert n_deleted == 1
    # Fresh row preserved
    rows = bus.latest()
    assert len(rows) == 1
    assert rows[0].signal_type == "t2"


def test_payload_round_trips_complex_types(tmp_path):
    """JSON serialization must handle nested dicts + lists + numbers."""
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "ratings": {"AAPL": 0.85, "MSFT": 0.91},
        "n": 42,
        "active": True,
    }
    bus.publish(scout="x", venue="alpaca", signal_type="t",
                  payload=payload)
    rows = bus.latest(venue="alpaca")
    assert rows[0].payload == payload


def test_unrelated_venues_isolated(tmp_path):
    bus = SignalBus(db_path=str(tmp_path / "bus.db"))
    bus.publish(scout="c", venue="coinbase", signal_type="x",
                  payload={"v": 1})
    bus.publish(scout="a", venue="alpaca", signal_type="x",
                  payload={"v": 2})
    cb = bus.get_fresh_for_strategy("coinbase")
    al = bus.get_fresh_for_strategy("alpaca")
    assert cb.get("x") == {"v": 1}
    assert al.get("x") == {"v": 2}
