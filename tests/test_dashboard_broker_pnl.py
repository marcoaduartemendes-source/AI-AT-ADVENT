"""Tests for per-broker realized-P&L attribution on the dashboard.

Production bug: legacy strategies (Momentum / MeanReversion /
VolatilityBreakout) trade live on Coinbase but are not in
ALL_STRATEGIES, so their realized P&L was being attributed to a
phantom "legacy" venue and Coinbase showed $0 realized on the
dashboard even when the bot was actively closing trades. The fix
adds an explicit LEGACY_STRATEGY_TO_VENUE override so the legacy
P&L lands on coinbase where it belongs.

These tests pin that behaviour down so the regression cannot return.
"""
from __future__ import annotations

from build_dashboard import (
    LEGACY_STRATEGY_TO_VENUE,
    _attribute_broker_pnl,
    _build_strategy_to_venue,
)


def _broker_snapshot(venue: str, **overrides) -> dict:
    """Mimic the shape returned by _broker_snapshot."""
    base = {
        "venue": venue,
        "cash_usd": 0.0, "buying_power_usd": 0.0, "equity_usd": 0.0,
        "invested_usd": 0.0, "unrealized_pnl_usd": 0.0,
        "n_positions": 0, "n_open_orders": 0,
        "available_pct": 0.0, "last_ok_at": None, "error": None,
    }
    base.update(overrides)
    return base


# ─── Legacy strategy attribution (the production bug) ─────────────────


def test_legacy_strategies_attribute_to_coinbase():
    """Momentum / MeanReversion / VolatilityBreakout trades must
    show up as Coinbase realized P&L, NOT in a phantom 'legacy'
    bucket."""
    by_broker = {
        "coinbase": _broker_snapshot("coinbase",
                                       cash_usd=5000, equity_usd=10000),
    }
    trades = [
        {"strategy": "Momentum",      "pnl_usd": 12.5,
         "timestamp": "2026-05-01T10:00:00+00:00"},
        {"strategy": "MeanReversion", "pnl_usd": -3.0,
         "timestamp": "2026-05-01T11:00:00+00:00"},
        {"strategy": "VolatilityBreakout", "pnl_usd": 8.0,
         "timestamp": "2026-05-01T12:00:00+00:00"},
    ]
    _attribute_broker_pnl(by_broker, trades, _build_strategy_to_venue())

    assert "legacy" not in by_broker, (
        "Legacy strategies should attribute to coinbase, not "
        "create a phantom 'legacy' bucket"
    )
    assert by_broker["coinbase"]["realized_pnl_usd"] == 17.5
    assert by_broker["coinbase"]["n_trades_closed"] == 3
    # total_pnl_usd = realized + unrealized
    assert by_broker["coinbase"]["total_pnl_usd"] == 17.5


def test_legacy_to_venue_map_covers_all_three():
    """Defensive: ensure all 3 known legacy strategies are mapped.
    If a 4th legacy strategy is ever revived without being added
    here, this test reminds us."""
    assert LEGACY_STRATEGY_TO_VENUE == {
        "Momentum": "coinbase",
        "MeanReversion": "coinbase",
        "VolatilityBreakout": "coinbase",
    }


# ─── Mixed-venue attribution ──────────────────────────────────────────


def test_orchestrator_strategies_attribute_to_their_venue():
    """tsmom_etf / risk_parity_etf trade on Alpaca → realized P&L
    must land on alpaca, not coinbase."""
    by_broker = {
        "coinbase": _broker_snapshot("coinbase", cash_usd=5000),
        "alpaca":   _broker_snapshot("alpaca", cash_usd=80000),
    }
    s2v = {
        "tsmom_etf": "alpaca",
        "risk_parity_etf": "alpaca",
        "Momentum": "coinbase",
    }
    trades = [
        {"strategy": "tsmom_etf", "pnl_usd": 50.0,
         "timestamp": "2026-05-01T10:00:00+00:00"},
        {"strategy": "risk_parity_etf", "pnl_usd": -20.0,
         "timestamp": "2026-05-01T11:00:00+00:00"},
        {"strategy": "Momentum", "pnl_usd": 5.0,
         "timestamp": "2026-05-01T12:00:00+00:00"},
    ]
    _attribute_broker_pnl(by_broker, trades, s2v)

    assert by_broker["alpaca"]["realized_pnl_usd"] == 30.0
    assert by_broker["alpaca"]["n_trades_closed"] == 2
    assert by_broker["coinbase"]["realized_pnl_usd"] == 5.0
    assert by_broker["coinbase"]["n_trades_closed"] == 1


def test_unrealized_pnl_preserved_in_total():
    """Existing unrealized_pnl_usd from _broker_snapshot must be
    carried into total_pnl_usd, not overwritten."""
    by_broker = {
        "alpaca": _broker_snapshot("alpaca",
                                     unrealized_pnl_usd=125.0,
                                     cash_usd=80000),
    }
    trades = [
        {"strategy": "tsmom_etf", "pnl_usd": 50.0,
         "timestamp": "2026-05-01T10:00:00+00:00"},
    ]
    _attribute_broker_pnl(by_broker, trades, {"tsmom_etf": "alpaca"})

    assert by_broker["alpaca"]["realized_pnl_usd"] == 50.0
    assert by_broker["alpaca"]["unrealized_pnl_usd"] == 125.0
    # Total = 50 + 125
    assert by_broker["alpaca"]["total_pnl_usd"] == 175.0


def test_open_trades_skipped():
    """Trades with pnl_usd = None (still open) must not contribute
    to realized P&L or trade count."""
    by_broker = {"alpaca": _broker_snapshot("alpaca")}
    trades = [
        {"strategy": "tsmom_etf", "pnl_usd": None, "side": "BUY",
         "timestamp": "2026-05-01T10:00:00+00:00"},
        {"strategy": "tsmom_etf", "pnl_usd": 25.0, "side": "SELL",
         "timestamp": "2026-05-01T15:00:00+00:00"},
    ]
    _attribute_broker_pnl(by_broker, trades, {"tsmom_etf": "alpaca"})

    assert by_broker["alpaca"]["realized_pnl_usd"] == 25.0
    assert by_broker["alpaca"]["n_trades_closed"] == 1


def test_unknown_strategy_falls_to_legacy_bucket():
    """A strategy that's neither in the registry nor in the
    legacy override should get its own 'legacy' bucket — not
    pollute coinbase or alpaca with mystery P&L."""
    by_broker = {
        "coinbase": _broker_snapshot("coinbase"),
        "alpaca":   _broker_snapshot("alpaca"),
    }
    s2v = {"tsmom_etf": "alpaca"}    # no entry for "MysteryStrat"
    trades = [
        {"strategy": "MysteryStrat", "pnl_usd": 99.0,
         "timestamp": "2026-05-01T10:00:00+00:00"},
    ]
    _attribute_broker_pnl(by_broker, trades, s2v)

    assert "legacy" in by_broker
    assert by_broker["legacy"]["realized_pnl_usd"] == 99.0
    assert by_broker["coinbase"]["realized_pnl_usd"] == 0.0
    assert by_broker["alpaca"]["realized_pnl_usd"] == 0.0


def test_total_pnl_is_realized_plus_unrealized_even_when_zero():
    """Brokers with no closed trades must still display a
    total_pnl_usd field (= unrealized) so the dashboard renders
    a number, not 'undefined'."""
    by_broker = {
        "kalshi": _broker_snapshot("kalshi", unrealized_pnl_usd=2.5),
    }
    _attribute_broker_pnl(by_broker, [], {})

    assert by_broker["kalshi"]["realized_pnl_usd"] == 0.0
    assert by_broker["kalshi"]["n_trades_closed"] == 0
    assert by_broker["kalshi"]["total_pnl_usd"] == 2.5
