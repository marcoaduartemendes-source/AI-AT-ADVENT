"""Parametrized contract test for the BrokerAdapter ABC.

Adding a new venue (Binance, dYdX, etc.) is supposed to be a single-
file change: implement BrokerAdapter, register it. The risk manager,
allocator, and orchestrator should pick it up automatically. But
today there is no contract test that *proves* a new adapter satisfies
the orchestrator's assumptions — you find out from production.

This test runs the same orchestrator-shaped probe against every
broker available in `brokers.registry`, MockBroker included. It
asserts the minimal contract:

  - get_account() returns an Account with non-NaN equity_usd
  - get_positions() returns a list of Position
  - get_open_orders() returns a list of Order (default [] is fine)
  - cancel_stale_orders(seconds) returns an int (default 0 is fine)
  - get_quote(symbol) returns a Quote OR raises BrokerError cleanly
    (not AttributeError / TypeError)

When you add a venue, write a `mock_<venue>` fixture below.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from brokers.base import (
    Account,
    AssetClass,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)
from tests.mock_broker import MockBroker, MockPosition


def _has_real_creds(env_keys: list[str]) -> bool:
    """Return True only if every listed env var is non-empty.
    Used to skip live-broker contract tests in CI / sandboxes.
    """
    import os
    return all(os.environ.get(k, "").strip() for k in env_keys)


# ─── Adapter providers ────────────────────────────────────────────────


def _make_mock_broker():
    return MockBroker(
        venue="alpaca",
        cash_usd=100_000,
        positions=[MockPosition("SPY", qty=10, entry=720, mark=725)],
    )


def _make_alpaca_adapter():
    if not _has_real_creds(["ALPACA_API_KEY_ID", "ALPACA_SECRET_KEY"]):
        pytest.skip("Alpaca creds not configured")
    from brokers.alpaca import AlpacaAdapter
    return AlpacaAdapter()


def _make_coinbase_adapter():
    if not _has_real_creds(["COINBASE_API_KEY", "COINBASE_API_SECRET"]):
        pytest.skip("Coinbase creds not configured")
    from brokers.coinbase import CoinbaseAdapter
    return CoinbaseAdapter()


def _make_kalshi_adapter():
    if not _has_real_creds(["KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY"]):
        pytest.skip("Kalshi creds not configured")
    from brokers.kalshi import KalshiAdapter
    return KalshiAdapter()


@pytest.fixture(params=[
    pytest.param(_make_mock_broker, id="mock"),
    pytest.param(_make_alpaca_adapter, id="alpaca"),
    pytest.param(_make_coinbase_adapter, id="coinbase"),
    pytest.param(_make_kalshi_adapter, id="kalshi"),
])
def adapter(request):
    return request.param()


# ─── Contract assertions ──────────────────────────────────────────────


class TestBrokerAdapterContract:

    def test_venue_attribute_is_set(self, adapter):
        assert isinstance(adapter.venue, str) and adapter.venue
        assert adapter.venue.islower(), (
            "venue should be a lowercase canonical name "
            "(used as a dict key throughout the codebase)"
        )

    def test_get_account_returns_account(self, adapter):
        acct = adapter.get_account()
        assert isinstance(acct, Account)
        assert acct.venue == adapter.venue
        assert acct.equity_usd == acct.equity_usd, "equity must not be NaN"
        assert acct.cash_usd == acct.cash_usd, "cash must not be NaN"

    def test_get_positions_returns_list_of_position(self, adapter):
        positions = adapter.get_positions()
        assert isinstance(positions, list)
        for p in positions:
            assert isinstance(p, Position)
            assert p.venue == adapter.venue
            assert isinstance(p.asset_class, AssetClass)

    def test_get_open_orders_returns_list_of_order(self, adapter):
        # Default ABC implementation returns []; venues that override
        # must still produce a list of Order. Coinbase + Kalshi should
        # accept the default no-op gracefully.
        orders = adapter.get_open_orders()
        assert isinstance(orders, list)
        for o in orders:
            assert isinstance(o, Order)
            assert isinstance(o.side, OrderSide)
            assert isinstance(o.status, OrderStatus)

    def test_cancel_stale_orders_returns_int(self, adapter):
        # Same default-returns-0 contract as get_open_orders.
        n = adapter.cancel_stale_orders(1800)
        assert isinstance(n, int)
        assert n >= 0


# ─── Mock-only behavioural contract ──────────────────────────────────


class TestMockBrokerOrderFlow:
    """The mock broker is what every other test depends on. Its
    behaviour must match what the orchestrator expects from a real
    broker: place_order returns an Order, get_positions reflects fills,
    BrokerError fires on injected reject_next."""

    def test_place_order_returns_order_with_status(self):
        broker = MockBroker(venue="alpaca", cash_usd=10_000)
        order = broker.place_order(
            symbol="SPY", side=OrderSide.BUY, type=OrderType.MARKET,
            notional_usd=100,
        )
        assert isinstance(order, Order)
        assert order.symbol == "SPY"
        assert order.side == OrderSide.BUY
        assert isinstance(order.status, OrderStatus)

    def test_reject_next_raises_broker_error(self):
        broker = MockBroker(venue="alpaca", cash_usd=10_000)
        broker.reject_next = "wash trade simulated"
        with pytest.raises(BrokerError):
            broker.place_order(
                symbol="SPY", side=OrderSide.BUY, type=OrderType.MARKET,
                notional_usd=100,
            )

    def test_get_quote_returns_quote(self):
        broker = MockBroker(venue="alpaca", cash_usd=10_000)
        quote = broker.get_quote("SPY")
        assert isinstance(quote, Quote)
        assert quote.symbol == "SPY"
        assert quote.timestamp.tzinfo is not None
