"""Unit tests for the helpers extracted from Orchestrator._handle_proposal
(audit fix #4). Each helper has one concern; we test each in isolation
so a future regression has a precise failing test, not a broad
end-to-end integration failure.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from brokers.base import AssetClass, OrderSide, OrderType, Position
from strategy_engine.base import TradeProposal
from strategy_engine.orchestrator import CycleReport, Orchestrator


def _make_orch_stub() -> Orchestrator:
    """Bare orchestrator with mocks just sufficient for helper tests."""
    o = Orchestrator.__new__(Orchestrator)
    o.brokers = {}
    o.strategies = {}
    o.cfg = MagicMock()
    o.cfg.is_dry = MagicMock(return_value=True)
    o.risk = MagicMock()
    o.risk.cached_positions = MagicMock(return_value=[])
    o.registry = MagicMock()
    o.registry.meta = MagicMock(return_value=None)
    o._pending_cache = {}
    o._tracker = None
    return o


# ─── _resolve_proposal_size ──────────────────────────────────────────


class TestResolveProposalSize:
    def test_uses_explicit_notional(self):
        o = _make_orch_stub()
        o._positions_for = lambda v: {}
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_usd=2500.0, confidence=0.8, reason="t",
        )
        state = MagicMock(equity_usd=100_000)
        notional, asset_class, existing = o._resolve_proposal_size(prop, state)
        assert notional == 2500.0
        assert asset_class is None      # registry.meta returned None
        assert existing == 0

    def test_falls_back_to_qty_x_limit(self):
        """When notional missing but quantity + limit_price set."""
        o = _make_orch_stub()
        o._positions_for = lambda v: {}
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=10, limit_price=400, confidence=0.8, reason="t",
        )
        state = MagicMock(equity_usd=100_000)
        notional, _, _ = o._resolve_proposal_size(prop, state)
        assert notional == 4000.0       # 10 × $400

    def test_falls_back_to_1pct_equity(self):
        o = _make_orch_stub()
        o._positions_for = lambda v: {}
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            confidence=0.8, reason="t",
        )
        state = MagicMock(equity_usd=100_000)
        notional, _, _ = o._resolve_proposal_size(prop, state)
        assert notional == 1000.0       # 1% of $100k

    def test_pulls_asset_class_from_registry(self):
        o = _make_orch_stub()
        o._positions_for = lambda v: {}
        meta = MagicMock(asset_classes=["EQUITY"])
        o.registry.meta = MagicMock(return_value=meta)
        prop = TradeProposal(
            strategy="rsi_mr", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_usd=100, confidence=0.8, reason="t",
        )
        state = MagicMock(equity_usd=100_000)
        _, asset_class, _ = o._resolve_proposal_size(prop, state)
        assert asset_class == "EQUITY"


# ─── _check_intracycle_wash ──────────────────────────────────────────


class TestCheckIntracycleWash:
    def test_no_pending_returns_false(self):
        o = _make_orch_stub()
        o._pending_orders_for = lambda v: {}
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_usd=100, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._check_intracycle_wash(prop, report) is False
        assert report.proposals_rejected == 0

    def test_opposite_side_pending_skips_proposal(self):
        """Pending BUY + new SELL proposal = wash-trade risk; skip."""
        o = _make_orch_stub()
        o._pending_orders_for = lambda v: {
            "SPY": {"n_pending": 1, "n_buy_pending": 1, "n_sell_pending": 0,
                    "buy_notional_usd": 1000, "sell_qty": 0.0},
        }
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=2, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._check_intracycle_wash(prop, report) is True
        assert report.proposals_rejected == 1

    def test_same_side_pending_passes_through(self):
        """Pending BUY + new BUY proposal = same direction; allow.
        This unblocks legitimate strategy aggregation (e.g. risk_parity
        + tsmom both buying SPY in the same cycle)."""
        o = _make_orch_stub()
        o._pending_orders_for = lambda v: {
            "SPY": {"n_pending": 1, "n_buy_pending": 1, "n_sell_pending": 0,
                    "buy_notional_usd": 1000, "sell_qty": 0.0},
        }
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_usd=500, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._check_intracycle_wash(prop, report) is False
        assert report.proposals_rejected == 0


# ─── _clamp_sell_quantity ────────────────────────────────────────────


class TestClampSellQuantity:
    def test_buy_passes_through(self):
        o = _make_orch_stub()
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="SPY",
            side=OrderSide.BUY, order_type=OrderType.MARKET,
            notional_usd=1000, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._clamp_sell_quantity(prop, report) is True

    def test_sell_clamps_to_90pct_of_available(self):
        o = _make_orch_stub()
        o.risk.cached_positions = MagicMock(return_value=[
            Position(
                venue="alpaca", symbol="TLT",
                asset_class=AssetClass.ETF,
                quantity=100, avg_entry_price=85,
                market_price=85, unrealized_pnl_usd=0,
                raw={"qty_available_parsed": 100},
            ),
        ])
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="TLT",
            side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=100, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._clamp_sell_quantity(prop, report) is True
        assert prop.quantity == pytest.approx(90.0)
        assert prop.notional_usd is None    # forced to qty path

    def test_sell_with_zero_available_skips(self):
        o = _make_orch_stub()
        o.risk.cached_positions = MagicMock(return_value=[
            Position(
                venue="alpaca", symbol="X",
                asset_class=AssetClass.ETF,
                quantity=0, avg_entry_price=0, market_price=0,
                unrealized_pnl_usd=0,
                raw={"qty_available_parsed": 0},
            ),
        ])
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="X",
            side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=10, confidence=0.8, reason="t",
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._clamp_sell_quantity(prop, report) is False
        assert report.proposals_rejected == 1

    def test_sell_under_max_passes_unchanged(self):
        o = _make_orch_stub()
        o.risk.cached_positions = MagicMock(return_value=[
            Position(
                venue="alpaca", symbol="GLD",
                asset_class=AssetClass.ETF,
                quantity=50, avg_entry_price=400, market_price=400,
                unrealized_pnl_usd=0,
                raw={"qty_available_parsed": 50},
            ),
        ])
        prop = TradeProposal(
            strategy="x", venue="alpaca", symbol="GLD",
            side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=10, confidence=0.8, reason="t",    # 10 << 50 × 0.9
        )
        report = CycleReport(timestamp=MagicMock())
        assert o._clamp_sell_quantity(prop, report) is True
        assert prop.quantity == 10
