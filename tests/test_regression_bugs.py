"""Regression tests for the 4 bugs that hit production today.

Each test FAILS on commits before the fix and PASSES on commits after.
If any one of these comes back, CI is the gate, not the user reading
the dashboard.

Bug index (in order shipped):
  1. Phantom -$5,746 PnL — _record_trade computed PnL with price=0
     because the broker hadn't reported a fill yet.
  2. Intra-cycle wash trade — _pending_cache wasn't updated after
     each successful place_order, so two strategies firing opposite
     sides on the same symbol in one cycle tripped Alpaca.
  3. Coinbase MARKET SELL needs qty — basis trade strategy was
     passing notional_usd to Coinbase SELL legs, which the API
     rejects with "Coinbase MARKET SELL requires quantity".
  4. Dashboard $0 P&L despite open positions — load_live_data
     summary was realized-only; unrealized PnL from open positions
     was silently dropped from the headline total.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, UTC
from unittest.mock import MagicMock

import pytest

# These imports run through conftest's sys.path setup.
from brokers.base import Order, OrderSide, OrderStatus, OrderType, Position, AssetClass


# ─────────────────────────────────────────────────────────────────────
# Bug #1 — phantom -$5,746 PnL from price=0 record
# ─────────────────────────────────────────────────────────────────────

class TestRecordTradeWithZeroPriceDoesNotComputePnL:
    """When the broker hasn't reported a fill yet (filled_avg_price is
    None), _record_trade MUST leave pnl_usd=NULL — never compute
    (0 - entry) * qty which produces a fake loss equal to the position
    value."""

    def test_orchestrator_record_trade_skips_pnl_when_no_fill(self, tmp_path, monkeypatch):
        from strategy_engine.orchestrator import Orchestrator
        from trading.performance import PerformanceTracker

        # Point the tracker at a tmp DB
        db = tmp_path / "trading_performance.db"
        monkeypatch.setenv("TRADING_DB_PATH", str(db))

        # Build the most minimal Orchestrator we can — risk, allocator,
        # strategies, brokers don't matter for this test.
        orch = Orchestrator.__new__(Orchestrator)
        orch._tracker = PerformanceTracker(str(db))
        orch.risk = MagicMock()
        orch.risk.cached_positions = MagicMock(return_value=[
            Position(
                venue="alpaca", symbol="TLT",
                asset_class=AssetClass.ETF,
                quantity=74.71, avg_entry_price=85.65,
                market_price=85.65, unrealized_pnl_usd=0.0,
            ),
        ])
        orch.cfg = MagicMock()
        orch.cfg.is_dry = MagicMock(return_value=False)

        # Build a closing-SELL proposal for which the order has NO fill
        # data yet — this is the exact shape that produced -$5,746.44.
        from strategy_engine.base import TradeProposal
        proposal = TradeProposal(
            strategy="tsmom_etf",
            venue="alpaca",
            symbol="TLT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=67.24,
            confidence=0.9,
            reason="trend exit",
            is_closing=True,
        )
        order = Order(
            venue="alpaca", order_id="abc-123", symbol="TLT",
            side=OrderSide.SELL, type=OrderType.MARKET,
            quantity=67.24, notional_usd=None, limit_price=None,
            status=OrderStatus.PENDING,
            filled_quantity=0.0,
            filled_avg_price=None,   # ← THE BUG: this was None at submit
        )
        decision = MagicMock()
        decision.approved_notional_usd = 6078.59

        orch._record_trade(proposal, order, decision)

        # Inspect the row that was written
        with sqlite3.connect(db) as conn:
            row = conn.execute(
                "SELECT price, pnl_usd FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
        price, pnl = row
        # Bug repro guard: pnl_usd MUST be NULL until a real fill
        # arrives. The old code wrote ~ -5746.44 here.
        assert pnl is None, (
            f"Expected pnl_usd=NULL pre-fill, got {pnl} "
            f"(price={price}). Phantom-loss bug regression."
        )


# ─────────────────────────────────────────────────────────────────────
# Bug #2 — intra-cycle wash trade
# ─────────────────────────────────────────────────────────────────────

class TestIntraCycleWashTrade:
    """When two strategies in one cycle place opposite-side orders on
    the same symbol (e.g. risk_parity BUY SPY then tsmom SELL SPY),
    the second one MUST be skipped. The pending_cache populated at
    cycle-start doesn't see the just-placed BUY, so we have to mark
    it ourselves via _mark_pending_intracycle."""

    def test_mark_pending_intracycle_inflates_n_pending(self):
        from strategy_engine.orchestrator import Orchestrator
        from strategy_engine.base import TradeProposal

        orch = Orchestrator.__new__(Orchestrator)
        orch._pending_cache = {"alpaca": {}}

        proposal = TradeProposal(
            strategy="risk_parity_etf",
            venue="alpaca",
            symbol="SPY",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            notional_usd=3829.66,
            confidence=0.85,
            reason="rebalance",
        )
        order = Order(
            venue="alpaca", order_id="ee4e-123", symbol="SPY",
            side=OrderSide.BUY, type=OrderType.MARKET,
            quantity=0, notional_usd=3829.66, limit_price=None,
            status=OrderStatus.PENDING,
        )
        decision = MagicMock()
        decision.approved_notional_usd = 3829.66

        orch._mark_pending_intracycle(proposal, order, decision)

        # The just-placed BUY must now show up as pending so the next
        # SELL on SPY in this cycle gets skipped.
        entry = orch._pending_cache["alpaca"]["SPY"]
        assert entry["n_pending"] == 1
        assert entry["buy_notional_usd"] == pytest.approx(3829.66)


# ─────────────────────────────────────────────────────────────────────
# Bug #3 — Coinbase MARKET SELL needs qty (basis trade)
# ─────────────────────────────────────────────────────────────────────

class TestCryptoBasisTradeUsesQtyForCoinbaseSells:
    """The basis trade's entry creates two opposite-side proposals
    (long spot + short future). For Coinbase, BUY takes notional_usd
    (quote_size) but SELL must take quantity (base_size). Passing
    notional_usd to a SELL is rejected by the broker with:
        BrokerError: "Coinbase MARKET SELL requires quantity"
    This test asserts that whenever the basis trade creates a SELL
    proposal, it carries `quantity` and not `notional_usd`."""

    def test_basis_trade_sell_legs_carry_quantity_not_notional(self, monkeypatch):
        from strategies.crypto_basis_trade import CryptoBasisTrade

        # Stub the public-products endpoint to return one ETH future
        # with positive basis, forcing the strategy to open a trade.
        # base_increment must be set or the strategy will round qty
        # down to 0 and skip — observed 2026-05-08 INVALID_SIZE_PRECISION
        # fix added base_increment-aware rounding.
        #
        # 2026-05-30: the expiry MUST be computed relative to now, not
        # hardcoded. A previously-hardcoded "29MAY26" silently expired and
        # the strategy correctly skipped the dead contract → the test
        # started failing with time, not with any code change. Generate a
        # contract ~90 days out so it's always live.
        from datetime import timedelta
        expiry = (datetime.now(UTC) + timedelta(days=90)).strftime("%d%b%y").upper()
        fake_products = [
            {"product_id": f"ET-{expiry}-CDE", "price": "3500",
             "base_increment": "0.001"},
        ]
        from strategies import crypto_basis_trade as cbt
        monkeypatch.setattr(cbt, "cached_get", lambda url, params=None, ttl_seconds=0: (
            {"products": fake_products}
            if "products" in url and not url.endswith("ETH-USD")
            else {"price": "3000"}  # spot price
        ))

        from strategy_engine.base import StrategyContext
        adapter = MagicMock()
        strat = CryptoBasisTrade(adapter)
        ctx = StrategyContext(
            timestamp=datetime.now(UTC),
            portfolio_equity_usd=10000,
            target_alloc_pct=0.02,
            target_alloc_usd=200,
            risk_multiplier=1.0,
            open_positions={},   # no positions → entry path
            scout_signals={},
            pending_orders={},
        )
        proposals = strat.compute(ctx)

        # The strategy should produce 2 proposals (spot+future legs).
        sells = [p for p in proposals if p.side == OrderSide.SELL]
        assert sells, (
            "Expected at least one SELL leg from basis trade entry. "
            "Bug #3 caused the SELL leg to be the future short."
        )
        for sell in sells:
            assert sell.quantity is not None and sell.quantity > 0, (
                f"SELL leg {sell.symbol} must carry quantity, not "
                f"notional_usd. Got quantity={sell.quantity}, "
                f"notional={sell.notional_usd}. Coinbase rejects "
                f"notional-only SELLs with HTTP 400."
            )


# ─────────────────────────────────────────────────────────────────────
# Bug #4 — dashboard summary excluded unrealized PnL
# ─────────────────────────────────────────────────────────────────────

# Removed: TestDashboardSummaryIncludesUnrealizedPnL — exercised the
# old 2120-line dashboard's _summarize / load_live_data internals.
# The replacement dashboard is server-rendered HTML with a single
# realized-P&L SQL aggregate; coverage now lives in tests/test_dashboard.py.


# ─────────────────────────────────────────────────────────────────────
# Bonus: independent FIFO recompute
# ─────────────────────────────────────────────────────────────────────

class TestFIFORecompute:
    """Sanity-check the second-brain that catches the next
    phantom-loss class of bug. A clean ledger should produce a
    matching DB total and FIFO recompute (drift ≤ $0.50)."""

    def test_clean_ledger_produces_zero_drift(self, tmp_path):
        from trading.recompute import recompute_realized_pnl_fifo

        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, product_id TEXT,
                side TEXT, amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL, dry_run INTEGER
            )
        """)
        # BUY 10 @ $100, then SELL 10 @ $110 → +$100 realized
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T10:00:00", "tsmom_etf", "SPY", "BUY",
             1000, 10, 100, "o1", None, 0),
        )
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T11:00:00", "tsmom_etf", "SPY", "SELL",
             1100, 10, 110, "o2", 100.0, 0),
        )
        conn.commit()
        conn.close()

        db_total, recomputed, drift = recompute_realized_pnl_fifo(str(db))
        assert db_total == pytest.approx(100.0)
        assert recomputed == pytest.approx(100.0)
        assert drift == {}    # no per-strategy drift > $0.50

    def test_drift_is_detected(self, tmp_path):
        """If the DB pnl_usd doesn't match what FIFO would derive,
        the function must report it in per_strategy_drift."""
        from trading.recompute import recompute_realized_pnl_fifo

        db = tmp_path / "test_drift.db"
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, product_id TEXT,
                side TEXT, amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL, dry_run INTEGER
            )
        """)
        # BUY 10 @ $100, SELL 10 @ $110 — but DB stores wrong PnL=+$200
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T10:00:00", "tsmom_etf", "SPY", "BUY",
             1000, 10, 100, "o1", None, 0),
        )
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-01T11:00:00", "tsmom_etf", "SPY", "SELL",
             1100, 10, 110, "o2", 200.0, 0),    # ← WRONG: should be 100
        )
        conn.commit()
        conn.close()

        db_total, recomputed, drift = recompute_realized_pnl_fifo(str(db))
        assert db_total == pytest.approx(200.0)
        assert recomputed == pytest.approx(100.0)
        assert "tsmom_etf" in drift
        assert drift["tsmom_etf"] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────
# Bug #5 — headline Realized P&L silently trusted a drifted ledger.
# On the droplet the stored pnl_usd summed to $+2.26 while a clean FIFO
# walk gave $+720.64 (a $718 drift), but the dashboard only ever showed
# the stored number — the recompute was logged where nobody looked. The
# dashboard must now flag a diverging headline instead of trusting it.
# ─────────────────────────────────────────────────────────────────────
class TestRealizedPnLDriftSurfacedOnDashboard:
    """The dashboard helper that cross-checks stored P&L against the
    FIFO recompute must (a) report no divergence for a clean ledger and
    (b) flag divergence when the stored number lies."""

    def _make_db(self, path, sell_pnl):
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, strategy TEXT, product_id TEXT,
                side TEXT, amount_usd REAL, quantity REAL, price REAL,
                order_id TEXT, pnl_usd REAL, dry_run INTEGER,
                fill_status TEXT
            )
        """)
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run,fill_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-20T10:00:00+00:00", "s1", "AAA", "BUY",
             100, 1, 100, "o1", None, 0, "FILLED"),
        )
        conn.execute(
            "INSERT INTO trades (timestamp,strategy,product_id,side,"
            "amount_usd,quantity,price,order_id,pnl_usd,dry_run,fill_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("2026-05-20T11:00:00+00:00", "s1", "AAA", "SELL",
             110, 1, 110, "o2", sell_pnl, 0, "FILLED"),
        )
        conn.commit()
        conn.close()

    def test_clean_ledger_not_flagged(self, tmp_path):
        import build_dashboard as bd
        db = tmp_path / "clean.db"
        self._make_db(db, sell_pnl=10.0)   # FIFO also derives +10 → match
        res = bd._realized_pnl_drift(str(db))
        assert res is not None
        assert res["diverged"] is False
        assert res["drift"] == pytest.approx(0.0)

    def test_drifted_ledger_is_flagged(self, tmp_path):
        import build_dashboard as bd
        db = tmp_path / "drift.db"
        # Stored pnl lies (+728) while FIFO derives +10 → $718 drift,
        # exactly the droplet failure mode.
        self._make_db(db, sell_pnl=728.0)
        res = bd._realized_pnl_drift(str(db))
        assert res is not None
        assert res["diverged"] is True
        assert res["drift"] == pytest.approx(718.0)
        assert res["db_total"] == pytest.approx(728.0)
        assert res["fifo_total"] == pytest.approx(10.0)

    def test_missing_db_returns_none(self, tmp_path):
        import build_dashboard as bd
        assert bd._realized_pnl_drift(str(tmp_path / "nope.db")) is None


# ─────────────────────────────────────────────────────────────────────
# Bug #6 — self-grade's alpha-vs-SPY axis was pinned to a fake 5.0.
# _grade_alpha_vs_benchmark() read flat fields (bot_return_pct_ann, …)
# that build_benchmark never writes. The real file nests returns under
# portfolio.windows / benchmarks.SPY.windows, so the reader always hit
# the "lacks comparable fields → 5.0" placeholder — inflating the grade
# and hiding a real -6pp/30d underperformance vs SPY. The user mandate
# is HONEST grading, so a flattering placeholder is itself the bug.
# ─────────────────────────────────────────────────────────────────────
class TestAlphaVsSpyReadsRealBenchmarkShape:
    def test_underperformance_scores_low(self, tmp_path, monkeypatch):
        import json
        import common.self_grade as sg
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "benchmark.json").write_text(json.dumps({
            "portfolio": {"windows": {"7": -0.1, "14": -1.3, "30": -1.65}},
            "benchmarks": {"SPY": {"windows": {"7": -0.2, "14": 0.9, "30": 4.52}}},
        }))
        monkeypatch.chdir(tmp_path)
        g, r = sg._grade_alpha_vs_benchmark()
        # -1.65 vs +4.52 → -6.17pp excess → bottom of the scale, NOT 5.0.
        assert g == 0.0
        assert "30d" in r and "excess" in r

    def test_outperformance_scores_high(self, tmp_path, monkeypatch):
        import json
        import common.self_grade as sg
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "benchmark.json").write_text(json.dumps({
            "portfolio": {"windows": {"30": 9.0}},
            "benchmarks": {"SPY": {"windows": {"30": 4.5}}},
        }))
        monkeypatch.chdir(tmp_path)
        g, _ = sg._grade_alpha_vs_benchmark()
        assert g == 10.0   # +4.5pp excess → capped top

    def test_missing_file_stays_neutral(self, tmp_path, monkeypatch):
        import common.self_grade as sg
        monkeypatch.chdir(tmp_path)   # no docs/benchmark.json
        g, _ = sg._grade_alpha_vs_benchmark()
        assert g == 5.0


# ─────────────────────────────────────────────────────────────────────
# Bug #7 — walk_forward read a non-existent "equity" key from the
# backtest equity_curve (points are {"t","pnl_cumulative"}), so every
# bar delta was 0, every Sharpe was None, and ALL strategies came back
# NO_DATA — pinning self-grade's overfit_resistance axis to 0 forever
# (observed 2026-05-22: "0 ROBUST / 0 OVERFIT_SUSPECT / 18 NO_DATA").
# ─────────────────────────────────────────────────────────────────────
class TestWalkForwardReadsCumulativePnL:
    def test_split_sharpe_uses_pnl_cumulative(self):
        from common.walk_forward import _split_sharpe, _verdict
        # A realistic rising-then-mixed cumulative-P&L curve with enough
        # points for both halves and ≥5 non-zero deltas each.
        cum = 0.0
        curve = []
        for i, step in enumerate([10, 12, -4, 8, 15, 9, -3, 11, 7, 13,
                                  6, -2, 9, 14, 8, 10, -5, 12, 9, 7]):
            cum += step
            curve.append({"t": f"2026-01-{i+1:02d}", "pnl_cumulative": cum})
        is_s, oos_s, is_t, oos_t = _split_sharpe(curve)
        assert is_s is not None and oos_s is not None, \
            "Sharpe must be computed from pnl_cumulative, not a missing key"
        assert is_t >= 5 and oos_t >= 5
        verdict, _ = _verdict(is_s, oos_s, is_t, oos_t)
        assert verdict != "NO_DATA", \
            f"a populated curve must yield a real verdict, got {verdict}"

    def test_legacy_equity_key_still_supported(self):
        from common.walk_forward import _split_sharpe
        curve = [{"t": f"d{i}", "equity": float(i * 5)} for i in range(20)]
        is_s, oos_s, is_t, oos_t = _split_sharpe(curve)
        # Monotonic curve → zero-variance deltas → Sharpe None, but the
        # key must be READ (non-zero trade counts prove it parsed).
        assert is_t > 0 and oos_t > 0


# ─────────────────────────────────────────────────────────────────────
# Bug #8 — Kalshi backtests read m["yes_close"], a field the API never
# returns, so yes_close_price was 0 for every market and all 1000
# settled candidates were rejected ("no_yes_close") — the Kalshi
# strategies could never validate or trade. The live adapter reads
# "last_price" (cents); the history parser must too.
# ─────────────────────────────────────────────────────────────────────
class TestKalshiSettledMarketParsing:
    def test_last_price_is_read_as_yes_close(self):
        from backtests.data.kalshi_history import _parse_yes_close
        # cents → 0-1
        assert _parse_yes_close({"last_price": 63}) == pytest.approx(0.63)
        # already-normalised fallback
        assert _parse_yes_close({"yes_close": 0.42}) == pytest.approx(0.42)
        # the old bug: no usable price → 0 (skipped downstream)
        assert _parse_yes_close({"ticker": "X"}) == 0.0

    def test_settlement_from_value_or_result(self):
        from backtests.data.kalshi_history import _parse_settlement
        assert _parse_settlement({"settlement_value": 100}) == 1.0
        assert _parse_settlement({"settlement_value": 0}) == 0.0
        assert _parse_settlement({"result": "yes"}) == 1.0
        assert _parse_settlement({"result": "no"}) == 0.0
        assert _parse_settlement({}) == 0.5   # unknown → void → skipped


# ─────────────────────────────────────────────────────────────────────
# Bug #9 — realized P&L was sourced from the stored pnl_usd column, which
# is computed at fill time from a single avg-cost entry and drifts on
# partial fills / stale broker cost-basis / orphan SELLs / phantom
# price=0 (the $718 droplet drift, and the corrupt allocator metrics that
# spuriously froze risk_parity_etf at "60d Sharpe=-5.62, DD=57758%").
# FIFO over the raw fill ledger is the canonical, auditable source.
# ─────────────────────────────────────────────────────────────────────
class TestFifoCanonicalRealized:
    def _db(self, tmp_path, rows):
        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "timestamp TEXT, strategy TEXT, product_id TEXT, side TEXT, "
                     "quantity REAL, price REAL, pnl_usd REAL, fill_status TEXT)")
        conn.executemany(
            "INSERT INTO trades (timestamp,strategy,product_id,side,quantity,"
            "price,pnl_usd,fill_status) VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        return str(db)

    def test_multilot_fifo_ignores_bogus_stored_pnl(self, tmp_path):
        from trading.recompute import fifo_realized_by_strategy
        db = self._db(tmp_path, [
            ("2026-05-01T10:00", "s1", "AAA", "BUY", 1, 100, None, "FILLED"),
            ("2026-05-01T11:00", "s1", "AAA", "BUY", 1, 105, None, "FILLED"),
            ("2026-05-02T10:00", "s1", "AAA", "SELL", 2, 110, 999.0, "FILLED"),
        ])
        # FIFO: (110-100) + (110-105) = 15, not the stored 999.
        assert fifo_realized_by_strategy(db) == {"s1": 15.0}

    def test_orphan_sell_and_phantom_excluded(self, tmp_path):
        from trading.recompute import fifo_realized_events
        db = self._db(tmp_path, [
            ("2026-05-03T10:00", "s2", "BBB", "SELL", 1, 50, 777.0, "FILLED"),  # orphan
            ("2026-05-04T10:00", "s3", "CCC", "BUY", 1, 0, None, "CANCELED"),   # phantom
        ])
        assert fifo_realized_events(db) == {}

    def test_allocator_metrics_use_fifo_not_stored(self, tmp_path):
        from allocator.metrics import StrategyPerformance
        # Stored pnl is a wild +5000 on one SELL, but FIFO says +10.
        db = self._db(tmp_path, [
            ("2026-05-01T10:00", "s1", "AAA", "BUY", 1, 100, None, "FILLED"),
            ("2026-05-02T10:00", "s1", "AAA", "SELL", 1, 110, 5000.0, "FILLED"),
        ])
        m = StrategyPerformance(db_path=db).metrics_for("s1", window_days=3650)
        assert m.total_pnl_usd == pytest.approx(10.0)   # FIFO, not 5000


# ─────────────────────────────────────────────────────────────────────
# Bug #10 — "<unattributed>" unrealized P&L. The dashboard matched a live
# broker position to the ledger by EXACT (venue, symbol) against the most
# recent BUY, so it dumped P&L into "<unattributed>" on symbol-format
# drift (Coinbase "BTC" vs ledger "BTC-USD"), multi-strategy symbols, or
# a wiped local DB. Fixed with FIFO open-lot attribution + symbol
# normalization + proportional split.
# ─────────────────────────────────────────────────────────────────────
class TestUnattributedPnLFixed:
    def _db(self, tmp_path, rows):
        db = tmp_path / "t.db"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "timestamp TEXT, strategy TEXT, product_id TEXT, side TEXT, "
                     "quantity REAL, price REAL, pnl_usd REAL, fill_status TEXT)")
        conn.executemany(
            "INSERT INTO trades (timestamp,strategy,product_id,side,quantity,"
            "price,pnl_usd,fill_status) VALUES (?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        return str(db)

    def test_symbol_normalization_collapses_formats(self):
        from trading.recompute import normalize_symbol
        assert normalize_symbol("BTC-USD") == "BTC"
        assert normalize_symbol("BTC/USD") == "BTC"
        assert normalize_symbol("btc") == "BTC"
        assert normalize_symbol("AAPL") == "AAPL"      # equities untouched

    def test_multi_strategy_open_lots_split(self, tmp_path):
        from trading.recompute import fifo_open_positions
        db = self._db(tmp_path, [
            ("t1", "sA", "BTC-USD", "BUY", 3, 100, None, "FILLED"),
            ("t2", "sB", "BTC-USD", "BUY", 2, 100, None, "FILLED"),
            ("t3", "sA", "BTC-USD", "SELL", 1, 110, 10.0, "FILLED"),
        ])
        lots = fifo_open_positions(db)
        assert lots == {"BTC": {"sA": 2.0, "sB": 2.0}}

    def test_closed_position_has_no_open_lot(self, tmp_path):
        from trading.recompute import fifo_open_positions
        db = self._db(tmp_path, [
            ("t1", "sC", "AAPL", "BUY", 10, 50, None, "FILLED"),
            ("t2", "sC", "AAPL", "SELL", 10, 55, 50.0, "FILLED"),
        ])
        assert "AAPL" not in fifo_open_positions(db)

    def test_dashboard_attribution_proportional_no_unattributed(self, tmp_path, monkeypatch):
        import build_dashboard as bd

        class _Pos:
            def __init__(self, symbol, upnl):
                self.symbol = symbol
                self.unrealized_pnl_usd = upnl

        class _Broker:
            def get_positions(self):
                # Broker reports a single netted BTC position (Coinbase
                # format "BTC") with +$100 unrealized.
                return [_Pos("BTC", 100.0)]

        monkeypatch.setattr(bd, "build_brokers", None, raising=False)
        monkeypatch.setattr("brokers.registry.build_brokers",
                            lambda: {"coinbase": _Broker()})
        # sA holds 3 open, sB holds 1 open (ledger stores "BTC-USD").
        monkeypatch.setattr(bd, "_open_lots_by_symbol",
                            lambda: {"BTC": {"sA": 3.0, "sB": 1.0}})
        out = bd._live_unrealized_by_strategy()
        assert out["sA"] == pytest.approx(75.0)   # 3/4 of +100
        assert out["sB"] == pytest.approx(25.0)   # 1/4 of +100
        assert not any("unattributed" in k for k in out)

    def test_genuinely_external_position_labelled_clearly(self, tmp_path, monkeypatch):
        import build_dashboard as bd

        class _Pos:
            symbol = "DOGE"
            unrealized_pnl_usd = 42.0

        class _Broker:
            def get_positions(self):
                return [_Pos()]

        monkeypatch.setattr("brokers.registry.build_brokers",
                            lambda: {"coinbase": _Broker()})
        monkeypatch.setattr(bd, "_open_lots_by_symbol", lambda: {})   # no ledger
        out = bd._live_unrealized_by_strategy()
        assert out == {"<unattributed: no ledger entry>": 42.0}


class TestActivityClassificationNotStale:
    """Reproduces the bug where the dashboard labelled every quiet
    strategy 'stale', conflating a low-turnover book correctly HOLDING
    positions with one whose every order is being REJECTED at the
    concentration cap and one that simply has no signal.

    Failing the test (a HOLDING book classified the same as a flat one)
    communicates the bug; passing communicates the invariant: activity
    is derived from real cycle counters, not from time-since-last-trade.
    """

    def _mk(self, **kw):
        from strategy_engine.orchestrator import StrategyOutcome
        return StrategyOutcome(strategy="x", venue="alpaca", **kw)

    def test_holding_book_is_not_waiting(self):
        from strategy_engine.orchestrator import Orchestrator
        # No proposals this cycle but the strategy owns 4 positions —
        # this is a risk-parity/dual-momentum book sitting tight, NOT dead.
        held = self._mk(proposed=0, target_alloc_usd=5000, held_positions=4)
        flat = self._mk(proposed=0, target_alloc_usd=5000, held_positions=0)
        assert Orchestrator._classify_activity(held) == "HOLDING"
        assert Orchestrator._classify_activity(flat) == "WAITING"
        # The two must be distinguishable — the whole point of the fix.
        assert Orchestrator._classify_activity(held) != \
               Orchestrator._classify_activity(flat)

    def test_fully_rejected_is_blocked_not_idle(self):
        from strategy_engine.orchestrator import Orchestrator
        # Produced 5 proposals, all rejected (e.g. ETF concentration cap).
        o = self._mk(proposed=5, rejected=5)
        assert Orchestrator._classify_activity(o) == "BLOCKED"

    def test_submit_beats_everything(self):
        from strategy_engine.orchestrator import Orchestrator
        o = self._mk(proposed=3, submitted=1, rejected=2, held_positions=9)
        assert Orchestrator._classify_activity(o) == "TRADING"

    def test_zero_allocation_is_no_alloc(self):
        from strategy_engine.orchestrator import Orchestrator
        o = self._mk(proposed=0, target_alloc_usd=0.0)
        assert Orchestrator._classify_activity(o) == "NO_ALLOC"

    def test_compute_error_is_error(self):
        from strategy_engine.orchestrator import Orchestrator
        o = self._mk(error="ValueError: boom", proposed=0)
        assert Orchestrator._classify_activity(o) == "ERROR"


class TestExecutionGradeNoFalseHealth:
    """Reproduces the bug where execution_quality fell back to 5/10
    when there were no submittable proposals — masking a silent book
    (zero throughput) as a passing grade. Failing the test (5/10 for
    a dead book) communicates the bug; passing communicates the
    invariant: 'no submissions ≠ healthy'.
    """

    def test_empty_cycle_history_scores_zero(self):
        from common.self_grade import _grade_execution
        g, _ = _grade_execution([])
        assert g == 0.0

    def test_all_dry_book_scores_zero_not_neutral(self):
        from common.self_grade import _grade_execution
        cycles = [{"timestamp": f"2026-06-05T0{i}:00:00+00:00",
                   "proposals_total": 5, "proposals_submitted": 0,
                   "proposals_dry": 5} for i in range(5)]
        g, reason = _grade_execution(cycles)
        assert g == 0.0
        assert "silent" in reason.lower() or "zero" in reason.lower()

    def test_real_submit_ratio_unchanged(self):
        from common.self_grade import _grade_execution
        cycles = [{"timestamp": f"2026-06-05T0{i}:00:00+00:00",
                   "proposals_total": 10, "proposals_submitted": 8,
                   "proposals_dry": 0} for i in range(5)]
        g, _ = _grade_execution(cycles)
        assert g == 8.0


class TestFullSystemReviewFixes:
    """Regression tests for the 2026-06-11 full-system review — the
    correctness/safety defects that were silently costing alpha or
    leaving the book unmanaged. Each test fails against the pre-fix code.
    """

    def test_alpaca_candles_return_most_recent(self, monkeypatch):
        """The bars API returns ascending-from-start capped at limit; the
        old code took the OLDEST num_candles and dropped today. Verify
        the adapter slices the TAIL so closes[-1] is the latest bar."""
        from brokers.alpaca import AlpacaAdapter
        # 300 ascending daily closes; we ask for 100.
        bars = [{"t": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                 "o": i, "h": i, "l": i, "c": float(i), "v": 1}
                for i in range(300)]
        a = AlpacaAdapter.__new__(AlpacaAdapter)
        a._get_cached_candles = lambda *x: None
        a._put_cached_candles = lambda *x: None
        a._get = lambda *x, **k: {"bars": bars}
        out = a.get_candles("SPY", "ONE_DAY", num_candles=100)
        assert len(out) == 100
        # Tail = most recent: last close must be 299 (newest), not 99.
        assert out[-1].close == 299.0
        assert out[0].close == 200.0

    def test_kill_switch_permits_closing_orders(self, tmp_path):
        """A latched KILL must still approve is_closing orders, or the
        emergency-close is a no-op and the book rides the drawdown."""
        from risk.manager import RiskManager, EquitySnapshotDB
        from risk.policies import RiskConfig, KillSwitchState
        from risk.manager import Decision
        from tests.mock_broker import MockBroker
        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        db.record_snapshot(100_000, note="peak")
        rm = RiskManager(brokers={"alpaca": MockBroker(
            venue="alpaca", cash_usd=80_000, equity_usd=80_000)},
            config=RiskConfig(), db=db)
        rm.arm_kill_switch(note="test")
        st = rm.compute_state(persist=False)
        assert st.kill_switch == KillSwitchState.KILL
        close = rm.check_order(
            notional_usd=5000, symbol="SPY",
            is_closing=True, state=st)
        assert close.decision == Decision.APPROVE, (
            "KILL must let closing orders through to flatten the book")
        open_ = rm.check_order(
            notional_usd=5000, symbol="SPY",
            is_closing=False, state=st)
        assert open_.decision == Decision.REJECT

    def test_reset_rebaselines_drawdown_peak(self, tmp_path):
        """After a manual reset, drawdown must be measured from the reset
        moment — not the untouched pre-drawdown all-time peak — so a
        genuine-DD KILL doesn't instantly re-latch."""
        import time
        from risk.manager import RiskManager, EquitySnapshotDB
        from risk.policies import RiskConfig, KillSwitchState
        from tests.mock_broker import MockBroker
        db = EquitySnapshotDB(str(tmp_path / "risk.db"))
        # Old peak $100k, book now flat at $85k (15% down).
        db.record_snapshot(100_000, note="old peak")
        rm = RiskManager(brokers={"alpaca": MockBroker(
            venue="alpaca", cash_usd=85_000, equity_usd=85_000)},
            config=RiskConfig(kill_dd_pct=0.15), db=db)
        # Reset re-baselines; record a post-reset snapshot at the low.
        rm.reset_kill_switch()
        time.sleep(0.01)
        db.record_snapshot(85_000, note="post-reset")
        st = rm.compute_state(persist=False)
        # Drawdown vs the post-reset baseline ($85k) is ~0, NOT 15%.
        assert st.kill_switch != KillSwitchState.KILL, (
            "reset must re-baseline the peak so KILL doesn't re-latch")

    def test_equity_and_etf_share_one_cap_bucket(self):
        """EQUITY and ETF must consume ONE budget, not two independent
        0.90 caps (which allowed 1.8x correlated beta)."""
        from risk.policies import RiskConfig
        cfg = RiskConfig()
        assert cfg.bucket_for("EQUITY") == cfg.bucket_for("ETF")
        # The shared cap is the tighter of the two members.
        assert cfg.cap_for_asset_class("EQUITY") == \
               cfg.cap_for_asset_class("ETF")

    def test_signal_bus_aggregates_event_rows(self, tmp_path, monkeypatch):
        """Event scouts publish one row per event under the same
        signal_type; the bus must deliver ALL of them as a list, not
        collapse to the newest one."""
        from scouts.signal_bus import SignalBus
        bus = SignalBus(db_path=str(tmp_path / "bus.db"))
        for tic in ("AAA", "BBB", "CCC"):
            bus.publish(scout="merger_arb", venue="alpaca",
                        signal_type="merger_arb_deal",
                        payload={"target": tic, "deal_price": 50.0},
                        ttl_seconds=3600)
        got = bus.get_fresh_for_strategy("alpaca")
        deals = got.get("merger_arb_deal")
        assert isinstance(deals, list)
        assert {d["target"] for d in deals} == {"AAA", "BBB", "CCC"}


class TestIRRReviewBatch2:
    """Regression tests for the 2026-06-11 IRR-focused fixes."""

    def test_autodemote_exempts_pass_from_live_bleed(self, tmp_path,
                                                     monkeypatch):
        """A validation-PASS strategy with a rough live stretch must NOT
        be zeroed by the live-bleed/overfit rules — the allocator's WATCH
        de-risking owns that case; full zeroing permanently benches a
        proven edge."""
        import json as _json
        import common.auto_demote as ad
        # PASS set includes our strategy.
        monkeypatch.setattr(
            "common.strategy_validation.passing_strategies",
            lambda *a, **k: {"good_pass"})
        # Validation: PASS; walk-forward: OVERFIT_SUSPECT (would trip
        # rule 2 if not exempt). No FAIL.
        (tmp_path / "docs").mkdir()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "docs" / "validation.json").write_text(_json.dumps({
            "strategies": {"good_pass": {"verdict": "PASS", "reason": "ok"}}}))
        (tmp_path / "docs" / "walk_forward.json").write_text(_json.dumps({
            "strategies": {"good_pass": {"verdict": "OVERFIT_SUSPECT",
                                          "reason": "x"}}}))
        (tmp_path / "docs" / "trades_recent.json").write_text("[]")
        overrides = ad.run_auto_demote(
            out_path=str(tmp_path / 'docs' / 'auto_overrides.json'))
        assert "good_pass" not in overrides, (
            "PASS strategy must be exempt from overfit/live-bleed zeroing")

    def test_vix_filter_reads_namespaced_key(self):
        """dividend_growth's VIX>30 gate must read macro_vix_regime.vix
        (the real published shape), not the phantom 'vix' key."""
        from unittest.mock import MagicMock
        from strategies.dividend_growth import DividendGrowth
        s = DividendGrowth(broker=MagicMock())
        ctx = type("C", (), {"scout_signals":
                              {"macro_vix_regime": {"vix": 35.0}}})()
        assert s._latest_vix(ctx) == 35.0
        ctx2 = type("C", (), {"scout_signals": {"vix": {"value": 35.0}}})()
        assert s._latest_vix(ctx2) is None   # old shape ignored

    def test_deploy_pass_reduces_cash_drag(self, tmp_path):
        """The allocator must deploy idle cash upward toward the target
        instead of leaving 20-30% fallow — while preserving max caps."""
        from allocator.allocator import AllocatorConfig, MetaAllocator
        from allocator.lifecycle import StrategyRegistry, StrategyMeta
        from allocator.metrics import StrategyPerformance
        reg = StrategyRegistry(str(tmp_path / "alloc.db"))
        for nm in ("a", "b"):
            reg.register(StrategyMeta(
                name=nm, asset_classes=["ETF"], venue="alpaca",
                target_alloc_pct=0.10, min_alloc_pct=0.0,
                max_alloc_pct=0.40))
        perf = StrategyPerformance(str(tmp_path / "perf.db"))
        alloc = MetaAllocator(reg, perf, AllocatorConfig())
        result = alloc.rebalance(100_000)
        total = sum(d.target_pct for d in result.decisions)
        # Cold sleeves would sit near ~0.10 (0.5×0.10×2) without the
        # deploy pass; with it, materially more capital is deployed.
        assert total > 0.20, f"cash drag not reduced: total={total:.3f}"
        for d in result.decisions:
            assert d.target_pct <= 0.40 + 1e-9   # caps respected
