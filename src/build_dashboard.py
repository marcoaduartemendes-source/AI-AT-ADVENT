#!/usr/bin/env python3
"""Build the trading dashboard.

Reads live trades from the SQLite DB and runs 7/15/30-day backtests against
historical candles, then renders one self-contained HTML file at
docs/index.html. Run on a schedule (or workflow_dispatch) from CI.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, UTC
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from backtest import (
    DEFAULT_FEE_BPS,
    backtest_strategy,
    fetch_coinbase_public_history,
    trade_to_dict,
)
from trading.performance import PerformanceTracker
from trading.strategies.mean_reversion import MeanReversionStrategy
from trading.strategies.momentum import MomentumStrategy
from trading.strategies.volatility_breakout import VolatilityBreakoutStrategy

# Optional imports — orchestrator state (added W1)
try:
    from allocator.lifecycle import StrategyRegistry
    _HAS_ORCHESTRATOR = True
except ImportError:
    _HAS_ORCHESTRATOR = False

# Per-strategy backtests for the new strategies
try:
    from backtests import backtest_all as _new_backtest_all, UNBACKTESTABLE
    _HAS_NEW_BACKTESTS = True
except ImportError:
    _HAS_NEW_BACKTESTS = False
    UNBACKTESTABLE = {}

# ─── Config ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dashboard")

PRODUCTS = [
    p.strip()
    for p in os.environ.get("TRADING_PRODUCTS", "BTC-USD,ETH-USD,SOL-USD").split(",")
    if p.strip()
]
GRANULARITY = os.environ.get("GRANULARITY", "FIVE_MINUTE")
MAX_TRADE_USD = float(os.environ.get("MAX_TRADE_USD", "20"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.04"))
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "900"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.6"))
FEE_BPS = float(os.environ.get("BACKTEST_FEE_BPS", str(DEFAULT_FEE_BPS)))
WINDOWS = [7, 15, 30]


def make_strategies(granularity: str):
    return [
        MomentumStrategy(products=PRODUCTS, granularity=granularity),
        MeanReversionStrategy(products=PRODUCTS, granularity=granularity),
        VolatilityBreakoutStrategy(products=PRODUCTS, granularity=granularity),
    ]


# ─── Live data ────────────────────────────────────────────────────────────────


def load_live_data() -> dict:
    """Pull all live trades + open positions from the bot's SQLite DB."""
    db_path = os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
    if not Path(db_path).exists():
        return {
            "trades": [],
            "open_positions": [],
            "by_strategy": {},
            "equity_curve": [],
            "summary": _empty_summary("Live"),
        }

    tracker = PerformanceTracker(db_path)
    all_trades = tracker.get_recent_trades(limit=10000)

    # Sanitize bad PnL records: trades recorded at submission time
    # before the broker reported a fill have price=0, which the old
    # orchestrator code combined with avg_entry_price to produce
    # gigantic phantom losses (e.g. (0 − $85) × 67 qty = −$5,746 on
    # a SELL TLT that actually filled near $100). Until those get
    # backfilled with real fill prices, treat them as un-realized.
    for t in all_trades:
        if (t.get("pnl_usd") is not None
                and (not t.get("price") or t.get("price") == 0)):
            t["pnl_usd"] = None

    # Sort ascending by time for an equity curve
    all_trades_sorted = sorted(all_trades, key=lambda t: t["timestamp"])

    # Equity curve: only closed (SELL with pnl) trades contribute
    eq: list[dict] = []
    cum = 0.0
    for t in all_trades_sorted:
        if t.get("pnl_usd") is not None:
            cum += t["pnl_usd"]
            eq.append({"t": t["timestamp"], "pnl_cumulative": cum})

    # Per-strategy summary — include legacy strategies AND the new
    # orchestrator-driven strategies so the dashboard shows all of them.
    by_strategy: dict[str, dict] = {}
    all_strategy_names = [
        # ── Production strategies (orchestrator-driven)
        "crypto_funding_carry", "risk_parity_etf", "kalshi_calibration_arb",
        "crypto_basis_trade", "tsmom_etf", "commodity_carry",
        "pead", "macro_kalshi", "crypto_xsmom", "vol_managed_overlay",
        # ── Legacy (retired in W2; kept here so historical trades remain visible)
        "Momentum", "MeanReversion", "VolatilityBreakout",
    ]
    # Pre-bucket trades by strategy in ONE pass instead of N×len(trades)
    # passes — was a hidden 13×10000 = 130k-iteration hot loop.
    trades_by_strategy: dict[str, list[dict]] = defaultdict(list)
    for t in all_trades:
        trades_by_strategy[t.get("strategy") or "<unknown>"].append(t)
    for name in all_strategy_names:
        strat_trades = trades_by_strategy.get(name, [])
        summary = _summarize(strat_trades)
        summary["strategy"] = name
        by_strategy[name] = {
            "summary": summary,
            "trades": strat_trades,
        }

    # Open positions — pull from BOTH the legacy open_positions table
    # AND the live broker adapters (Alpaca/Coinbase/Kalshi). Live broker
    # positions are the source of truth for current holdings.
    # NOTE: positions MUST be loaded before the summary so we can fold
    # unrealized P&L into the headline number — without that the
    # dashboard shows $0 P&L even when we hold $25k of paper positions
    # gaining/losing money in real time.
    open_pos_raw = []
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM open_positions").fetchall()
    for r in rows:
        open_pos_raw.append(dict(r))
    conn.close()
    # Live broker positions + per-broker account snapshots so the
    # dashboard can render a "money on each broker" view.
    # Parallel fetch across venues — three brokers were sequential
    # (~6 blocking HTTP round-trips), now all three run concurrently.
    by_broker: dict[str, dict] = {}
    open_orders_all: list[dict] = []
    try:
        from brokers.registry import build_brokers
        brokers = build_brokers()
        with ThreadPoolExecutor(max_workers=max(1, len(brokers))) as ex:
            futures = {
                ex.submit(_broker_snapshot, name, adapter): name
                for name, adapter in brokers.items()
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    snap = fut.result(timeout=30)
                except Exception as e:
                    logger.warning(f"broker snapshot {name} failed: {e}")
                    continue
                # Fold positions into the legacy open_pos_raw list so
                # downstream summary aggregation keeps working.
                for pos in snap.pop("positions", []):
                    open_pos_raw.append({
                        "strategy": f"<{name}>",
                        "product_id": pos["symbol"],
                        "quantity": pos["quantity"],
                        "cost_basis_usd": pos["cost_basis_usd"],
                        "entry_price": pos["entry_price"],
                        "entry_time": "",
                        "market_price": pos["market_price"],
                        "unrealized_pnl_usd": pos["unrealized_pnl_usd"],
                        "venue": pos["venue"],
                    })
                open_orders_all.extend(snap.pop("open_orders", []))
                by_broker[name] = snap
    except Exception as e:
        logger.warning(f"Could not fetch live broker positions: {e}")

    # Top-level totals — REALIZED + UNREALIZED so the dashboard reflects
    # the actual P&L state (open positions moving up/down right now),
    # not just trades that have round-tripped. Day P&L from an
    # equity-snapshot lookup happens later (after orchestrator state).
    base = _summarize(all_trades)
    realized_pnl = base["total_pnl_usd"]
    unrealized_pnl = sum(p.get("unrealized_pnl_usd", 0) or 0 for p in open_pos_raw)
    market_value = sum(
        (p.get("quantity", 0) or 0) * (p.get("market_price", 0) or 0)
        for p in open_pos_raw
    )
    total_pnl = realized_pnl + unrealized_pnl
    summary = {
        "label": "Live",
        **base,
        # total_pnl_usd is the headline number on the dashboard — keep
        # it as realized+unrealized so the user sees real-time P&L.
        "total_pnl_usd": total_pnl,
        "realized_pnl_usd": realized_pnl,
        "unrealized_pnl_usd": unrealized_pnl,
        "market_value_usd": market_value,
        "n_open_positions": len(open_pos_raw),
        "return_on_volume_pct": (
            total_pnl / base["entry_volume_usd"] * 100
            if base["entry_volume_usd"] > 0 else 0.0
        ),
    }

    # Per-product (per-symbol) P&L breakdown — closes the "is BTC up
    # while ETH bleeds?" gap. Combines realized P&L from closed trades
    # with unrealized P&L from open positions, keyed by product_id.
    by_product: dict[str, dict] = {}
    for t in all_trades:
        sym = t.get("product_id") or "<unknown>"
        b = by_product.setdefault(sym, {
            "symbol": sym,
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "n_trades": 0,
            "n_buy": 0, "n_sell": 0,
            "entry_volume_usd": 0.0,
            "quantity_open": 0.0,
            "market_value_usd": 0.0,
        })
        if t.get("pnl_usd") is not None:
            b["realized_pnl_usd"] += t["pnl_usd"]
            b["n_trades"] += 1
        if t.get("side") == "BUY":
            b["n_buy"] += 1
            b["entry_volume_usd"] += (t.get("amount_usd") or 0)
        elif t.get("side") == "SELL":
            b["n_sell"] += 1
    for p in open_pos_raw:
        sym = p.get("product_id") or "<unknown>"
        b = by_product.setdefault(sym, {
            "symbol": sym,
            "realized_pnl_usd": 0.0,
            "unrealized_pnl_usd": 0.0,
            "n_trades": 0, "n_buy": 0, "n_sell": 0,
            "entry_volume_usd": 0.0,
            "quantity_open": 0.0,
            "market_value_usd": 0.0,
        })
        b["unrealized_pnl_usd"] += (p.get("unrealized_pnl_usd") or 0)
        b["quantity_open"] += (p.get("quantity") or 0)
        b["market_value_usd"] += ((p.get("quantity") or 0) *
                                    (p.get("market_price") or 0))
    for b in by_product.values():
        b["total_pnl_usd"] = b["realized_pnl_usd"] + b["unrealized_pnl_usd"]
    by_product_list = sorted(
        by_product.values(),
        key=lambda x: abs(x["total_pnl_usd"]),
        reverse=True,
    )

    return {
        "trades": all_trades_sorted,
        "open_positions": open_pos_raw,
        "open_orders": open_orders_all,
        "by_strategy": by_strategy,
        "by_broker": by_broker,
        "by_product": by_product_list,
        "equity_curve": eq,
        "summary": summary,
    }


def load_orchestrator_state() -> dict:
    """Pull current strategy pod state + risk snapshot from the orchestrator
    DBs. Returns an empty/default payload if the orchestrator hasn't run yet
    (e.g. on first dashboard build before the new system has cycled)."""
    out = {
        "strategies": [],
        "risk": None,
        "lifecycle_events": [],
    }
    if not _HAS_ORCHESTRATOR:
        return out

    # Allocator state
    try:
        registry = StrategyRegistry()
        latest = registry.latest_allocations()
        for name, row in latest.items():
            out["strategies"].append({
                "name": name,
                "state": row.get("state"),
                "target_pct": float(row.get("target_pct") or 0),
                "target_usd": float(row.get("target_usd") or 0),
                "sharpe": row.get("sharpe"),
                "drawdown_pct": row.get("drawdown_pct"),
                "reason": row.get("reason"),
                "timestamp": row.get("timestamp"),
            })
        out["lifecycle_events"] = registry.lifecycle_events(limit=20)
    except Exception as exc:
        logger.warning(f"Could not load allocator state: {exc}")

    # Risk state — read the latest snapshot AND the full series so the
    # dashboard can render a mark-to-market equity / drawdown curve.
    try:
        import sqlite3
        risk_db = os.environ.get("RISK_DB_PATH", "data/risk_state.db")
        if Path(risk_db).exists():
            conn = sqlite3.connect(risk_db)
            conn.row_factory = sqlite3.Row
            # Latest snapshot
            row = conn.execute(
                "SELECT timestamp, equity_usd, note FROM equity_snapshots "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            peak = conn.execute(
                "SELECT MAX(equity_usd) AS peak FROM equity_snapshots"
            ).fetchone()
            ks = conn.execute(
                "SELECT timestamp, state, drawdown_pct, note FROM kill_switch_events "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            # Full history of kill-switch trips (helps spot patterns)
            ks_history_rows = conn.execute(
                "SELECT timestamp, state, drawdown_pct, note "
                "FROM kill_switch_events ORDER BY id DESC LIMIT 30"
            ).fetchall()
            # Equity snapshot series — cap at 1000 points to keep the
            # dashboard payload small. 1k 5-min snapshots is ~3.5 days
            # of history, plenty for the current view.
            series_rows = conn.execute(
                "SELECT timestamp, equity_usd FROM equity_snapshots "
                "ORDER BY id DESC LIMIT 1000"
            ).fetchall()
            conn.close()

            equity_series = [
                {"t": r["timestamp"], "equity": float(r["equity_usd"])}
                for r in reversed(series_rows)  # oldest → newest
            ]
            # Drawdown series derived in one pass
            dd_series = []
            running_peak = 0.0
            for pt in equity_series:
                if pt["equity"] > running_peak:
                    running_peak = pt["equity"]
                dd_pct = ((running_peak - pt["equity"]) / running_peak * 100
                          if running_peak > 0 else 0.0)
                dd_series.append({"t": pt["t"], "dd_pct": dd_pct})

            # Day P&L: equity now vs. earliest snapshot in last 24h
            day_pnl = None
            if equity_series:
                latest_ts_str = equity_series[-1]["t"]
                try:
                    latest_ts = datetime.fromisoformat(
                        latest_ts_str.replace("Z", "+00:00")
                    )
                    day_ago = latest_ts.timestamp() - 86400
                    last_eq = equity_series[-1]["equity"]
                    # Walk back to the first sample older than 24h
                    base_eq = None
                    for pt in equity_series:
                        try:
                            t = datetime.fromisoformat(
                                pt["t"].replace("Z", "+00:00")
                            ).timestamp()
                        except Exception:
                            continue
                        if t >= day_ago:
                            base_eq = pt["equity"]
                            break
                    if base_eq is not None and base_eq > 0:
                        day_pnl = {
                            "absolute_usd": last_eq - base_eq,
                            "pct": (last_eq - base_eq) / base_eq * 100,
                        }
                except Exception:
                    pass

            if row:
                eq = float(row["equity_usd"])
                pk = float(peak["peak"]) if peak and peak["peak"] else eq
                out["risk"] = {
                    "equity_usd": eq,
                    "peak_equity_usd": pk,
                    "drawdown_pct": (pk - eq) / pk if pk > 0 else 0.0,
                    "last_snapshot": row["timestamp"],
                    "last_kill_switch_state": ks["state"] if ks else "NORMAL",
                    "last_kill_switch_at": ks["timestamp"] if ks else None,
                    "equity_series": equity_series,
                    "drawdown_series": dd_series,
                    "kill_switch_history": [dict(r) for r in ks_history_rows],
                    "day_pnl": day_pnl,
                }
    except Exception as exc:
        logger.warning(f"Could not load risk state: {exc}")

    return out


def load_latest_strategic_review() -> dict:
    """Pull the most-recent Opus review from data/strategic_review.db."""
    out: dict = {}
    db_path = os.environ.get("REVIEW_DB_PATH", "data/strategic_review.db")
    if not Path(db_path).exists():
        return out
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT 1"
        ).fetchone()
        history = conn.execute(
            "SELECT id, timestamp, overall_health, summary FROM reviews "
            "ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        if row:
            payload = {}
            try:
                payload = json.loads(row["payload_json"])
            except Exception:
                pass
            out = {
                "timestamp": row["timestamp"],
                "overall_health": row["overall_health"],
                "summary": row["summary"],
                "risk_multiplier_rec": row["risk_mult_rec"],
                "risk_multiplier_reason": row["risk_mult_reason"],
                "model_used": row["model_used"],
                "cost_usd": row["cost_usd"],
                "strategy_actions": payload.get("strategy_actions", []),
                "investigate": payload.get("investigate", []),
                "history": [dict(r) for r in history],
            }
    except Exception as exc:
        logger.warning(f"Could not load strategic review: {exc}")
    return out


def _empty_summary(label: str) -> dict:
    return {
        "label": label,
        "n_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl_usd": 0.0,
        "entry_volume_usd": 0.0,
        "return_on_volume_pct": 0.0,
        "avg_pnl_usd": 0.0,   # JS reads this on every row; missing field caused undefined.toFixed
    }


def _summarize(trades: list[dict]) -> dict:
    """Single source of truth for trade aggregates.

    Replaces 4 near-identical aggregation blocks that drifted apart over
    time (per-strategy live, headline live, per-strategy backtest,
    backtest merge). Computes wins / losses / win rate / total P&L /
    entry volume / return on volume / Sharpe / max drawdown from a list
    of trade dicts that may contain `pnl_usd=None` for not-yet-closed.
    """
    closed = [t for t in trades if t.get("pnl_usd") is not None]
    n = len(closed)
    wins = sum(1 for t in closed if t["pnl_usd"] > 0)
    losses = n - wins
    total_pnl = sum(t["pnl_usd"] for t in closed)
    avg_pnl = (total_pnl / n) if n else 0.0
    entry_volume = sum(
        (t.get("amount_usd") or 0) for t in trades if t.get("side") == "BUY"
    )
    rov = (total_pnl / entry_volume * 100) if entry_volume > 0 else 0.0
    sharpe = 0.0
    max_dd = 0.0
    if n >= 3:
        pnls = [t["pnl_usd"] for t in closed]
        mu = total_pnl / n
        var = sum((p - mu) ** 2 for p in pnls) / max(n - 1, 1)
        std = var ** 0.5
        sharpe = (mu / std * (n ** 0.5)) if std > 0 else 0.0
        cum = 0.0
        peak = 0.0
        for p in pnls:
            cum += p
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd
    return {
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / n) if n else 0.0,
        "total_pnl_usd": total_pnl,
        "avg_pnl_usd": avg_pnl,
        "entry_volume_usd": entry_volume,
        "return_on_volume_pct": rov,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def _broker_snapshot(venue: str, adapter) -> dict:
    """Pull account + positions + open orders for one broker.

    Designed to be called concurrently from a ThreadPoolExecutor — each
    venue is independent. Catches errors per-call so one broker failure
    doesn't poison the whole snapshot.
    """
    entry: dict = {
        "venue": venue,
        "cash_usd": 0.0,
        "buying_power_usd": 0.0,
        "equity_usd": 0.0,
        "invested_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "n_positions": 0,
        "n_open_orders": 0,
        "available_pct": 0.0,
        "last_ok_at": None,
        "error": None,
        "positions": [],
        "open_orders": [],
    }
    errs = []
    # Account
    try:
        acct = adapter.get_account()
        entry["cash_usd"] = float(acct.cash_usd or 0)
        entry["buying_power_usd"] = float(acct.buying_power_usd or 0)
        entry["equity_usd"] = float(acct.equity_usd or 0)
    except Exception as e:
        errs.append(f"account: {type(e).__name__}: {str(e)[:80]}")
    # Positions
    try:
        positions = adapter.get_positions() or []
    except Exception as e:
        positions = []
        errs.append(f"positions: {type(e).__name__}: {str(e)[:80]}")
    for p in positions:
        mkt = (p.quantity or 0) * (p.market_price or 0)
        entry["invested_usd"] += mkt
        entry["unrealized_pnl_usd"] += (p.unrealized_pnl_usd or 0)
        entry["n_positions"] += 1
        entry["positions"].append({
            "symbol": p.symbol,
            "quantity": p.quantity,
            "entry_price": p.avg_entry_price,
            "market_price": p.market_price,
            "market_value_usd": mkt,
            "cost_basis_usd": p.quantity * p.avg_entry_price,
            "unrealized_pnl_usd": p.unrealized_pnl_usd,
            "venue": venue,
        })
    # Open orders (used for the new "Pending orders" panel)
    if hasattr(adapter, "get_open_orders"):
        try:
            opens = adapter.get_open_orders() or []
            for o in opens:
                entry["open_orders"].append({
                    "venue": venue,
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                    "quantity": o.quantity,
                    "notional_usd": o.notional_usd,
                    "limit_price": o.limit_price,
                    "status": (o.status.value if hasattr(o.status, "value")
                                else str(o.status)),
                    "submitted_at": (o.submitted_at.isoformat()
                                      if o.submitted_at else None),
                })
            entry["n_open_orders"] = len(opens)
        except Exception as e:
            errs.append(f"open_orders: {type(e).__name__}: {str(e)[:80]}")
    if entry["equity_usd"] > 0:
        entry["available_pct"] = entry["cash_usd"] / entry["equity_usd"] * 100
    if not errs:
        entry["last_ok_at"] = datetime.now(UTC).isoformat()
    else:
        entry["error"] = " | ".join(errs)[:300]
    return entry


# ─── Backtest data ────────────────────────────────────────────────────────────


def run_backtest_window(days: int, candles_by_product: dict) -> dict:
    """Run all 3 strategies on each product for the past `days` days."""
    strategies = make_strategies(GRANULARITY)

    by_strategy: dict[str, dict] = {}
    all_trades: list[dict] = []
    combined_eq_points: list[dict] = []

    cutoff_ts = time.time() - days * 86400

    for strat in strategies:
        all_results = []
        for product_id in PRODUCTS:
            full_candles = candles_by_product.get(product_id)
            if full_candles is None or len(full_candles) == 0:
                continue
            # Slice candles to the window (need lookback before window start)
            # Find start index where candle_ts >= cutoff_ts
            ts_col = full_candles[:, 0]
            start_idx = max(0, int((ts_col >= cutoff_ts).argmax()) - strat.lookback)
            if start_idx >= len(full_candles) - strat.lookback:
                continue
            window_candles = full_candles[start_idx:]
            res = backtest_strategy(
                strat,
                product_id,
                window_candles,
                window_days=days,
                max_trade_usd=MAX_TRADE_USD,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
                cooldown_seconds=COOLDOWN_SECONDS,
                min_confidence=MIN_CONFIDENCE,
                fee_bps=FEE_BPS,
            )
            all_results.append(res)
            for t in res.trades:
                d = trade_to_dict(t)
                d["strategy"] = strat.name
                all_trades.append(d)

        # Aggregate this strategy
        flat_trades = [t for r in all_results for t in r.trades]
        closed = [t for t in flat_trades if t.pnl_usd is not None]
        total_pnl = sum(t.pnl_usd for t in closed) if closed else 0.0
        entry_volume = sum(t.amount_usd for t in flat_trades)
        wins = sum(1 for t in closed if t.pnl_usd > 0)

        by_strategy[strat.name] = {
            "summary": {
                "strategy": strat.name,
                "n_trades": len(closed),
                "wins": wins,
                "losses": len(closed) - wins,
                "win_rate": (wins / len(closed)) if closed else 0.0,
                "total_pnl_usd": total_pnl,
                "entry_volume_usd": entry_volume,
                "return_on_volume_pct": (
                    total_pnl / entry_volume * 100 if entry_volume > 0 else 0.0
                ),
                "avg_pnl_usd": (total_pnl / len(closed)) if closed else 0.0,
            },
            "trades": [trade_to_dict(t) for t in flat_trades],
        }

    # Build combined equity curve
    closed_all = [t for t in all_trades if t.get("pnl_usd") is not None]
    closed_all.sort(key=lambda t: t.get("close_time") or "")
    cum = 0.0
    for t in closed_all:
        cum += t["pnl_usd"]
        combined_eq_points.append({"t": t["close_time"], "pnl_cumulative": cum})

    # Top-level summary
    total_pnl = sum(t["pnl_usd"] for t in closed_all)
    entry_volume = sum(t["amount_usd"] for t in all_trades)
    wins = sum(1 for t in closed_all if t["pnl_usd"] > 0)

    summary = {
        "label": f"{days}-day backtest",
        "n_trades": len(closed_all),
        "wins": wins,
        "losses": len(closed_all) - wins,
        "win_rate": (wins / len(closed_all)) if closed_all else 0.0,
        "total_pnl_usd": total_pnl,
        "entry_volume_usd": entry_volume,
        "return_on_volume_pct": (
            total_pnl / entry_volume * 100 if entry_volume > 0 else 0.0
        ),
    }

    return {
        "trades": all_trades,
        "by_strategy": by_strategy,
        "equity_curve": combined_eq_points,
        "summary": summary,
    }


# ─── HTML ─────────────────────────────────────────────────────────────────────


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Crypto Trading Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel-2: #1f242c;
    --border: #30363d; --text: #e6edf3; --muted: #7d8590;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; margin: 0; padding: 0; }
  header { padding: 24px 32px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
  h1 { margin: 0; font-size: 20px; font-weight: 600; }
  .meta { color: var(--muted); font-size: 13px; }
  .tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); padding: 0 32px; background: var(--panel); position: sticky; top: 0; z-index: 10; }
  .tab { padding: 14px 20px; cursor: pointer; color: var(--muted); border-bottom: 2px solid transparent; font-size: 14px; user-select: none; }
  .tab.active { color: var(--text); border-bottom-color: var(--blue); }
  .tab:hover { color: var(--text); }
  .panel-wrap { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
  .kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .kpi-value { font-size: 22px; font-weight: 600; }
  .kpi-sub { color: var(--muted); font-size: 12px; margin-top: 4px; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
  .chart-wrap { position: relative; height: 280px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .panel h2 { margin: 0 0 12px 0; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; }
  .pill.buy { background: rgba(63,185,80,0.15); color: var(--green); }
  .pill.sell { background: rgba(248,81,73,0.15); color: var(--red); }
  .pill.stop { background: rgba(248,81,73,0.15); color: var(--red); }
  .pill.tp { background: rgba(63,185,80,0.15); color: var(--green); }
  .pill.sig { background: rgba(88,166,255,0.15); color: var(--blue); }
  details { background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 8px 12px; }
  details summary { cursor: pointer; color: var(--muted); font-size: 13px; user-select: none; }
  details[open] { padding-bottom: 12px; }
  .scroll { max-height: 360px; overflow: auto; }
  .pods-wrap { padding: 16px 32px 0 32px; max-width: 1400px; margin: 0 auto; }
  .pods-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
  .pod { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .pod-name { font-size: 14px; font-weight: 600; margin-bottom: 4px; }
  .pod-state { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 500; margin-bottom: 8px; }
  .pod-state.ACTIVE { background: rgba(63,185,80,0.15); color: var(--green); }
  .pod-state.WATCH { background: rgba(218,165,32,0.15); color: #d4a64c; }
  .pod-state.FROZEN { background: rgba(248,81,73,0.15); color: var(--red); }
  .pod-state.RETIRED { background: rgba(125,133,144,0.15); color: var(--muted); }
  .pod-row { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
  .pod-row .label { color: var(--muted); }
  .risk-banner { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; display: flex; gap: 24px; flex-wrap: wrap; }
  .risk-banner .item { display: flex; gap: 8px; font-size: 13px; }
  .risk-banner .item .label { color: var(--muted); }
</style>
</head>
<body>
<header>
  <div>
    <h1>Crypto Trading Dashboard</h1>
    <div class="meta">Generated <span id="generated-at"></span> · <a style="color: var(--blue); text-decoration: none;" id="repo-link" href="#">View workflow</a></div>
  </div>
  <div class="meta">
    Stop: <span id="cfg-stop"></span> · Take: <span id="cfg-tp"></span> · Cooldown: <span id="cfg-cd"></span> · Max/trade: <span id="cfg-mt"></span>
  </div>
</header>
<div id="pods-banner"></div>
<div id="strategic-review-banner"></div>
<div id="strategy-pnl-overview"></div>
<div class="tabs" id="tabs"></div>
<div id="panels"></div>

<script>
const DATA = __DATA__;
const TABS = ["live", "d7", "d15", "d30"];
const TAB_LABELS = {live: "Live", d7: "7-day backtest", d15: "15-day backtest", d30: "30-day backtest"};

const fmtUSD = (v) => (v == null) ? "—" : (v >= 0 ? "+$" : "-$") + Math.abs(v).toFixed(2);
const fmtPct = (v) => (v == null) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
const fmtTime = (iso) => { if (!iso) return "—"; const d = new Date(iso); return d.toISOString().slice(0,16).replace("T"," "); };
const cls = (v) => (v == null) ? "" : (v >= 0 ? "pos" : "neg");

document.getElementById("generated-at").textContent = fmtTime(DATA.generated_at);
document.getElementById("cfg-stop").textContent = (DATA.config.stop_loss_pct * 100).toFixed(1) + "%";
document.getElementById("cfg-tp").textContent = (DATA.config.take_profit_pct * 100).toFixed(1) + "%";
document.getElementById("cfg-cd").textContent = (DATA.config.cooldown_seconds / 60) + " min";
document.getElementById("cfg-mt").textContent = "$" + DATA.config.max_trade_usd.toFixed(0);
const repo = DATA.config.repo;
document.getElementById("repo-link").href = repo ? `https://github.com/${repo}/actions` : "#";

const tabsEl = document.getElementById("tabs");
TABS.forEach(t => {
  const el = document.createElement("div");
  el.className = "tab" + (t === "live" ? " active" : "");
  el.textContent = TAB_LABELS[t];
  el.dataset.tab = t;
  el.onclick = () => activate(t);
  tabsEl.appendChild(el);
});

const panelsEl = document.getElementById("panels");
const charts = {};

function activate(tab) {
  document.querySelectorAll(".tab").forEach(e => e.classList.toggle("active", e.dataset.tab === tab));
  render(tab);
}

function kpi(label, value, sub, klass) {
  return `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value ${klass||""}">${value}</div>${sub ? `<div class="kpi-sub">${sub}</div>` : ""}</div>`;
}

function render(tab) {
  const d = DATA[tab];
  const s = d.summary;
  const isLive = tab === "live";

  let html = `<div class="panel-wrap">`;

  // ─── System health banner (live tab only) ────────────────────────
  // Shows whether the orchestrator + brokers are healthy. Green when
  // all brokers are reachable and the orchestrator cycled in the last
  // 15 min; yellow if stale; red if any broker errored.
  if (isLive) {
    const now = Date.now();
    const lastRiskTs = (DATA.pods?.risk?.last_snapshot) || null;
    const orchAgeMin = lastRiskTs
      ? Math.floor((now - new Date(lastRiskTs).getTime()) / 60000)
      : null;
    const brokers = Object.values(DATA.live.by_broker || {});
    const errBrokers = brokers.filter(b => b.error);
    const okBrokers = brokers.filter(b => !b.error);
    let healthClass = "pos";
    let healthText = "All systems operational";
    if (errBrokers.length) {
      healthClass = "neg";
      healthText = `${errBrokers.length} broker error${errBrokers.length>1?"s":""}: ${errBrokers.map(b=>b.venue).join(", ")}`;
    } else if (orchAgeMin !== null && orchAgeMin > 15) {
      healthClass = "neg";
      healthText = `Orchestrator stale (last cycle ${orchAgeMin} min ago)`;
    } else if (orchAgeMin !== null && orchAgeMin > 8) {
      healthClass = "";
      healthText = `Orchestrator: last cycle ${orchAgeMin} min ago`;
    }
    const orchStatus = (orchAgeMin === null)
      ? "no cycle data"
      : `last cycle ${orchAgeMin} min ago`;
    html += `<div class="panel" style="margin-bottom: 12px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
          <strong class="${healthClass}">● ${healthText}</strong>
          <div style="color: var(--muted); font-size: 12px; margin-top: 4px;">
            Orchestrator: ${orchStatus} · Brokers OK: ${okBrokers.length}/${brokers.length}
          </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">`;
    brokers.forEach(b => {
      const ageStr = b.last_ok_at
        ? `${Math.floor((now - new Date(b.last_ok_at).getTime())/60000)}m ago`
        : "—";
      const klass = b.error ? "stop" : "tp";
      const tip = b.error ? b.error : `last OK: ${ageStr}`;
      html += `<span class="pill ${klass}" title="${tip}">${b.venue}: ${b.error ? "err" : ageStr}</span>`;
    });
    html += `</div></div></div>`;
  }

  html += `<div class="kpi-row">`;
  if (isLive) {
    // Live tab: show realized + unrealized broken out so the user sees
    // both "what closed trades have netted" and "what open positions
    // are doing right now". Total P&L is the sum.
    const totalPnl = s.total_pnl_usd;
    const realPnl  = (s.realized_pnl_usd ?? s.total_pnl_usd);
    const unrPnl   = (s.unrealized_pnl_usd ?? 0);
    const mktVal   = (s.market_value_usd ?? 0);
    const nOpen    = (s.n_open_positions ?? DATA.live.open_positions.length);
    const dayPnl   = DATA.pods?.risk?.day_pnl;
    html += kpi("Total P&L",
                fmtUSD(totalPnl),
                `realized + unrealized`,
                cls(totalPnl));
    if (dayPnl) {
      html += kpi("Day P&L",
                  fmtUSD(dayPnl.absolute_usd),
                  `${dayPnl.pct >= 0 ? "+" : ""}${dayPnl.pct.toFixed(2)}% in 24h`,
                  cls(dayPnl.absolute_usd));
    }
    html += kpi("Unrealized P&L",
                fmtUSD(unrPnl),
                `${nOpen} open position${nOpen===1?"":"s"} · $${mktVal.toFixed(0)} market value`,
                cls(unrPnl));
    html += kpi("Realized P&L",
                fmtUSD(realPnl),
                `${s.n_trades} closed trade${s.n_trades===1?"":"s"}`,
                cls(realPnl));
    html += kpi("Win rate", (s.win_rate * 100).toFixed(1) + "%", `${s.wins}W / ${s.losses}L`);
    // Drawdown card
    const dd = DATA.pods?.risk?.drawdown_pct;
    if (dd !== null && dd !== undefined) {
      html += kpi("Drawdown",
                  (dd * 100).toFixed(2) + "%",
                  "from peak equity",
                  dd > 0.05 ? "neg" : "");
    }
  } else {
    html += kpi("Total P&L", fmtUSD(s.total_pnl_usd), `${s.n_trades} closed trades`, cls(s.total_pnl_usd));
    html += kpi("Return on volume", fmtPct(s.return_on_volume_pct), `over $${s.entry_volume_usd.toFixed(0)} traded`, cls(s.return_on_volume_pct));
    html += kpi("Win rate", (s.win_rate * 100).toFixed(1) + "%", `${s.wins}W / ${s.losses}L`);
    html += kpi("Trade volume", "$" + s.entry_volume_usd.toFixed(0), `entry notional`);
  }
  html += `</div>`;

  // ─── Equity curve + drawdown chart (live tab only) ────────────────
  // Plots the full mark-to-market equity series from risk_state.db
  // alongside the underwater (drawdown) curve. This is the single
  // chart most useful for spotting whether the bot is gaining money,
  // bleeding money, or stuck flat — and for catching drawdown trips.
  if (isLive && DATA.pods?.risk?.equity_series?.length > 1) {
    const eq = DATA.orchestrator.risk.equity_series;
    const ddSeries = DATA.orchestrator.risk.drawdown_series || [];
    html += `<div class="panel">
      <h2>Mark-to-market equity</h2>
      <div style="position: relative; height: 280px;">
        <canvas id="mtm-equity-chart"></canvas>
      </div>
    </div>`;
    if (ddSeries.length > 1) {
      html += `<div class="panel">
        <h2>Drawdown (underwater curve)</h2>
        <div style="position: relative; height: 200px;">
          <canvas id="mtm-dd-chart"></canvas>
        </div>
      </div>`;
    }
    // Defer chart creation until the canvas is in the DOM.
    setTimeout(() => {
      const eqCtx = document.getElementById("mtm-equity-chart");
      if (eqCtx) {
        new Chart(eqCtx, {
          type: "line",
          data: {
            labels: eq.map(p => p.t.slice(0,16).replace("T"," ")),
            datasets: [{
              label: "Equity ($)",
              data: eq.map(p => p.equity),
              borderColor: "#4ec9b0",
              backgroundColor: "rgba(78, 201, 176, 0.10)",
              fill: true, tension: 0.2, pointRadius: 0, borderWidth: 2,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { maxTicksLimit: 8, color: "#7d8590" }, grid: { color: "rgba(125,133,144,0.1)" }},
              y: { ticks: { color: "#7d8590", callback: v => "$"+v.toFixed(0) }, grid: { color: "rgba(125,133,144,0.1)" }},
            },
          },
        });
      }
      const ddCtx = document.getElementById("mtm-dd-chart");
      if (ddCtx) {
        new Chart(ddCtx, {
          type: "line",
          data: {
            labels: ddSeries.map(p => p.t.slice(0,16).replace("T"," ")),
            datasets: [{
              label: "Drawdown (%)",
              data: ddSeries.map(p => -p.dd_pct),  // negative for visual
              borderColor: "#f47174",
              backgroundColor: "rgba(244, 113, 116, 0.18)",
              fill: true, tension: 0.2, pointRadius: 0, borderWidth: 2,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { maxTicksLimit: 8, color: "#7d8590" }, grid: { color: "rgba(125,133,144,0.1)" }},
              y: { ticks: { color: "#7d8590", callback: v => v.toFixed(1)+"%" }, grid: { color: "rgba(125,133,144,0.1)" }},
            },
          },
        });
      }
    }, 50);
  }

  // Per-broker breakdown (live tab only) — show cash available,
  // amount invested (market value of positions), and unrealized P&L
  // for each broker. Lets the user see at a glance how much capital
  // is sitting idle vs deployed on each venue.
  if (isLive && d.by_broker && Object.keys(d.by_broker).length) {
    html += `<div class="panel"><h2>By broker</h2>`;
    html += `<table><thead><tr>
      <th>Broker</th>
      <th class="num">Cash available</th>
      <th class="num">Buying power</th>
      <th class="num">Invested</th>
      <th class="num">Equity</th>
      <th class="num">Unrealized P&L</th>
      <th class="num">Positions</th>
      <th class="num">Utilisation</th>
    </tr></thead><tbody>`;
    Object.values(d.by_broker).forEach(b => {
      const utilPct = (b.equity_usd > 0)
        ? ((b.invested_usd / b.equity_usd) * 100)
        : 0;
      const errBadge = b.error
        ? ` <span class="pill stop" title="${b.error}">err</span>`
        : "";
      html += `<tr>
        <td><strong>${b.venue}</strong>${errBadge}</td>
        <td class="num">${fmtUSD(b.cash_usd)}</td>
        <td class="num">${fmtUSD(b.buying_power_usd)}</td>
        <td class="num">${fmtUSD(b.invested_usd)}</td>
        <td class="num">${fmtUSD(b.equity_usd)}</td>
        <td class="num ${cls(b.unrealized_pnl_usd)}">${fmtUSD(b.unrealized_pnl_usd)}</td>
        <td class="num">${b.n_positions}</td>
        <td class="num">${utilPct.toFixed(1)}%</td>
      </tr>`;
    });
    // Totals row
    const tot = Object.values(d.by_broker).reduce((a, b) => ({
      cash_usd: a.cash_usd + b.cash_usd,
      buying_power_usd: a.buying_power_usd + b.buying_power_usd,
      invested_usd: a.invested_usd + b.invested_usd,
      equity_usd: a.equity_usd + b.equity_usd,
      unrealized_pnl_usd: a.unrealized_pnl_usd + b.unrealized_pnl_usd,
      n_positions: a.n_positions + b.n_positions,
    }), {cash_usd:0, buying_power_usd:0, invested_usd:0, equity_usd:0,
        unrealized_pnl_usd:0, n_positions:0});
    const totUtil = (tot.equity_usd > 0)
      ? ((tot.invested_usd / tot.equity_usd) * 100)
      : 0;
    html += `<tr style="border-top: 2px solid var(--border); font-weight: 600;">
      <td>TOTAL</td>
      <td class="num">${fmtUSD(tot.cash_usd)}</td>
      <td class="num">${fmtUSD(tot.buying_power_usd)}</td>
      <td class="num">${fmtUSD(tot.invested_usd)}</td>
      <td class="num">${fmtUSD(tot.equity_usd)}</td>
      <td class="num ${cls(tot.unrealized_pnl_usd)}">${fmtUSD(tot.unrealized_pnl_usd)}</td>
      <td class="num">${tot.n_positions}</td>
      <td class="num">${totUtil.toFixed(1)}%</td>
    </tr>`;
    html += `</tbody></table></div>`;
  }

  // ─── Pending orders (live tab only) ──────────────────────────────
  // Surfaces every open order across brokers so the user can see
  // what's actually queued. Stale orders (>30 min) are highlighted —
  // the orchestrator auto-cancels them but a pile here means a
  // strategy is firing too aggressively or the broker is stuck.
  if (isLive && DATA.live.open_orders && DATA.live.open_orders.length) {
    const ords = DATA.live.open_orders.slice().sort((a,b) =>
      (b.submitted_at || "").localeCompare(a.submitted_at || ""));
    html += `<div class="panel"><h2>Pending orders <span style="color:var(--muted); font-weight:400; font-size:14px;">(${ords.length})</span></h2>`;
    html += `<table><thead><tr>
      <th>Broker</th><th>Symbol</th><th>Side</th>
      <th class="num">Qty / Notional</th><th class="num">Limit</th>
      <th>Status</th><th>Submitted</th><th>Age</th>
    </tr></thead><tbody>`;
    const now = Date.now();
    ords.slice(0, 50).forEach(o => {
      const age = o.submitted_at
        ? Math.floor((now - new Date(o.submitted_at).getTime()) / 60000)
        : null;
      const stale = age !== null && age > 30;
      const sideKlass = (o.side === "BUY") ? "pos" : "neg";
      const qtyOrNotional = o.notional_usd
        ? fmtUSD(o.notional_usd)
        : (o.quantity != null ? o.quantity.toFixed(4) : "—");
      html += `<tr${stale ? ' style="background: rgba(244,113,116,0.06);"' : ''}>
        <td>${o.venue}</td>
        <td><strong>${o.symbol}</strong></td>
        <td class="${sideKlass}">${o.side}</td>
        <td class="num">${qtyOrNotional}</td>
        <td class="num">${o.limit_price ? "$"+o.limit_price.toFixed(2) : "MKT"}</td>
        <td><span class="pill" style="background: rgba(125,133,144,0.15); color: var(--muted);">${o.status}</span></td>
        <td>${o.submitted_at ? fmtTime(o.submitted_at) : "—"}</td>
        <td class="num ${stale ? 'neg' : ''}">${age !== null ? age + "m" : "—"}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
    if (ords.length > 50) {
      html += `<div style="color: var(--muted); font-size: 12px; padding: 8px;">Showing 50 of ${ords.length} pending orders.</div>`;
    }
    html += `</div>`;
  }

  // ─── P&L by product / symbol (live tab only) ──────────────────────
  // Closes the "is BTC up while ETH bleeds?" hole. Combines realized
  // P&L from closed trades with unrealized from open positions, ranks
  // by absolute total P&L so winners and biggest losers float to top.
  if (isLive && DATA.live.by_product && DATA.live.by_product.length) {
    const prods = DATA.live.by_product
      .filter(p => p.total_pnl_usd !== 0 || p.market_value_usd > 0 || p.n_trades > 0);
    if (prods.length) {
      html += `<div class="panel"><h2>P&L by product</h2>`;
      html += `<table><thead><tr>
        <th>Symbol</th>
        <th class="num">Realized</th>
        <th class="num">Unrealized</th>
        <th class="num">Total P&L</th>
        <th class="num">Open qty</th>
        <th class="num">Market value</th>
        <th class="num">Trades</th>
        <th class="num">Volume</th>
      </tr></thead><tbody>`;
      prods.slice(0, 25).forEach(p => {
        html += `<tr>
          <td><strong>${p.symbol}</strong></td>
          <td class="num ${cls(p.realized_pnl_usd)}">${fmtUSD(p.realized_pnl_usd)}</td>
          <td class="num ${cls(p.unrealized_pnl_usd)}">${fmtUSD(p.unrealized_pnl_usd)}</td>
          <td class="num ${cls(p.total_pnl_usd)}">${fmtUSD(p.total_pnl_usd)}</td>
          <td class="num">${p.quantity_open ? p.quantity_open.toFixed(4) : "—"}</td>
          <td class="num">${p.market_value_usd ? fmtUSD(p.market_value_usd).replace(/^\+\$/,"$") : "—"}</td>
          <td class="num">${p.n_trades} (${p.n_buy}B/${p.n_sell}S)</td>
          <td class="num">$${p.entry_volume_usd.toFixed(0)}</td>
        </tr>`;
      });
      if (prods.length > 25) {
        html += `<tr><td colspan="8" style="color: var(--muted); font-size: 12px; text-align: center;">... and ${prods.length - 25} more</td></tr>`;
      }
      html += `</tbody></table></div>`;
    }
  }

  // Per-strategy P&L overview (always visible; cards)
  html += `<div class="panel"><h2>P&L by strategy</h2>`;
  html += `<div class="kpi-row" style="grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));">`;
  Object.values(d.by_strategy).forEach(st => {
    const x = st.summary;
    const sub = `${x.n_trades} closed · ${(x.win_rate * 100).toFixed(0)}% win`;
    html += kpi(x.strategy, fmtUSD(x.total_pnl_usd), sub, cls(x.total_pnl_usd));
  });
  html += `</div></div>`;

  // Build a name → lifecycle-state lookup so each row shows whether
  // the strategy is LIVE / DRY / FROZEN / RETIRED.
  const podByName = {};
  (DATA.pods?.strategies || []).forEach(p => { podByName[p.name] = p; });
  const lifecyclePill = (state) => {
    if (!state) return "";
    const klass = state === "ACTIVE" ? "tp"
                : state === "FROZEN" ? "stop"
                : state === "RETIRED" ? "stop"
                : "";
    return `<span class="pill ${klass}" style="margin-left:6px;font-size:10px;">${state}</span>`;
  };

  // Per-strategy summary table + EXPANDABLE TRADE DETAILS per row
  html += `<div class="panel"><h2>By strategy — click a row to see every trade</h2>`;
  html += `<table><thead><tr>
    <th>Strategy</th><th>State</th><th>Trades</th><th>Win rate</th><th class="num">Total P&L</th>
    <th class="num">Volume</th><th class="num">Return on volume</th><th class="num">Avg/trade</th></tr></thead><tbody>`;
  const stratList = Object.values(d.by_strategy);
  stratList.forEach((st, idx) => {
    const x = st.summary;
    const safeId = `${tab}-strat-${idx}`;
    const pod = podByName[x.strategy];
    const stateCell = pod
      ? `${lifecyclePill(pod.state)} <span style="color:var(--muted);font-size:11px;">${(pod.target_pct*100).toFixed(1)}%</span>`
      : `<span style="color:var(--muted);font-size:11px;">—</span>`;
    html += `<tr style="cursor: pointer;" onclick="document.getElementById('${safeId}').open = !document.getElementById('${safeId}').open">
      <td><strong>${x.strategy}</strong></td>
      <td>${stateCell}</td>
      <td>${x.n_trades} (${x.wins}W/${x.losses}L)</td>
      <td>${(x.win_rate * 100).toFixed(1)}%</td>
      <td class="num ${cls(x.total_pnl_usd)}">${fmtUSD(x.total_pnl_usd)}</td>
      <td class="num">$${x.entry_volume_usd.toFixed(2)}</td>
      <td class="num ${cls(x.return_on_volume_pct)}">${fmtPct(x.return_on_volume_pct)}</td>
      <td class="num ${cls(x.avg_pnl_usd)}">${fmtUSD(x.avg_pnl_usd)}</td>
    </tr>`;
    // Expandable trade detail row
    const stTrades = (st.trades || []).slice().sort((a,b) => {
      const ta = a.close_time || a.open_time || a.timestamp;
      const tb = b.close_time || b.open_time || b.timestamp;
      return tb.localeCompare(ta);
    });
    html += `<tr><td colspan="8" style="padding: 0; background: var(--panel-2);">
      <details id="${safeId}" style="border: none; background: transparent; padding: 0;">
        <summary style="padding: 10px 14px;">${stTrades.length} trade(s) for <strong>${x.strategy}</strong></summary>
        <div class="scroll" style="max-height: 280px;"><table style="margin: 0;"><thead><tr>
          <th>Time</th><th>Product</th><th>Side</th><th class="num">Price</th>
          <th class="num">Qty</th><th class="num">USD</th><th class="num">P&L</th><th>Reason</th></tr></thead><tbody>`;
    stTrades.slice(0, 200).forEach(t => {
      const time = fmtTime(t.close_time || t.open_time || t.timestamp);
      const side = t.side || (t.exit_reason ? "SELL" : "BUY");
      const price = t.exit_price ?? t.entry_price ?? t.price;
      const qty = t.quantity ?? 0;
      const usd = t.amount_usd ?? 0;
      const pnl = t.pnl_usd;
      let reason = t.exit_reason || t.reason || "";
      let rPill = "";
      if (reason === "stop_loss") rPill = `<span class="pill stop">STOP</span>`;
      else if (reason === "take_profit") rPill = `<span class="pill tp">TP</span>`;
      else if (reason === "signal") rPill = `<span class="pill sig">SIGNAL</span>`;
      else if (reason) rPill = `<span class="pill" style="background: rgba(125,133,144,0.15); color: var(--muted);">${reason}</span>`;
      html += `<tr>
        <td>${time}</td><td>${t.product_id || ""}</td>
        <td><span class="pill ${side === "BUY" ? "buy" : "sell"}">${side}</span></td>
        <td class="num">$${price ? Number(price).toFixed(4) : "—"}</td>
        <td class="num">${qty ? Number(qty).toFixed(6) : "—"}</td>
        <td class="num">$${Number(usd).toFixed(2)}</td>
        <td class="num ${cls(pnl)}">${fmtUSD(pnl)}</td>
        <td>${rPill}</td>
      </tr>`;
    });
    if (stTrades.length === 0) {
      html += `<tr><td colspan="8" style="padding: 12px; color: var(--muted); text-align: center;">No trades yet for this strategy.</td></tr>`;
    }
    html += `</tbody></table></div></details></td></tr>`;
  });
  html += `</tbody></table></div>`;

  // Charts
  html += `<div class="grid-2">
    <div class="panel"><h2>Equity curve (cumulative P&L)</h2><div class="chart-wrap"><canvas id="${tab}-eq"></canvas></div></div>
    <div class="panel"><h2>P&L by strategy</h2><div class="chart-wrap"><canvas id="${tab}-bar"></canvas></div></div>
  </div>`;

  // Open positions for live
  if (isLive && DATA.live.open_positions.length) {
    html += `<div class="panel"><h2>Open positions</h2><table><thead><tr>
      <th>Strategy</th><th>Product</th><th class="num">Quantity</th><th class="num">Cost basis</th><th class="num">Entry price</th><th>Entry time</th></tr></thead><tbody>`;
    DATA.live.open_positions.forEach(p => {
      html += `<tr><td>${p.strategy}</td><td>${p.product_id}</td>
        <td class="num">${p.quantity.toFixed(8)}</td>
        <td class="num">$${p.cost_basis_usd.toFixed(2)}</td>
        <td class="num">$${p.entry_price.toFixed(4)}</td>
        <td>${fmtTime(p.entry_time)}</td></tr>`;
    });
    html += `</tbody></table></div>`;
  }

  // Trades table (collapsed)
  html += `<div class="panel"><h2>Trades</h2><details${isLive && d.trades.length < 50 ? " open" : ""}>
    <summary>${d.trades.length} trades — click to expand</summary>
    <div class="scroll" style="margin-top: 12px;"><table><thead><tr>
      <th>Time</th><th>Strategy</th><th>Product</th><th>Side</th><th class="num">Price</th>
      <th class="num">Qty</th><th class="num">USD</th><th class="num">P&L</th><th>Reason</th></tr></thead><tbody>`;
  const tradesSorted = [...d.trades].sort((a, b) => {
    const ta = a.close_time || a.open_time || a.timestamp;
    const tb = b.close_time || b.open_time || b.timestamp;
    return tb.localeCompare(ta);
  });
  tradesSorted.slice(0, 500).forEach(t => {
    const time = fmtTime(t.close_time || t.open_time || t.timestamp);
    const side = t.side || (t.exit_reason ? "SELL" : "BUY");
    const price = t.exit_price ?? t.entry_price ?? t.price;
    const qty = t.quantity ?? 0;
    const usd = t.amount_usd ?? 0;
    const pnl = t.pnl_usd;
    let reason = t.exit_reason || t.reason || "";
    let rPill = "";
    if (reason === "stop_loss") rPill = `<span class="pill stop">STOP</span>`;
    else if (reason === "take_profit") rPill = `<span class="pill tp">TP</span>`;
    else if (reason === "signal") rPill = `<span class="pill sig">SIGNAL</span>`;
    html += `<tr>
      <td>${time}</td><td>${t.strategy || ""}</td><td>${t.product_id}</td>
      <td><span class="pill ${side === "BUY" ? "buy" : "sell"}">${side}</span></td>
      <td class="num">$${price ? Number(price).toFixed(4) : "—"}</td>
      <td class="num">${qty ? Number(qty).toFixed(6) : "—"}</td>
      <td class="num">$${Number(usd).toFixed(2)}</td>
      <td class="num ${cls(pnl)}">${fmtUSD(pnl)}</td>
      <td>${rPill}</td>
    </tr>`;
  });
  html += `</tbody></table></div></details></div>`;

  html += `</div>`;
  panelsEl.innerHTML = html;

  // Render charts
  Chart.defaults.color = "#7d8590";
  Chart.defaults.borderColor = "#30363d";

  const eqCtx = document.getElementById(`${tab}-eq`);
  if (eqCtx) {
    const labels = d.equity_curve.map(p => p.t);
    const data = d.equity_curve.map(p => p.pnl_cumulative);
    new Chart(eqCtx, {
      type: "line",
      data: {
        labels,
        datasets: [{ data, label: "Cumulative P&L", borderColor: "#58a6ff", backgroundColor: "rgba(88,166,255,0.1)", fill: true, tension: 0.1, pointRadius: 0 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { maxTicksLimit: 8, callback: function(v) { return fmtTime(this.getLabelForValue(v)); } }, grid: { color: "#30363d" } },
          y: { ticks: { callback: v => "$" + v.toFixed(2) }, grid: { color: "#30363d" } }
        }
      }
    });
  }

  const barCtx = document.getElementById(`${tab}-bar`);
  if (barCtx) {
    const names = Object.keys(d.by_strategy);
    const pnls = names.map(n => d.by_strategy[n].summary.total_pnl_usd);
    new Chart(barCtx, {
      type: "bar",
      data: {
        labels: names,
        datasets: [{ data: pnls, backgroundColor: pnls.map(v => v >= 0 ? "rgba(63,185,80,0.6)" : "rgba(248,81,73,0.6)"), borderColor: pnls.map(v => v >= 0 ? "#3fb950" : "#f85149"), borderWidth: 1 }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false } },
          y: { ticks: { callback: v => "$" + v.toFixed(2) }, grid: { color: "#30363d" } }
        }
      }
    });
  }
}

function renderPods() {
  const pods = DATA.pods || {};
  const strategies = pods.strategies || [];
  const risk = pods.risk;
  const wrap = document.getElementById("pods-banner");

  let html = `<div class="pods-wrap">`;

  if (!strategies.length && !risk) {
    html += `<div class="risk-banner" style="border-style: dashed;">`;
    html += `<div class="item"><span class="label">Multi-Asset Orchestrator:</span><strong>scaffolded · waiting for first live cycle</strong></div>`;
    html += `<div class="item"><span class="label">Strategies:</span>crypto_funding_carry · risk_parity_etf · kalshi_calibration_arb</div>`;
    html += `<div class="item"><span class="label">Run cadence:</span>every 5 min (DRY)</div>`;
    html += `</div></div>`;
    wrap.innerHTML = html;
    return;
  }


  if (risk) {
    const ddCls = risk.drawdown_pct >= 0.10 ? "neg" : (risk.drawdown_pct >= 0.05 ? "" : "pos");
    html += `<div class="risk-banner">`;
    html += `<div class="item"><span class="label">Equity:</span><strong>$${risk.equity_usd.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</strong></div>`;
    html += `<div class="item"><span class="label">Peak:</span><strong>$${risk.peak_equity_usd.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</strong></div>`;
    html += `<div class="item"><span class="label">Drawdown:</span><strong class="${ddCls}">${(risk.drawdown_pct * 100).toFixed(2)}%</strong></div>`;
    html += `<div class="item"><span class="label">Kill switch:</span><strong>${risk.last_kill_switch_state || "NORMAL"}</strong></div>`;
    if (risk.last_snapshot) {
      html += `<div class="item"><span class="label">Snapshot:</span>${fmtTime(risk.last_snapshot)}</div>`;
    }
    html += `</div>`;
  }

  if (strategies.length) {
    html += `<div class="pods-grid">`;
    strategies.forEach(s => {
      html += `<div class="pod">`;
      html += `<div class="pod-name">${s.name}</div>`;
      html += `<div class="pod-state ${s.state}">${s.state}</div>`;
      html += `<div class="pod-row"><span class="label">Target %</span><span>${(s.target_pct * 100).toFixed(1)}%</span></div>`;
      html += `<div class="pod-row"><span class="label">Target $</span><span>$${(s.target_usd || 0).toLocaleString(undefined,{maximumFractionDigits:0})}</span></div>`;
      if (s.sharpe != null) {
        html += `<div class="pod-row"><span class="label">Sharpe (60d)</span><span class="${s.sharpe >= 0 ? 'pos' : 'neg'}">${s.sharpe.toFixed(2)}</span></div>`;
      }
      if (s.drawdown_pct != null) {
        html += `<div class="pod-row"><span class="label">DD</span><span>${(s.drawdown_pct * 100).toFixed(1)}%</span></div>`;
      }
      if (s.reason) {
        html += `<div class="pod-row" style="font-size:11px;color:var(--muted);"><span>${s.reason}</span></div>`;
      }
      html += `</div>`;
    });
    html += `</div>`;
  }

  html += `</div>`;
  wrap.innerHTML = html;
}

renderPods();

function renderStrategicReview() {
  const wrap = document.getElementById("strategic-review-banner");
  const r = DATA.strategic_review || {};
  if (!r.timestamp) { wrap.innerHTML = ""; return; }

  const colors = {GREEN: "#3fb950", YELLOW: "#d4a64c", RED: "#f85149"};
  const color = colors[r.overall_health] || "#7d8590";

  const repo = (DATA.config && DATA.config.repo) || "marcoaduartemendes-source/AI-AT-ADVENT";
  const applyUrl = `https://github.com/${repo}/actions/workflows/apply_review.yml`;

  let html = `<div class="panel-wrap" style="padding-top: 12px;">`;
  html += `<div class="panel" style="border-left: 4px solid ${color};">`;
  html += `<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">`;
  html += `<h2 style="margin: 0;">Strategic Review · ${r.overall_health || "—"}</h2>`;
  html += `<a href="${applyUrl}" target="_blank" style="display: inline-block; padding: 8px 14px; background: var(--blue); color: white; border-radius: 6px; font-size: 13px; font-weight: 500; text-decoration: none;">⚙ Apply latest review →</a>`;
  html += `</div>`;
  html += `<div class="meta" style="margin: 4px 0 10px 0;">`;
  html += `${fmtTime(r.timestamp)} · `;
  html += `${r.model_used || "—"} · `;
  html += `risk multiplier rec: <strong>${r.risk_multiplier_rec ? r.risk_multiplier_rec.toFixed(2) : "1.00"}x</strong>`;
  if (r.cost_usd != null) html += ` · cost \$${r.cost_usd.toFixed(4)}`;
  html += `</div>`;

  if (r.summary) {
    html += `<p style="font-size: 14px; line-height: 1.5; margin: 0 0 12px 0;">${r.summary}</p>`;
  }
  if (r.risk_multiplier_reason) {
    html += `<p style="font-size: 13px; color: var(--muted); margin: 0 0 12px 0;">Risk multiplier rationale: ${r.risk_multiplier_reason}</p>`;
  }

  if (r.strategy_actions && r.strategy_actions.length) {
    html += `<details${r.overall_health === "RED" ? " open" : ""}>`;
    html += `<summary style="font-size: 13px;">${r.strategy_actions.length} strategy action(s) — click to expand</summary>`;
    html += `<table style="margin-top: 10px;"><thead><tr>
      <th>Strategy</th><th>Action</th><th class="num">Target %</th><th class="num">Conf</th><th>Reason</th></tr></thead><tbody>`;
    r.strategy_actions.forEach(a => {
      const tgt = (a.target_alloc_pct != null) ? (a.target_alloc_pct * 100).toFixed(1) + "%" : "—";
      const actClass = ({
        FREEZE: "neg", RETIRE: "neg", DECREASE: "neg",
        ACTIVATE: "pos", INCREASE: "pos",
      })[a.action] || "";
      html += `<tr><td><strong>${a.strategy || ""}</strong></td>
        <td><span class="pill" style="background: rgba(125,133,144,0.15); color: var(--text);">${a.action || "—"}</span></td>
        <td class="num">${tgt}</td>
        <td class="num">${a.confidence != null ? a.confidence.toFixed(2) : "—"}</td>
        <td style="font-size: 12px;">${a.reason || ""}</td></tr>`;
    });
    html += `</tbody></table></details>`;
  }

  if (r.investigate && r.investigate.length) {
    html += `<details style="margin-top: 8px;"><summary style="font-size: 13px;">Items to investigate (${r.investigate.length})</summary><ul style="margin: 8px 0; padding-left: 20px; font-size: 13px; color: var(--text);">`;
    r.investigate.forEach(item => { html += `<li>${item}</li>`; });
    html += `</ul></details>`;
  }

  html += `</div></div>`;
  wrap.innerHTML = html;
}

renderStrategicReview();

function renderStrategyPnLOverview() {
  // Always-visible banner: each strategy's lifetime LIVE P&L (real trades only)
  const wrap = document.getElementById("strategy-pnl-overview");
  const live = DATA.live || {};
  const byStrat = live.by_strategy || {};
  const names = Object.keys(byStrat);
  if (!names.length) { wrap.innerHTML = ""; return; }

  let totalPnl = 0, totalTrades = 0, totalWins = 0, totalVol = 0;
  names.forEach(n => {
    const s = byStrat[n].summary || {};
    totalPnl += (s.total_pnl_usd || 0);
    totalTrades += (s.n_trades || 0);
    totalWins += (s.wins || 0);
    totalVol += (s.entry_volume_usd || 0);
  });

  let html = `<div class="panel-wrap" style="padding-top: 12px;">`;
  html += `<div class="panel"><h2>Strategy P&L overview — lifetime live trading</h2>`;
  // Top KPI strip
  html += `<div class="kpi-row" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 16px;">`;
  html += kpi("Total P&L (live)", fmtUSD(totalPnl), `${totalTrades} trades`, cls(totalPnl));
  html += kpi("Volume traded", "$" + totalVol.toFixed(0), "lifetime entry notional");
  html += kpi("Win rate", totalTrades ? (totalWins / totalTrades * 100).toFixed(1) + "%" : "—",
              totalTrades ? `${totalWins}W / ${totalTrades - totalWins}L` : "no trades yet");
  html += `</div>`;
  // Per-strategy mini-cards
  html += `<div class="kpi-row" style="grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));">`;
  names.forEach(n => {
    const s = byStrat[n].summary || {};
    const pnl = s.total_pnl_usd || 0;
    const sub = `${s.n_trades || 0} trades · ${((s.win_rate || 0) * 100).toFixed(0)}% win`;
    html += kpi(n, fmtUSD(pnl), sub, cls(pnl));
  });
  html += `</div>`;
  html += `</div></div>`;

  wrap.innerHTML = html;
}

renderStrategyPnLOverview();
activate("live");
</script>
</body>
</html>
"""


def render_html(payload: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_blob = json.dumps(payload, default=str)
    html = HTML_TEMPLATE.replace("__DATA__", json_blob)
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {out_path} ({len(html):,} bytes)")


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    config = {
        "stop_loss_pct": STOP_LOSS_PCT,
        "take_profit_pct": TAKE_PROFIT_PCT,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "max_trade_usd": MAX_TRADE_USD,
        "min_confidence": MIN_CONFIDENCE,
        "products": PRODUCTS,
        "granularity": GRANULARITY,
        "fee_bps": FEE_BPS,
        "repo": repo,
    }

    logger.info("Loading live trades from SQLite…")
    live = load_live_data()
    logger.info(f"  Live: {live['summary']['n_trades']} closed trades, "
                f"${live['summary']['total_pnl_usd']:+.2f} P&L")

    logger.info("Fetching historical candles for backtests…")
    candles_by_product: dict = {}
    for pid in PRODUCTS:
        try:
            candles = fetch_coinbase_public_history(pid, GRANULARITY, days=max(WINDOWS) + 2)
            candles_by_product[pid] = candles
            logger.info(f"  {pid}: {len(candles)} candles")
        except Exception as exc:
            logger.error(f"  {pid}: failed — {exc}")
            candles_by_product[pid] = None

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "live": live,
        "pods": load_orchestrator_state(),
        "strategic_review": load_latest_strategic_review(),
    }
    for days in WINDOWS:
        logger.info(f"Running {days}-day backtest (legacy strategies)…")
        result = run_backtest_window(days, candles_by_product)

        # Layer in the new orchestrator strategies' backtests
        if _HAS_NEW_BACKTESTS:
            try:
                new_results = _new_backtest_all(days)
                for sname, summary in new_results.items():
                    if summary.n_trades == 0 and not summary.trades and summary.note:
                        # Strategy has no backtest data yet — record placeholder
                        result["by_strategy"][sname] = {
                            "summary": {
                                "strategy": sname,
                                "n_trades": 0, "wins": 0, "losses": 0,
                                "win_rate": 0.0, "total_pnl_usd": 0.0,
                                "entry_volume_usd": 0.0,
                                "return_on_volume_pct": 0.0,
                                "avg_pnl_usd": 0.0,
                                "note": summary.note,
                            },
                            "trades": [],
                        }
                    else:
                        result["by_strategy"][sname] = {
                            "summary": {
                                "strategy": sname,
                                "n_trades": summary.n_trades,
                                "wins": summary.n_wins,
                                "losses": summary.n_losses,
                                "win_rate": summary.win_rate,
                                "total_pnl_usd": summary.total_pnl_usd,
                                "entry_volume_usd": summary.entry_volume_usd,
                                "return_on_volume_pct": summary.return_on_volume_pct,
                                "avg_pnl_usd": summary.avg_pnl_usd,
                                "sharpe": summary.sharpe,
                                "max_drawdown": summary.max_drawdown_usd,
                                "note": summary.note,
                            },
                            "trades": summary.trades,
                        }
                    result["trades"].extend(summary.trades)
            except Exception as e:
                logger.warning(f"  new-strategy backtest failed for {days}d: {e}")

        # Strategies still missing get a placeholder entry so the dashboard
        # always shows all 10
        for sname, why in UNBACKTESTABLE.items():
            if sname not in result["by_strategy"]:
                result["by_strategy"][sname] = {
                    "summary": {
                        "strategy": sname,
                        "n_trades": 0, "wins": 0, "losses": 0,
                        "win_rate": 0.0, "total_pnl_usd": 0.0,
                        "entry_volume_usd": 0.0,
                        "return_on_volume_pct": 0.0,
                        "avg_pnl_usd": 0.0,
                        "note": why,
                    },
                    "trades": [],
                }

        # Recompute window-level summary now that we've added new strategies
        all_trades = []
        for st in result["by_strategy"].values():
            all_trades.extend(st.get("trades", []))
        closed_all = [t for t in all_trades if t.get("pnl_usd") is not None]
        wins_all = sum(1 for t in closed_all if t["pnl_usd"] > 0)
        total_pnl = sum(t["pnl_usd"] for t in closed_all)
        entry_vol = sum(t.get("amount_usd", 0)
                          for t in all_trades if t.get("side") == "BUY")
        result["summary"].update({
            "n_trades": len(closed_all),
            "wins": wins_all, "losses": len(closed_all) - wins_all,
            "win_rate": (wins_all / len(closed_all)) if closed_all else 0.0,
            "total_pnl_usd": total_pnl,
            "entry_volume_usd": entry_vol,
            "return_on_volume_pct": (total_pnl / entry_vol * 100) if entry_vol else 0.0,
        })

        logger.info(f"  {days}d: {result['summary']['n_trades']} trades, "
                    f"${result['summary']['total_pnl_usd']:+.2f} P&L "
                    f"({result['summary']['return_on_volume_pct']:+.2f}% RoV)")
        payload[f"d{days}"] = result

    render_html(payload, Path("docs/index.html"))


if __name__ == "__main__":
    main()
