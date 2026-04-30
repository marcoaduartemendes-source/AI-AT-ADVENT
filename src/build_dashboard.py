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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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


def load_live_data() -> Dict:
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
    # Sort ascending by time for an equity curve
    all_trades_sorted = sorted(all_trades, key=lambda t: t["timestamp"])

    # Equity curve: only closed (SELL with pnl) trades contribute
    eq: List[Dict] = []
    cum = 0.0
    for t in all_trades_sorted:
        if t.get("pnl_usd") is not None:
            cum += t["pnl_usd"]
            eq.append({"t": t["timestamp"], "pnl_cumulative": cum})

    # Per-strategy summary
    by_strategy: Dict[str, Dict] = {}
    for name in ["Momentum", "MeanReversion", "VolatilityBreakout"]:
        m = tracker.get_metrics(name)
        strat_trades = [t for t in all_trades if t["strategy"] == name]
        entry_volume = sum(
            t["amount_usd"] for t in strat_trades if t["side"] == "BUY"
        )
        by_strategy[name] = {
            "summary": {
                "strategy": name,
                "n_trades": m["closed_trades"],
                "wins": m["wins"],
                "losses": m["losses"],
                "win_rate": m["win_rate"],
                "total_pnl_usd": m["total_pnl"],
                "entry_volume_usd": entry_volume,
                "return_on_volume_pct": (
                    m["total_pnl"] / entry_volume * 100 if entry_volume > 0 else 0.0
                ),
                "avg_pnl_usd": m["avg_pnl"],
                "sharpe": m["sharpe"],
                "max_drawdown": m["max_drawdown"],
            },
            "trades": strat_trades,
        }

    # Top-level totals
    closed = [t for t in all_trades if t.get("pnl_usd") is not None]
    total_pnl = sum(t["pnl_usd"] for t in closed)
    entry_volume = sum(t["amount_usd"] for t in all_trades if t["side"] == "BUY")
    summary = {
        "label": "Live",
        "n_trades": len(closed),
        "wins": sum(1 for t in closed if t["pnl_usd"] > 0),
        "losses": sum(1 for t in closed if t["pnl_usd"] <= 0),
        "win_rate": (
            sum(1 for t in closed if t["pnl_usd"] > 0) / len(closed) if closed else 0.0
        ),
        "total_pnl_usd": total_pnl,
        "entry_volume_usd": entry_volume,
        "return_on_volume_pct": (
            total_pnl / entry_volume * 100 if entry_volume > 0 else 0.0
        ),
    }

    # Open positions
    open_pos_raw = []
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM open_positions").fetchall()
    for r in rows:
        open_pos_raw.append(dict(r))
    conn.close()

    return {
        "trades": all_trades_sorted,
        "open_positions": open_pos_raw,
        "by_strategy": by_strategy,
        "equity_curve": eq,
        "summary": summary,
    }


def load_orchestrator_state() -> Dict:
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

    # Risk state — read the latest snapshot
    try:
        import sqlite3
        risk_db = os.environ.get("RISK_DB_PATH", "data/risk_state.db")
        if Path(risk_db).exists():
            conn = sqlite3.connect(risk_db)
            conn.row_factory = sqlite3.Row
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
            conn.close()
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
                }
    except Exception as exc:
        logger.warning(f"Could not load risk state: {exc}")

    return out


def _empty_summary(label: str) -> Dict:
    return {
        "label": label,
        "n_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "total_pnl_usd": 0.0,
        "entry_volume_usd": 0.0,
        "return_on_volume_pct": 0.0,
    }


# ─── Backtest data ────────────────────────────────────────────────────────────


def run_backtest_window(days: int, candles_by_product: Dict) -> Dict:
    """Run all 3 strategies on each product for the past `days` days."""
    strategies = make_strategies(GRANULARITY)

    by_strategy: Dict[str, Dict] = {}
    all_trades: List[Dict] = []
    combined_eq_points: List[Dict] = []

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
  html += `<div class="kpi-row">`;
  html += kpi("Total P&L", fmtUSD(s.total_pnl_usd), `${s.n_trades} closed trades`, cls(s.total_pnl_usd));
  html += kpi("Return on volume", fmtPct(s.return_on_volume_pct), `over $${s.entry_volume_usd.toFixed(0)} traded`, cls(s.return_on_volume_pct));
  html += kpi("Win rate", (s.win_rate * 100).toFixed(1) + "%", `${s.wins}W / ${s.losses}L`);
  html += kpi("Trade volume", "$" + s.entry_volume_usd.toFixed(0), `entry notional`);
  if (isLive && DATA.live.open_positions.length) {
    html += kpi("Open positions", DATA.live.open_positions.length, `live exposure`);
  }
  html += `</div>`;

  // Per-strategy summary
  html += `<div class="panel"><h2>By strategy</h2><table><thead><tr>
    <th>Strategy</th><th>Trades</th><th>Win rate</th><th class="num">Total P&L</th>
    <th class="num">Volume</th><th class="num">Return on volume</th><th class="num">Avg/trade</th></tr></thead><tbody>`;
  Object.values(d.by_strategy).forEach(st => {
    const x = st.summary;
    html += `<tr>
      <td>${x.strategy}</td>
      <td>${x.n_trades} (${x.wins}W/${x.losses}L)</td>
      <td>${(x.win_rate * 100).toFixed(1)}%</td>
      <td class="num ${cls(x.total_pnl_usd)}">${fmtUSD(x.total_pnl_usd)}</td>
      <td class="num">$${x.entry_volume_usd.toFixed(2)}</td>
      <td class="num ${cls(x.return_on_volume_pct)}">${fmtPct(x.return_on_volume_pct)}</td>
      <td class="num ${cls(x.avg_pnl_usd)}">${fmtUSD(x.avg_pnl_usd)}</td>
    </tr>`;
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
activate("live");
</script>
</body>
</html>
"""


def render_html(payload: Dict, out_path: Path):
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
    candles_by_product: Dict = {}
    for pid in PRODUCTS:
        try:
            candles = fetch_coinbase_public_history(pid, GRANULARITY, days=max(WINDOWS) + 2)
            candles_by_product[pid] = candles
            logger.info(f"  {pid}: {len(candles)} candles")
        except Exception as exc:
            logger.error(f"  {pid}: failed — {exc}")
            candles_by_product[pid] = None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "live": live,
        "pods": load_orchestrator_state(),
    }
    for days in WINDOWS:
        logger.info(f"Running {days}-day backtest…")
        result = run_backtest_window(days, candles_by_product)
        logger.info(f"  {days}d: {result['summary']['n_trades']} trades, "
                    f"${result['summary']['total_pnl_usd']:+.2f} P&L "
                    f"({result['summary']['return_on_volume_pct']:+.2f}% RoV)")
        payload[f"d{days}"] = result

    render_html(payload, Path("docs/index.html"))


if __name__ == "__main__":
    main()
