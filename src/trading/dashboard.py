"""Generates a self-contained mobile-first HTML dashboard from trading SQLite data."""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────

def _pnl_class(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "neu")

def _pnl_fmt(v: float) -> str:
    return f"${v:+.2f}"

def _pct_fmt(v: float) -> str:
    return f"{v:.1f}%"


# ── Metrics helper ────────────────────────────────────────────────────────────

def _metrics(pnls: List[float]) -> Dict:
    if not pnls:
        return dict(closed=0, wins=0, losses=0, win_rate=0.0,
                    total=0.0, avg=0.0, best=0.0, worst=0.0,
                    sharpe=None, max_dd=0.0)
    a = np.array(pnls)
    wins = int((a > 0).sum())
    sharpe = (
        round(float(a.mean() / a.std() * np.sqrt(252)), 2)
        if len(a) > 1 and a.std() > 0 else None
    )
    cum = np.cumsum(a)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())
    return dict(
        closed=len(pnls), wins=wins, losses=len(pnls) - wins,
        win_rate=wins / len(pnls) * 100,
        total=round(float(a.sum()), 2),
        avg=round(float(a.mean()), 2),
        best=round(float(a.max()), 2),
        worst=round(float(a.min()), 2),
        sharpe=sharpe, max_dd=round(max_dd, 2),
    )


# ── HTML helpers ──────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "Momentum":          "#58a6ff",
    "MeanReversion":     "#bc8cff",
    "VolatilityBreakout":"#ffa657",
}

STRATEGY_ICONS = {
    "Momentum":          "📈",
    "MeanReversion":     "↩️",
    "VolatilityBreakout":"💥",
}

STRATEGY_LABELS = {
    "Momentum":          "Momentum",
    "MeanReversion":     "Mean Reversion",
    "VolatilityBreakout":"Volatility Breakout",
}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e17;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:15px;line-height:1.5}
a{color:inherit;text-decoration:none}

/* Header */
.hdr{background:#131823;border-bottom:1px solid #1e2533;padding:18px 20px 14px}
.hdr h1{font-size:1.25rem;font-weight:700;color:#f1f5f9}
.hdr .sub{font-size:0.78rem;color:#64748b;margin-top:3px}
.mode-badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:0.72rem;font-weight:600;margin-left:8px;vertical-align:middle}
.mode-paper{background:#1c3a5e;color:#58a6ff}
.mode-live{background:#3d1515;color:#f87171}
.mode-sim{background:#1e2d1e;color:#86efac}

/* Layout */
.wrap{max-width:700px;margin:0 auto;padding:16px}
.section{margin-bottom:28px}
.section-title{font-size:0.85rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin-bottom:12px}

/* Summary strip */
.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:24px}
.s-card{background:#131823;border:1px solid #1e2533;border-radius:10px;padding:14px 12px;text-align:center}
.s-card .lbl{font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.s-card .val{font-size:1.55rem;font-weight:700;margin-top:4px}

/* Colours */
.pos{color:#22c55e}
.neg{color:#ef4444}
.neu{color:#94a3b8}

/* Strategy cards */
.strat-grid{display:grid;gap:12px}
.strat-card{background:#131823;border:1px solid #1e2533;border-radius:10px;padding:16px}
.strat-hdr{display:flex;align-items:center;margin-bottom:14px}
.strat-hdr .icon{font-size:1.2rem;margin-right:10px}
.strat-hdr .name{font-size:1rem;font-weight:600;color:#f1f5f9}
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #1e2533;font-size:0.88rem}
.stat-row:last-child{border-bottom:none}
.stat-lbl{color:#64748b}
.stat-val{font-weight:500}
.winbar-wrap{background:#1e2533;border-radius:4px;height:6px;width:90px;margin-top:2px}
.winbar{height:6px;border-radius:4px;background:#22c55e}

/* Tables */
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{width:100%;border-collapse:collapse;font-size:0.85rem;min-width:420px}
th{padding:10px 12px;background:#131823;color:#64748b;font-weight:500;text-align:left;border-bottom:1px solid #1e2533;white-space:nowrap}
td{padding:10px 12px;border-bottom:1px solid #1a2030;vertical-align:middle}
tr:last-child td{border-bottom:none}
.empty{text-align:center;color:#475569;padding:28px 0;font-size:0.9rem}

/* Badges */
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:600;white-space:nowrap}
.buy-b{background:#14291a;color:#4ade80}
.sell-b{background:#2d1515;color:#f87171}
.paper-b{background:#1c2a3e;color:#93c5fd;font-size:0.68rem}

/* Chart */
.chart-wrap{background:#131823;border:1px solid #1e2533;border-radius:10px;padding:16px}

/* Footer */
.footer{text-align:center;color:#334155;font-size:0.75rem;padding:24px 0 32px}

@media(max-width:400px){
  .summary{grid-template-columns:1fr 1fr}
  .s-card:last-child{grid-column:span 2}
  .s-card .val{font-size:1.3rem}
}
"""

def _positions_table(positions) -> str:
    if not positions:
        return '<div class="empty">No open positions</div>'
    rows = ""
    for p in positions:
        asset = p["product_id"].split("-")[0]
        strat = STRATEGY_LABELS.get(p["strategy"], p["strategy"])
        rows += f"""<tr>
            <td><strong>{asset}</strong></td>
            <td style="color:#94a3b8;font-size:0.82rem">{strat}</td>
            <td>${p['entry_price']:,.2f}</td>
            <td style="color:#94a3b8;font-size:0.82rem">{p['entry_time'][:16].replace('T',' ')}</td>
            <td>${p['cost_basis_usd']:.2f}</td>
        </tr>"""
    return f"""<table>
        <thead><tr><th>Asset</th><th>Strategy</th><th>Entry Price</th><th>Opened</th><th>Invested</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _trades_table(trades) -> str:
    if not trades:
        return '<div class="empty">No trades yet — bot will trade when signals fire</div>'
    rows = ""
    for t in trades[:25]:
        asset = t["product_id"].split("-")[0]
        side_cls = "buy-b" if t["side"] == "BUY" else "sell-b"
        pnl_html = (
            f'<span class="{_pnl_class(t["pnl_usd"])}">{_pnl_fmt(t["pnl_usd"])}</span>'
            if t["pnl_usd"] is not None else '<span class="neu">—</span>'
        )
        paper = '<span class="badge paper-b">PAPER</span>' if t["dry_run"] else ""
        strat = STRATEGY_LABELS.get(t["strategy"], t["strategy"])
        rows += f"""<tr>
            <td style="color:#94a3b8;font-size:0.8rem;white-space:nowrap">{t['timestamp'][:16].replace('T',' ')}</td>
            <td>{asset}</td>
            <td style="font-size:0.78rem;color:#94a3b8">{strat}</td>
            <td><span class="badge {side_cls}">{t['side']}</span> {paper}</td>
            <td>${t['price']:,.2f}</td>
            <td>${t['amount_usd']:.2f}</td>
            <td>{pnl_html}</td>
        </tr>"""
    return f"""<table>
        <thead><tr><th>Date</th><th>Asset</th><th>Strategy</th><th>Side</th><th>Price</th><th>Amount</th><th>P&L</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _strategy_card(name: str, m: Dict) -> str:
    color = STRATEGY_COLORS.get(name, "#94a3b8")
    icon = STRATEGY_ICONS.get(name, "⚡")
    label = STRATEGY_LABELS.get(name, name)
    pnl_cls = _pnl_class(m["total"])
    wr = min(m["win_rate"], 100)
    sharpe_row = (
        f'<div class="stat-row"><span class="stat-lbl">Sharpe Ratio</span>'
        f'<span class="stat-val" style="color:{color}">{m["sharpe"]}</span></div>'
        if m["sharpe"] is not None else ""
    )
    return f"""<div class="strat-card" style="border-left:3px solid {color}">
    <div class="strat-hdr">
        <span class="icon">{icon}</span>
        <span class="name">{label}</span>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Total P&amp;L</span>
        <span class="stat-val {pnl_cls}" style="font-size:1.1rem">{_pnl_fmt(m['total'])}</span>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Win Rate</span>
        <div style="text-align:right">
            <div style="color:#94a3b8;font-size:0.85rem">{_pct_fmt(wr)} ({m['wins']}W / {m['losses']}L)</div>
            <div class="winbar-wrap" style="margin-left:auto">
                <div class="winbar" style="width:{wr}%"></div>
            </div>
        </div>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Closed Trades</span>
        <span class="stat-val">{m['closed']}</span>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Avg P&amp;L / Trade</span>
        <span class="stat-val {_pnl_class(m['avg'])}">{_pnl_fmt(m['avg'])}</span>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Best / Worst</span>
        <span class="stat-val"><span class="pos">{_pnl_fmt(m['best'])}</span> / <span class="neg">{_pnl_fmt(m['worst'])}</span></span>
    </div>
    <div class="stat-row">
        <span class="stat-lbl">Max Drawdown</span>
        <span class="stat-val neg">-{_pnl_fmt(m['max_dd'])}</span>
    </div>
    {sharpe_row}
</div>"""


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dashboard(
    db_path: str,
    output_path: str = "dashboard/index.html",
    mode: str = "PAPER TRADING",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    all_trades     = [dict(r) for r in conn.execute("SELECT * FROM trades ORDER BY timestamp DESC").fetchall()]
    open_positions = [dict(r) for r in conn.execute("SELECT * FROM open_positions ORDER BY strategy, product_id").fetchall()]

    strategies = ["Momentum", "MeanReversion", "VolatilityBreakout"]
    metrics    = {}
    pnl_series = {}

    for s in strategies:
        rows = conn.execute(
            "SELECT timestamp, pnl_usd FROM trades WHERE strategy=? AND side='SELL' ORDER BY timestamp", (s,)
        ).fetchall()
        pnls = [r["pnl_usd"] for r in rows if r["pnl_usd"] is not None]
        metrics[s] = _metrics(pnls)

        cumulative = 0.0
        series = []
        for r in rows:
            cumulative += r["pnl_usd"] or 0
            series.append({"x": r["timestamp"][:16].replace("T", " "), "y": round(cumulative, 2)})
        pnl_series[s] = series

    conn.close()

    # Overall totals
    all_pnls   = [t["pnl_usd"] for t in all_trades if t["pnl_usd"] is not None]
    total_pnl  = sum(all_pnls) if all_pnls else 0.0
    total_closed = len(all_pnls)
    wins       = sum(1 for p in all_pnls if p > 0)
    win_rate   = wins / total_closed * 100 if total_closed else 0.0

    pnl_cls    = _pnl_class(total_pnl)
    mode_cls   = "mode-paper" if "PAPER" in mode else ("mode-sim" if "SIM" in mode else "mode-live")

    strat_cards = "\n".join(_strategy_card(s, metrics[s]) for s in strategies)
    now = datetime.utcnow().strftime("%b %d %Y  %H:%M UTC")

    chart_datasets = []
    for s in strategies:
        c = STRATEGY_COLORS.get(s, "#94a3b8")
        chart_datasets.append({
            "label": STRATEGY_LABELS.get(s, s),
            "data": pnl_series[s],
            "borderColor": c,
            "backgroundColor": c + "22",
            "fill": True,
            "tension": 0.4,
            "pointRadius": 3,
        })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#0a0e17">
<title>Crypto Trading Bot</title>
<style>{CSS}</style>
</head>
<body>

<div class="hdr">
  <h1>🤖 Crypto Trading Bot <span class="mode-badge {mode_cls}">{mode}</span></h1>
  <div class="sub">Updated {now} · refreshes each hour · <a href="." style="color:#58a6ff">↻ reload</a></div>
</div>

<div class="wrap">

  <!-- Summary -->
  <div class="summary">
    <div class="s-card">
      <div class="lbl">Total P&amp;L</div>
      <div class="val {pnl_cls}">{_pnl_fmt(total_pnl)}</div>
    </div>
    <div class="s-card">
      <div class="lbl">Closed Trades</div>
      <div class="val">{total_closed}</div>
    </div>
    <div class="s-card">
      <div class="lbl">Win Rate</div>
      <div class="val {'pos' if win_rate >= 50 else 'neg'}">{_pct_fmt(win_rate)}</div>
    </div>
  </div>

  <!-- Open Positions -->
  <div class="section">
    <div class="section-title">📊 Open Positions ({len(open_positions)})</div>
    <div class="tbl-wrap">
      {_positions_table(open_positions)}
    </div>
  </div>

  <!-- Strategies -->
  <div class="section">
    <div class="section-title">⚡ Strategy Performance</div>
    <div class="strat-grid">
      {strat_cards}
    </div>
  </div>

  <!-- P&L Chart -->
  <div class="section">
    <div class="section-title">📈 Cumulative P&amp;L</div>
    <div class="chart-wrap">
      <canvas id="chart"></canvas>
    </div>
  </div>

  <!-- Trades -->
  <div class="section">
    <div class="section-title">🔄 Recent Trades</div>
    <div class="tbl-wrap">
      {_trades_table(all_trades)}
    </div>
  </div>

</div>

<div class="footer">Crypto Trading Bot · $20 max per trade · Auto-updates hourly</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const datasets = {json.dumps(chart_datasets)};
const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{ datasets }},
  options: {{
    responsive: true,
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }},
      tooltip: {{
        backgroundColor: '#131823',
        borderColor: '#1e2533',
        borderWidth: 1,
        titleColor: '#e2e8f0',
        bodyColor: '#94a3b8',
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: $${{ctx.parsed.y >= 0 ? '+' : ''}}${{ctx.parsed.y.toFixed(2)}}`
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'category',
        ticks: {{ color: '#475569', maxRotation: 30, maxTicksLimit: 8, font: {{ size: 11 }} }},
        grid: {{ color: '#1e2533' }}
      }},
      y: {{
        ticks: {{
          color: '#475569',
          font: {{ size: 11 }},
          callback: v => '$' + (v >= 0 ? '+' : '') + v.toFixed(2)
        }},
        grid: {{ color: '#1e2533' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
