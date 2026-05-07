"""Minimal trading dashboard — strategies ranked by P&L since inception.

Single page, server-rendered (no JS framework). Reads from the same
SQLite DBs the orchestrator writes to:
  - data/trading_performance.db   (trades + per-strategy P&L)
  - data/risk_state.db            (equity snapshots + kill-switch events)

Layout:
  1. Header banner with the kill-switch state (color-coded), portfolio
     equity, and total all-time realized P&L.
  2. One table: every strategy, sorted by total realized P&L descending.
     Columns: strategy · venue · mode (DRY/PAPER/LIVE) · trades · wins · win rate · realized P&L · last trade.

Output: docs/index.html  (committed by the dashboard.yml workflow only
when the *content* changes — see the timestamp-stripped hash gate).

This replaces a 2120-line dashboard the audit flagged as overcomplex
("rebuild the dashboard, too complex; just want a view of all strategies
ranked by performance with P&L since beginning of investment if it's
paper or live money and kill switch").
"""
from __future__ import annotations

import html
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dashboard")


# ─── Mode classification (DRY / PAPER / LIVE) ───────────────────────────


def _flag(name: str) -> bool | None:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    return v.lower() not in ("false", "0", "no")


def _strategy_mode(name: str, venue: str, live_strategies: set[str]) -> str:
    """Classify a strategy as DRY / PAPER / LIVE.

      DRY   — orders not submitted to the broker (logged only).
      PAPER — orders submitted to a paper / sandbox account (no real money).
      LIVE  — real money on the line.
    """
    global_dry = _flag("DRY_RUN") if _flag("DRY_RUN") is not None else True
    venue_dry_map = {
        "coinbase": _flag("DRY_RUN_COINBASE"),
        "alpaca":   _flag("DRY_RUN_ALPACA"),
        "kalshi":   _flag("DRY_RUN_KALSHI"),
    }
    venue_dry = venue_dry_map.get(venue)
    if venue_dry is None:
        venue_dry = global_dry

    # Per-strategy LIVE override beats every DRY flag.
    if name in live_strategies:
        return "LIVE"

    if venue_dry:
        return "DRY"

    # Submitting orders — is it paper or real money?
    if venue == "coinbase":
        return "LIVE"   # Coinbase has no paper account
    if venue == "alpaca":
        ep = os.environ.get("ALPACA_ENDPOINT", "").lower()
        return "PAPER" if "paper" in ep else "LIVE"
    if venue == "kalshi":
        ep = os.environ.get("KALSHI_ENDPOINT", "").lower()
        return "PAPER" if ("demo" in ep or "sandbox" in ep) else "LIVE"
    return "DRY"


# ─── Data loaders ───────────────────────────────────────────────────────


def _strategy_meta() -> dict[str, dict]:
    """Strategy registry: {name: {venue, asset_classes, description}}.
    Empty on import failure (e.g. partial deploy)."""
    out: dict[str, dict] = {}
    try:
        from run_orchestrator import ALL_STRATEGIES
        for m in ALL_STRATEGIES:
            out[m.name] = {
                "venue": m.venue,
                "asset_classes": list(m.asset_classes),
                "description": getattr(m, "description", ""),
            }
    except ImportError:
        pass
    return out


def _per_strategy_pnl(db_path: str) -> dict[str, dict]:
    """Per-strategy aggregates from the trades ledger.

    Returns {strategy: {realized_pnl_usd, n_trades, n_closed, wins, losses,
                         win_rate, last_trade_at, days_since_last_trade}}.
    Only rows with fill_status='FILLED' contribute to realized P&L —
    that's the audit-fix invariant (#2) the orchestrator enforces.
    """
    out: dict[str, dict] = {}
    if not Path(db_path).exists():
        return out
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT strategy,
                   COUNT(*)                                     AS n_trades,
                   SUM(CASE WHEN side='SELL' AND pnl_usd IS NOT NULL
                            AND fill_status='FILLED' THEN 1 ELSE 0 END)  AS n_closed,
                   SUM(CASE WHEN side='SELL' AND pnl_usd > 0
                            AND fill_status='FILLED' THEN 1 ELSE 0 END)  AS wins,
                   SUM(CASE WHEN side='SELL' AND pnl_usd < 0
                            AND fill_status='FILLED' THEN 1 ELSE 0 END)  AS losses,
                   SUM(CASE WHEN side='SELL' AND fill_status='FILLED'
                            THEN COALESCE(pnl_usd, 0) ELSE 0 END)        AS realized_pnl_usd,
                   MAX(timestamp)                               AS last_trade_at
              FROM trades
             GROUP BY strategy
            """
        ).fetchall()
    now = datetime.now(UTC)
    for r in rows:
        last_ts = r["last_trade_at"]
        days_since = None
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(
                    last_ts.replace("Z", "+00:00")
                ).astimezone(UTC)
                days_since = round((now - last_dt).total_seconds() / 86400, 1)
            except (ValueError, TypeError):
                pass
        n_closed = int(r["n_closed"] or 0)
        wins = int(r["wins"] or 0)
        out[r["strategy"]] = {
            "n_trades":         int(r["n_trades"] or 0),
            "n_closed":         n_closed,
            "wins":             wins,
            "losses":           int(r["losses"] or 0),
            "win_rate":         (wins / n_closed) if n_closed else 0.0,
            "realized_pnl_usd": float(r["realized_pnl_usd"] or 0.0),
            "last_trade_at":    last_ts,
            "days_since":       days_since,
        }
    return out


def _risk_snapshot() -> dict:
    """Latest equity + kill-switch event from risk_state.db."""
    out = {
        "equity_usd": 0.0,
        "peak_equity_usd": 0.0,
        "drawdown_pct": 0.0,
        "kill_switch": "UNKNOWN",
        "kill_switch_at": None,
        "snapshot_at": None,
    }
    risk_db = os.environ.get("RISK_DB_PATH", "data/risk_state.db")
    if not Path(risk_db).exists():
        return out
    try:
        with sqlite3.connect(risk_db) as conn:
            conn.row_factory = sqlite3.Row
            eq = conn.execute(
                "SELECT timestamp, equity_usd FROM equity_snapshots "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            peak = conn.execute(
                "SELECT MAX(equity_usd) AS p FROM equity_snapshots"
            ).fetchone()
            ks = conn.execute(
                "SELECT timestamp, state FROM kill_switch_events "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if eq:
            equity = float(eq["equity_usd"])
            pk = float(peak["p"]) if peak and peak["p"] else equity
            out.update({
                "equity_usd": equity,
                "peak_equity_usd": pk,
                "drawdown_pct": (pk - equity) / pk if pk > 0 else 0.0,
                "snapshot_at": eq["timestamp"],
                "kill_switch": ks["state"] if ks else "NORMAL",
                "kill_switch_at": ks["timestamp"] if ks else None,
            })
        else:
            out["kill_switch"] = "NO-DATA"
    except sqlite3.Error as e:
        logger.warning(f"risk_state.db read failed: {e}")
    return out


# ─── Rendering ──────────────────────────────────────────────────────────


_MODE_BADGE = {
    "LIVE":  ("#b91c1c", "💰 LIVE"),
    "PAPER": ("#1e40af", "🧪 PAPER"),
    "DRY":   ("#4b5563", "🟦 DRY"),
}

_KS_COLOR = {
    "NORMAL":   ("#166534", "✅"),
    "WARNING":  ("#92400e", "⚠️"),
    "CRITICAL": ("#9a3412", "🟠"),
    "KILL":     ("#7f1d1d", "🛑"),
    "NO-DATA":  ("#4b5563", "❓"),
    "UNKNOWN":  ("#4b5563", "❓"),
}


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _row_html(name: str, meta: dict, pnl: dict, mode: str) -> str:
    venue = meta.get("venue", "—") if meta else "—"
    mode_color, mode_label = _MODE_BADGE.get(mode, ("#4b5563", mode))
    pnl_v = pnl.get("realized_pnl_usd", 0.0)
    pnl_color = "#166534" if pnl_v > 0 else ("#7f1d1d" if pnl_v < 0 else "#4b5563")
    if pnl.get("days_since") is not None:
        last_label = f"{pnl['days_since']:g}d ago"
    else:
        last_label = "never"
    return (
        f"<tr>"
        f"<td><strong>{html.escape(name)}</strong>"
        + (f"<br><span class=desc>{html.escape(meta.get('description',''))}</span>" if meta else "")
        + f"</td>"
        f"<td>{html.escape(venue)}</td>"
        f"<td><span class=badge style=\"background:{mode_color}\">{mode_label}</span></td>"
        f"<td class=num>{pnl.get('n_trades', 0)}</td>"
        f"<td class=num>{pnl.get('n_closed', 0)}</td>"
        f"<td class=num>{_fmt_pct(pnl.get('win_rate', 0.0))}</td>"
        f"<td class=num style=\"color:{pnl_color};font-weight:600\">{_fmt_money(pnl_v)}</td>"
        f"<td>{html.escape(last_label)}</td>"
        f"</tr>"
    )


def _render_errors_section(errors: list[dict]) -> str:
    """Render the recent-errors panel; empty section when no errors."""
    if not errors:
        return ""
    rows = []
    for e in errors:
        scope = html.escape(e.get("scope") or "")
        strat = html.escape(e.get("strategy") or "")
        venue = html.escape(e.get("venue") or "")
        exc_type = html.escape(e.get("exc_type") or "")
        exc_msg = html.escape((e.get("exc_message") or "")[:200])
        ts = html.escape(e.get("timestamp") or "")
        # Stack trace is collapsed in a <details> to keep the dashboard
        # tight; click to expand. Truncate to 2KB so a runaway loop
        # doesn't bloat the HTML.
        tb = html.escape((e.get("traceback") or "")[:2000])
        rows.append(
            f"<tr>"
            f"<td><time data-ts='error'>{ts}</time></td>"
            f"<td><code>{scope}</code></td>"
            f"<td>{strat}</td>"
            f"<td>{venue}</td>"
            f"<td><strong>{exc_type}</strong>: {exc_msg}"
            f"<details><summary>traceback</summary>"
            f"<pre style='font-size:11px;overflow-x:auto'>{tb}</pre>"
            f"</details></td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows)
    return f"""
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Recent errors ({len(errors)})
</h2>
<table>
  <thead>
    <tr><th>When</th><th>Scope</th><th>Strategy</th>
        <th>Venue</th><th>Exception</th></tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>"""


def _recent_errors(limit: int = 10) -> list[dict]:
    """Pull the most recent N stack traces from errors.db. Empty
    when the DB doesn't exist (first deploy) or the import fails."""
    try:
        from common.errors_db import recent_errors
        return recent_errors(limit=limit)
    except Exception:
        return []


def render_dashboard(out_path: Path = Path("docs/index.html")) -> None:
    db_path = os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
    pnl = _per_strategy_pnl(db_path)
    risk = _risk_snapshot()
    metas = _strategy_meta()
    errors = _recent_errors(10)
    live_strategies = {
        s.strip() for s in os.environ.get("LIVE_STRATEGIES", "").split(",")
        if s.strip()
    }

    # Union: every strategy from the registry + every strategy that has
    # ever traded (catches renamed / retired strategies still in the
    # ledger so historical rows remain visible).
    names = sorted(set(metas.keys()) | set(pnl.keys()))

    rows = []
    for n in names:
        m = metas.get(n, {})
        venue = m.get("venue", "")
        mode = _strategy_mode(n, venue, live_strategies)
        rows.append((n, m, pnl.get(n, {}), mode))

    # Sort by realized P&L desc; strategies with no closed trades go to
    # the bottom alphabetically.
    rows.sort(key=lambda r: (
        -1 if r[2].get("n_closed", 0) else 0,
        -(r[2].get("realized_pnl_usd", 0.0)),
        r[0],
    ))

    total_pnl = sum(r[2].get("realized_pnl_usd", 0.0) for r in rows)
    total_closed = sum(r[2].get("n_closed", 0) for r in rows)
    total_wins = sum(r[2].get("wins", 0) for r in rows)
    portfolio_winrate = (total_wins / total_closed) if total_closed else 0.0
    pnl_color = ("#166534" if total_pnl > 0
                 else "#7f1d1d" if total_pnl < 0 else "#4b5563")

    ks = (risk.get("kill_switch") or "UNKNOWN").upper()
    ks_color, ks_emoji = _KS_COLOR.get(ks, _KS_COLOR["UNKNOWN"])

    body_rows = "\n".join(_row_html(n, m, p, md) for n, m, p, md in rows)
    if not rows:
        body_rows = (
            "<tr><td colspan=8 style='text-align:center;color:#6b7280;"
            "padding:24px'>No strategies registered or no trades yet.</td></tr>"
        )

    generated_at = datetime.now(UTC).isoformat()
    snapshot_at = risk.get("snapshot_at") or "—"
    ks_at = risk.get("kill_switch_at") or ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-AT-ADVENT — Strategy Performance</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
          system-ui, sans-serif; max-width: 1100px; margin: 24px auto;
          padding: 0 16px; color: #111827; background: #f9fafb; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 16px; }}
  .ks-banner {{ padding: 14px 16px; border-radius: 8px; color: white;
                font-weight: 600; font-size: 16px; margin-bottom: 16px;
                display: flex; justify-content: space-between; align-items: center; }}
  .ks-banner small {{ font-weight: 400; opacity: 0.85; font-size: 12px; }}
  .totals {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin-bottom: 20px; }}
  .stat {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px;
           padding: 12px 16px; }}
  .stat .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase;
                  letter-spacing: 0.04em; }}
  .stat .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden;
           font-size: 14px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #f3f4f6; }}
  th {{ background: #f3f4f6; font-size: 12px; text-transform: uppercase;
        letter-spacing: 0.04em; color: #6b7280; }}
  tr:last-child td {{ border-bottom: 0; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px;
            color: white; font-size: 11px; font-weight: 600;
            letter-spacing: 0.03em; }}
  .desc {{ color: #6b7280; font-size: 11px; font-weight: 400; }}
  footer {{ color: #9ca3af; font-size: 12px; margin-top: 16px; text-align: center; }}
</style>
</head>
<body>

<h1>AI-AT-ADVENT — Strategy Performance</h1>
<div class="meta">
  Snapshot: <time data-ts="snapshot">{html.escape(snapshot_at)}</time>
  · Updated every 15 min · <a href="https://github.com/marcoaduartemendes-source/ai-at-advent/actions" target=_blank>Workflows</a>
</div>

<div class="ks-banner" style="background:{ks_color}">
  <span>{ks_emoji} Kill switch: {html.escape(ks)}</span>
  <small><time data-ts="kill-switch">{html.escape(ks_at)}</time></small>
</div>

<div class="totals">
  <div class="stat">
    <div class="label">Portfolio equity</div>
    <div class="value">{_fmt_money(risk.get('equity_usd', 0.0))}</div>
  </div>
  <div class="stat">
    <div class="label">All-time realized P&amp;L</div>
    <div class="value" style="color:{pnl_color}">{_fmt_money(total_pnl)}</div>
  </div>
  <div class="stat">
    <div class="label">Closed trades</div>
    <div class="value">{total_closed:,}</div>
  </div>
  <div class="stat">
    <div class="label">Win rate</div>
    <div class="value">{_fmt_pct(portfolio_winrate)}</div>
  </div>
  <div class="stat">
    <div class="label">Drawdown from peak</div>
    <div class="value">{_fmt_pct(risk.get('drawdown_pct', 0.0))}</div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>Strategy</th>
      <th>Venue</th>
      <th>Mode</th>
      <th class=num>Total trades</th>
      <th class=num>Closed</th>
      <th class=num>Win rate</th>
      <th class=num>Realized P&amp;L</th>
      <th>Last trade</th>
    </tr>
  </thead>
  <tbody>
{body_rows}
  </tbody>
</table>

{_render_errors_section(errors)}

<footer>Generated <time data-ts="generated">{generated_at}</time></footer>

</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    logger.info(
        f"Wrote {out_path} ({len(html_doc):,} bytes, "
        f"{len(rows)} strategies, total realized P&L ${total_pnl:+.2f})"
    )


# ─── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    try:
        render_dashboard()
        return 0
    except Exception:
        logger.exception("Dashboard build failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
