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

import time as _time
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
    _NY_TZ = _ZoneInfo("America/New_York")
except Exception:
    _NY_TZ = UTC


def _ny_converter(*args):
    """See run_orchestrator._ny_converter — same shim, same reason."""
    secs = None
    for a in args:
        if isinstance(a, (int, float)):
            secs = a
            break
    if secs is None:
        secs = _time.time()
    return datetime.fromtimestamp(secs, tz=UTC).astimezone(_NY_TZ).timetuple()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s ET [%(levelname)s] %(name)s: %(message)s",
)
logging.Formatter.converter = staticmethod(_ny_converter)
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

    # Per-strategy LIVE override beats every DRY flag — but only
    # when ALLOW_LIVE_TRADING is truthy. Mirrors the orchestrator's
    # two-key gate. Accepts "1", "true", "yes" (case-insensitive)
    # so a user who set the var to "true" doesn't get a misleading
    # DRY badge.
    allow_live = (os.environ.get("ALLOW_LIVE_TRADING", "")
                  .strip().lower() in ("1", "true", "yes"))
    if name in live_strategies and allow_live:
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


def _config_diagnostic() -> dict:
    """Returns a snapshot of the env-var state that drives mode
    classification. Surfaced on the dashboard so the user can see
    AT A GLANCE why strategies are landing in the mode they're in
    instead of having to read the source code.
    """
    live_strats_raw = os.environ.get("LIVE_STRATEGIES", "").strip()
    live_strats = {s.strip() for s in live_strats_raw.split(",") if s.strip()}
    return {
        "DRY_RUN": os.environ.get("DRY_RUN", "(unset → defaults true)"),
        "DRY_RUN_COINBASE": os.environ.get("DRY_RUN_COINBASE", "(unset → falls back to DRY_RUN)"),
        "DRY_RUN_ALPACA": os.environ.get("DRY_RUN_ALPACA", "(unset → falls back to DRY_RUN)"),
        "ALLOW_LIVE_TRADING": os.environ.get("ALLOW_LIVE_TRADING", "(unset → blocks LIVE_STRATEGIES)"),
        "LIVE_STRATEGIES": live_strats_raw or "(unset → no per-strategy override)",
        "_live_strats_set": live_strats,
        "_allow_live": (os.environ.get("ALLOW_LIVE_TRADING", "")
                         .strip().lower() in ("1", "true", "yes")),
    }


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


def _live_unrealized_by_strategy() -> dict[str, float]:
    """Return per-strategy unrealized P&L by querying live broker
    positions and attributing each symbol to the strategy that
    last opened it (per the trades ledger).

    Empty dict when broker creds aren't configured or nothing is
    open; the dashboard treats absent strategies as 0 unrealized.
    """
    out: dict[str, float] = {}
    try:
        from brokers.registry import build_brokers
        brokers = build_brokers()
    except Exception:
        return out
    if not brokers:
        return out

    # Map (venue, symbol) -> strategy by looking at the most recent
    # opening BUY for that pair in the trades ledger.
    db_path = os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
    sym_to_strategy: dict[tuple[str, str], str] = {}
    if Path(db_path).exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                for r in conn.execute(
                    "SELECT venue, product_id, strategy "
                    "  FROM trades "
                    " WHERE side = 'BUY' AND venue IS NOT NULL "
                    "   AND product_id IS NOT NULL "
                    "   AND strategy IS NOT NULL "
                    " ORDER BY id ASC"
                ).fetchall():
                    sym_to_strategy[(r["venue"], r["product_id"])] = r["strategy"]
        except sqlite3.Error as e:
            logger.debug(f"sym_to_strategy query failed: {e}")

    for venue, adapter in brokers.items():
        try:
            positions = adapter.get_positions()
        except Exception as e:
            logger.debug(f"[{venue}] get_positions for unrealized: {e}")
            continue
        for p in positions:
            unrealized = float(p.unrealized_pnl_usd or 0.0)
            if unrealized == 0.0:
                continue
            strategy = sym_to_strategy.get((venue, p.symbol))
            if strategy is None:
                # Position not attributable to any tracked strategy;
                # bucket under "<unattributed>" so the dashboard total
                # still reflects it.
                strategy = "<unattributed>"
            out[strategy] = out.get(strategy, 0.0) + unrealized
    return out


def _recent_trades(limit: int = 50) -> list[dict]:
    """Last N trades from trading_performance.db, newest first.

    Surfaces what's actually being submitted/filled so the user
    doesn't have to dig through GH Actions logs to see "is the bot
    actually trading?". Includes DRY proposals too — they're tagged
    in the rendered table.
    """
    db_path = os.environ.get(
        "TRADING_DB_PATH", "data/trading_performance.db"
    )
    if not Path(db_path).exists():
        return []
    out: list[dict] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT timestamp, strategy, product_id, side, "
                "       amount_usd, quantity, price, order_id, "
                "       pnl_usd, dry_run, fill_status, venue "
                "  FROM trades "
                " ORDER BY id DESC "
                f" LIMIT {int(limit)}"
            ).fetchall()
        for r in rows:
            out.append({
                "timestamp":  r["timestamp"],
                "strategy":   r["strategy"],
                "symbol":     r["product_id"],
                "side":       r["side"],
                "amount_usd": float(r["amount_usd"] or 0),
                "quantity":   float(r["quantity"] or 0),
                "price":      float(r["price"] or 0),
                "order_id":   r["order_id"],
                "pnl_usd":    (float(r["pnl_usd"]) if r["pnl_usd"] is not None else None),
                "dry_run":    bool(r["dry_run"]),
                "fill_status": r["fill_status"] or "UNKNOWN",
                "venue":      r["venue"] or "",
            })
    except sqlite3.Error as e:
        logger.warning(f"recent_trades read failed: {e}")
    return out


def _recent_cycles(limit: int = 5) -> list[dict]:
    """Last N cycle diagnostics for the dashboard's 'Cycle activity'
    panel. Empty when the table doesn't exist yet (first deploy after
    PR #16 lands)."""
    db_path = os.environ.get(
        "TRADING_DB_PATH", "data/trading_performance.db"
    )
    if not Path(db_path).exists():
        return []
    out: list[dict] = []
    try:
        import json as _json
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT timestamp, cycle_seconds, proposals_total, "
                    "       proposals_submitted, n_errors, venue_health, "
                    "       strategy_outcomes "
                    "  FROM cycle_diagnostics "
                    " ORDER BY id DESC "
                    f" LIMIT {int(limit)}"
                ).fetchall()
            except sqlite3.OperationalError:
                # Table not yet created (orchestrator hasn't run with
                # the PR #16 schema yet). Empty list is a fine fallback.
                return []
        for r in rows:
            try:
                vh = _json.loads(r["venue_health"]) or {}
            except Exception:
                vh = {}
            try:
                so = _json.loads(r["strategy_outcomes"]) or {}
            except Exception:
                so = {}
            out.append({
                "timestamp": r["timestamp"],
                "cycle_seconds": float(r["cycle_seconds"] or 0),
                "proposals_total": int(r["proposals_total"] or 0),
                "proposals_submitted": int(r["proposals_submitted"] or 0),
                "n_errors": int(r["n_errors"] or 0),
                "venue_health": vh,
                "strategy_outcomes": so,
            })
    except sqlite3.Error as e:
        logger.warning(f"recent_cycles read failed: {e}")
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
    realized = pnl.get("realized_pnl_usd", 0.0)
    unrealized = pnl.get("unrealized_pnl_usd", 0.0)
    total = realized + unrealized
    realized_color = "#166534" if realized > 0 else ("#7f1d1d" if realized < 0 else "#4b5563")
    unrealized_color = "#166534" if unrealized > 0 else ("#7f1d1d" if unrealized < 0 else "#4b5563")
    total_color = "#166534" if total > 0 else ("#7f1d1d" if total < 0 else "#4b5563")
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
        f"<td class=num>{pnl.get('n_closed', 0)}</td>"
        f"<td class=num>{_fmt_pct(pnl.get('win_rate', 0.0))}</td>"
        f"<td class=num style=\"color:{realized_color}\">{_fmt_money(realized)}</td>"
        f"<td class=num style=\"color:{unrealized_color}\">{_fmt_money(unrealized)}</td>"
        f"<td class=num style=\"color:{total_color};font-weight:600\">{_fmt_money(total)}</td>"
        f"<td>{html.escape(last_label)}</td>"
        f"</tr>"
    )


def _render_mode_diagnostic(diag: dict, venue_modes: list[tuple[str, str]]) -> str:
    """Banner that explains the current mode-classification state.

    Shows: the relevant env vars, their parsed values, and what each
    venue's representative strategy was classified as. Click "details"
    to see the raw env-var dump. The dashboard always shows this
    banner so "why is venue X in mode Y?" never requires reading code.
    """
    rows = []
    for v, mode in venue_modes:
        color, label = _MODE_BADGE.get(mode, ("#4b5563", mode))
        rows.append(
            f"<tr><td><strong>{html.escape(v)}</strong></td>"
            f"<td><span class=badge style=\"background:{color}\">{label}</span></td></tr>"
        )
    rows_html = "\n".join(rows) or "<tr><td colspan=2>(no venues active)</td></tr>"

    # Pre-compute warnings the user is most likely to need
    warnings = []
    live_set = diag.get("_live_strats_set") or set()
    allow_live = diag.get("_allow_live")
    if live_set and not allow_live:
        warnings.append(
            "⚠ <strong>LIVE_STRATEGIES is set but ALLOW_LIVE_TRADING is not truthy</strong> — "
            "the runtime safety gate is forcing all listed strategies to DRY. "
            "Set repo Variable <code>ALLOW_LIVE_TRADING=1</code> (or true/yes) to honour the override."
        )
    if not live_set and allow_live:
        warnings.append(
            "ℹ ALLOW_LIVE_TRADING is truthy but LIVE_STRATEGIES is empty — "
            "no strategy will trade real money on Coinbase via the per-strategy override. "
            "(The DRY_RUN_COINBASE=false flag still routes the whole venue live.)"
        )

    warnings_html = ""
    if warnings:
        warnings_html = (
            "<div style='background:#fef3c7;border:1px solid #f59e0b;"
            "padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:12px'>"
            + "<br>".join(warnings) + "</div>"
        )

    env_lines = "\n".join(
        f"<tr><td><code>{html.escape(k)}</code></td>"
        f"<td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in diag.items() if not k.startswith("_")
    )

    return f"""
<div style="background:#f3f4f6;border:1px solid #e5e7eb;border-radius:8px;
            padding:10px 14px;margin-bottom:14px;font-size:13px;">
  <strong>Mode by venue</strong>
  <table style="margin-top:6px;font-size:12px;">{rows_html}</table>
  {warnings_html}
  <details style="margin-top:6px">
    <summary style="cursor:pointer;color:#6b7280;font-size:11px">
      Why these modes? (click to expand env-var state)
    </summary>
    <table style="margin-top:6px;font-size:11px;">{env_lines}</table>
    <p style="font-size:11px;color:#6b7280;margin:6px 0 0">
      Set repo Variables at
      <a href="https://github.com/marcoaduartemendes-source/ai-at-advent/settings/variables/actions" target=_blank>
        Settings → Secrets and variables → Actions → Variables
      </a>.
    </p>
  </details>
</div>
"""


def _render_cycle_diagnostics(cycles: list[dict]) -> str:
    """Render the per-cycle 'is the bot alive?' + per-strategy
    'why didn't X trade?' diagnostic panels.

    User feedback 2026-05-08: "feels like a black box where I can't
    see what's happening". This panel surfaces every layer of the
    cycle so the operator can answer:
      - Did the cycle run?  → timestamp + cycle_seconds
      - How many proposals?  → proposals_total
      - How many made it to the broker? → proposals_submitted
      - Per-strategy: was it FROZEN, did it produce nothing, did it
        DRY-log, was it rejected?
    """
    if not cycles:
        return """
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Cycle activity
</h2>
<p style="color:#6b7280; font-size:13px;">
  No cycle diagnostics yet — the orchestrator hasn't run with the
  diagnostics schema enabled. Wait one cron tick (≤5 min) after this
  build commits.
</p>"""

    latest = cycles[0]
    # Venue health badges
    vh = latest.get("venue_health") or {}
    vh_html = " · ".join(
        f'<span style="background:{"#15803d" if v == "ok" else "#7f1d1d"};color:white;padding:2px 8px;border-radius:4px;font-size:11px">{html.escape(name)}: {html.escape(v)}</span>'
        for name, v in sorted(vh.items())
    ) or "(no venues active)"

    # Per-strategy outcome table
    outcomes = latest.get("strategy_outcomes") or {}
    rows = []
    for name in sorted(outcomes.keys()):
        o = outcomes[name]
        skip = o.get("skip_reasons") or []
        err = o.get("error") or ""
        # Color the row by outcome severity
        if err:
            row_color = "#fee2e2"   # error → red tint
            why = f'<span style="color:#7f1d1d">{html.escape(err[:120])}</span>'
        elif o.get("submitted", 0) > 0:
            row_color = "#dcfce7"   # submitted → green
            why = f'<span style="color:#166534">submitted {o["submitted"]} order(s)</span>'
        elif o.get("dry_logged", 0) > 0:
            row_color = "#dbeafe"   # DRY → blue
            why = f'<span style="color:#1e40af">{o["dry_logged"]} DRY-logged</span>'
        elif skip:
            row_color = "#f3f4f6"   # skipped → gray
            why = html.escape(", ".join(skip[:3]))
        else:
            row_color = "white"
            why = '<span style="color:#6b7280">—</span>'
        rows.append(
            f'<tr style="background:{row_color}">'
            f"<td><strong>{html.escape(name)}</strong></td>"
            f"<td>{html.escape(o.get('venue', ''))}</td>"
            f"<td>{html.escape(o.get('state', ''))}</td>"
            f"<td style='text-align:right'>{o.get('target_alloc_pct', 0)*100:.1f}%</td>"
            f"<td style='text-align:right'>${o.get('target_alloc_usd', 0):.0f}</td>"
            f"<td style='text-align:right'>{o.get('proposed', 0)}</td>"
            f"<td style='text-align:right'>{o.get('approved', 0)}</td>"
            f"<td style='text-align:right'>{o.get('rejected', 0)}</td>"
            f"<td style='text-align:right'>{o.get('submitted', 0)}</td>"
            f"<td>{why}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows) or '<tr><td colspan="10">(no strategies ran)</td></tr>'

    # Cycle history sparkline-ish summary
    history_rows = []
    for c in cycles:
        history_rows.append(
            f"<tr>"
            f'<td><time data-ts="cycle">{html.escape(c["timestamp"])}</time></td>'
            f'<td style="text-align:right">{c["cycle_seconds"]:.1f}s</td>'
            f'<td style="text-align:right">{c["proposals_total"]}</td>'
            f'<td style="text-align:right">{c["proposals_submitted"]}</td>'
            f'<td style="text-align:right">{c["n_errors"]}</td>'
            f"</tr>"
        )
    history_html = "\n".join(history_rows)

    return f"""
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Cycle activity — last cycle was {html.escape(latest["timestamp"])} ({latest["cycle_seconds"]:.1f}s)
</h2>
<div style="margin-bottom:12px">
  <strong>Venue health:</strong> {vh_html}<br>
  <strong>Last 5 cycles:</strong>
  <table style="font-size:11px;margin-top:4px">
    <thead><tr><th>When</th><th>Duration</th><th>Proposed</th><th>Submitted</th><th>Errors</th></tr></thead>
    <tbody>{history_html}</tbody>
  </table>
</div>

<h3 style="font-size: 14px; margin-top: 16px; margin-bottom: 6px;">
  Per-strategy outcome (last cycle) — answers "why didn't X trade?"
</h3>
<table style="font-size:11px;">
  <thead><tr><th>Strategy</th><th>Venue</th><th>State</th>
      <th style='text-align:right'>Target%</th><th style='text-align:right'>Target$</th>
      <th style='text-align:right'>Proposed</th><th style='text-align:right'>Approved</th>
      <th style='text-align:right'>Rejected</th><th style='text-align:right'>Submitted</th>
      <th>Outcome / why</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


def _render_recent_trades(trades: list[dict]) -> str:
    """Render the most-recent-trades panel.

    User feedback 2026-05-08: "I want to see all the trades on the
    dashboard". This panel is the primary 'is the bot trading?' view —
    if it's empty, no trades have been recorded recently. If it shows
    DRY rows for the venues you expect to be live, your config is off.
    """
    if not trades:
        return """
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Recent trades
</h2>
<p style="color:#6b7280; font-size:13px;">
  No trades recorded yet. If the orchestrator is running but this
  stays empty, check the Notify-on-failure webhook for errors —
  most likely a strategy is short-circuiting before it can submit.
</p>"""

    def _row(t):
        ts = html.escape(t.get("timestamp") or "")
        strat = html.escape(t.get("strategy") or "")
        sym = html.escape(t.get("symbol") or "")
        side = html.escape(t.get("side") or "")
        side_color = "#166534" if side == "BUY" else "#7f1d1d"
        venue = html.escape(t.get("venue") or "")
        amount = t.get("amount_usd", 0)
        qty = t.get("quantity", 0)
        price = t.get("price", 0)
        status = html.escape(t.get("fill_status") or "")
        # DRY rows shouldn't be confused with live fills.
        mode_pill = ('<span style="background:#4b5563;color:white;'
                     'padding:1px 6px;border-radius:4px;font-size:11px;">DRY</span>'
                     if t.get("dry_run") else
                     '<span style="background:#15803d;color:white;'
                     'padding:1px 6px;border-radius:4px;font-size:11px;">LIVE</span>')
        pnl = t.get("pnl_usd")
        if pnl is None:
            pnl_cell = "—"
        else:
            pnl_color = "#166534" if pnl > 0 else ("#7f1d1d" if pnl < 0 else "#4b5563")
            pnl_cell = f'<span style="color:{pnl_color}">{_fmt_money(pnl)}</span>'
        # Status pill
        status_color = {
            "FILLED": "#15803d", "PENDING": "#d97706",
            "PARTIALLY_FILLED": "#d97706",
            "CANCELED": "#6b7280", "REJECTED": "#7f1d1d",
        }.get(status.upper(), "#6b7280")
        return (
            f"<tr>"
            f"<td><time data-ts='trade'>{ts}</time></td>"
            f"<td>{strat}</td>"
            f"<td>{venue}</td>"
            f"<td><code>{sym}</code></td>"
            f'<td style="color:{side_color};font-weight:600">{side}</td>'
            f"<td>{qty:.6f}</td>"
            f'<td style="text-align:right">{_fmt_money(amount)}</td>'
            f"<td>{_fmt_money(price) if price else '—'}</td>"
            f'<td><span style="background:{status_color};color:white;'
            f'padding:1px 6px;border-radius:4px;font-size:11px">{status}</span></td>'
            f"<td>{mode_pill}</td>"
            f'<td style="text-align:right">{pnl_cell}</td>'
            f"</tr>"
        )
    rows_html = "\n".join(_row(t) for t in trades)
    return f"""
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Recent trades ({len(trades)})
</h2>
<table style="font-size:12px;">
  <thead>
    <tr><th>When</th><th>Strategy</th><th>Venue</th><th>Symbol</th>
        <th>Side</th><th>Qty</th><th>Notional</th><th>Price</th>
        <th>Status</th><th>Mode</th><th>P&amp;L</th></tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>"""


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


def _recent_errors(limit: int = 10, valid_strategies: set | None = None
                    ) -> list[dict]:
    """Pull the most recent N stack traces from errors.db. Empty
    when the DB doesn't exist (first deploy) or the import fails.

    `valid_strategies` filters out rows whose strategy isn't in the
    runtime registry — defensive against a dev's local pytest run
    leaking test-strategy errors into a committed docs/index.html.
    Without this, the test "broken" strategy errors leaked to the
    user's dashboard 2026-05-08.
    """
    try:
        from common.errors_db import recent_errors
        rows = recent_errors(limit=limit * 3 if valid_strategies else limit)
    except Exception:
        return []
    if valid_strategies is None:
        return rows[:limit]
    out = []
    for r in rows:
        s = r.get("strategy") or ""
        # Empty-strategy rows are non-strategy errors (orchestrator,
        # broker layer) — keep them. Otherwise gate on the registry.
        if not s or s in valid_strategies:
            out.append(r)
        if len(out) >= limit:
            break
    return out


def render_dashboard(out_path: Path = Path("docs/index.html")) -> None:
    db_path = os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
    pnl = _per_strategy_pnl(db_path)
    risk = _risk_snapshot()
    metas = _strategy_meta()
    # Gate the errors panel on the live strategy registry so a dev's
    # local pytest run doesn't leak test-strategy errors to the
    # dashboard. Falls back to "no filter" if metas is empty (cold
    # start) — better to over-show than under-show in that case.
    valid_strategies = set(metas.keys()) if metas else None
    errors = _recent_errors(10, valid_strategies=valid_strategies)
    # Last 50 trades — surfaced on the dashboard so the user can answer
    # "is the bot actually trading right now?" without digging through
    # the GH Actions log.
    trades_recent = _recent_trades(50)
    # Per-cycle diagnostics — answers "is the cycle running?" + "why
    # didn't strategy X trade?". The single biggest visibility win
    # against the user's "feels like a black box" complaint.
    cycles_recent = _recent_cycles(5)
    # Live unrealized P&L per strategy — pulled from broker positions
    # at render time. Best-effort; absent or empty when creds missing.
    unrealized_by_strategy = _live_unrealized_by_strategy()
    # Fold the unrealized into the per-strategy view so the table can
    # show it alongside realized.
    for s, u in unrealized_by_strategy.items():
        if s not in pnl:
            pnl[s] = {
                "n_trades": 0, "n_closed": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "realized_pnl_usd": 0.0,
                "last_trade_at": None, "days_since": None,
            }
        pnl[s]["unrealized_pnl_usd"] = u
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

    total_realized = sum(r[2].get("realized_pnl_usd", 0.0) for r in rows)
    total_unrealized = sum(unrealized_by_strategy.values())
    total_pnl = total_realized + total_unrealized

    # Mode diagnostic: shows the env-var state that drives DRY/PAPER/LIVE
    # classification so the user can debug "why is X showing DRY?" without
    # reading source.
    diag = _config_diagnostic()
    venue_modes_summary = []
    for v in ("coinbase", "alpaca", "kalshi"):
        # Pick a representative strategy on this venue to derive its mode
        sample = next(
            (n for n, m, _, _ in rows if m and m.get("venue") == v),
            None,
        )
        if sample:
            mode = _strategy_mode(sample, v, diag["_live_strats_set"])
            venue_modes_summary.append((v, mode))
    total_closed = sum(r[2].get("n_closed", 0) for r in rows)
    total_wins = sum(r[2].get("wins", 0) for r in rows)
    portfolio_winrate = (total_wins / total_closed) if total_closed else 0.0

    def _color_for(v: float) -> str:
        return "#166534" if v > 0 else ("#7f1d1d" if v < 0 else "#4b5563")
    pnl_color = _color_for(total_pnl)
    realized_color = _color_for(total_realized)
    unrealized_color = _color_for(total_unrealized)

    ks = (risk.get("kill_switch") or "UNKNOWN").upper()
    ks_color, ks_emoji = _KS_COLOR.get(ks, _KS_COLOR["UNKNOWN"])

    body_rows = "\n".join(_row_html(n, m, p, md) for n, m, p, md in rows)
    if not rows:
        body_rows = (
            "<tr><td colspan=9 style='text-align:center;color:#6b7280;"
            "padding:24px'>No strategies registered or no trades yet.</td></tr>"
        )

    def _to_et(iso_ts: str | None) -> str:
        """Convert an ISO-8601 UTC timestamp to America/New_York for
        display. Falls back to the original string on parse failure."""
        if not iso_ts or iso_ts == "—":
            return "—"
        try:
            dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            return dt.astimezone(_NY_TZ).strftime("%Y-%m-%d %H:%M ET")
        except (ValueError, TypeError):
            return iso_ts

    generated_at = datetime.now(UTC).astimezone(_NY_TZ).strftime("%Y-%m-%d %H:%M ET")
    snapshot_at = _to_et(risk.get("snapshot_at"))
    ks_at = _to_et(risk.get("kill_switch_at") or None) if risk.get("kill_switch_at") else ""

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Auto-reload every 30s. Underlying file updates as fast as the
     dashboard cron fires (5 min on GitHub Actions; 30s on the VPS
     systemd timer if enabled). The page will pick up new data on
     the next reload after a build commits. -->
<meta http-equiv="refresh" content="30">
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
  .ks-actions {{ display: flex; gap: 8px; }}
  .ks-btn {{ display: inline-block; padding: 6px 12px; border-radius: 6px;
             font-size: 12px; font-weight: 600; text-decoration: none;
             border: 1px solid rgba(255,255,255,0.4); }}
  .ks-arm   {{ background: rgba(239, 68, 68, 0.95); color: white; }}
  .ks-reset {{ background: rgba(34, 197, 94, 0.95); color: white; }}
  .ks-btn:hover {{ filter: brightness(1.1); }}
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
  <span class="ks-actions">
    <a class="ks-btn ks-arm" target="_blank"
       href="https://github.com/marcoaduartemendes-source/ai-at-advent/actions/workflows/kill_switch.yml"
       title="Open the kill_switch workflow with action=arm preselected. Triggers an immediate close of every position on the next cycle.">🛑 ARM KILL</a>
    <a class="ks-btn ks-reset" target="_blank"
       href="https://github.com/marcoaduartemendes-source/ai-at-advent/actions/workflows/kill_switch.yml"
       title="Reset kill switch to NORMAL — strategies resume trading on the next cycle.">✅ RESET</a>
  </span>
</div>

{_render_mode_diagnostic(diag, venue_modes_summary)}

<div class="totals">
  <div class="stat" style="grid-column: span 2; border: 2px solid {pnl_color};">
    <div class="label">Total P&amp;L (realized + unrealized)</div>
    <div class="value" style="color:{pnl_color}; font-size: 28px;">{_fmt_money(total_pnl)}</div>
  </div>
  <div class="stat">
    <div class="label">Realized P&amp;L</div>
    <div class="value" style="color:{realized_color}">{_fmt_money(total_realized)}</div>
  </div>
  <div class="stat">
    <div class="label">Unrealized P&amp;L</div>
    <div class="value" style="color:{unrealized_color}">{_fmt_money(total_unrealized)}</div>
  </div>
  <div class="stat">
    <div class="label">Portfolio equity</div>
    <div class="value">{_fmt_money(risk.get('equity_usd', 0.0))}</div>
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
      <th class=num>Closed</th>
      <th class=num>Win rate</th>
      <th class=num>Realized</th>
      <th class=num>Unrealized</th>
      <th class=num>Total P&amp;L</th>
      <th>Last trade</th>
    </tr>
  </thead>
  <tbody>
{body_rows}
  </tbody>
</table>

{_render_cycle_diagnostics(cycles_recent)}

{_render_recent_trades(trades_recent)}

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
    # Healthchecks dead-man's-switch ping. Without this, Healthchecks.io
    # never sees a successful run and fires the dashboard alert every
    # cycle even when the build completes cleanly. Best-effort: a
    # failed ping never blocks the build itself.
    try:
        from common.heartbeat import ping_fail, ping_start, ping_success
    except Exception:
        ping_start = ping_success = ping_fail = lambda *a, **kw: False
    try:
        ping_start("dashboard")
    except Exception:
        pass
    try:
        render_dashboard()
        try:
            ping_success("dashboard", message="ok")
        except Exception:
            pass
        return 0
    except Exception as e:
        logger.exception("Dashboard build failed")
        try:
            ping_fail("dashboard", message=f"{type(e).__name__}: {str(e)[:160]}")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
