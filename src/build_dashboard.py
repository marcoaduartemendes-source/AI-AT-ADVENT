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
                "group": getattr(m, "group", "OTHER"),
                "leverage_x": float(getattr(m, "leverage_x", 1.0)),
                "target_alloc_pct": float(getattr(m, "target_alloc_pct", 0.0)),
                "max_alloc_pct": float(getattr(m, "max_alloc_pct", 0.0)),
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

    # Overlay canonical FIFO realized P&L + win stats. The stored pnl_usd
    # column (summed above) drifts on partial fills / stale cost-basis /
    # phantom price=0; the FIFO walk over raw fills is the auditable
    # truth, so the dashboard shows that. n_trades/last_trade keep the
    # SQL values (they're counts/timestamps, not P&L).
    try:
        from trading.recompute import fifo_realized_events
        for strat, evs in fifo_realized_events(db_path).items():
            n_closed = len(evs)
            wins = sum(1 for e in evs if e["pnl_usd"] > 0)
            losses = sum(1 for e in evs if e["pnl_usd"] < 0)
            d = out.setdefault(strat, {
                "n_trades": n_closed, "last_trade_at": None, "days_since": None,
            })
            d["realized_pnl_usd"] = round(sum(e["pnl_usd"] for e in evs), 2)
            d["n_closed"] = n_closed
            d["wins"] = wins
            d["losses"] = losses
            d["win_rate"] = (wins / n_closed) if n_closed else 0.0
    except Exception as e:
        logger.debug(f"FIFO realized overlay failed: {e}")
    return out


# Headline-drift threshold: below this the DB↔FIFO difference is float
# rounding noise, not a real bookkeeping bug. The recompute module flags
# per-strategy drift at $0.50; the headline aggregates many strategies so
# we use a slightly looser $1.00 before painting the scary banner.
PNL_DRIFT_THRESHOLD_USD = 1.00


def _realized_pnl_drift(db_path: str) -> dict | None:
    """Cross-check the stored realized P&L against an independent FIFO
    recompute over the raw trade ledger.

    The headline "Realized P&L" comes from the stored ``pnl_usd`` column,
    which is written at fill time from the broker's avg-entry basis. When
    the broker re-cost-bases a position, or an orphan SELL slips in, that
    stored number drifts from a clean FIFO walk. We surfaced the $718
    droplet drift (DB $+2.26 vs FIFO $+720.64) only in the orchestrator
    log — invisible to anyone reading the dashboard. Surface it here so a
    diverging headline is obviously flagged instead of silently trusted.

    Returns {db_total, fifo_total, drift, per_strategy, diverged} or None
    if the recompute can't run (missing DB / import). Never raises — a
    dashboard build must not die on a diagnostic.
    """
    if not Path(db_path).exists():
        return None
    try:
        from trading.recompute import recompute_realized_pnl_fifo
        db_total, fifo_total, per_strategy = recompute_realized_pnl_fifo(db_path)
    except Exception as e:  # pragma: no cover - diagnostic must never break build
        logger.warning(f"realized_pnl_drift recompute failed: {e}")
        return None
    drift = round(db_total - fifo_total, 2)
    return {
        "db_total": db_total,
        "fifo_total": fifo_total,
        "drift": drift,
        "per_strategy": per_strategy,
        "diverged": abs(drift) > PNL_DRIFT_THRESHOLD_USD,
    }


def _open_lots_by_symbol() -> dict[str, dict[str, float]]:
    """{normalized_symbol: {strategy: open_qty}} for attribution.

    Primary: FIFO open lots from the local trades ledger. Fallback (when
    the local DB is empty/fresh — e.g. a rebuilt droplet): reconstruct
    from Supabase's persistent trade history, which survives local DB
    resets. This is what kills "<unattributed>" after a DB wipe.
    """
    db_path = os.environ.get("TRADING_DB_PATH", "data/trading_performance.db")
    if Path(db_path).exists():
        try:
            from trading.recompute import fifo_open_positions
            lots = fifo_open_positions(db_path)
            if lots:
                return lots
        except Exception as e:
            logger.debug(f"fifo_open_positions failed: {e}")

    # Supabase fallback — rebuild a FIFO from the persisted ledger.
    try:
        from common.supabase_store import SupabaseStore
        from trading.recompute import normalize_symbol
        sb = SupabaseStore()
        if not sb.is_configured():
            return {}
        rows = sb.recent_trades(limit=5000)
        from collections import defaultdict, deque
        books: dict[tuple[str, str], deque] = defaultdict(deque)
        for r in sorted(rows, key=lambda x: str(x.get("timestamp", ""))):
            qty = float(r.get("quantity") or 0)
            px = float(r.get("price") or 0)
            if qty <= 0 or px <= 0:
                continue
            key = (r.get("strategy"), normalize_symbol(r.get("product_id")))
            if (r.get("side") or "").upper() == "BUY":
                books[key].append(qty)
            else:
                rem = qty
                while rem > 0 and books[key]:
                    take = min(books[key][0], rem)
                    books[key][0] -= take
                    rem -= take
                    if books[key][0] <= 1e-12:
                        books[key].popleft()
        out: dict[str, dict[str, float]] = {}
        for (strat, sym), q in books.items():
            tot = sum(q)
            if tot > 1e-9 and strat:
                out.setdefault(sym, {})[strat] = tot
        return out
    except Exception as e:
        logger.debug(f"supabase open-lots fallback failed: {e}")
        return {}


def _live_unrealized_by_strategy() -> dict[str, float]:
    """Per-strategy unrealized P&L from live broker positions, attributed
    via FIFO open lots (symbol-normalized, multi-strategy proportional).

    Each broker position's unrealized P&L is split across the strategies
    holding open lots in that (normalized) symbol, proportional to each
    strategy's open quantity. A position with NO matching ledger lots —
    genuinely external or opened before tracking — is labelled
    "<unattributed: no ledger entry>" so it's honest, not silent. Empty
    dict when broker creds aren't configured.
    """
    out: dict[str, float] = {}
    try:
        from brokers.registry import build_brokers
        brokers = build_brokers()
    except Exception:
        return out
    if not brokers:
        return out

    from trading.recompute import normalize_symbol
    open_lots = _open_lots_by_symbol()       # {norm_symbol: {strategy: qty}}

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
            weights = open_lots.get(normalize_symbol(p.symbol))
            if weights:
                total_qty = sum(weights.values())
                # Split this position's unrealized across the strategies
                # that hold it, proportional to open quantity.
                for strat, qty in weights.items():
                    share = (qty / total_qty) if total_qty > 0 else 0.0
                    out[strat] = out.get(strat, 0.0) + unrealized * share
            else:
                out["<unattributed: no ledger entry>"] = (
                    out.get("<unattributed: no ledger entry>", 0.0)
                    + unrealized)
    return out


# Built-in leverage factor of known leveraged ETFs. A $1k TQQQ position
# is $3k of economic Nasdaq exposure, so counting it at face notional
# UNDERSTATES true leverage — which matters now that leveraged_momentum
# and leveraged_champions hold these. Multiply face notional by these
# to get ECONOMIC exposure for the gross-leverage stat.
_LEVERAGED_ETF_FACTOR: dict[str, float] = {
    "TQQQ": 3.0, "UPRO": 3.0, "SOXL": 3.0, "TNA": 3.0, "TMF": 3.0,
    "SPXL": 3.0, "TECL": 3.0, "FAS": 3.0, "LABU": 3.0, "UDOW": 3.0,
    "QLD": 2.0, "SSO": 2.0, "UGL": 2.0,
}


def _portfolio_leverage(equity_usd: float) -> float | None:
    """Gross ECONOMIC leverage = Σ(|market_value| × etf_leverage_factor)
    ÷ equity across every live broker position. None when no brokers/
    creds or zero equity.

    Leveraged-ETF positions are scaled by their built-in factor (TQQQ ×3,
    SSO ×2, …) so the figure reflects true economic exposure — a 3x
    sleeve held at $2k face shows as $6k of exposure, not $2k. Plain
    positions count at face (factor 1.0). Answers "how levered am I
    really?" — the leveraged_momentum / leveraged_champions sleeves and
    any margin all surface here.
    """
    if not equity_usd or equity_usd <= 0:
        return None
    try:
        from brokers.registry import build_brokers
        brokers = build_brokers()
    except Exception:
        return None
    if not brokers:
        return None
    gross = 0.0
    saw_position = False
    for venue, adapter in brokers.items():
        try:
            for p in adapter.get_positions():
                face = abs(float(p.market_price or 0.0)
                           * float(p.quantity or 0.0))
                factor = _LEVERAGED_ETF_FACTOR.get(
                    str(getattr(p, "symbol", "")).upper(), 1.0)
                gross += face * factor
                saw_position = True
        except Exception as e:
            logger.debug(f"[{venue}] get_positions for leverage: {e}")
            continue
    if not saw_position:
        return 0.0
    return gross / equity_usd


def _recent_trades(limit: int = 50) -> list[dict]:
    """Last N trades, newest first.

    Same two-source pattern as _recent_cycles:
      1. docs/trades_recent.json (committed by orchestrator) — primary
      2. trading_performance.db.trades — fallback
    """
    json_path = Path("docs/trades_recent.json")
    if json_path.exists():
        try:
            import json as _json
            rows = _json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return rows[:limit]
        except Exception as e:
            logger.warning(f"trades_recent.json read failed: {e}")

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


def _read_heartbeat() -> dict | None:
    """Read the orchestrator heartbeat. Disambiguates 'not running'
    from 'running but diagnostics empty'."""
    db_path = os.environ.get(
        "TRADING_DB_PATH", "data/trading_performance.db"
    )
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT timestamp, git_sha FROM cycle_heartbeat WHERE id = 1"
            ).fetchone()
        if row:
            return {"timestamp": row["timestamp"], "git_sha": row["git_sha"]}
    except sqlite3.OperationalError:
        # Table doesn't exist yet (first deploy after this PR).
        return None
    except Exception as e:
        logger.warning(f"heartbeat read failed: {e}")
    return None


def _recent_cycles(limit: int = 5) -> list[dict]:
    """Last N cycle diagnostics for the dashboard's 'Cycle activity'
    panel. Empty when the table doesn't exist yet (first deploy after
    PR #16 lands).

    Two sources, in order:
      1. docs/cycle_status.json (committed by orchestrator) — primary
         source after PR with this comment lands. Bypasses the
         actions/cache flow entirely.
      2. trading_performance.db.cycle_diagnostics (cache-restored) —
         fallback for backwards-compat with cycles written before
         the JSON-dump path existed.
    """
    # Primary: docs/cycle_status.json
    json_path = Path("docs/cycle_status.json")
    if json_path.exists():
        try:
            import json as _json
            buf = _json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(buf, list) and buf:
                # JSON has newest at end; reverse for newest-first
                # then trim to limit
                rev = list(reversed(buf))[:limit]
                # Normalize: rename proposals_submitted (already named)
                return rev
        except Exception as e:
            logger.warning(f"cycle_status.json read failed: {e}")

    # Fallback: SQLite cycle_diagnostics
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


def _latest_activity_by_strategy(cycles: list[dict]) -> dict[str, dict]:
    """Pull the most recent ACTIVITY classification per strategy from the
    cycle-diagnostics ring buffer.

    Walks cycles newest→oldest and takes the first non-empty `activity`
    for each strategy, so a strategy that was skipped (VENUE_CLOSED) on
    the very last tick still shows its last *meaningful* status. Returns
    {strategy: {activity, held_positions, held_usd, reject_reasons}}.
    """
    out: dict[str, dict] = {}
    # cycles arrive oldest→newest from _recent_cycles; reverse to prefer
    # the freshest meaningful status.
    for cyc in reversed(cycles or []):
        for name, o in (cyc.get("strategy_outcomes") or {}).items():
            if not isinstance(o, dict):
                continue
            act = o.get("activity") or ""
            if name in out and out[name].get("activity"):
                continue  # already have a fresher meaningful status
            # Skip pure VENUE_CLOSED until we've found something better,
            # but keep it as a fallback so the cell isn't blank.
            if not act:
                continue
            out.setdefault(name, {
                "activity": act,
                "held_positions": o.get("held_positions", 0),
                "held_usd": o.get("held_usd", 0.0),
                "reject_reasons": o.get("reject_reasons", []) or [],
                "target_alloc_usd": o.get("target_alloc_usd", 0.0),
                "target_alloc_pct": o.get("target_alloc_pct", 0.0),
            })
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


# Strategy-group display metadata: emoji + human label + display order
# (top = highest-conviction / largest sleeves, bottom = experimental
# / overlay / informational). Drives the group-header rows in the
# strategy table.
_GROUP_DISPLAY: dict[str, tuple[str, str, int]] = {
    "TREND":         ("📈", "Trend-following",            10),
    "FACTOR":        ("📊", "Equity factor",              20),
    "MACRO":         ("🌍", "Global macro",               25),
    "CARRY":         ("💰", "Carry & yield",              30),
    "DEFENSIVE":     ("🛡️", "Defensive / risk-parity",   40),
    "MEAN_REVERSION":("🔄", "Mean reversion",             50),
    "EVENT":         ("📰", "Event-driven",               55),
    "CRYPTO":        ("₿",  "Crypto",                     60),
    "PREDICTION":    ("🎯", "Prediction markets",         70),
    "LEVERAGED":     ("⚡", "Leveraged (3x)",             80),
    "OVERLAY":       ("⚙️", "Overlays (no trades)",       90),
    "OTHER":         ("•",  "Other",                       95),
}


def _row_html(name: str, meta: dict, pnl: dict, mode: str, rank: int | None = None) -> str:
    venue = meta.get("venue", "—") if meta else "—"
    mode_color, mode_label = _MODE_BADGE.get(mode, ("#4b5563", mode))
    realized = pnl.get("realized_pnl_usd", 0.0)
    unrealized = pnl.get("unrealized_pnl_usd", 0.0)
    total = realized + unrealized
    realized_color = "#166534" if realized > 0 else ("#7f1d1d" if realized < 0 else "#4b5563")
    unrealized_color = "#166534" if unrealized > 0 else ("#7f1d1d" if unrealized < 0 else "#4b5563")
    total_color = "#166534" if total > 0 else ("#7f1d1d" if total < 0 else "#4b5563")
    days_since = pnl.get("days_since")
    if days_since is not None:
        last_label = f"{days_since:g}d ago"
        # Health colour for "Last trade": green ≤3d, amber 4-14d, red >14d
        last_color = ("#166534" if days_since <= 3
                      else "#b45309" if days_since <= 14
                      else "#7f1d1d")
    else:
        last_label = "never"
        last_color = "#7f1d1d"
    # Only rank strategies that have actually traded; idle ones show "—".
    rank_label = str(rank) if (rank is not None and pnl.get("n_closed", 0)) else "—"
    lev = float((meta or {}).get("leverage_x", 1.0))
    lev_color = "#7f1d1d" if lev > 1.5 else "#6b7280"
    lev_label = f"{lev:g}x"
    # Allocated capital: prefer the live per-cycle figure; fall back to
    # the configured target_pct (× $100k paper book) so the column is
    # populated even before the first cycle records an allocation.
    alloc_usd = (pnl or {}).get("target_alloc_usd") or 0.0
    alloc_pct = ((pnl or {}).get("target_alloc_pct")
                 or (meta or {}).get("target_alloc_pct") or 0.0)
    if not alloc_usd and alloc_pct:
        alloc_usd = alloc_pct * 100_000.0   # paper-book baseline estimate
    alloc_label = (f"{_fmt_money(alloc_usd)}"
                   if alloc_usd else
                   (f"{alloc_pct*100:.1f}%" if alloc_pct else "—"))
    return (
        f"<tr>"
        f"<td class=num style=\"color:#6b7280\">{rank_label}</td>"
        f"<td><strong>{html.escape(name)}</strong>"
        + (f"<br><span class=desc>{html.escape(meta.get('description',''))}</span>" if meta else "")
        + f"</td>"
        f"<td>{html.escape(venue)}</td>"
        f"<td class=num style=\"color:{lev_color};font-weight:600\">{lev_label}</td>"
        f"<td class=num title=\"target {alloc_pct*100:.1f}%\">{alloc_label}</td>"
        f"<td><span class=badge style=\"background:{mode_color}\">{mode_label}</span></td>"
        f"<td class=num>{pnl.get('n_closed', 0)}</td>"
        f"<td class=num>{_fmt_pct(pnl.get('win_rate', 0.0))}</td>"
        f"<td class=num style=\"color:{realized_color}\">{_fmt_money(realized)}</td>"
        f"<td class=num style=\"color:{unrealized_color}\">{_fmt_money(unrealized)}</td>"
        f"<td class=num style=\"color:{total_color};font-weight:600\">{_fmt_money(total)}</td>"
        f"<td>{_activity_badge(pnl)}</td>"
        f"<td style=\"color:{last_color};font-weight:500\">{html.escape(last_label)}</td>"
        f"</tr>"
    )


# Activity status → (background, text colour, label). The dashboard's
# honesty fix: distinguishes a correctly-HOLDING low-turnover book from a
# BLOCKED one whose orders are all rejected, from a WAITING (no-signal)
# one — instead of lumping all three under a scary "stale".
_ACTIVITY_STYLE: dict[str, tuple[str, str]] = {
    "TRADING":      ("#dcfce7", "#166534"),
    "HOLDING":      ("#dbeafe", "#1e40af"),
    "BLOCKED":      ("#fee2e2", "#991b1b"),
    "WAITING":      ("#f3f4f6", "#6b7280"),
    "NO_ALLOC":     ("#fef9c3", "#854d0e"),
    "FROZEN":       ("#e5e7eb", "#374151"),
    "VENUE_CLOSED": ("#f3f4f6", "#9ca3af"),
    "ERROR":        ("#fee2e2", "#991b1b"),
}


def _activity_badge(pnl: dict) -> str:
    """Render the live activity status as a coloured chip, with a tooltip
    that explains HOLDING (held positions) or BLOCKED (reject reason)."""
    act = (pnl or {}).get("activity") or ""
    if not act:
        return '<span style="color:#d1d5db">—</span>'
    bg, fg = _ACTIVITY_STYLE.get(act, ("#f3f4f6", "#6b7280"))
    held_n = (pnl or {}).get("held_positions", 0) or 0
    held_usd = (pnl or {}).get("held_usd", 0.0) or 0.0
    title = act
    sub = ""
    if act == "HOLDING" and held_n:
        sub = f"{held_n}@{_fmt_money(held_usd)}"
        title = f"Holding {held_n} positions worth {_fmt_money(held_usd)} — low-turnover, working as designed"
    elif act == "BLOCKED":
        rr = (pnl or {}).get("reject_reasons") or []
        title = "All proposals rejected: " + ("; ".join(rr[-2:]) if rr else "risk/cap veto")
    elif act == "WAITING":
        title = "No signal / regime gate not met — flat, awaiting entry"
    elif act == "NO_ALLOC":
        title = "Allocator gave $0 this cycle"
    badge = (
        f'<span class=badge style="background:{bg};color:{fg}" '
        f'title="{html.escape(title)}">{html.escape(act)}</span>'
    )
    if sub:
        badge += f'<br><span class=desc>{html.escape(sub)}</span>'
    return badge


def _group_header_row(group: str, members: list[tuple[str, dict, dict, str]]) -> str:
    """Visual section header inside the strategy table: emoji + name +
    count + group P&L subtotal. Spans all 10 columns."""
    emoji, label, _ = _GROUP_DISPLAY.get(group, _GROUP_DISPLAY["OTHER"])
    subtotal = sum(
        (p.get("realized_pnl_usd", 0.0) + p.get("unrealized_pnl_usd", 0.0))
        for _, _, p, _ in members
    )
    color = "#166534" if subtotal > 0 else ("#7f1d1d" if subtotal < 0 else "#4b5563")
    return (
        f'<tr data-group-header="1" style="background:#eef2f7">'
        f'<td colspan=10 style="padding:8px 12px;font-weight:600;'
        f'color:#1f2937;font-size:13px;'
        f'border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb">'
        f'{emoji}&nbsp;&nbsp;{html.escape(label)}'
        f'<span class="group-pill">{html.escape(group)}</span>'
        f'<span style="font-weight:400;color:#6b7280;margin-left:8px">'
        f'· {len(members)} strateg{"y" if len(members)==1 else "ies"}</span>'
        f'</td>'
        f'<td class=num style="color:{color};font-weight:600;'
        f'border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb">'
        f'{_fmt_money(subtotal)}</td>'
        f'<td colspan=2 style="border-top:1px solid #e5e7eb;'
        f'border-bottom:1px solid #e5e7eb"></td></tr>'
    )


def _grouped_body_rows(rows: list[tuple[str, dict, dict, str]]) -> str:
    """Insert group-header rows between rank-sorted strategies so the
    table reads like a clean roster organized by edge type."""
    if not rows:
        return ""
    # Bucket by group (stable order: by group display priority, then
    # original rank inside each group — already sorted by total P&L desc).
    by_group: dict[str, list[tuple]] = {}
    for r in rows:
        g = (r[1].get("group") or "OTHER") if r[1] else "OTHER"
        by_group.setdefault(g, []).append(r)
    out_html: list[str] = []
    rank = 0
    for g in sorted(by_group, key=lambda gg: _GROUP_DISPLAY.get(gg, _GROUP_DISPLAY["OTHER"])[2]):
        members = by_group[g]
        out_html.append(_group_header_row(g, members))
        for n, m, p, md in members:
            rank += 1
            out_html.append(_row_html(n, m, p, md, rank=rank))
    return "\n".join(out_html)


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


def _system_status_line(cycles: list[dict],
                          heartbeat: dict | None = None) -> str:
    """One-line "is the bot alive and what did it do today" header.

    Renders right under the H1 so the operator gets the answer in the
    first 2 seconds without scrolling. Counts only cycles from today
    (UTC) so the numbers don't grow unbounded.

    When `cycles` is empty but `heartbeat` exists, we know the
    orchestrator IS running (heartbeat is written before anything
    else can fail) but diagnostics persistence is broken. That's
    a different message than "never ran".
    """
    from datetime import datetime as _dt, timedelta as _td
    if not cycles:
        # Heartbeat present → orchestrator IS running but diagnostics
        # write is failing. Different problem than "never ran".
        if heartbeat:
            try:
                hb_ts = _dt.fromisoformat(
                    heartbeat["timestamp"].replace("Z", "+00:00")
                ).replace(tzinfo=None)
                hb_ago = _dt.utcnow() - hb_ts
                hb_str = (
                    f"{int(hb_ago.total_seconds())}s ago" if hb_ago < _td(minutes=1)
                    else f"{int(hb_ago.total_seconds()/60)}m ago" if hb_ago < _td(hours=1)
                    else f"{int(hb_ago.total_seconds()/3600)}h ago"
                )
            except Exception:
                hb_str = "unknown"
            sha = heartbeat.get("git_sha") or ""
            return (
                f'<div style="background:#fef3c7;border:1px solid #f59e0b;'
                f'padding:8px 12px;border-radius:6px;margin-bottom:12px;'
                f'font-size:13px">'
                f"🟡 <strong>Orchestrator alive</strong> "
                f"(heartbeat {hb_str}, sha {sha}) but no cycle "
                f"diagnostics written yet. The orchestrator is "
                f"starting cycles but failing before the diagnostics "
                f"persist call. Check Actions tab for the orchestrator "
                f"workflow's stdout."
                f"</div>"
            )
        return (
            '<div style="background:#fef3c7;border:1px solid #f59e0b;'
            'padding:8px 12px;border-radius:6px;margin-bottom:12px;'
            'font-size:13px">'
            "⏳ <strong>Bootstrapping</strong> — orchestrator hasn't "
            "written its first cycle row yet. Wait ~5 min after the "
            "next cron tick."
            "</div>"
        )
    from datetime import datetime as _dt, timedelta as _td
    now = _dt.utcnow()
    today = now.date()
    today_cycles = [c for c in cycles
                    if c["timestamp"][:10] == today.isoformat()]
    last = cycles[0]
    try:
        last_ts = _dt.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        ago = now - last_ts.replace(tzinfo=None)
        ago_str = (
            f"{int(ago.total_seconds())}s ago" if ago < _td(minutes=1)
            else f"{int(ago.total_seconds()/60)}m ago" if ago < _td(hours=1)
            else f"{int(ago.total_seconds()/3600)}h ago"
        )
    except Exception:
        ago_str = "unknown"
    n_today = len(today_cycles)
    err_today = sum(c["n_errors"] for c in today_cycles)
    sub_today = sum(c["proposals_submitted"] for c in today_cycles)
    prop_today = sum(c["proposals_total"] for c in today_cycles)
    status_color = "#15803d" if ago < _td(minutes=15) else "#d97706" if ago < _td(hours=1) else "#7f1d1d"
    status_icon = "🟢" if ago < _td(minutes=15) else "🟠" if ago < _td(hours=1) else "🔴"
    return (
        f'<div style="background:white;border:2px solid {status_color};'
        f'padding:10px 14px;border-radius:8px;margin-bottom:14px;'
        f'font-size:13px;display:flex;flex-wrap:wrap;gap:18px;'
        f'align-items:center">'
        f'<span style="font-size:18px">{status_icon}</span>'
        f'<span><strong>Last cycle:</strong> {ago_str}'
        f' ({last["cycle_seconds"]:.1f}s)</span>'
        f'<span><strong>Today:</strong> {n_today} cycles · '
        f'{prop_today} proposed · {sub_today} submitted · '
        f'{err_today} errors</span>'
        f'</div>'
    )


def _cycles_since_last(cycles: list[dict]) -> dict[str, dict]:
    """Compute, per strategy, how many cycles since last
    proposal/submit/error. Cycles are newest-first.

    Returns dict keyed by strategy name with:
      {"since_proposed": N, "since_submitted": N, "since_error": N}

    N is the index in the buffer where it last happened (0 = current
    cycle), or -1 if never observed in the buffer window.
    """
    out: dict[str, dict] = {}
    for i, c in enumerate(cycles):
        for sname, o in (c.get("strategy_outcomes") or {}).items():
            if sname not in out:
                out[sname] = {
                    "since_proposed": -1,
                    "since_submitted": -1,
                    "since_error": -1,
                }
            entry = out[sname]
            if entry["since_proposed"] < 0 and (o.get("proposed", 0) or 0) > 0:
                entry["since_proposed"] = i
            if entry["since_submitted"] < 0 and (o.get("submitted", 0) or 0) > 0:
                entry["since_submitted"] = i
            if entry["since_error"] < 0 and (o.get("error") or "").strip():
                entry["since_error"] = i
    return out


def _broker_reachability_streaks(cycles: list[dict]) -> dict[str, int]:
    """Per-venue: current consecutive-unreachable streak (0 if last
    cycle was 'ok'). Cycles are newest-first."""
    streaks: dict[str, int] = {}
    if not cycles:
        return streaks
    # Get all venues mentioned across the buffer
    venues = set()
    for c in cycles:
        venues.update((c.get("venue_health") or {}).keys())
    for venue in venues:
        streak = 0
        for c in cycles:
            status = (c.get("venue_health") or {}).get(venue)
            if status == "ok":
                break
            streak += 1
        streaks[venue] = streak
    return streaks


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
    # Venue health badges with reachability streak suffix.
    # Streak count tells the operator if a venue's been down briefly
    # (1 cycle = transient) vs persistently (5+ = real outage).
    streaks = _broker_reachability_streaks(cycles)
    vh = latest.get("venue_health") or {}

    def _venue_badge(vname: str, status: str) -> str:
        ok = status == "ok"
        bg = "#15803d" if ok else "#7f1d1d"
        suffix = ""
        if not ok:
            n = streaks.get(vname, 1)
            suffix = f" ({n} cycle{'' if n == 1 else 's'})"
        return (
            f'<span style="background:{bg};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:11px">'
            f'{html.escape(vname)}: {html.escape(status)}{suffix}'
            f'</span>'
        )
    vh_html = " · ".join(
        _venue_badge(name, v) for name, v in sorted(vh.items())
    ) or "(no venues active)"

    # Per-strategy outcome table — also compute "cycles since last X"
    # for each strategy so the operator can spot a strategy that's
    # gone idle (proposed N cycles ago, hasn't moved since).
    cycles_since = _cycles_since_last(cycles)
    outcomes = latest.get("strategy_outcomes") or {}
    rows = []
    for name in sorted(outcomes.keys()):
        o = outcomes[name]
        skip = o.get("skip_reasons") or []
        rej_reasons = o.get("reject_reasons") or []
        exec_errs = o.get("execute_errors") or []
        err = o.get("error") or ""
        # Color the row by outcome severity. Order matters: explicit
        # execute errors win over generic dry_logged because the
        # outcome counter can't disambiguate the two perfectly.
        if err:
            row_color = "#fee2e2"   # compute() error → red tint
            why = f'<span style="color:#7f1d1d">{html.escape(err[:120])}</span>'
        elif exec_errs:
            row_color = "#fee2e2"   # execution error → red tint
            why = f'<span style="color:#7f1d1d" title="{html.escape(" | ".join(exec_errs))}">EXEC: {html.escape(exec_errs[0][:120])}</span>'
        elif o.get("submitted", 0) > 0:
            row_color = "#dcfce7"   # submitted → green
            why = f'<span style="color:#166534">submitted {o["submitted"]} order(s)</span>'
        elif rej_reasons:
            row_color = "#fef3c7"   # rejected → amber
            why = f'<span style="color:#92400e" title="{html.escape(" | ".join(rej_reasons))}">REJ: {html.escape(rej_reasons[0][:120])}</span>'
        elif o.get("dry_logged", 0) > 0:
            row_color = "#dbeafe"   # DRY → blue
            why = f'<span style="color:#1e40af">{o["dry_logged"]} DRY-logged</span>'
        elif skip:
            row_color = "#f3f4f6"   # skipped → gray
            why = html.escape(", ".join(skip[:3]))
        else:
            row_color = "white"
            why = '<span style="color:#6b7280">—</span>'
        # "Cycles since last *" — shows -1 (formatted as "—") if
        # the strategy hasn't done it inside the buffer window.
        cs = cycles_since.get(name, {})
        sub_cells = cs.get("since_submitted", -1)
        prop_cells = cs.get("since_proposed", -1)
        last_sub_str = "—" if sub_cells < 0 else (
            "now" if sub_cells == 0 else f"{sub_cells}c ago"
        )
        last_prop_str = "—" if prop_cells < 0 else (
            "now" if prop_cells == 0 else f"{prop_cells}c ago"
        )
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
            f"<td style='text-align:right;font-size:10px;color:#6b7280'>{last_prop_str}</td>"
            f"<td style='text-align:right;font-size:10px;color:#6b7280'>{last_sub_str}</td>"
            f"<td>{why}</td>"
            f"</tr>"
        )
    rows_html = "\n".join(rows) or '<tr><td colspan="12">(no strategies ran)</td></tr>'

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
      <th style='text-align:right;font-size:10px' title='Cycles since last proposal'>Last prop.</th>
      <th style='text-align:right;font-size:10px' title='Cycles since last successful submit'>Last sub.</th>
      <th>Outcome / why</th></tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""


def _suggest_actions(cycles: list[dict], trades: list[dict],
                       diag: dict) -> list[str]:
    """Look at the data and produce actionable suggestions.

    Each item is an HTML string ready to render. Empty list when
    everything looks healthy. The list is the operator-facing
    answer to "what should I do right now?" — surfaces the few
    things they can act on without reading source code.
    """
    out: list[str] = []
    # No cycles ever recorded.
    if not cycles:
        out.append(
            "🟡 <strong>Bootstrapping</strong> — no cycle diagnostics "
            "yet. Wait one cron tick. If this persists past 10 min, "
            "check the Actions tab for orchestrator workflow failures."
        )
        return out

    latest = cycles[0]
    # Stale cycle (last write > 15 min ago)
    from datetime import datetime as _dt, timedelta as _td
    try:
        last_ts = _dt.fromisoformat(
            latest["timestamp"].replace("Z", "+00:00")
        ).replace(tzinfo=None)
        if _dt.utcnow() - last_ts > _td(minutes=15):
            out.append(
                "🔴 <strong>Cycles have stalled</strong> — last cycle "
                f'was {latest["timestamp"]}, more than 15 min ago. '
                "Check the Actions tab for orchestrator failures."
            )
    except Exception:
        pass

    # Strategies with 0 proposals across last 5 cycles → likely
    # config issue (missing API key, no signal, frozen).
    if cycles and latest.get("strategy_outcomes"):
        chronic_idle = []
        for sname, _ in latest["strategy_outcomes"].items():
            never_proposed = all(
                (c.get("strategy_outcomes", {}).get(sname, {}).get("proposed", 0) or 0) == 0
                for c in cycles
            )
            if never_proposed:
                chronic_idle.append(sname)
        if chronic_idle and len(chronic_idle) > 2:
            out.append(
                "🟡 <strong>{} strategies have proposed nothing in "
                "the last {} cycles</strong>: <code>{}</code>. "
                "Likely causes: missing API key, FROZEN state, or "
                "no signal. Check the per-strategy outcome panel "
                "for the specific reason.".format(
                    len(chronic_idle), len(cycles),
                    ", ".join(sorted(chronic_idle)[:6])
                    + (" …" if len(chronic_idle) > 6 else "")
                )
            )

    # No trades recorded at all
    n_filled = sum(
        1 for t in trades
        if t.get("fill_status", "").upper() == "FILLED" and not t.get("dry_run")
    )
    if not trades:
        out.append(
            "🟡 <strong>No trades recorded yet</strong>. "
            "If a strategy proposed-and-submitted in the panel below "
            "but trades.db is empty, check the dead-letter table: "
            "<code>SELECT * FROM record_trade_dead_letter</code>"
        )
    elif n_filled == 0 and any(not t.get("dry_run") for t in trades):
        out.append(
            "🟠 <strong>Live orders submitted but none filled yet</strong>. "
            "Check broker side: insufficient funds, market closed, "
            "or rate limits. Coinbase wallet balance is the usual "
            "culprit for INSUFFICIENT_FUND errors."
        )

    # ALLOW_LIVE_TRADING is set but LIVE_STRATEGIES is empty
    live_set = diag.get("_live_strats_set") or set()
    allow_live = diag.get("_allow_live")
    if allow_live and not live_set:
        out.append(
            "ℹ <strong>ALLOW_LIVE_TRADING is on but LIVE_STRATEGIES is empty</strong>. "
            "Per-strategy LIVE override is moot; only the per-venue flags drive live trading. "
            "If that's intentional, ignore this message."
        )

    return out


def _read_benchmark() -> dict | None:
    """docs/benchmark.json written by the orchestrator's
    write_benchmark_json(). None if not yet produced."""
    p = Path("docs/benchmark.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"benchmark.json read failed: {e}")
        return None


def _read_data_quality() -> dict | None:
    p = Path("docs/data_quality.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"data_quality.json read failed: {e}")
        return None


def _read_portfolio_intel() -> dict | None:
    p = Path("docs/portfolio_intel.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"portfolio_intel.json read failed: {e}")
        return None


def _render_portfolio_intel(pi: dict | None) -> str:
    if not pi:
        return ""
    concentrated = pi.get("concentrated")
    border = "#b45309" if concentrated else "#15803d"
    enb = pi.get("effective_bets")
    avg = pi.get("avg_pairwise_corr")
    rows = "".join(
        f"<tr><td>{html.escape(p['a'])}</td><td>{html.escape(p['b'])}</td>"
        f"<td class=num>{p['corr']:+.2f}</td></tr>"
        for p in (pi.get("top_correlated_pairs") or [])[:6]
    )
    pairs_tbl = (
        f"<table style='margin-top:8px;font-size:12px'><thead><tr>"
        f"<th>Strategy</th><th>Strategy</th><th class=num>Corr</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>" if rows else "")
    return (
        f'<section id="portfolio-intel"><h2>Portfolio intelligence — '
        f'diversification</h2>'
        f'<div style="background:white;border:2px solid {border};'
        f'border-radius:8px;padding:12px 14px;margin-bottom:14px">'
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:14px">'
        f'<span><strong>Effective bets:</strong> '
        f'{enb if enb is not None else "—"} of '
        f'{pi.get("n_strategies_compared","—")}</span>'
        f'<span><strong>Avg pairwise corr:</strong> '
        f'{f"{avg:+.2f}" if avg is not None else "—"}</span></div>'
        f'<p style="font-size:13px;margin:8px 0 0">'
        f'{html.escape(pi.get("verdict",""))}</p>'
        f'{pairs_tbl}</div></section>')


def _render_data_quality(dq: dict | None) -> str:
    if not dq:
        return ""
    rows = []
    for r in (dq.get("rows") or []):
        status = r.get("status", "?")
        bg = {"OK": "#dcfce7", "STALE": "#fef3c7",
              "MISSING": "#fee2e2", "INVALID": "#fee2e2",
              "INCONSISTENT": "#fee2e2", "EMPTY": "#f3f4f6"
              }.get(status, "white")
        key = r.get("file") or r.get("check") or "?"
        rows.append(
            f'<tr style="background:{bg}">'
            f"<td><code>{html.escape(str(key))}</code></td>"
            f"<td>{html.escape(status)}</td>"
            f'<td style="text-align:right">'
            f'{r.get("age_hours","—")}</td>'
            f'<td style="font-size:11px;color:#4b5563">'
            f'{html.escape(str(r.get("reason",""))[:160])}</td>'
            "</tr>"
        )
    score = dq.get("score", "?")
    return f"""
<h2 id="data-quality" style="font-size:16px;margin-top:28px;margin-bottom:8px">
  Data quality — {score}/10
  <span style="font-weight:400;color:#6b7280;font-size:12px">
  (as of {html.escape((dq.get("as_of") or "")[:19])} —
  audits every JSON the dashboard reads each cycle)</span>
</h2>
<table style="font-size:12px">
  <thead><tr><th>File / Check</th><th>Status</th>
      <th style="text-align:right">Age (h)</th>
      <th>Notes</th></tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>"""


def _read_self_grade() -> dict | None:
    p = Path("docs/self_grade.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"self_grade.json read failed: {e}")
        return None


def _read_hedge_funds() -> dict | None:
    p = Path("docs/hedge_fund_13f.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"hedge_fund_13f.json read failed: {e}")
        return None


def _render_grade_hero(g: dict | None) -> str:
    """Hero card: overall grade + per-axis bars + narrative cards."""
    if not g:
        return (
            '<div class="grade-hero">'
            '<div class="score"><div class="big">—</div>'
            '<div class="of">/10</div>'
            '<div class="target">grade pending first cycle</div></div>'
            '<div class="right">'
            '<h3>Self-grade</h3>'
            '<p style="font-size:12px;color:#cbd5e1;margin:4px 0">'
            "Will populate after the next orchestrator cycle runs "
            "<code>run_self_grade()</code>.</p>"
            "</div></div>"
        )
    overall = g.get("overall_grade", 0)
    target = g.get("target_grade", 9.0)
    comps = g.get("components") or {}
    nar = g.get("narrative") or {}

    def _bar_color(s):
        if s >= 8: return "#22c55e"
        if s >= 6: return "#84cc16"
        if s >= 4: return "#facc15"
        if s >= 2: return "#f97316"
        return "#ef4444"

    bars = []
    for k, v in comps.items():
        s = v.get("score", 0)
        pct = max(0, min(100, s * 10))
        col = _bar_color(s)
        bars.append(
            f'<div class="bar-row">'
            f'<div class="label">{html.escape(k.replace("_"," "))}</div>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct}%;background:{col}"></div></div>'
            f'<div class="score-cell">{s}</div>'
            f"</div>"
        )

    def _cards(label, key, color):
        items = nar.get(key) or []
        if not items:
            return ""
        lis = "".join(f"<li>{html.escape(str(x))}</li>" for x in items[:4])
        return (
            f'<div class="card"><h4 style="color:{color}">{label}</h4>'
            f'<ul>{lis}</ul></div>'
        )

    score_color = _bar_color(overall)
    return f"""
<div class="grade-hero" id="grade">
  <div class="score">
    <div class="big" style="color:{score_color}">{overall}</div>
    <div class="of">/ 10</div>
    <div class="target">target ≥ {target}</div>
  </div>
  <div class="right">
    <h3>Self-grade — {html.escape((g.get("as_of") or "")[:10])}</h3>
    <div class="bars">{''.join(bars)}</div>
    <div class="narrative-grid">
      {_cards("Tried", "tried", "#a5b4fc")}
      {_cards("Worked", "worked", "#86efac")}
      {_cards("Failed", "failed", "#fca5a5")}
      {_cards("Next", "next", "#fcd34d")}
    </div>
  </div>
</div>"""


def _render_hedge_funds(hf: dict | None) -> str:
    """Hedge-fund 13F panel — most recent filings, link out."""
    if not hf or not (hf.get("filings") or []):
        return ""
    rows = []
    for f in (hf.get("filings") or [])[:10]:
        rows.append(
            f"<tr>"
            f"<td><strong>{html.escape(f.get('fund',''))}</strong></td>"
            f"<td>{html.escape(f.get('filed_date',''))}</td>"
            f"<td>{html.escape(f.get('form',''))}</td>"
            f'<td><a href="{html.escape(f.get("primary_doc_url",""))}" '
            f'target="_blank" rel=noopener>open filing</a></td>'
            f"</tr>"
        )
    return f"""
<h2 id="hedge-funds" style="font-size:16px;margin-top:28px;margin-bottom:8px">
  Top alpha funds — most recent 13F filings
  <span style="font-weight:400;color:#6b7280;font-size:12px">
  (as of {html.escape((hf.get("as_of") or "")[:19])} — quarterly,
  filed 45d after quarter-end)</span>
</h2>
<p style="color:#6b7280;font-size:11px;margin:0 0 6px">
  Renaissance / Two Sigma / Citadel / Millennium / Bridgewater /
  AQR / DE Shaw. We learn from what they're holding; this scout
  surfaces filings — actual holdings parsing is a follow-up.
</p>
<table style="font-size:12px">
  <thead><tr><th>Fund</th><th>Filed</th><th>Form</th><th>Filing</th></tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>"""


def _render_nav() -> str:
    """Sticky in-page nav so the user can jump to any panel."""
    return ('<div class="nav">'
            '<strong style="color:#111827">JUMP →</strong>'
            '<a href="#grade">Self-grade</a>'
            '<a href="#improvements">Action queue</a>'
            '<a href="#validation">Validation</a>'
            '<a href="#hedge-funds">Hedge funds</a>'
            '<a href="#data-quality">Data quality</a>'
            '<a href="#strategies">Strategies</a>'
            '<a href="#trades">Trades</a>'
            "</div>")


def _read_improvements() -> dict | None:
    """docs/improvements.json written by run_performance_review()."""
    p = Path("docs/improvements.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"improvements.json read failed: {e}")
        return None


def _render_improvements(imp: dict | None) -> str:
    """Performance-review action queue — what to fix next, ranked."""
    if not imp:
        return ""
    rows = imp.get("strategies") or []
    pri_color = {1: "#fee2e2", 2: "#fde68a", 3: "#fef3c7",
                  4: "#dbeafe", 5: "#dcfce7", 6: "#f3f4f6", 9: "white"}
    pri_label = {1: "P1 LIVE BLEED", 2: "P2 OVERFIT", 3: "P3 FAIL",
                  4: "P4 GATHERING", 5: "P5 OK", 6: "P6 UNPROVEN"}
    body = []
    for r in rows[:30]:
        bg = pri_color.get(r["priority"], "white")
        live_s = r.get("live_sharpe_30d")
        live_str = (f"{live_s:+.2f}" if isinstance(live_s, (int, float))
                     else "—")
        bt_s = r.get("backtest_sharpe_5y")
        bt_str = (f"{bt_s:+.2f}" if isinstance(bt_s, (int, float))
                   else "—")
        body.append(
            f'<tr style="background:{bg}">'
            f'<td><strong>{html.escape(r["strategy"])}</strong></td>'
            f"<td>{pri_label.get(r['priority'],'?')}</td>"
            f"<td>{html.escape(r.get('backtest_verdict',''))}</td>"
            f"<td>{html.escape(r.get('walk_forward_verdict',''))}</td>"
            f'<td style="text-align:right">{bt_str}</td>'
            f'<td style="text-align:right">{live_str}</td>'
            f'<td style="text-align:right">{r.get("live_n_trades_30d","—")}</td>'
            f'<td style="text-align:right">${r.get("live_pnl_30d",0):+.2f}</td>'
            f'<td style="font-size:11px;color:#4b5563">'
            f'{html.escape(r.get("action_reason","")[:130])}</td>'
            "</tr>"
        )
    pc = imp.get("priority_counts") or {}
    setup = imp.get("setup_issues") or []
    setup_html = ""
    if setup:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in setup)
        setup_html = (
            '<div style="background:#fee2e2;padding:8px 12px;'
            'margin:6px 0;border-radius:4px;font-size:12px">'
            f"<strong>Setup issues (recurring in last 20 cycles):</strong>"
            f"<ul style='margin:4px 0 0 16px'>{items}</ul></div>")
    return f"""
<h2 id="improvements" style="font-size:16px;margin-top:28px;margin-bottom:8px">
  Performance review — autonomous action queue
  <span style="font-weight:400;color:#6b7280;font-size:12px">
  (as of {html.escape((imp.get("as_of") or "")[:19])} —
  P1:{pc.get('p1',0)} bleed,
  P2:{pc.get('p2',0)} overfit,
  P3:{pc.get('p3',0)} fail,
  P5:{pc.get('p5',0)} ok)</span>
</h2>
<p style="color:#6b7280;font-size:11px;margin:0 0 6px">
  Ranks every strategy by what to do next. P1 (red) =
  live capital being eroded NOW (PASS in backtest but UNDER live).
  P2 (amber) = backtest is suspect (walk-forward says overfit).
  P3 (yellow) = validation FAIL, freeze or retire.
  {html.escape(imp.get("method_notes") or "")}
</p>
{setup_html}
<table style="font-size:12px">
  <thead><tr>
    <th>Strategy</th><th>Action</th>
    <th>Backtest</th><th>Walk-Fwd</th>
    <th style="text-align:right">Sharpe BT</th>
    <th style="text-align:right">Sharpe 30d</th>
    <th style="text-align:right">N 30d</th>
    <th style="text-align:right">P&L 30d</th>
    <th>Why</th>
  </tr></thead>
  <tbody>
{chr(10).join(body)}
  </tbody>
</table>"""


def _read_validation() -> dict | None:
    """docs/validation.json written by run_validation(). None if not
    yet produced."""
    p = Path("docs/validation.json")
    if not p.exists():
        return None
    try:
        import json as _json
        return _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"validation.json read failed: {e}")
        return None


def _render_validation(val: dict | None) -> str:
    """Strategy validation gate — PASS/FAIL/UNPROVEN per strategy from
    1/2/5y fee-aware backtests. The evidence layer the user's
    retirement-critical mandate demands."""
    if not val:
        return (
            '<h2 style="font-size:16px;margin-top:28px;margin-bottom:8px">'
            "Strategy validation</h2>"
            '<p style="color:#6b7280;font-size:13px">Not yet run. The '
            "orchestrator backtests every strategy across 1/2/5y once "
            "per day; the verdict appears here.</p>"
        )
    strategies = val.get("strategies") or {}
    n_pass = val.get("n_pass", 0)
    n_tot = val.get("n_strategies", len(strategies))
    rub = val.get("rubric", {})
    order = {"PASS": 0, "FAIL": 1, "UNPROVEN": 2, "NO_DATA": 3}
    rows = []
    for name in sorted(strategies,
                        key=lambda n: (order.get(
                            strategies[n].get("verdict"), 9), n)):
        v = strategies[name]
        verdict = v.get("verdict", "?")
        bg = {"PASS": "#dcfce7", "FAIL": "#fee2e2",
              "UNPROVEN": "#fef3c7", "NO_DATA": "#f3f4f6"}.get(
            verdict, "white")
        fg = {"PASS": "#166534", "FAIL": "#7f1d1d",
              "UNPROVEN": "#92400e", "NO_DATA": "#6b7280"}.get(
            verdict, "#111")
        s5 = v.get("sharpe_5y")
        s5s = f"{s5:+.2f}" if isinstance(s5, (int, float)) else "—"
        rows.append(
            f'<tr style="background:{bg}">'
            f"<td><strong>{html.escape(name)}</strong></td>"
            f'<td style="color:{fg};font-weight:700">{verdict}</td>'
            f'<td style="text-align:right">{s5s}</td>'
            f'<td style="text-align:right">{v.get("n_trades_5y","—")}</td>'
            f'<td style="text-align:right">{v.get("return_on_volume_pct","—")}</td>'
            f'<td style="font-size:11px;color:#4b5563">'
            f'{html.escape(str(v.get("reason",""))[:140])}</td>'
            f"</tr>"
        )
    rubric_txt = (
        f"PASS rubric: 5y Sharpe ≥ {rub.get('min_sharpe_5y','?')}, "
        f"≥ {rub.get('min_trades_5y','?')} trades, "
        f"return-on-volume > {rub.get('min_return_on_volume_pct','?')}%, "
        f"positive in ≥ {rub.get('min_positive_windows','?')}/3 windows"
    )
    return f"""
<h2 id="validation" style="font-size:16px;margin-top:28px;margin-bottom:8px">
  Strategy validation — {n_pass}/{n_tot} PASS
  <span style="font-weight:400;color:#6b7280;font-size:12px">
  (backtested 1/2/5y, fee-aware; as of {html.escape((val.get("as_of") or "")[:19])})</span>
</h2>
<p style="color:#6b7280;font-size:11px;margin:0 0 6px">{html.escape(rubric_txt)}.
Only PASS strategies are eligible for live promotion; everything
else stays paper-only until it proves an edge.</p>
<table style="font-size:12px">
  <thead><tr><th>Strategy</th><th>Verdict</th>
      <th style="text-align:right">Sharpe 5y</th>
      <th style="text-align:right">Trades 5y</th>
      <th style="text-align:right">RoV %</th>
      <th>Why</th></tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>"""


def _render_benchmark(bench: dict | None) -> str:
    """Portfolio trailing return vs SPY / QQQ / BTC over 7/14/30d."""
    if not bench:
        return ""
    port = bench.get("portfolio") or {}
    pw = port.get("windows") or {}
    bm = bench.get("benchmarks") or {}
    windows = [str(w) for w in bench.get("windows", [7, 14, 30])]

    if not pw and not bm:
        return (
            '<h2 style="font-size:16px;margin-top:28px;margin-bottom:8px">'
            "Performance vs market</h2>"
            '<p style="color:#6b7280;font-size:13px">Benchmark not yet '
            "computed (needs ≥1 equity snapshot + FMP_API_KEY). Will "
            "populate within a cycle or two.</p>"
        )

    def _cell(v) -> str:
        if v is None:
            return '<td style="text-align:right;color:#9ca3af">—</td>'
        color = "#166534" if v > 0 else ("#7f1d1d" if v < 0 else "#4b5563")
        return (f'<td style="text-align:right;color:{color};'
                f'font-weight:600">{v:+.2f}%</td>')

    hdr = "".join(f"<th style='text-align:right'>{w}d</th>" for w in windows)
    rows = []
    # Portfolio row first, bolded
    eq = port.get("current_equity")
    eq_str = f" (equity ${eq:,.0f})" if eq else ""
    rows.append(
        f"<tr style='background:#eff6ff'>"
        f"<td><strong>Portfolio</strong>{eq_str}</td>"
        + "".join(_cell(pw.get(w)) for w in windows)
        + "</tr>"
    )
    for sym, info in bm.items():
        bw = info.get("windows") or {}
        rows.append(
            f"<tr><td>{html.escape(info.get('name', sym))} "
            f"<code style='color:#6b7280'>{html.escape(sym)}</code></td>"
            + "".join(_cell(bw.get(w)) for w in windows)
            + "</tr>"
        )
    # Alpha row: portfolio minus SPY per window
    spy = (bm.get("SPY") or {}).get("windows") or {}
    if spy and pw:
        def _alpha(w):
            a, b = pw.get(w), spy.get(w)
            if a is None or b is None:
                return None
            return round(a - b, 3)
        rows.append(
            "<tr style='background:#fefce8'>"
            "<td><strong>Alpha vs S&amp;P 500</strong></td>"
            + "".join(_cell(_alpha(w)) for w in windows)
            + "</tr>"
        )
    as_of = html.escape((bench.get("as_of") or "")[:19])
    return f"""
<h2 style="font-size:16px;margin-top:28px;margin-bottom:8px">
  Performance vs market <span style="font-weight:400;color:#6b7280;font-size:12px">(as of {as_of})</span>
</h2>
<table style="font-size:12px">
  <thead><tr><th>Series</th>{hdr}</tr></thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
</table>
<p style="color:#6b7280;font-size:11px;margin-top:4px">
  Trailing total return. Portfolio = paper+live equity curve from
  risk_state.db. Benchmarks = FMP adjusted closes. Alpha = portfolio − SPY.
</p>"""


def _render_suggestions(suggestions: list[str]) -> str:
    if not suggestions:
        return ""
    items = "\n".join(f"<li>{s}</li>" for s in suggestions)
    return f"""
<h2 style="font-size: 16px; margin-top: 28px; margin-bottom: 8px;">
  Suggestions ({len(suggestions)})
</h2>
<ul style="padding-left:20px;font-size:13px;line-height:1.5">
{items}
</ul>"""


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
    # Independent FIFO cross-check of the stored realized P&L. None when
    # there's no ledger to walk; surfaced on the headline card so a
    # diverging number is flagged, not silently trusted.
    pnl_drift = _realized_pnl_drift(db_path)
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
    heartbeat = _read_heartbeat()
    benchmark = _read_benchmark()
    validation = _read_validation()
    improvements = _read_improvements()
    self_grade = _read_self_grade()
    hedge_funds = _read_hedge_funds()
    data_quality = _read_data_quality()
    portfolio_intel = _read_portfolio_intel()
    # Live unrealized P&L per strategy — pulled from broker positions
    # at render time. Best-effort; absent or empty when creds missing.
    unrealized_by_strategy = _live_unrealized_by_strategy()
    # Gross leverage (Σ|notional| ÷ equity) — live from broker positions.
    portfolio_leverage = _portfolio_leverage(risk.get("equity_usd", 0.0))
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
    # Fold the latest live ACTIVITY classification (HOLDING / BLOCKED /
    # WAITING / TRADING / …) from the most recent cycle into the
    # per-strategy view. This is the fix for "everything looks stale":
    # a low-turnover book HOLDING positions is no longer indistinguishable
    # from a strategy whose every order is being rejected at the cap.
    activity_by_strategy = _latest_activity_by_strategy(cycles_recent)
    # Only enrich strategies still in the registry or with ledger history —
    # don't resurrect retired strategies that merely linger in the cycle
    # ring buffer.
    _known = set(metas.keys()) | set(pnl.keys())
    for s, info in activity_by_strategy.items():
        if s not in _known:
            continue
        pnl.setdefault(s, {
            "n_trades": 0, "n_closed": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "realized_pnl_usd": 0.0,
            "last_trade_at": None, "days_since": None,
        })
        pnl[s]["activity"] = info.get("activity", "")
        pnl[s]["held_positions"] = info.get("held_positions", 0)
        pnl[s]["held_usd"] = info.get("held_usd", 0.0)
        pnl[s]["reject_reasons"] = info.get("reject_reasons", [])
        pnl[s]["target_alloc_usd"] = info.get("target_alloc_usd", 0.0)
        pnl[s]["target_alloc_pct"] = info.get("target_alloc_pct", 0.0)
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

    # Rank by TOTAL P&L (realized + unrealized) desc — the bottom-line
    # performance the user asked to rank on. Strategies with no closed
    # trades sink to the bottom alphabetically (they're unranked).
    def _total_pnl(p: dict) -> float:
        return p.get("realized_pnl_usd", 0.0) + p.get("unrealized_pnl_usd", 0.0)
    rows.sort(key=lambda r: (
        -1 if r[2].get("n_closed", 0) else 0,
        -_total_pnl(r[2]),
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
    # Activity-health rollup: how many strategies are actually trading.
    # Helps the operator spot the "many sleeves dormant" failure mode at
    # a glance instead of scanning the per-strategy "Last trade" column.
    n_active = sum(
        1 for r in rows
        if r[2].get("days_since") is not None and r[2]["days_since"] <= 3
    )
    n_stale = sum(
        1 for r in rows
        if r[2].get("days_since") is not None
        and 3 < r[2]["days_since"] <= 14
    )
    n_idle = sum(
        1 for r in rows
        if r[2].get("days_since") is None or r[2]["days_since"] > 14
    )
    n_never = sum(1 for r in rows if r[2].get("days_since") is None)
    activity_color = ("#166534" if n_active >= 0.5 * len(rows)
                      else "#b45309" if n_active >= 0.25 * len(rows)
                      else "#7f1d1d")

    def _color_for(v: float) -> str:
        return "#166534" if v > 0 else ("#7f1d1d" if v < 0 else "#4b5563")
    pnl_color = _color_for(total_pnl)
    realized_color = _color_for(total_realized)
    unrealized_color = _color_for(total_unrealized)

    # Realized-P&L trust badge: when the stored ledger total disagrees with
    # the independent FIFO recompute by more than $1, the headline number is
    # NOT trustworthy — show both and flag it loudly so nobody reports a
    # bogus figure (this is the $718 droplet drift made visible).
    realized_badge = ""
    if pnl_drift and pnl_drift["diverged"]:
        realized_color = "#b45309"  # amber — number is in dispute
        realized_badge = (
            f'<div style="font-size:11px;color:#b45309;margin-top:4px;'
            f'line-height:1.3">⚠ FIFO recompute disagrees by '
            f'{_fmt_money(pnl_drift["drift"])}<br>'
            f'(ledger {_fmt_money(pnl_drift["db_total"])} vs FIFO '
            f'{_fmt_money(pnl_drift["fifo_total"])})</div>'
        )
    elif pnl_drift:
        realized_badge = (
            '<div style="font-size:11px;color:#15803d;margin-top:4px">'
            '✓ FIFO-reconciled</div>'
        )

    # Gross-leverage cell. Threshold off the configured cap when we can
    # read it, else a sane 2.0x. >cap = red, >1.25x = amber, else green.
    try:
        from risk.policies import RiskConfig
        _lev_cap = RiskConfig.from_env().leverage_cap or 2.0
    except Exception:
        _lev_cap = 2.0
    if portfolio_leverage is None:
        lev_value, lev_color = "—", "#4b5563"
    else:
        lev_value = f"{portfolio_leverage:.2f}×"
        lev_color = ("#7f1d1d" if portfolio_leverage > _lev_cap
                     else "#b45309" if portfolio_leverage > 1.25
                     else "#166534")

    ks = (risk.get("kill_switch") or "UNKNOWN").upper()
    ks_color, ks_emoji = _KS_COLOR.get(ks, _KS_COLOR["UNKNOWN"])
    # When the switch is latched, the #1 operator question is "why is
    # nothing trading?" — answer it inline so it's never a mystery again.
    if ks == "KILL":
        ks_explainer = ("&nbsp;— ALL new orders are HALTED while latched. "
                        "This is why no trades are flowing. Reset to resume "
                        "(positions are NOT auto-closed by the latch).")
    elif ks == "CRITICAL":
        ks_explainer = ("&nbsp;— closing-only mode: new entries blocked, "
                        "exits still allowed.")
    else:
        ks_explainer = ""

    body_rows = _grouped_body_rows(rows)
    if not rows:
        body_rows = (
            "<tr><td colspan=13 style='text-align:center;color:#6b7280;"
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
  /* ── 2026-05-20 professional refresh ─────────────────────────── */
  body {{ max-width: 1180px; }}
  h1 {{ font-weight: 700; letter-spacing: -0.01em; }}
  h2 {{ font-weight: 600; letter-spacing: -0.01em; padding-bottom: 4px;
        border-bottom: 1px solid #e5e7eb; }}
  .nav {{ position: sticky; top: 0; background: #f9fafb; z-index: 50;
          padding: 6px 0; margin: 0 -4px 16px; border-bottom: 1px solid #e5e7eb;
          font-size: 12px; display: flex; gap: 14px; flex-wrap: wrap;
          backdrop-filter: blur(8px); }}
  .nav a {{ color: #4b5563; text-decoration: none; padding: 2px 6px;
            border-radius: 4px; }}
  .nav a:hover {{ background: #e5e7eb; color: #111827; }}
  .grade-hero {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                 color: white; border-radius: 12px; padding: 18px 22px;
                 margin: 0 0 16px; display: grid;
                 grid-template-columns: 200px 1fr; gap: 22px;
                 box-shadow: 0 4px 12px rgba(15,23,42,0.10); }}
  .grade-hero .score {{ display: flex; flex-direction: column;
                          align-items: center; justify-content: center;
                          border-right: 1px solid rgba(255,255,255,0.15);
                          padding-right: 12px; }}
  .grade-hero .score .big {{ font-size: 56px; font-weight: 700;
                                line-height: 1; letter-spacing: -0.03em;
                                font-variant-numeric: tabular-nums; }}
  .grade-hero .score .of {{ font-size: 14px; color: #94a3b8;
                              margin-top: 2px; }}
  .grade-hero .score .target {{ font-size: 11px; color: #cbd5e1;
                                  margin-top: 8px; text-align: center; }}
  .grade-hero .right h3 {{ margin: 0 0 8px; font-size: 13px;
                              font-weight: 600; color: #cbd5e1;
                              text-transform: uppercase;
                              letter-spacing: 0.06em; }}
  .grade-hero .bars {{ display: grid; gap: 5px; }}
  .grade-hero .bar-row {{ display: grid;
                             grid-template-columns: 150px 1fr 30px;
                             gap: 8px; align-items: center;
                             font-size: 12px; }}
  .grade-hero .bar-row .label {{ color: #cbd5e1; }}
  .grade-hero .bar-track {{ background: rgba(255,255,255,0.10);
                              height: 7px; border-radius: 4px;
                              overflow: hidden; }}
  .grade-hero .bar-fill {{ height: 100%; border-radius: 4px; }}
  .grade-hero .bar-row .score-cell {{ text-align: right;
                                         font-variant-numeric: tabular-nums;
                                         font-weight: 600; color: white; }}
  .narrative-grid {{ display: grid;
                      grid-template-columns: repeat(auto-fit, minmax(220px,1fr));
                      gap: 10px; margin: 8px 0 0; }}
  .narrative-grid .card {{ background: rgba(255,255,255,0.06);
                              border: 1px solid rgba(255,255,255,0.10);
                              border-radius: 6px; padding: 8px 10px;
                              font-size: 11px; }}
  .narrative-grid .card h4 {{ margin: 0 0 4px; font-size: 11px;
                                font-weight: 600;
                                text-transform: uppercase;
                                letter-spacing: 0.06em; color: #94a3b8; }}
  .narrative-grid .card ul {{ margin: 0; padding-left: 14px;
                                color: #e2e8f0; }}
  td.num, th.num, .num-cell {{ font-variant-numeric: tabular-nums; }}
  /* ── 2026-06-03 UI polish: sticky headers, hover, density ─────── */
  table {{ box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
  thead th {{ position: sticky; top: 36px; z-index: 10;
              background: #f3f4f6; box-shadow: inset 0 -1px 0 #e5e7eb; }}
  tbody tr:hover td {{ background: #fafbfc; }}
  tbody tr[data-group-header] {{ position: sticky; top: 72px; z-index: 5; }}
  .stat {{ transition: transform 120ms ease, box-shadow 120ms ease; }}
  .stat:hover {{ transform: translateY(-1px);
                 box-shadow: 0 4px 10px rgba(15,23,42,0.06); }}
  .stat .value {{ letter-spacing: -0.01em; }}
  h2 {{ margin-top: 28px; font-size: 16px; color: #1f2937; }}
  .group-pill {{ display: inline-block; padding: 1px 7px; border-radius: 9999px;
                 background: #eef2ff; color: #4338ca; font-size: 10px;
                 font-weight: 600; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-left: 6px; vertical-align: middle; }}
  @media (max-width: 720px) {{
    body {{ margin: 12px auto; padding: 0 10px; }}
    .grade-hero {{ grid-template-columns: 1fr; }}
    .grade-hero .score {{ border-right: 0;
                            border-bottom: 1px solid rgba(255,255,255,0.15);
                            padding-right: 0; padding-bottom: 10px; }}
    table {{ font-size: 12px; }}
    th, td {{ padding: 7px 6px; }}
    .nav {{ font-size: 11px; gap: 8px; }}
  }}
</style>
</head>
<body>

<h1>AI-AT-ADVENT — Performance Control</h1>
<div class="meta">
  Snapshot: <time data-ts="snapshot">{html.escape(snapshot_at)}</time>
  · Updated every 5 min · <a href="https://github.com/marcoaduartemendes-source/ai-at-advent/actions" target=_blank>Workflows</a>
</div>
{_render_nav()}
{_render_grade_hero(self_grade)}
{_system_status_line(cycles_recent, heartbeat)}

<div class="ks-banner" style="background:{ks_color}">
  <span>{ks_emoji} Kill switch: {html.escape(ks)}<small style="font-weight:400">{ks_explainer}</small></span>
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
    {realized_badge}
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
  <div class="stat" title="Gross notional / equity across all live broker positions. Cap {_lev_cap:.1f}×.">
    <div class="label">Gross leverage</div>
    <div class="value" style="color:{lev_color}">{lev_value}</div>
  </div>
  <div class="stat" title="Strategies traded in the last 3d / 4-14d / never or &gt;14d. Helps spot dormant sleeves at a glance.">
    <div class="label">Activity (3d / stale / idle)</div>
    <div class="value" style="color:{activity_color};font-size:18px">
      <span style="color:#166534">{n_active}</span>
      <span style="color:#9ca3af">·</span>
      <span style="color:#b45309">{n_stale}</span>
      <span style="color:#9ca3af">·</span>
      <span style="color:#7f1d1d">{n_idle}</span>
      <span style="color:#9ca3af;font-size:11px;font-weight:400">({n_never} never)</span>
    </div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th class=num title="Rank by total P&amp;L (traded strategies only)">#</th>
      <th>Strategy</th>
      <th>Venue</th>
      <th class=num title="Notional leverage: 1x = unlevered, 3x = 3x ETF sleeves">Lev</th>
      <th class=num title="Capital allocated to this strategy (target). The allocator Sharpe-tilts around this baseline; the live figure comes from the last cycle.">Alloc $</th>
      <th>Mode</th>
      <th class=num>Closed</th>
      <th class=num>Win rate</th>
      <th class=num>Realized</th>
      <th class=num>Unrealized</th>
      <th class=num>Total P&amp;L</th>
      <th title="Live status from the last cycle: TRADING (submitted), HOLDING (owns positions, low-turnover), BLOCKED (proposals rejected at cap/risk), WAITING (no signal), NO_ALLOC, FROZEN">Status</th>
      <th>Last trade</th>
    </tr>
  </thead>
  <tbody>
{body_rows}
  </tbody>
</table>

{_render_cycle_diagnostics(cycles_recent)}

{_render_benchmark(benchmark)}

{_render_validation(validation)}

{_render_improvements(improvements)}

{_render_hedge_funds(hedge_funds)}

{_render_portfolio_intel(portfolio_intel)}

{_render_data_quality(data_quality)}

{_render_suggestions(_suggest_actions(cycles_recent, trades_recent, diag))}

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
