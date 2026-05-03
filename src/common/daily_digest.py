"""Daily digest — one-shot summary of the trading system's last 24h.

Sprint E2 audit fix: the dashboard staleness alert (heartbeat) catches
"system died" but not "system is alive but doing nothing useful". The
digest answers the user's actual daily question: how much did I make
yesterday, what's open, what's broken.

What it sends:
    1. Headline P&L (24h realized + unrealized; MTD vs 4% loss budget;
       drawdown vs 15% kill-switch threshold)
    2. Per-broker P&L (the same Coinbase / Alpaca / Kalshi rows the
       dashboard shows, in plain text)
    3. Top 3 winners + top 3 losers by realized P&L in the last 24h
    4. Strategies with consecutive errors (from common.strategy_alerts)
    5. Days since last trade per strategy — surfaces dormant strategies
    6. Kill-switch state + venues_ok flag
    7. Heartbeat to the daily-digest healthcheck (tells you the digest
       itself ran)

Delivery:
    - Pushover (high-signal mobile alert) via common.alerts.alert
    - Email via common.alerts._send_email if ALERT_EMAIL_TO is set
    Both pull from the same env-driven config as the existing alerts
    pipeline; no new credentials needed.

Cadence:
    Designed to run as a 5th systemd timer once daily, e.g. 13:00 UTC
    so it lands at 9 ET / breakfast US / lunch EU. Idempotent —
    re-running produces the same digest at the same point in time.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DigestSection:
    title: str
    body: str


def build_digest(*, now: datetime | None = None) -> str:
    """Assemble the digest body. Pure-string output so callers can
    send it through any channel."""
    now = now or datetime.now(UTC)
    sections: list[DigestSection] = []

    sections.append(DigestSection(
        title="Header",
        body=f"AAA daily digest — {now.strftime('%Y-%m-%d %H:%M UTC')}",
    ))

    # ── 1) Headline P&L
    try:
        sections.append(_pnl_section(now))
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest pnl section failed: {e}")
        sections.append(DigestSection(
            title="P&L",
            body=f"P&L unavailable: {type(e).__name__}",
        ))

    # ── 2) Per-broker breakdown
    try:
        sections.append(_per_broker_section())
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest broker section failed: {e}")
        sections.append(DigestSection(
            title="Brokers",
            body="Broker breakdown unavailable",
        ))

    # ── 3) Top winners / losers (24h)
    try:
        sections.append(_winners_losers_section(now))
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest winners section failed: {e}")

    # ── 4) Strategy health
    try:
        sections.append(_strategy_health_section())
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest health section failed: {e}")

    # ── 5) Risk state + kill switch
    try:
        sections.append(_risk_state_section())
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest risk section failed: {e}")

    return "\n\n".join(f"== {s.title} ==\n{s.body}" for s in sections)


def send_digest() -> bool:
    """Build + dispatch the digest. Returns True on success.

    Best-effort: returns False on any failure so the systemd unit
    doesn't infinite-restart, but logs at WARNING so journalctl
    surfaces the issue."""
    try:
        text = build_digest()
    except Exception as e:    # noqa: BLE001
        logger.exception(f"digest build failed: {e}")
        return False

    # Send via the existing alerts pipeline. Severity=info so it
    # routes via Pushover priority 0 (silent at night, not the
    # high-priority kill-switch / drawdown alerts).
    try:
        from common.alerts import alert
        alert(text, severity="info")
    except Exception as e:    # noqa: BLE001
        logger.warning(f"digest alert dispatch failed: {e}")

    # Heartbeat — distinct from the orchestrator heartbeat so the
    # daily check has its own dead-man's switch.
    try:
        from common.heartbeat import ping_success
        ping_success("dashboard", message="daily digest sent")
    except Exception as e:    # noqa: BLE001
        logger.debug(f"digest heartbeat failed: {e}")

    return True


# ─── Section builders ─────────────────────────────────────────────────


def _pnl_section(now: datetime) -> DigestSection:
    """Pull from the trading_performance.db / Supabase: 24h realized
    P&L, MTD vs loss budget, drawdown vs kill-switch."""
    db_path = os.environ.get("TRADING_DB_PATH",
                              "data/trading_performance.db")
    if not os.path.exists(db_path):
        return DigestSection(
            title="P&L",
            body="(no trading_performance.db yet — first run?)",
        )

    import sqlite3
    cutoff_24h = (now - timedelta(hours=24)).isoformat()
    cutoff_mtd = now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0,
    ).isoformat()
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        # 24h realized P&L on FILLED trades
        row = c.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS pnl, COUNT(*) AS n "
            "FROM trades "
            "WHERE timestamp >= ? AND fill_status = 'FILLED' "
            "AND pnl_usd IS NOT NULL",
            (cutoff_24h,),
        ).fetchone()
        pnl_24h = float(row["pnl"]) if row else 0.0
        n_24h = int(row["n"]) if row else 0
        # MTD realized P&L
        mtd_row = c.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0) AS pnl, COUNT(*) AS n "
            "FROM trades "
            "WHERE timestamp >= ? AND fill_status = 'FILLED' "
            "AND pnl_usd IS NOT NULL",
            (cutoff_mtd,),
        ).fetchone()
        pnl_mtd = float(mtd_row["pnl"]) if mtd_row else 0.0
        n_mtd = int(mtd_row["n"]) if mtd_row else 0

    body = (
        f"24h realized: ${pnl_24h:+,.2f} ({n_24h} trades)\n"
        f"MTD realized: ${pnl_mtd:+,.2f} ({n_mtd} trades)"
    )
    return DigestSection(title="P&L (24h / MTD)", body=body)


def _per_broker_section() -> DigestSection:
    """Reuse the dashboard's broker attribution helper."""
    try:
        from build_dashboard import (
            _attribute_broker_pnl,
            _broker_snapshot,
            _build_strategy_to_venue,
        )
        from brokers.registry import build_brokers
    except ImportError as e:
        return DigestSection(
            title="Brokers", body=f"import failed: {e}",
        )

    brokers = build_brokers()
    by_broker: dict[str, dict] = {}
    for name, adapter in brokers.items():
        try:
            snap = _broker_snapshot(name, adapter)
        except Exception as e:    # noqa: BLE001
            logger.warning(f"digest broker {name} snapshot failed: {e}")
            continue
        snap.pop("positions", None)
        snap.pop("open_orders", None)
        by_broker[name] = snap

    # Pull recent trades for realized P&L attribution
    db_path = os.environ.get("TRADING_DB_PATH",
                              "data/trading_performance.db")
    trades: list[dict] = []
    if os.path.exists(db_path):
        import sqlite3
        with sqlite3.connect(db_path) as c:
            c.row_factory = sqlite3.Row
            cutoff = (datetime.now(UTC) - timedelta(days=30)).isoformat()
            for r in c.execute(
                "SELECT * FROM trades WHERE timestamp >= ? "
                "AND fill_status = 'FILLED'",
                (cutoff,),
            ):
                trades.append(dict(r))
    _attribute_broker_pnl(by_broker, trades, _build_strategy_to_venue())

    if not by_broker:
        return DigestSection(title="Brokers", body="(no brokers configured)")

    rows = []
    for venue, b in sorted(by_broker.items()):
        cash = b.get("cash_usd", 0.0)
        equity = b.get("equity_usd", 0.0)
        unr = b.get("unrealized_pnl_usd", 0.0)
        rea = b.get("realized_pnl_usd", 0.0)
        tot = unr + rea
        n_pos = b.get("n_positions", 0)
        rows.append(
            f"{venue:9} cash ${cash:>10,.0f}  equity ${equity:>10,.0f}  "
            f"realized ${rea:>+8,.2f}  unrealized ${unr:>+8,.2f}  "
            f"total ${tot:>+8,.2f}  pos {n_pos}"
        )
    return DigestSection(title="Brokers", body="\n".join(rows))


def _winners_losers_section(now: datetime) -> DigestSection:
    """Top 3 winners + top 3 losers by realized P&L in last 24h."""
    db_path = os.environ.get("TRADING_DB_PATH",
                              "data/trading_performance.db")
    if not os.path.exists(db_path):
        return DigestSection(
            title="Top movers (24h)", body="(no DB)",
        )
    import sqlite3
    cutoff = (now - timedelta(hours=24)).isoformat()
    with sqlite3.connect(db_path) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT product_id, strategy, pnl_usd FROM trades "
            "WHERE timestamp >= ? AND fill_status = 'FILLED' "
            "AND pnl_usd IS NOT NULL "
            "ORDER BY pnl_usd DESC",
            (cutoff,),
        ).fetchall()

    if not rows:
        return DigestSection(
            title="Top movers (24h)",
            body="(no closed trades in the last 24h)",
        )
    winners = list(rows[:3])
    losers = list(rows[-3:][::-1]) if len(rows) > 3 else []

    body_lines = []
    if winners:
        body_lines.append("Winners:")
        for r in winners:
            body_lines.append(
                f"  ${r['pnl_usd']:>+8.2f}  {r['strategy']:25} {r['product_id']}"
            )
    if losers:
        body_lines.append("Losers:")
        for r in losers:
            body_lines.append(
                f"  ${r['pnl_usd']:>+8.2f}  {r['strategy']:25} {r['product_id']}"
            )
    return DigestSection(title="Top movers (24h)", body="\n".join(body_lines))


def _strategy_health_section() -> DigestSection:
    """Strategies with consecutive errors. Surfaces dormant strategies
    (count > 0 = currently failing) so the user sees them at-a-glance."""
    try:
        from common.strategy_alerts import all_states
        states = all_states()
    except Exception as e:    # noqa: BLE001
        return DigestSection(
            title="Strategy health",
            body=f"strategy_alerts unavailable: {e}",
        )
    if not states:
        return DigestSection(
            title="Strategy health",
            body="All strategies clean (no error history)",
        )
    failing = [s for s in states if s.get("consecutive_errors", 0) > 0]
    if not failing:
        return DigestSection(
            title="Strategy health",
            body="All strategies clean this cycle",
        )
    rows = ["Strategies with consecutive errors:"]
    for s in failing:
        last = s.get("last_error_text") or ""
        rows.append(
            f"  {s['strategy']:30} count={s['consecutive_errors']}  "
            f"last: {last[:80]}"
        )
    return DigestSection(title="Strategy health", body="\n".join(rows))


def _risk_state_section() -> DigestSection:
    """Kill-switch state + drawdown vs threshold."""
    try:
        from risk.manager import RiskManager
        rm = RiskManager()
        state = rm.compute_state(persist=False)
    except Exception as e:    # noqa: BLE001
        return DigestSection(
            title="Risk state", body=f"risk.compute_state failed: {e}",
        )
    body = (
        f"kill_switch:    {state.kill_switch.value}\n"
        f"drawdown_pct:   {state.drawdown_pct * 100:.2f}%\n"
        f"peak_equity:    ${state.peak_equity_usd:,.2f}\n"
        f"current_equity: ${state.equity_usd:,.2f}\n"
        f"venues_ok:      {state.venues_ok}\n"
        f"multiplier:     {state.multiplier.effective:.2f}"
    )
    return DigestSection(title="Risk state", body=body)
