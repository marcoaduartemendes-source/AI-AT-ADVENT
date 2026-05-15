"""Benchmark comparison — portfolio vs market indexes.

Runs inside the orchestrator job (where FMP_API_KEY + risk_state.db
both exist) and writes docs/benchmark.json. The dashboard renders it.

Why this lives here and not in the dashboard build: the dashboard
workflow doesn't restore risk_state.db's equity history reliably,
and FMP rate-limits mean we want ONE fetch per cycle, not per
dashboard rebuild. Computing here keeps the data fetch co-located
with the equity curve it's compared against.

Trailing-return method: (equity_now / equity_N_days_ago) - 1, using
the closest equity snapshot at/just-before the window start. Index
returns use FMP adjusted daily closes over the same calendar window
so dividends/splits don't distort the comparison.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# What we benchmark against. SPY = S&P 500, QQQ = Nasdaq-100,
# BTC-USD = crypto beta (the bot trades crypto too).
_BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq-100",
    "BTC-USD": "Bitcoin",
}
_WINDOWS = (7, 14, 30)


def _equity_at_or_before(conn: sqlite3.Connection, cutoff: datetime
                          ) -> float | None:
    """Closest equity snapshot at/just-before `cutoff`."""
    row = conn.execute(
        "SELECT equity_usd FROM equity_snapshots "
        "WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1",
        (cutoff.isoformat(),),
    ).fetchone()
    if row:
        return float(row[0])
    # No snapshot before cutoff — fall back to the earliest we have
    row = conn.execute(
        "SELECT equity_usd FROM equity_snapshots "
        "ORDER BY timestamp ASC LIMIT 1"
    ).fetchone()
    return float(row[0]) if row else None


def _portfolio_returns(risk_db: str) -> dict:
    """Trailing portfolio return per window from equity_snapshots."""
    out: dict = {"current_equity": None, "windows": {}}
    if not Path(risk_db).exists():
        return out
    try:
        with sqlite3.connect(risk_db) as conn:
            latest = conn.execute(
                "SELECT timestamp, equity_usd FROM equity_snapshots "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not latest:
                return out
            now_ts = datetime.fromisoformat(latest[0])
            now_eq = float(latest[1])
            out["current_equity"] = round(now_eq, 2)
            for days in _WINDOWS:
                start_eq = _equity_at_or_before(
                    conn, now_ts - timedelta(days=days)
                )
                if start_eq and start_eq > 0:
                    out["windows"][str(days)] = round(
                        (now_eq / start_eq - 1.0) * 100, 3
                    )
                else:
                    out["windows"][str(days)] = None
    except sqlite3.Error as e:
        logger.warning(f"benchmark: equity read failed: {e}")
    return out


def _index_returns() -> dict:
    """Trailing index return per window via FMP adjusted closes.

    Empty dict when FMP isn't configured — the dashboard then shows
    "benchmark unavailable (set FMP_API_KEY)" instead of fake numbers.
    """
    out: dict = {}
    try:
        from backtests.data.fmp import FMPClient
        client = FMPClient()
        if not client.is_configured():
            return out
    except Exception as e:
        logger.debug(f"benchmark: FMP client unavailable: {e}")
        return out

    today = datetime.now(UTC).date()
    # Pull 40 calendar days so the 30d window has a clean anchor even
    # across weekends/holidays.
    frm = today - timedelta(days=40)
    for sym in _BENCHMARKS:
        try:
            bars = client.daily_bars(sym, frm, today)
        except Exception as e:
            logger.debug(f"benchmark: {sym} fetch failed: {e}")
            continue
        if not bars:
            continue
        bars.sort(key=lambda b: b.date)
        last_close = bars[-1].close
        sym_out: dict = {}
        for days in _WINDOWS:
            target = bars[-1].date - timedelta(days=days)
            # closest bar at/just-before target
            anchor = None
            for b in bars:
                if b.date <= target:
                    anchor = b
                else:
                    break
            if anchor and anchor.close > 0:
                sym_out[str(days)] = round(
                    (last_close / anchor.close - 1.0) * 100, 3
                )
            else:
                sym_out[str(days)] = None
        out[sym] = {"name": _BENCHMARKS[sym], "windows": sym_out}
    return out


def write_benchmark_json(
    risk_db: str | None = None,
    out_path: str = "docs/benchmark.json",
) -> dict:
    """Compute portfolio-vs-index trailing returns, write JSON, return
    the payload. Best-effort: never raises (orchestrator calls this
    from its finally block)."""
    risk_db = risk_db or os.environ.get(
        "RISK_DB_PATH", "data/risk_state.db"
    )
    payload = {
        "as_of": datetime.now(UTC).isoformat(),
        "portfolio": _portfolio_returns(risk_db),
        "benchmarks": _index_returns(),
        "windows": list(_WINDOWS),
    }
    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception as e:
        logger.warning(f"benchmark: write failed: {e}")
    return payload
