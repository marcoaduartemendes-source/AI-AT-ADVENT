"""Insider-cluster scout — detects coordinated open-market insider buying.

REAL EDGE:
    Cohen-Malloy-Pomorski, "Decoding Inside Information" (JF 2012):
    opportunistic (non-routine) insider purchases predict ~7%/yr
    abnormal returns. The cluster variant — MULTIPLE distinct insiders
    buying the same stock within days — is the strongest documented
    flavor (a lone CEO buy can be signalling; three executives writing
    personal checks the same week is conviction).

    Edge persistence: Form 4 must be filed within 2 business days of the
    trade (regulator-mandated trigger, same family as the 13D edge), and
    the names involved are typically small/mid-caps where institutional
    capital can't size — exactly where a small book has the advantage.

DATA:
    FMP insider-trading feed (the droplet already carries FMP_API_KEY).
    Tries the current `/stable` route first, then the legacy v4 RSS
    route, and emits nothing gracefully when the plan lacks both.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)

# ≥ this many DISTINCT insiders buying within the window = a cluster.
MIN_INSIDERS = 3
# Trailing window for the cluster test.
WINDOW_DAYS = 7
# Ignore token purchases — sub-$10k buys are noise / option exercises.
MIN_TOTAL_USD = 50_000.0


def _parse_purchases(rows: list) -> dict[str, dict]:
    """Group open-market purchases by symbol over the trailing window.

    Returns {symbol: {"insiders": set, "total_usd": float, "latest": str}}.
    """
    cutoff = datetime.now(UTC) - timedelta(days=WINDOW_DAYS)
    by_symbol: dict[str, dict] = defaultdict(
        lambda: {"insiders": set(), "total_usd": 0.0, "latest": ""})
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        # FMP labels open-market buys "P-Purchase" (transaction code P).
        ttype = (r.get("transactionType") or r.get("acquistionOrDisposition")
                 or "").upper()
        if not ttype.startswith("P"):
            continue
        sym = (r.get("symbol") or "").upper()
        who = (r.get("reportingName") or r.get("reportingCik") or "").strip()
        if not sym or not who:
            continue
        when_raw = (r.get("transactionDate") or r.get("filingDate") or "")
        try:
            when = datetime.fromisoformat(str(when_raw)[:19])
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
        if when < cutoff:
            continue
        try:
            qty = float(r.get("securitiesTransacted") or 0)
            px = float(r.get("price") or 0)
        except (TypeError, ValueError):
            qty, px = 0.0, 0.0
        d = by_symbol[sym]
        d["insiders"].add(who)
        d["total_usd"] += abs(qty * px)
        d["latest"] = max(d["latest"], when.isoformat())
    return dict(by_symbol)


class InsiderClusterScout(ScoutAgent):
    """Detects ≥3 distinct insiders buying the same stock within 7 days."""

    name = "insider_cluster"

    def scan(self) -> list[ScoutSignal]:
        try:
            from backtests.data.fmp import FMPClient
            client = FMPClient()
        except Exception as e:
            logger.debug(f"[insider_cluster] FMP unavailable: {e}")
            return []
        if not client.is_configured():
            return []

        rows = []
        for path, params in (
            ("insider-trading/latest", {"page": 0, "limit": 500}),
            ("insider-trading", {"page": 0, "limit": 500}),
        ):
            try:
                resp = client._get(path, params=params,
                                    cache_ttl_seconds=1800)
                if isinstance(resp, list) and resp:
                    rows = resp
                    break
            except Exception as e:
                logger.debug(f"[insider_cluster] {path}: {e}")
                continue
        if not rows:
            return []

        clusters = _parse_purchases(rows)
        signals: list[ScoutSignal] = []
        for sym, d in clusters.items():
            if len(d["insiders"]) < MIN_INSIDERS:
                continue
            if d["total_usd"] < MIN_TOTAL_USD:
                continue
            signals.append(ScoutSignal(
                venue="alpaca",
                signal_type="insider_cluster_buy",
                payload={
                    "ticker": sym,
                    "n_insiders": len(d["insiders"]),
                    "total_usd": round(d["total_usd"], 2),
                    "latest_trade": d["latest"],
                },
                # The documented drift runs weeks-months; keep the signal
                # actionable for 5 trading days, the strategy holds longer.
                ttl_seconds=5 * 24 * 3600,
            ))
        if signals:
            logger.info(
                f"[insider_cluster] {len(signals)} cluster-buy signals: "
                f"{[s.payload['ticker'] for s in signals]}")
        return signals
