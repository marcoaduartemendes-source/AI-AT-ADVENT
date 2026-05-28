"""Independent realized-PnL recompute via FIFO BUY/SELL matching.

This is the "second brain" that disagrees with the orchestrator's
broker-attributed PnL when something has gone wrong. It reads ONLY
from the trade ledger (timestamp, side, quantity, price) and ignores
avg_entry_price from the broker — so a divergence means either:

  - The orchestrator computed the wrong PnL when it backfilled a fill
    (e.g. broker reported a stale avg_entry_price)
  - The broker re-cost-basised a position that we computed against
    the old basis
  - We have an orphan SELL with no matching BUY
  - A BUY was never recorded but a SELL was

Any of these is a real bug — log it loudly, surface on the dashboard
once we wire that, and **never silently overwrite the orchestrator's
number** with this one. They're independent witnesses; the human
decides who's right.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class _Lot:
    qty: float
    price: float


def recompute_realized_pnl_fifo(
    db_path: str,
) -> tuple[float, float, dict[str, float]]:
    """Re-derive realized PnL from the raw trade ledger via FIFO match.

    Returns (db_total, recomputed_total, per_strategy_drift) where:
      - db_total          = SUM(pnl_usd) from the trades table
      - recomputed_total  = independent FIFO walk
      - per_strategy_drift = {strategy: db_pnl - recomputed_pnl}
                              for any strategy that disagrees by > $0.50

    Implementation notes:
      - Trades are processed in timestamp order, per (strategy, product_id)
      - BUY adds a lot to the per-(strategy, product_id) FIFO queue
      - SELL pops lots until the SELL's qty is consumed, accumulating
        (sell_px - lot_px) * matched_qty into realized PnL
      - Trades with price=0 or NULL are SKIPPED (not yet filled — they
        contribute zero to either side, so drift is unaffected)
      - SELLs with no matching BUY (orphans) are recorded as a drift
        signal: counted in `db_total` if pnl_usd is set, but counted
        as 0 in `recomputed_total` since FIFO has no basis. The
        difference shows up in per_strategy_drift.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT timestamp, strategy, product_id, side, quantity,
                   price, pnl_usd
              FROM trades
             WHERE price IS NOT NULL AND price > 0
             ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
        all_with_pnl = conn.execute(
            "SELECT strategy, COALESCE(SUM(pnl_usd), 0) AS s "
            "FROM trades WHERE pnl_usd IS NOT NULL GROUP BY strategy"
        ).fetchall()
    finally:
        conn.close()

    # FIFO queue per (strategy, product_id)
    books: dict[tuple[str, str], deque[_Lot]] = defaultdict(deque)
    realized: dict[str, float] = defaultdict(float)
    for r in rows:
        key = (r["strategy"], r["product_id"])
        qty = float(r["quantity"] or 0)
        px = float(r["price"] or 0)
        if qty <= 0 or px <= 0:
            continue
        side = r["side"]
        if side == "BUY":
            books[key].append(_Lot(qty=qty, price=px))
            continue
        # SELL — match against the FIFO queue
        remaining = qty
        while remaining > 0 and books[key]:
            lot = books[key][0]
            matched = min(lot.qty, remaining)
            realized[r["strategy"]] += (px - lot.price) * matched
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= 1e-12:
                books[key].popleft()
        # If `remaining` > 0 here, this is an orphan SELL — no matching
        # BUY in the ledger. Don't add to realized; the drift will show
        # up vs. the DB total which DID record a (possibly bogus) PnL.

    recomputed_total = sum(realized.values())
    db_total = sum(float(r["s"] or 0) for r in all_with_pnl)

    # Per-strategy drift for the alert message
    db_by_strat = {r["strategy"]: float(r["s"] or 0) for r in all_with_pnl}
    drift: dict[str, float] = {}
    all_strategies = set(db_by_strat) | set(realized)
    for s in all_strategies:
        d = db_by_strat.get(s, 0) - realized.get(s, 0)
        if abs(d) > 0.50:
            drift[s] = round(d, 2)

    return round(db_total, 2), round(recomputed_total, 2), drift


def fifo_realized_events(db_path: str) -> dict[str, list[dict]]:
    """Per-strategy realized-PnL close events via FIFO lot matching over
    the RAW fill ledger — the canonical, auditable realized P&L.

    Returns {strategy: [{"timestamp", "pnl_usd", "product_id"}, …]} with
    one event per SELL that matched against a prior BUY lot. This is the
    phantom-proof source of truth the dashboard and the allocator's
    metrics should use instead of the stored pnl_usd column, which is
    computed at fill time from a single avg-cost entry and drifts on
    partial fills, stale broker cost-basis, orphan SELLs, and the
    price=0 phantom-loss bug.

    Rules (match recompute_realized_pnl_fifo so the drift check stays
    consistent):
      • Only price>0 rows count (CANCELED/unfilled have price=0).
      • Trades processed in (timestamp, id) order per (strategy, product).
      • BUY pushes a lot; SELL pops lots FIFO, realizing (sell-lot)×qty.
      • An orphan SELL (no matching BUY) realizes nothing and emits NO
        event — it's not a real round trip (vs the stored column, which
        recorded a bogus number for it).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT timestamp, strategy, product_id, side, quantity, price
              FROM trades
             WHERE price IS NOT NULL AND price > 0
             ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    books: dict[tuple[str, str], deque[_Lot]] = defaultdict(deque)
    events: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["strategy"], r["product_id"])
        qty = float(r["quantity"] or 0)
        px = float(r["price"] or 0)
        if qty <= 0 or px <= 0:
            continue
        if r["side"] == "BUY":
            books[key].append(_Lot(qty=qty, price=px))
            continue
        # SELL — FIFO match.
        remaining = qty
        realized = 0.0
        while remaining > 0 and books[key]:
            lot = books[key][0]
            matched = min(lot.qty, remaining)
            realized += (px - lot.price) * matched
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= 1e-12:
                books[key].popleft()
        if remaining < qty:   # matched at least part of a real round trip
            events[r["strategy"]].append({
                "timestamp": r["timestamp"],
                "pnl_usd": realized,
                "product_id": r["product_id"],
            })
    return dict(events)


def fifo_realized_by_strategy(db_path: str) -> dict[str, float]:
    """Total FIFO realized P&L per strategy (sum of close events)."""
    return {s: round(sum(e["pnl_usd"] for e in evs), 2)
            for s, evs in fifo_realized_events(db_path).items()}


def normalize_symbol(sym: str) -> str:
    """Canonical key for matching a broker position to a ledger row.

    The #1 cause of "<unattributed>" P&L was format drift between how a
    venue reports an open position and how product_id was stored:
        Coinbase position "BTC"    vs ledger "BTC-USD"
        Alpaca   crypto   "BTC/USD" vs "BTCUSD"
    Collapse all of these to the base asset. Equity/ETF tickers (no
    separator) pass through unchanged.
    """
    s = str(sym or "").upper().strip().replace("/", "-")
    if "-" in s:                      # crypto pair → base asset
        s = s.split("-", 1)[0]
    return s


def fifo_open_positions(db_path: str) -> dict[str, dict[str, float]]:
    """Per-(normalized-symbol) open inventory split BY strategy, via the
    same FIFO walk used for realized P&L.

    Returns {norm_symbol: {strategy: open_qty}} for every symbol with a
    net-long open position in the ledger. The authoritative source for
    attributing a live broker position's unrealized P&L back to the
    strategies that opened it — proportional to each strategy's remaining
    open lots — so it never lands in "<unattributed>" merely because two
    strategies share a symbol or the symbol format differs.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT strategy, product_id, side, quantity, price
              FROM trades
             WHERE price IS NOT NULL AND price > 0
             ORDER BY timestamp ASC, id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    books: dict[tuple[str, str], deque[_Lot]] = defaultdict(deque)
    for r in rows:
        qty = float(r["quantity"] or 0)
        px = float(r["price"] or 0)
        if qty <= 0 or px <= 0:
            continue
        key = (r["strategy"], normalize_symbol(r["product_id"]))
        if r["side"] == "BUY":
            books[key].append(_Lot(qty=qty, price=px))
            continue
        remaining = qty
        while remaining > 0 and books[key]:
            lot = books[key][0]
            matched = min(lot.qty, remaining)
            lot.qty -= matched
            remaining -= matched
            if lot.qty <= 1e-12:
                books[key].popleft()

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (strategy, sym), lots in books.items():
        open_qty = sum(lot.qty for lot in lots)
        if open_qty > 1e-9:
            out[sym][strategy] = round(open_qty, 10)
    return dict(out)
