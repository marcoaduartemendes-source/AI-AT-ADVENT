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
    qty: float          # POSITIVE magnitude, always
    price: float        # effective entry price, fee-adjusted
    sign: int = 1       # +1 = long lot, -1 = short lot


def _book_sign(book: deque[_Lot]) -> int:
    """+1 if the book is net long, -1 if net short, 0 if flat.

    Invariant: a single (strategy, product) book never holds long and
    short lots simultaneously — `_apply_fill` closes against the opposite
    side before opening a new one.
    """
    return book[0].sign if book else 0


def _apply_fill(
    book: deque[_Lot], fill_sign: int, qty: float, px: float, fee: float,
) -> tuple[float, float]:
    """Apply one fill to a SIGNED FIFO book. Returns (realized, closed_qty).

    2026-08-05 LONG/SHORT UPGRADE. The previous walk was long-only: BUY
    pushed a lot, SELL popped lots, and a SELL with no matching BUY was
    discarded as an "orphan". A short entry IS an opening SELL with no
    prior BUY, so under that walk every short entry vanished from the
    books and the covering BUY was recorded as a NEW LONG LOT. A complete,
    profitable short round trip would therefore have booked $0.00 realized
    P&L and left a phantom long position on the ledger forever.

    That is the same failure class as the phantom-loss and the $0.50
    book graded 10/10 (docs/AUDIT_INCEPTION.md): accounting that quietly
    reports a number unrelated to what happened. Shorting could not be
    enabled anywhere until this was signed.

    The generalisation is one line of arithmetic — realized P&L on a
    close is `lot.sign * (exit_price - entry_price) * qty` — which
    reduces to the old `(sell - buy) * qty` for long lots and gives
    `(entry - cover) * qty` for short lots.

    Fee convention (unchanged for longs, extended to shorts): fees on the
    OPENING half fold into the lot's effective entry price, fees on the
    CLOSING half subtract from realized. A long pays up (entry + fee), a
    short receives less (entry - fee), which is `px + sign * fee_per_unit`
    in both cases. A fill that both closes and opens pro-rates its fee
    between the two halves by matched quantity.
    """
    realized = 0.0
    remaining = qty
    closed_qty = 0.0

    if _book_sign(book) not in (0, fill_sign):
        # Opposite side on the book — this fill closes before it opens.
        while remaining > 1e-12 and book:
            lot = book[0]
            matched = min(lot.qty, remaining)
            realized += lot.sign * (px - lot.price) * matched
            lot.qty -= matched
            remaining -= matched
            closed_qty += matched
            if lot.qty <= 1e-12:
                book.popleft()

    fee_close = (fee * (closed_qty / qty)) if qty > 0 else 0.0
    realized -= fee_close

    if remaining > 1e-12:
        fee_open = fee - fee_close
        eff_px = px + fill_sign * (fee_open / remaining)
        book.append(_Lot(qty=remaining, price=eff_px, sign=fill_sign))

    return realized, closed_qty


def _has_fees_column(conn: sqlite3.Connection) -> bool:
    """True if trades.fees_usd exists (migration 003). Lets the FIFO
    recompute net fees where available and degrade cleanly on a
    pre-migration DB (e.g. a test fixture that skips migrations)."""
    try:
        cols = conn.execute("PRAGMA table_info(trades)").fetchall()
        return any((c[1] if not hasattr(c, "keys") else c["name"])
                   == "fees_usd" for c in cols)
    except sqlite3.Error:
        return False


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
      - SIGNED FIFO per (strategy, product_id): a fill closes any
        opposite-side lots first, then opens same-side lots with what
        remains. See `_apply_fill`.
      - Trades with price=0 or NULL are SKIPPED (not yet filled — they
        contribute zero to either side, so drift is unaffected)
      - A SELL with no prior BUY now OPENS A SHORT rather than being
        treated as an orphan. That is required for long/short strategies
        to account correctly, but it does cost a diagnostic: a genuinely
        erroneous orphan SELL from a long-only strategy (missing BUY row,
        double-recorded exit) no longer shows up here as drift — it looks
        like a deliberate short. The guard against that moved to where it
        belongs: the orchestrator refuses an opening SELL from a strategy
        not declared short-capable, so a phantom short cannot be created
        in the first place.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _fee_col = _has_fees_column(conn)
        _fee_sel = "COALESCE(fees_usd, 0) AS fees_usd" if _fee_col \
            else "0 AS fees_usd"
        rows = conn.execute(
            f"""
            SELECT timestamp, strategy, product_id, side, quantity,
                   price, pnl_usd, {_fee_sel}
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
        fee = float(r["fees_usd"] or 0)
        fill_sign = 1 if r["side"] == "BUY" else -1
        pnl, _closed = _apply_fill(books[key], fill_sign, qty, px, fee)
        realized[r["strategy"]] += pnl

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
      • SIGNED FIFO (2026-08-05): a fill closes opposite-side lots first,
        then opens same-side lots with the remainder. Realized P&L on a
        close is lot.sign × (exit − entry) × qty, so a long round trip
        gives (sell − buy) and a short gives (entry − cover).
      • A purely OPENING fill realizes nothing and emits no event. That
        now includes an opening short (a SELL with no prior BUY), which
        the old long-only walk discarded as an "orphan" — silently
        erasing the entry leg of every short.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        _fee_col = _has_fees_column(conn)
        _fee_sel = "COALESCE(fees_usd, 0) AS fees_usd" if _fee_col \
            else "0 AS fees_usd"
        rows = conn.execute(
            f"""
            SELECT timestamp, strategy, product_id, side, quantity, price,
                   {_fee_sel}
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
        fee = float(r["fees_usd"] or 0)
        fill_sign = 1 if r["side"] == "BUY" else -1
        realized, closed_qty = _apply_fill(
            books[key], fill_sign, qty, px, fee)
        # An event is emitted only when the fill CLOSED something — i.e.
        # a completed round trip, long or short. A pure opening fill
        # (including an opening short, which the long-only walk used to
        # discard as an "orphan SELL") realizes nothing and emits nothing.
        if closed_qty > 1e-12:
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
        _apply_fill(books[key], 1 if r["side"] == "BUY" else -1,
                    qty, px, 0.0)

    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (strategy, sym), lots in books.items():
        # SIGNED net quantity — negative for a short book. Callers that
        # weight by these must handle the sign (see build_dashboard's
        # proportional unrealized split).
        open_qty = sum(lot.sign * lot.qty for lot in lots)
        if abs(open_qty) > 1e-9:
            out[sym][strategy] = round(open_qty, 10)
    return dict(out)
