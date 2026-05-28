"""Tests for the portfolio-intelligence correlation/diversification layer."""
from __future__ import annotations

import datetime as dt
import math
import sqlite3


def _db(tmp_path, specs):
    """specs: {strategy: [daily_pnl,...]} → a trades DB of BUY/SELL pairs."""
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                 "timestamp TEXT, strategy TEXT, product_id TEXT, side TEXT, "
                 "quantity REAL, price REAL, pnl_usd REAL, fill_status TEXT)")
    base = dt.date.today() - dt.timedelta(days=len(next(iter(specs.values()))) + 1)
    rows = []
    for strat, series in specs.items():
        for d, pnl in enumerate(series):
            ts = (base + dt.timedelta(days=d)).isoformat() + "T12:00:00+00:00"
            rows.append((ts, strat, "X", "BUY", 1, 100, None, "FILLED"))
            rows.append((ts, strat, "X", "SELL", 1, 100 + pnl, pnl, "FILLED"))
    conn.executemany(
        "INSERT INTO trades (timestamp,strategy,product_id,side,quantity,"
        "price,pnl_usd,fill_status) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return str(db)


def test_correlated_book_flagged_concentrated(tmp_path):
    from common.portfolio_intel import run_portfolio_intel
    swing = [math.sin(d / 3) * 100 for d in range(40)]
    # Three strategies all moving together → high avg corr, low ENB.
    db = _db(tmp_path, {"A": swing,
                        "B": [x * 0.98 for x in swing],
                        "C": [x * 1.02 for x in swing]})
    r = run_portfolio_intel(db_path=db, window_days=120,
                            out_path=str(tmp_path / "pi.json"))
    assert r["avg_pairwise_corr"] > 0.9
    assert r["concentrated"] is True
    assert r["effective_bets"] < 1.5          # really ~1 bet
    assert "CONCENTRATED" in r["verdict"]


def test_uncorrelated_book_is_diversified(tmp_path):
    from common.portfolio_intel import run_portfolio_intel
    # Distinct deterministic patterns with low mutual correlation.
    a = [math.sin(d / 3) * 100 for d in range(40)]
    b = [math.cos(d / 7) * 100 for d in range(40)]
    c = [((d * 37) % 11 - 5) * 20 for d in range(40)]
    db = _db(tmp_path, {"A": a, "B": b, "C": c})
    r = run_portfolio_intel(db_path=db, window_days=120,
                            out_path=str(tmp_path / "pi.json"))
    assert r["avg_pairwise_corr"] < 0.6
    assert r["effective_bets"] > 1.5
    assert r["concentrated"] is False


def test_insufficient_history_is_honest(tmp_path):
    from common.portfolio_intel import run_portfolio_intel
    db = _db(tmp_path, {"A": [10, 20, 30]})   # < MIN_OVERLAP_DAYS
    r = run_portfolio_intel(db_path=db, window_days=120,
                            out_path=str(tmp_path / "pi.json"))
    assert r["avg_pairwise_corr"] is None
    assert "Not enough" in r["verdict"]


def test_pearson_edge_cases():
    from common.portfolio_intel import _pearson
    assert _pearson([1, 2], [1, 2]) is None             # too few points
    flat = [5.0] * 15
    assert _pearson(flat, list(range(15))) is None      # zero variance
    perfect = _pearson(list(range(15)), list(range(15)))
    assert perfect is not None and abs(perfect - 1.0) < 1e-9
