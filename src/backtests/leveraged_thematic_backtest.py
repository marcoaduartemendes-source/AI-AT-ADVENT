"""Backtests for the two user-requested strategies.

leveraged_momentum  — 3x ETF trend with regime gate + hard stop.
thematic_growth     — Within-theme 6m-momentum picker.

Both follow the BacktestSummary contract used by runner._STRATEGY_BACKTESTS
so the validation harness can score them across 1/2/5y windows.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from .runner import BacktestSummary, _FEE_RATE, _equity_curve_to_summary, _yahoo_history

logger = logging.getLogger(__name__)


# ─── leveraged_momentum ───────────────────────────────────────────────

_LEV_PAIRS = [   # (3x ETF actually held, underlying for regime check)
    ("TQQQ", "QQQ"),
    ("UPRO", "SPY"),
    ("SOXL", "SOXX"),
    ("TNA",  "IWM"),
]
_LEV_SMA = 200
_LEV_VOL_LOOKBACK = 60
_LEV_VOL_CEIL = 0.022
_LEV_STOP_PCT = 0.15
_LEV_REBAL = 21          # check entry/exit signals monthly
_LEV_TGT_USD = 1000.0    # per-leg notional (mirrors tsmom_etf sizing)


def backtest_leveraged_momentum(window_days: int) -> BacktestSummary:
    """Per-month: for each (3x, underlying), if underlying > 200d SMA
    AND 60d realised vol ≤ 0.022, hold the 3x leg for the next month.
    Intra-month: stop out if the 3x position drops -15% from entry.
    Fees: 10bps round-trip (runner._FEE_RATE)."""
    need = _LEV_SMA + _LEV_VOL_LOOKBACK + window_days + 30
    hist3x: dict[str, np.ndarray] = {}
    histund: dict[str, np.ndarray] = {}
    for lev, und in _LEV_PAIRS:
        hist3x[lev] = _yahoo_history(lev, need)
        histund[und] = _yahoo_history(und, need)
    # Drop pairs missing either history (e.g. SOXL pre-2010).
    pairs = [(lev, und) for lev, und in _LEV_PAIRS
             if len(hist3x.get(lev, [])) >= _LEV_SMA + 5
             and len(histund.get(und, [])) >= _LEV_SMA + 5]
    if not pairs:
        return BacktestSummary(strategy="leveraged_momentum",
                                window_days=window_days,
                                note="no Yahoo history for 3x ETFs")

    trades: list[dict] = []
    positions: dict[str, dict] = {}      # lev_sym -> {qty, entry_price, entry_time}
    entry_volume = 0.0

    # All series should be aligned by Yahoo (calendar days); use the
    # shortest as the iteration index to avoid OOB.
    n_bars = min(len(hist3x[lev]) for lev, _ in pairs)
    base_idx = max(n_bars - window_days, _LEV_SMA + _LEV_VOL_LOOKBACK + 5)

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            float(hist3x[pairs[0][0]][i, 0]), tz=UTC)

        # ── 1) Intra-bar hard stops on open 3x positions.
        for lev in list(positions):
            cur = float(hist3x[lev][i, 4])
            ent = positions[lev]["entry_price"]
            if cur <= ent * (1 - _LEV_STOP_PCT):
                pos = positions.pop(lev)
                gross = pos["qty"] * (cur - pos["entry_price"])
                fees = (pos["qty"] * pos["entry_price"]
                         + pos["qty"] * cur) * _FEE_RATE
                trades.append({
                    "strategy": "leveraged_momentum", "side": "SELL",
                    "product_id": lev, "amount_usd": pos["qty"] * cur,
                    "quantity": pos["qty"], "entry_price": pos["entry_price"],
                    "exit_price": cur, "open_time": pos["entry_time"],
                    "close_time": bar_time.isoformat(),
                    "pnl_usd": gross - fees, "exit_reason": "hard_stop",
                })

        # ── 2) Monthly rebalance of entries/regime-exits.
        if (i - base_idx) % _LEV_REBAL != 0:
            continue
        for lev, und in pairs:
            und_closes = histund[und][:i + 1, 4]
            if len(und_closes) < _LEV_SMA + _LEV_VOL_LOOKBACK + 1:
                continue
            sma200 = und_closes[-_LEV_SMA:].mean()
            rets = np.diff(np.log(und_closes[-_LEV_VOL_LOOKBACK - 1:]))
            rv = float(rets.std())
            regime_on = (und_closes[-1] >= sma200) and (rv <= _LEV_VOL_CEIL)
            holding = lev in positions
            lev_price = float(hist3x[lev][i, 4])

            if regime_on and not holding:
                qty = _LEV_TGT_USD / lev_price
                positions[lev] = {"qty": qty, "entry_price": lev_price,
                                    "entry_time": bar_time.isoformat()}
                entry_volume += _LEV_TGT_USD
                trades.append({
                    "strategy": "leveraged_momentum", "side": "BUY",
                    "product_id": lev, "amount_usd": _LEV_TGT_USD,
                    "quantity": qty, "entry_price": lev_price,
                    "open_time": bar_time.isoformat(),
                    "reason": f"regime on ({und}>SMA200, vol={rv:.3f})",
                })
            elif (not regime_on) and holding:
                pos = positions.pop(lev)
                gross = pos["qty"] * (lev_price - pos["entry_price"])
                fees = (pos["qty"] * pos["entry_price"]
                         + pos["qty"] * lev_price) * _FEE_RATE
                trades.append({
                    "strategy": "leveraged_momentum", "side": "SELL",
                    "product_id": lev, "amount_usd": pos["qty"] * lev_price,
                    "quantity": pos["qty"], "entry_price": pos["entry_price"],
                    "exit_price": lev_price, "open_time": pos["entry_time"],
                    "close_time": bar_time.isoformat(),
                    "pnl_usd": gross - fees, "exit_reason": "regime_off",
                })
    return _equity_curve_to_summary("leveraged_momentum", window_days,
                                     trades, entry_volume)


# ─── thematic_growth ──────────────────────────────────────────────────

_THEMES: dict[str, list[str]] = {
    "ai_compute":         ["NVDA", "AMD", "AVGO", "TSM", "ASML",
                            "AMAT", "KLAC", "LRCX", "MRVL", "ARM", "MU"],
    "ai_power":           ["CEG", "VST", "NRG", "GEV", "ETN", "PWR"],
    "cybersecurity":      ["PANW", "CRWD", "ZS", "FTNT", "S"],
    "defense":            ["LMT", "RTX", "NOC", "GD", "HII"],
    "obesity_glp1":       ["LLY", "NVO"],
    "robotics":           ["ISRG", "ROK", "ABBT"],
    "quantum_spec":       ["IONQ", "RGTI"],
    "clean_energy":       ["NEE", "FSLR", "ENPH", "RUN", "ICLN", "BEP"],
    "reshoring_chips":    ["TXN", "INTC", "ON", "ENTG", "ROP", "ETN"],
    "lithium_ev_supply":  ["ALB", "LIT", "TSLA", "RIVN", "PCRFY"],
    "space_economy":      ["RKLB", "ASTS", "LMT", "BA", "IRDM"],
    "biotech_innovation": ["REGN", "VRTX", "MRNA", "BNTX", "ALNY"],
    "water":              ["AWK", "XYL", "PHO", "FIW"],
}
_THEME_WEIGHT: dict[str, float] = {
    "ai_compute": 0.22, "ai_power": 0.16, "cybersecurity": 0.10,
    "defense": 0.08, "obesity_glp1": 0.08, "robotics": 0.06,
    "quantum_spec": 0.04, "clean_energy": 0.06,
    "reshoring_chips": 0.06, "lithium_ev_supply": 0.04,
    "space_economy": 0.03, "biotech_innovation": 0.04,
    "water": 0.03,
}
_THM_SMA = 200
_THM_MOM = 126
_THM_TOP = 2
_THM_REBAL = 14
_THM_SLEEVE = 10000.0     # nominal sleeve sized so per-pick notionals
                            # match other equity strategies' ~$300-500


def backtest_thematic_growth(window_days: int) -> BacktestSummary:
    """Every 14 days: per theme, rank members by 6m return among those
    above 200d SMA; hold top-2 per theme. Cross-theme weights from
    _THEME_WEIGHT (sums to ~0.95). 10bps round-trip fees."""
    all_syms = sorted({s for lst in _THEMES.values() for s in lst})
    need = _THM_SMA + _THM_MOM + window_days + 30
    hist: dict[str, np.ndarray] = {s: _yahoo_history(s, need) for s in all_syms}
    hist = {s: h for s, h in hist.items()
            if len(h) >= _THM_SMA + _THM_MOM + 5}
    if not hist:
        return BacktestSummary(strategy="thematic_growth",
                                window_days=window_days,
                                note="no Yahoo history for theme universe")

    trades: list[dict] = []
    positions: dict[str, dict] = {}        # sym -> {qty, entry_price, entry_time}
    entry_volume = 0.0

    # Use the longest-history symbol as the iteration spine; per-symbol
    # signal eval guards length individually.
    spine = max(hist, key=lambda s: len(hist[s]))
    n_bars = len(hist[spine])
    base_idx = max(n_bars - window_days, _THM_SMA + _THM_MOM + 5)

    for i in range(base_idx, n_bars):
        if (i - base_idx) % _THM_REBAL != 0:
            continue
        bar_time = datetime.fromtimestamp(float(hist[spine][i, 0]), tz=UTC)
        # Per-theme picks
        target: dict[str, float] = {}  # sym -> usd alloc
        for theme, members in _THEMES.items():
            scored: list[tuple[str, float]] = []
            for s in members:
                arr = hist.get(s)
                if arr is None or len(arr) <= i:
                    continue
                closes = arr[:i + 1, 4]
                if len(closes) < _THM_SMA + _THM_MOM:
                    continue
                if closes[-1] < closes[-_THM_SMA:].mean():
                    continue
                p0 = closes[-_THM_MOM]
                if p0 <= 0:
                    continue
                scored.append((s, float(closes[-1] / p0 - 1.0)))
            scored.sort(key=lambda x: x[1], reverse=True)
            picks = scored[:_THM_TOP]
            if not picks:
                continue
            per = _THM_SLEEVE * _THEME_WEIGHT.get(theme, 0.0) / len(picks)
            for s, _m in picks:
                target[s] = max(target.get(s, 0.0), per)

        target_set = set(target)
        # ── Exits
        for s in list(positions):
            if s in target_set:
                continue
            pos = positions.pop(s)
            arr = hist.get(s)
            if arr is None or i >= len(arr):
                continue
            cur = float(arr[i, 4])
            gross = pos["qty"] * (cur - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * cur) * _FEE_RATE
            trades.append({
                "strategy": "thematic_growth", "side": "SELL",
                "product_id": s, "amount_usd": pos["qty"] * cur,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": cur, "open_time": pos["entry_time"],
                "close_time": bar_time.isoformat(),
                "pnl_usd": gross - fees, "exit_reason": "rerank_out",
            })
        # ── Entries
        for s, usd in target.items():
            if s in positions:
                continue
            arr = hist.get(s)
            if arr is None or i >= len(arr):
                continue
            cur = float(arr[i, 4])
            if cur <= 0 or usd <= 0:
                continue
            qty = usd / cur
            positions[s] = {"qty": qty, "entry_price": cur,
                              "entry_time": bar_time.isoformat()}
            entry_volume += usd
            trades.append({
                "strategy": "thematic_growth", "side": "BUY",
                "product_id": s, "amount_usd": usd, "quantity": qty,
                "entry_price": cur, "open_time": bar_time.isoformat(),
                "reason": "top-2 in theme by 6m mom",
            })
    return _equity_curve_to_summary("thematic_growth", window_days,
                                     trades, entry_volume)
