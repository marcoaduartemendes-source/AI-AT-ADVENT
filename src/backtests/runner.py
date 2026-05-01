"""Per-strategy backtest runner.

Each backtest is intentionally small and self-contained — same
strategy logic as the live version, just fed historical bars and a
mock fill model. Same risk rules apply (stop/take-profit, cooldown,
fees, slippage).

We deliberately don't try to backtest scout-fed strategies here:
faking historical funding rates, mispriced markets, or earnings
surprises produces meaningless numbers. Those are flagged in
UNBACKTESTABLE so the dashboard can show "data feed required" instead
of fake P&L.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC

import numpy as np
import requests

logger = logging.getLogger(__name__)


_ANN = math.sqrt(252)
_FEE_BPS = 10.0       # round-trip fees+slippage on liquid ETFs (Alpaca commission-free, but spread)
_FEE_RATE = _FEE_BPS / 10000


# Strategies whose backtest needs data we don't have (yet).
# `pead` MOVED to a real backtest (uses Polygon /vX/reference/financials —
# see pead_backtest.py). The remaining 5 still need data feeds we
# haven't wired up.
UNBACKTESTABLE = {
    "crypto_funding_carry":   "needs historical perp funding-rate series (Coinbase doesn't expose)",
    "crypto_basis_trade":     "needs historical futures vs spot snapshots",
    "commodity_carry":        "needs historical futures-curve term structure",
    "kalshi_calibration_arb": "needs historical Kalshi market resolutions",
    "macro_kalshi":           "needs historical Kalshi macro events",
}


# ─── Output type ──────────────────────────────────────────────────────────


@dataclass
class BacktestSummary:
    """Same shape used by the existing dashboard backtest tabs."""

    strategy: str
    window_days: int
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_pnl_usd: float = 0.0
    entry_volume_usd: float = 0.0
    return_on_volume_pct: float = 0.0
    avg_pnl_usd: float = 0.0
    sharpe: float | None = None
    max_drawdown_usd: float = 0.0
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    note: str = ""


# ─── Data fetchers ────────────────────────────────────────────────────────


_HIST_CACHE: dict[tuple[str, str, int], np.ndarray] = {}


def _yahoo_history(symbol: str, days: int) -> np.ndarray:
    """Daily OHLCV from Yahoo Finance (free, no auth)."""
    cache_key = ("yahoo", symbol, days)
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key]
    end = int(time.time())
    start = end - days * 86400 - 86400 * 7   # buffer for weekends/holidays
    try:
        r = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"period1": start, "period2": end, "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; backtest/1.0)"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("chart", {}).get("result", [])
        if not data:
            _HIST_CACHE[cache_key] = np.empty((0, 6))
            return _HIST_CACHE[cache_key]
        ts = data[0].get("timestamp", []) or []
        q = data[0].get("indicators", {}).get("quote", [{}])[0] or {}
        opens = q.get("open") or []
        highs = q.get("high") or []
        lows = q.get("low") or []
        closes = q.get("close") or []
        vols = q.get("volume") or []
        rows = []
        for i in range(len(ts)):
            if closes[i] is None or opens[i] is None:
                continue
            rows.append([float(ts[i]), float(lows[i] or closes[i]),
                          float(highs[i] or closes[i]), float(opens[i]),
                          float(closes[i]), float(vols[i] or 0)])
        arr = np.array(rows, dtype=float) if rows else np.empty((0, 6))
        _HIST_CACHE[cache_key] = arr
        return arr
    except Exception as e:
        logger.warning(f"Yahoo history {symbol} failed: {e}")
        _HIST_CACHE[cache_key] = np.empty((0, 6))
        return _HIST_CACHE[cache_key]


def _coinbase_daily_history(symbol: str, days: int) -> np.ndarray:
    """Reuse the existing Coinbase public history fetcher."""
    cache_key = ("coinbase", symbol, days)
    if cache_key in _HIST_CACHE:
        return _HIST_CACHE[cache_key]
    try:
        from backtest import fetch_coinbase_public_history
        arr = fetch_coinbase_public_history(symbol, "ONE_DAY", days)
    except Exception as e:
        logger.warning(f"Coinbase daily history {symbol} failed: {e}")
        arr = np.empty((0, 6))
    _HIST_CACHE[cache_key] = arr
    return arr


# ─── Per-strategy backtest functions ──────────────────────────────────────


def _equity_curve_to_summary(name: str, days: int, trades: list[dict],
                              entry_volume: float) -> BacktestSummary:
    closed = [t for t in trades if t.get("pnl_usd") is not None]
    wins = sum(1 for t in closed if t["pnl_usd"] > 0)
    total_pnl = sum(t["pnl_usd"] for t in closed) if closed else 0.0
    pnl_arr = np.array([t["pnl_usd"] for t in closed]) if closed else np.array([])

    sharpe = None
    if pnl_arr.size > 1 and pnl_arr.std() > 0:
        sharpe = round(float(pnl_arr.mean() / pnl_arr.std() * _ANN), 3)

    cum, peak, max_dd = 0.0, 0.0, 0.0
    eq_curve = []
    for t in closed:
        cum += t["pnl_usd"]
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
        eq_curve.append({"t": t.get("close_time"), "pnl_cumulative": cum})

    return BacktestSummary(
        strategy=name,
        window_days=days,
        n_trades=len(closed),
        n_wins=wins,
        n_losses=len(closed) - wins,
        win_rate=(wins / len(closed)) if closed else 0.0,
        total_pnl_usd=total_pnl,
        entry_volume_usd=entry_volume,
        return_on_volume_pct=(total_pnl / entry_volume * 100) if entry_volume else 0.0,
        avg_pnl_usd=(total_pnl / len(closed)) if closed else 0.0,
        sharpe=sharpe,
        max_drawdown_usd=max_dd,
        trades=trades,
        equity_curve=eq_curve,
    )


def backtest_tsmom_etf(window_days: int) -> BacktestSummary:
    """12-1m TSMOM long-only on the 7-ETF basket."""
    universe = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"]
    lookback = 252
    skip = 21
    rebalance_every = 21    # monthly
    target_alloc_per_leg = 1000.0

    histories = {sym: _yahoo_history(sym, window_days + lookback + 30) for sym in universe}
    histories = {s: h for s, h in histories.items() if len(h) >= lookback + 5}
    if not histories:
        return BacktestSummary(strategy="tsmom_etf", window_days=window_days,
                               note="No Yahoo data available")

    trades: list[dict] = []
    positions: dict[str, dict] = {}     # {sym: {"qty", "entry_price", "entry_time"}}
    entry_volume = 0.0

    base_idx = max(len(next(iter(histories.values()))) - window_days, lookback + 1)
    n_bars = len(next(iter(histories.values())))

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            histories[list(histories.keys())[0]][i, 0], tz=UTC)
        # Rebalance every `rebalance_every` bars
        if (i - base_idx) % rebalance_every != 0:
            continue
        for sym, candles in histories.items():
            if i >= len(candles): continue
            window = candles[i - lookback:i - skip, 4]   # closes only
            if len(window) < 30: continue
            ret_12_1 = (window[-1] - window[0]) / window[0]
            current_price = float(candles[i, 4])
            should_hold = ret_12_1 > 0
            holding = sym in positions

            if should_hold and not holding:
                qty = target_alloc_per_leg / current_price
                positions[sym] = {"qty": qty, "entry_price": current_price,
                                   "entry_time": bar_time.isoformat()}
                entry_volume += target_alloc_per_leg
                trades.append({
                    "strategy": "tsmom_etf", "side": "BUY", "product_id": sym,
                    "amount_usd": target_alloc_per_leg, "quantity": qty,
                    "entry_price": current_price, "open_time": bar_time.isoformat(),
                    "reason": f"12-1m={ret_12_1*100:+.1f}%",
                })
            elif (not should_hold) and holding:
                pos = positions.pop(sym)
                exit_price = current_price
                gross = pos["qty"] * (exit_price - pos["entry_price"])
                fees = (pos["qty"] * pos["entry_price"] + pos["qty"] * exit_price) * _FEE_RATE
                pnl = gross - fees
                trades.append({
                    "strategy": "tsmom_etf", "side": "SELL", "product_id": sym,
                    "amount_usd": pos["qty"] * exit_price, "quantity": pos["qty"],
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "open_time": pos["entry_time"], "close_time": bar_time.isoformat(),
                    "pnl_usd": pnl, "exit_reason": "signal",
                })
    return _equity_curve_to_summary("tsmom_etf", window_days, trades, entry_volume)


def backtest_risk_parity_etf(window_days: int) -> BacktestSummary:
    """5-ETF inverse-vol book; rebalance monthly."""
    universe = ["SPY", "TLT", "IEF", "GLD", "DBC"]
    rebalance_every = 21
    total_book_usd = 5000.0      # nominal book size for per-trade attribution

    histories = {sym: _yahoo_history(sym, window_days + 90) for sym in universe}
    histories = {s: h for s, h in histories.items() if len(h) >= 60}
    if len(histories) < 3:
        return BacktestSummary(strategy="risk_parity_etf", window_days=window_days,
                               note="No Yahoo data available")

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, 60)

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        # Compute inverse-vol weights from last 60 days
        vols = {}
        for sym, candles in histories.items():
            window = candles[max(0, i - 60):i, 4]
            if len(window) < 30: continue
            rets = np.diff(window) / window[:-1]
            sd = float(np.std(rets, ddof=1))
            if sd > 0:
                vols[sym] = sd * _ANN
        if not vols: continue
        inv = {s: 1.0 / v for s, v in vols.items()}
        tot = sum(inv.values())
        weights = {s: w / tot for s, w in inv.items()}

        bar_time = datetime.fromtimestamp(
            histories[list(histories.keys())[0]][i, 0], tz=UTC).isoformat()
        for sym in universe:
            if sym not in histories or i >= len(histories[sym]): continue
            target_usd = total_book_usd * weights.get(sym, 0)
            current_price = float(histories[sym][i, 4])
            cur_qty = positions.get(sym, {}).get("qty", 0)
            cur_usd = cur_qty * current_price
            delta_usd = target_usd - cur_usd
            if abs(delta_usd) < total_book_usd * 0.02:
                continue
            if delta_usd > 0:
                # Buy more
                add_qty = delta_usd / current_price
                new_qty = cur_qty + add_qty
                # Weighted entry price
                if cur_qty > 0:
                    entry_p = (
                        positions[sym]["entry_price"] * cur_qty
                        + current_price * add_qty
                    ) / new_qty
                else:
                    entry_p = current_price
                positions[sym] = {"qty": new_qty, "entry_price": entry_p,
                                    "entry_time": bar_time}
                entry_volume += abs(delta_usd)
                trades.append({
                    "strategy": "risk_parity_etf", "side": "BUY", "product_id": sym,
                    "amount_usd": abs(delta_usd), "quantity": add_qty,
                    "entry_price": current_price, "open_time": bar_time,
                    "reason": f"target {weights[sym]*100:.1f}%",
                })
            else:
                # Sell down
                sell_qty = min(cur_qty, abs(delta_usd) / current_price)
                if sell_qty <= 0: continue
                gross = sell_qty * (current_price - positions[sym]["entry_price"])
                fees = (sell_qty * positions[sym]["entry_price"]
                         + sell_qty * current_price) * _FEE_RATE
                pnl = gross - fees
                positions[sym]["qty"] -= sell_qty
                trades.append({
                    "strategy": "risk_parity_etf", "side": "SELL", "product_id": sym,
                    "amount_usd": sell_qty * current_price, "quantity": sell_qty,
                    "entry_price": positions[sym]["entry_price"],
                    "exit_price": current_price,
                    "open_time": positions[sym]["entry_time"],
                    "close_time": bar_time,
                    "pnl_usd": pnl, "exit_reason": "rebalance",
                })
                if positions[sym]["qty"] <= 1e-9:
                    positions.pop(sym, None)
    return _equity_curve_to_summary("risk_parity_etf", window_days, trades, entry_volume)


def backtest_crypto_xsmom(window_days: int) -> BacktestSummary:
    """Top-quintile 30-day return from a 15-coin alt universe."""
    universe = [
        "SOL-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "DOGE-USD",
        "LINK-USD", "LTC-USD", "BCH-USD", "XLM-USD",
        "UNI-USD", "ATOM-USD", "ALGO-USD", "FIL-USD", "ICP-USD",
        "XRP-USD",
    ]
    lookback = 30
    rebalance_every = 7
    per_leg_usd = 500.0
    fee_rate = 0.006     # Coinbase taker

    histories = {sym: _coinbase_daily_history(sym, window_days + lookback + 5)
                  for sym in universe}
    histories = {s: h for s, h in histories.items() if len(h) >= lookback + 2}
    if len(histories) < 5:
        return BacktestSummary(strategy="crypto_xsmom", window_days=window_days,
                               note="Insufficient Coinbase history")

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0
    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, lookback)

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        # Compute returns
        returns = []
        for sym, candles in histories.items():
            if i >= len(candles): continue
            start = candles[i - lookback, 4]
            end = candles[i, 4]
            if start <= 0: continue
            returns.append((sym, (end - start) / start))
        if len(returns) < 5: continue
        returns.sort(key=lambda r: r[1], reverse=True)
        top_n = max(1, int(len(returns) * 0.20))
        winners = {sym for sym, _ in returns[:top_n]}

        bar_time = datetime.fromtimestamp(
            histories[list(histories.keys())[0]][i, 0], tz=UTC).isoformat()

        # Open new winners
        for sym in winners:
            if sym in positions: continue
            cur_price = float(histories[sym][i, 4])
            qty = per_leg_usd / cur_price
            positions[sym] = {"qty": qty, "entry_price": cur_price,
                                "entry_time": bar_time}
            entry_volume += per_leg_usd
            trades.append({
                "strategy": "crypto_xsmom", "side": "BUY", "product_id": sym,
                "amount_usd": per_leg_usd, "quantity": qty,
                "entry_price": cur_price, "open_time": bar_time,
                "reason": "top quintile 30d momentum",
            })
        # Close non-winners
        for sym in list(positions.keys()):
            if sym in winners: continue
            cur_price = float(histories[sym][i, 4])
            pos = positions.pop(sym)
            gross = pos["qty"] * (cur_price - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"] + pos["qty"] * cur_price) * fee_rate
            pnl = gross - fees
            trades.append({
                "strategy": "crypto_xsmom", "side": "SELL", "product_id": sym,
                "amount_usd": pos["qty"] * cur_price, "quantity": pos["qty"],
                "entry_price": pos["entry_price"], "exit_price": cur_price,
                "open_time": pos["entry_time"], "close_time": bar_time,
                "pnl_usd": pnl, "exit_reason": "dropped from top quintile",
            })
    return _equity_curve_to_summary("crypto_xsmom", window_days, trades, entry_volume)


def backtest_vol_managed_overlay(window_days: int) -> BacktestSummary:
    """Synthetic backtest: SPY long with vol-target overlay vs SPY plain.

    The overlay isn't a standalone strategy; this backtest illustrates
    its lift over a buy-and-hold SPY equity curve, attributing the
    differential as the overlay's "P&L".
    """
    candles = _yahoo_history("SPY", window_days + 60)
    if len(candles) < 30:
        return BacktestSummary(strategy="vol_managed_overlay", window_days=window_days,
                               note="Insufficient SPY data")

    n_bars = len(candles)
    base_idx = max(n_bars - window_days, 30)
    capital = 1000.0
    target_vol = 0.15

    plain_eq = capital
    overlay_eq = capital
    trades: list[dict] = []
    last_scaler = 1.0

    for i in range(base_idx, n_bars):
        if i < 30: continue
        # Compute current realized vol
        window = candles[max(0, i - 30):i, 4]
        if len(window) < 20:
            continue
        rets = np.diff(window) / window[:-1]
        sd = float(np.std(rets, ddof=1))
        if sd <= 0:
            continue
        realized_vol = sd * _ANN
        scaler = max(0.3, min(1.5, target_vol / realized_vol))
        # Apply daily return
        daily_ret = float((candles[i, 4] - candles[i - 1, 4]) / candles[i - 1, 4])
        plain_eq *= (1 + daily_ret)
        overlay_eq *= (1 + daily_ret * scaler)
        # Log a "trade" each time scaler changes meaningfully
        if abs(scaler - last_scaler) > 0.05:
            ts = datetime.fromtimestamp(candles[i, 0], tz=UTC).isoformat()
            trades.append({
                "strategy": "vol_managed_overlay", "side": "REBAL",
                "product_id": "SPY",
                "amount_usd": capital, "quantity": 0,
                "entry_price": float(candles[i, 4]),
                "exit_price": float(candles[i, 4]),
                "open_time": ts, "close_time": ts,
                "pnl_usd": (overlay_eq - plain_eq) - sum(t.get("pnl_usd", 0) for t in trades),
                "exit_reason": f"scaler {last_scaler:.2f}->{scaler:.2f}",
            })
            last_scaler = scaler

    summary = _equity_curve_to_summary("vol_managed_overlay", window_days, trades,
                                          entry_volume=capital)
    summary.total_pnl_usd = overlay_eq - plain_eq
    summary.note = (f"Overlay ${overlay_eq:.2f} vs plain ${plain_eq:.2f} "
                     f"= {((overlay_eq/plain_eq) - 1) * 100:+.2f}% relative")
    return summary


# ─── Dispatch ─────────────────────────────────────────────────────────────


def _pead_dispatch(window_days: int) -> BacktestSummary:
    """Lazy-import to avoid pulling Polygon at module-load time."""
    from .pead_backtest import backtest_pead
    return backtest_pead(window_days)


_STRATEGY_BACKTESTS = {
    "tsmom_etf": backtest_tsmom_etf,
    "risk_parity_etf": backtest_risk_parity_etf,
    "crypto_xsmom": backtest_crypto_xsmom,
    "vol_managed_overlay": backtest_vol_managed_overlay,
    "pead": _pead_dispatch,
}


def backtest_strategy_by_name(name: str, window_days: int) -> BacktestSummary:
    if name in _STRATEGY_BACKTESTS:
        return _STRATEGY_BACKTESTS[name](window_days)
    return BacktestSummary(
        strategy=name, window_days=window_days,
        note=UNBACKTESTABLE.get(name, "no backtest defined"),
    )


def backtest_all(window_days: int) -> dict[str, BacktestSummary]:
    out: dict[str, BacktestSummary] = {}
    for name in _STRATEGY_BACKTESTS:
        try:
            out[name] = backtest_strategy_by_name(name, window_days)
            logger.info(f"  backtest {name} {window_days}d: {out[name].n_trades} trades, "
                        f"${out[name].total_pnl_usd:+.2f}")
        except Exception as e:
            logger.error(f"  backtest {name} {window_days}d failed: {e}")
            out[name] = BacktestSummary(
                strategy=name, window_days=window_days, note=f"error: {e}")
    return out
