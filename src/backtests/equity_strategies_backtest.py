"""Equity-strategy backtests using Yahoo daily bars.

Five strategies in one module — each follows the same template
(walk daily bars, apply entry/exit logic, summarize via
runner._equity_curve_to_summary). Yahoo is free, no auth.

Strategies implemented:
  - rsi_mean_reversion    Connors RSI(2) on S&P 100 universe
  - bollinger_breakout    BB upper-band breakout on liquid mid+caps
  - gap_trading           Open-gap reversion on liquid stocks
  - low_vol_anomaly       Long lowest-vol quintile of S&P 500 ETF basket
  - turn_of_month         SPY long, last 3 + first 2 trading days

Each entry-point: backtest_<name>(window_days) → BacktestSummary.

Limitations honestly noted:
  - Yahoo daily bars don't carry true open quotes for gap_trading;
    we approximate via the bar's open which is close-to-close-after-
    night-session. Live strategy uses Alpaca's tick data — this
    backtest's gap signals will differ by ~10 bps.
  - rsi_mean_reversion and bollinger_breakout assume same-day fills
    at close. Real fills are next-bar open with slippage; the
    backtest's results are therefore optimistic by ~5-10 bps per
    round-trip.
  - turn_of_month uses calendar days; real market calendar (with
    NYSE holiday calendar) would be cleaner.

These backtests are calibration tools, not P&L predictors. Use
them to confirm directional edge + discount the live numbers.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np

from .runner import (
    _ANN,
    _FEE_RATE,
    BacktestSummary,
    _equity_curve_to_summary,
    _yahoo_history,
)

logger = logging.getLogger(__name__)


# ─── Common universes (matches live strategies) ───────────────────────


_RSI_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "JPM", "BAC", "V", "MA", "GS", "MS",
    "JNJ", "UNH", "LLY", "ABBV",
    "CAT", "GE", "XOM", "CVX",
    "WMT", "COST", "HD", "MCD", "KO", "PEP", "PG",
    "DIS", "T", "NEE",
]

_BOLLINGER_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "CRM", "ORCL", "ADBE", "NFLX", "INTC", "QCOM",
    "JPM", "GS", "BAC", "MS", "V", "MA",
    "LLY", "UNH", "ABBV",
    "XOM", "CVX",
    "WMT", "COST", "HD", "MCD",
    "DIS", "NKE", "BA", "CAT",
]

_LOW_VOL_UNIVERSE = [
    # Sector-ETF basket — equivalent to S&P 500 by sector
    "XLK", "XLV", "XLF", "XLE", "XLI", "XLP", "XLY", "XLU", "XLRE",
    "XLB", "XLC",
    # Plus a few low-vol names for granularity
    "JNJ", "PG", "KO", "PEP", "WMT", "COST",
]


# ─── 1) rsi_mean_reversion ────────────────────────────────────────────


def backtest_rsi_mean_reversion(window_days: int) -> BacktestSummary:
    """Connors RSI(2) < 10 entry, RSI(2) > 65 or 5d hold or -3% stop exit.

    Mirrors src/strategies/rsi_mean_reversion.py thresholds; the live
    strategy's stop-loss (Sprint B2 fix) is enforced here too."""
    rsi_period, rsi_oversold, rsi_overbought = 2, 10, 65
    sma_period, max_hold_days = 200, 5
    stop_loss_pct, max_concurrent = 0.03, 5
    size_per_slot = 1000.0

    histories = _load_universe(_RSI_UNIVERSE,
                                  window_days + sma_period + 30)
    if len(histories) < 5:
        return BacktestSummary(
            strategy="rsi_mean_reversion", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, sma_period + rsi_period)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # ── Exits first
        for sym in list(positions.keys()):
            pos = positions[sym]
            if i >= len(histories[sym]):
                continue
            close = float(histories[sym][i, 4])
            should_exit, reason = False, ""
            ret = (close - pos["entry_price"]) / pos["entry_price"]
            if ret <= -stop_loss_pct:
                should_exit, reason = True, f"STOP {ret*100:.2f}%"
            elif (i - pos["entry_idx"]) >= max_hold_days:
                should_exit, reason = True, f">{max_hold_days}d hold"
            else:
                rsi = _rsi(histories[sym][:i + 1, 4], rsi_period)
                if rsi is not None and rsi >= rsi_overbought:
                    should_exit, reason = True, f"RSI={rsi:.0f}"
            if should_exit:
                gross = pos["qty"] * (close - pos["entry_price"])
                fees = (pos["qty"] * pos["entry_price"]
                         + pos["qty"] * close) * _FEE_RATE
                trades.append({
                    "strategy": "rsi_mean_reversion", "side": "SELL",
                    "product_id": sym,
                    "amount_usd": pos["qty"] * close, "quantity": pos["qty"],
                    "entry_price": pos["entry_price"], "exit_price": close,
                    "open_time": pos["entry_time"], "close_time": bar_time,
                    "pnl_usd": gross - fees, "exit_reason": reason,
                })
                positions.pop(sym)

        # ── Entries: max 5 concurrent
        slots = max_concurrent - len(positions)
        if slots <= 0:
            continue
        candidates: list[tuple[str, float, float]] = []
        for sym, candles in histories.items():
            if sym in positions or i >= len(candles):
                continue
            closes = candles[:i + 1, 4]
            rsi = _rsi(closes, rsi_period)
            if rsi is None or rsi > rsi_oversold:
                continue
            sma = _sma(closes, sma_period)
            if sma is None:
                continue
            last = float(candles[i, 4])
            if last < sma:
                continue
            score = (rsi_oversold - rsi) + (last / sma - 1.0) * 100
            candidates.append((sym, score, last))
        candidates.sort(key=lambda c: c[1], reverse=True)
        for sym, _, last in candidates[:slots]:
            qty = size_per_slot / last
            positions[sym] = {
                "qty": qty, "entry_price": last, "entry_idx": i,
                "entry_time": bar_time,
            }
            entry_volume += size_per_slot
            trades.append({
                "strategy": "rsi_mean_reversion", "side": "BUY",
                "product_id": sym, "amount_usd": size_per_slot,
                "quantity": qty, "entry_price": last,
                "open_time": bar_time, "reason": "RSI<10, >SMA200",
            })

    return _equity_curve_to_summary(
        "rsi_mean_reversion", window_days, trades, entry_volume,
    )


# ─── 2) bollinger_breakout ────────────────────────────────────────────


def backtest_bollinger_breakout(window_days: int) -> BacktestSummary:
    """20d Bollinger upper-band breakout, exit on cross below mid-band.
    Mirrors src/strategies/bollinger_breakout.py."""
    band_period, band_stddev = 20, 2.0
    max_concurrent, size_per_slot = 5, 4000.0

    histories = _load_universe(_BOLLINGER_UNIVERSE,
                                  window_days + band_period + 10)
    if len(histories) < 5:
        return BacktestSummary(
            strategy="bollinger_breakout", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, band_period + 5)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # ── Exits: close below mid-band
        for sym in list(positions.keys()):
            pos = positions[sym]
            if i >= len(histories[sym]):
                continue
            closes = histories[sym][:i + 1, 4]
            mid = _sma(closes, band_period)
            close = float(closes[-1])
            if mid is None:
                continue
            if close < mid:
                gross = pos["qty"] * (close - pos["entry_price"])
                fees = (pos["qty"] * pos["entry_price"]
                         + pos["qty"] * close) * _FEE_RATE
                trades.append({
                    "strategy": "bollinger_breakout", "side": "SELL",
                    "product_id": sym, "amount_usd": pos["qty"] * close,
                    "quantity": pos["qty"], "entry_price": pos["entry_price"],
                    "exit_price": close, "open_time": pos["entry_time"],
                    "close_time": bar_time, "pnl_usd": gross - fees,
                    "exit_reason": "below mid-band",
                })
                positions.pop(sym)

        # ── Entries: close > upper band
        slots = max_concurrent - len(positions)
        if slots <= 0:
            continue
        for sym, candles in histories.items():
            if sym in positions or i >= len(candles) or slots <= 0:
                continue
            closes = candles[:i + 1, 4]
            mid = _sma(closes, band_period)
            if mid is None or len(closes) < band_period:
                continue
            window = closes[-band_period:]
            sd = float(np.std(window, ddof=1))
            upper = mid + band_stddev * sd
            close = float(closes[-1])
            prev_close = float(closes[-2]) if len(closes) >= 2 else close
            # Breakout: today's close > upper AND yesterday's close <= upper
            if close > upper and prev_close <= upper:
                qty = size_per_slot / close
                positions[sym] = {
                    "qty": qty, "entry_price": close, "entry_time": bar_time,
                }
                entry_volume += size_per_slot
                trades.append({
                    "strategy": "bollinger_breakout", "side": "BUY",
                    "product_id": sym, "amount_usd": size_per_slot,
                    "quantity": qty, "entry_price": close,
                    "open_time": bar_time,
                    "reason": f"close ${close:.2f} > upper ${upper:.2f}",
                })
                slots -= 1

    return _equity_curve_to_summary(
        "bollinger_breakout", window_days, trades, entry_volume,
    )


# ─── 3) gap_trading ───────────────────────────────────────────────────


def backtest_gap_trading(window_days: int) -> BacktestSummary:
    """Gap reversion: when a stock opens >1.5% from the prior close,
    fade the gap by holding 1 day and exiting at next close."""
    gap_pct_min, max_concurrent = 0.015, 3
    size_per_slot = 3000.0

    histories = _load_universe(_RSI_UNIVERSE, window_days + 30)
    if len(histories) < 5:
        return BacktestSummary(
            strategy="gap_trading", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, 5)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # ── Exit: next-close exit (1-day hold)
        for sym in list(positions.keys()):
            pos = positions[sym]
            if i >= len(histories[sym]):
                continue
            close = float(histories[sym][i, 4])
            gross = pos["qty"] * (close - pos["entry_price"]) * pos["dir"]
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * close) * _FEE_RATE
            trades.append({
                "strategy": "gap_trading",
                "side": "SELL" if pos["dir"] > 0 else "BUY-COVER",
                "product_id": sym, "amount_usd": pos["qty"] * close,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": close, "open_time": pos["entry_time"],
                "close_time": bar_time, "pnl_usd": gross - fees,
                "exit_reason": "1d hold",
            })
            positions.pop(sym)

        # ── Entries: gap > 1.5%, enter at OPEN (proxied), fade
        slots = max_concurrent - len(positions)
        if slots <= 0:
            continue
        for sym, candles in histories.items():
            if sym in positions or i >= len(candles) or slots <= 0:
                continue
            if i < 1:
                continue
            prev_close = float(candles[i - 1, 4])
            today_open = float(candles[i, 3])
            if prev_close <= 0:
                continue
            gap = (today_open - prev_close) / prev_close
            if abs(gap) < gap_pct_min:
                continue
            # Fade the gap: gap up → SHORT (dir=-1); gap down → LONG (dir=+1)
            direction = -1 if gap > 0 else 1
            qty = size_per_slot / today_open
            positions[sym] = {
                "qty": qty, "entry_price": today_open, "dir": direction,
                "entry_time": bar_time,
            }
            entry_volume += size_per_slot
            trades.append({
                "strategy": "gap_trading",
                "side": "BUY" if direction > 0 else "SHORT",
                "product_id": sym, "amount_usd": size_per_slot,
                "quantity": qty, "entry_price": today_open,
                "open_time": bar_time,
                "reason": f"gap {gap*100:+.2f}% — fade",
            })
            slots -= 1

    return _equity_curve_to_summary(
        "gap_trading", window_days, trades, entry_volume,
    )


# ─── 4) low_vol_anomaly ───────────────────────────────────────────────


def backtest_low_vol_anomaly(window_days: int) -> BacktestSummary:
    """Long the lowest-vol quintile of a sector-ETF basket; rebalance
    monthly. Documented edge: low-vol stocks earn higher risk-adjusted
    returns than CAPM predicts (Frazzini & Pedersen, 2014)."""
    vol_lookback, rebalance_every = 30, 21
    total_book_usd = 5000.0

    histories = _load_universe(_LOW_VOL_UNIVERSE,
                                  window_days + vol_lookback + 10)
    if len(histories) < 4:
        return BacktestSummary(
            strategy="low_vol_anomaly", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, vol_lookback + 1)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # Rank by realized vol over lookback window
        vols: list[tuple[str, float, float]] = []
        for sym, candles in histories.items():
            if i >= len(candles):
                continue
            window = candles[max(0, i - vol_lookback):i, 4]
            if len(window) < vol_lookback // 2:
                continue
            rets = np.diff(window) / window[:-1]
            sd = float(np.std(rets, ddof=1)) * _ANN
            vols.append((sym, sd, float(candles[i, 4])))
        if len(vols) < 4:
            continue

        # Bottom quintile by vol = our longs
        vols.sort(key=lambda v: v[1])
        n_keep = max(1, len(vols) // 5)
        winners = {v[0] for v in vols[:n_keep]}
        per_leg_usd = total_book_usd / max(1, n_keep)

        # Close positions not in winners
        for sym in list(positions.keys()):
            if sym in winners:
                continue
            pos = positions.pop(sym)
            if i >= len(histories[sym]):
                continue
            close = float(histories[sym][i, 4])
            gross = pos["qty"] * (close - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * close) * _FEE_RATE
            trades.append({
                "strategy": "low_vol_anomaly", "side": "SELL",
                "product_id": sym, "amount_usd": pos["qty"] * close,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": close, "open_time": pos["entry_time"],
                "close_time": bar_time, "pnl_usd": gross - fees,
                "exit_reason": "no longer bottom-quintile vol",
            })

        # Open new winners
        for sym in winners:
            if sym in positions:
                continue
            close = next(v[2] for v in vols if v[0] == sym)
            qty = per_leg_usd / close
            positions[sym] = {
                "qty": qty, "entry_price": close, "entry_time": bar_time,
            }
            entry_volume += per_leg_usd
            trades.append({
                "strategy": "low_vol_anomaly", "side": "BUY",
                "product_id": sym, "amount_usd": per_leg_usd,
                "quantity": qty, "entry_price": close,
                "open_time": bar_time, "reason": "bottom-quintile vol",
            })

    return _equity_curve_to_summary(
        "low_vol_anomaly", window_days, trades, entry_volume,
    )


# ─── 5) turn_of_month ─────────────────────────────────────────────────


def backtest_turn_of_month(window_days: int) -> BacktestSummary:
    """Long SPY on the last 3 trading days of the month + first 2 of
    the next. Documented anomaly: month-end pension/401k rebalancing
    flows lift equities disproportionately (Lakonishok & Smidt, 1988
    revisited; Etula et al., 2020)."""
    candles = _yahoo_history("SPY", window_days + 60)
    if len(candles) < 30:
        return BacktestSummary(
            strategy="turn_of_month", window_days=window_days,
            note="Insufficient SPY data",
        )

    n_bars = len(candles)
    base_idx = max(n_bars - window_days, 5)
    size_usd = 5000.0

    trades: list[dict] = []
    in_position = False
    entry_idx = 0
    entry_price = 0.0
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(candles[i, 0], tz=UTC)
        in_window = _is_turn_of_month(bar_time, candles, i)

        if in_window and not in_position:
            in_position = True
            entry_idx = i
            entry_price = float(candles[i, 4])
            qty = size_usd / entry_price
            entry_volume += size_usd
            trades.append({
                "strategy": "turn_of_month", "side": "BUY",
                "product_id": "SPY", "amount_usd": size_usd,
                "quantity": qty, "entry_price": entry_price,
                "open_time": bar_time.isoformat(),
                "reason": "turn-of-month entry",
            })
        elif (not in_window) and in_position:
            in_position = False
            close = float(candles[i, 4])
            qty = size_usd / entry_price
            gross = qty * (close - entry_price)
            fees = (qty * entry_price + qty * close) * _FEE_RATE
            trades.append({
                "strategy": "turn_of_month", "side": "SELL",
                "product_id": "SPY", "amount_usd": qty * close,
                "quantity": qty, "entry_price": entry_price,
                "exit_price": close,
                "open_time": datetime.fromtimestamp(
                    candles[entry_idx, 0], tz=UTC,
                ).isoformat(),
                "close_time": bar_time.isoformat(),
                "pnl_usd": gross - fees,
                "exit_reason": "turn-of-month window closed",
            })

    return _equity_curve_to_summary(
        "turn_of_month", window_days, trades, entry_volume,
    )


# ─── 6) sector_rotation ───────────────────────────────────────────────


_SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                 "XLI", "XLB", "XLU", "XLRE", "XLC"]


def backtest_sector_rotation(window_days: int) -> BacktestSummary:
    """Top-N SPDR sector ETFs by 90d trailing return; rebalance monthly.
    Mirrors src/strategies/sector_rotation.py: TOP_N=3, LOOKBACK=90."""
    lookback, top_n = 90, 3
    rebalance_every = 21
    per_leg_usd = 4000.0

    histories = _load_universe(_SECTOR_ETFS, window_days + lookback + 10)
    if len(histories) < 4:
        return BacktestSummary(
            strategy="sector_rotation", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, lookback + 1)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # Rank by 90-day return
        scored: list[tuple[str, float, float]] = []
        for sym, candles in histories.items():
            if i >= len(candles) or i - lookback < 0:
                continue
            start = float(candles[i - lookback, 4])
            end = float(candles[i, 4])
            if start <= 0:
                continue
            ret = (end - start) / start
            if ret < 0:    # don't long sectors with negative trend
                continue
            scored.append((sym, ret, end))
        if not scored:
            continue
        scored.sort(key=lambda r: r[1], reverse=True)
        winners = {s for s, _, _ in scored[:top_n]}

        # Close non-winners
        for sym in list(positions.keys()):
            if sym in winners or i >= len(histories[sym]):
                continue
            pos = positions.pop(sym)
            close = float(histories[sym][i, 4])
            gross = pos["qty"] * (close - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * close) * _FEE_RATE
            trades.append({
                "strategy": "sector_rotation", "side": "SELL",
                "product_id": sym, "amount_usd": pos["qty"] * close,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": close, "open_time": pos["entry_time"],
                "close_time": bar_time, "pnl_usd": gross - fees,
                "exit_reason": "out of top-3",
            })
        # Open new winners
        for sym, _, last in scored[:top_n]:
            if sym in positions:
                continue
            qty = per_leg_usd / last
            positions[sym] = {
                "qty": qty, "entry_price": last, "entry_time": bar_time,
            }
            entry_volume += per_leg_usd
            trades.append({
                "strategy": "sector_rotation", "side": "BUY",
                "product_id": sym, "amount_usd": per_leg_usd,
                "quantity": qty, "entry_price": last,
                "open_time": bar_time, "reason": "top-3 by 90d return",
            })

    return _equity_curve_to_summary(
        "sector_rotation", window_days, trades, entry_volume,
    )


# ─── 7) pairs_trading ─────────────────────────────────────────────────


_PAIRS = [
    ("KO", "PEP"), ("V", "MA"), ("GS", "MS"),
    ("JPM", "BAC"), ("HD", "LOW"), ("CVX", "XOM"),
]


def backtest_pairs_trading(window_days: int) -> BacktestSummary:
    """Stat-arb on 6 classic correlated pairs. When the price ratio
    A/B drifts > 2σ from its 60-day mean, long the cheap leg / short
    the expensive leg; close on mean revert."""
    lookback = 60
    z_entry, z_exit = 2.0, 0.5
    per_leg_usd = 1500.0
    fee_rate = _FEE_RATE

    all_syms = sorted({s for pair in _PAIRS for s in pair})
    histories = _load_universe(all_syms, window_days + lookback + 10)
    if len(histories) < 6:
        return BacktestSummary(
            strategy="pairs_trading", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, lookback + 1)

    trades: list[dict] = []
    # positions[(a,b)] = {"long": sym, "short": sym, "long_qty", "short_qty",
    #                    "entry_long", "entry_short", "entry_idx", "entry_time"}
    positions: dict[tuple[str, str], dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        for a, b in _PAIRS:
            if a not in histories or b not in histories:
                continue
            ca = histories[a]
            cb = histories[b]
            if i >= len(ca) or i >= len(cb):
                continue
            window_a = ca[max(0, i - lookback):i, 4]
            window_b = cb[max(0, i - lookback):i, 4]
            if len(window_a) < lookback or len(window_b) < lookback:
                continue
            mask = (window_a > 0) & (window_b > 0)
            ratios = window_a[mask] / window_b[mask]
            if len(ratios) < lookback // 2:
                continue
            mean = float(ratios.mean())
            sd = float(ratios.std(ddof=1))
            if sd <= 0:
                continue
            cur_a = float(ca[i, 4])
            cur_b = float(cb[i, 4])
            if cur_b <= 0:
                continue
            cur_ratio = cur_a / cur_b
            z = (cur_ratio - mean) / sd

            pos = positions.get((a, b))

            if pos is None and abs(z) >= z_entry:
                # Long the cheap leg, short the expensive leg
                if z > 0:
                    # ratio too high → A is rich, B is cheap → long B, short A
                    long_sym, short_sym = b, a
                    long_px, short_px = cur_b, cur_a
                else:
                    long_sym, short_sym = a, b
                    long_px, short_px = cur_a, cur_b
                long_qty = per_leg_usd / long_px
                short_qty = per_leg_usd / short_px
                positions[(a, b)] = {
                    "long": long_sym, "short": short_sym,
                    "long_qty": long_qty, "short_qty": short_qty,
                    "entry_long": long_px, "entry_short": short_px,
                    "entry_time": bar_time, "entry_z": z,
                }
                entry_volume += 2 * per_leg_usd
                trades.append({
                    "strategy": "pairs_trading", "side": "PAIR-OPEN",
                    "product_id": f"{long_sym}/{short_sym}",
                    "amount_usd": 2 * per_leg_usd,
                    "quantity": long_qty + short_qty,
                    "entry_price": cur_ratio,
                    "open_time": bar_time,
                    "reason": f"z={z:+.2f}",
                })
            elif pos is not None and abs(z) <= z_exit:
                # Mean-revert: close both legs
                long_close = (cur_a if pos["long"] == a else cur_b)
                short_close = (cur_b if pos["long"] == a else cur_a)
                long_pnl = pos["long_qty"] * (long_close - pos["entry_long"])
                short_pnl = pos["short_qty"] * (pos["entry_short"] - short_close)
                gross = long_pnl + short_pnl
                fees = 4 * per_leg_usd * fee_rate    # 4 legs total
                trades.append({
                    "strategy": "pairs_trading", "side": "PAIR-CLOSE",
                    "product_id": f"{pos['long']}/{pos['short']}",
                    "amount_usd": 2 * per_leg_usd,
                    "quantity": pos["long_qty"] + pos["short_qty"],
                    "entry_price": pos["entry_z"],
                    "exit_price": z,
                    "open_time": pos["entry_time"], "close_time": bar_time,
                    "pnl_usd": gross - fees,
                    "exit_reason": f"|z|={abs(z):.2f} ≤ {z_exit}",
                })
                del positions[(a, b)]

    return _equity_curve_to_summary(
        "pairs_trading", window_days, trades, entry_volume,
    )


# ─── 8) dividend_growth ───────────────────────────────────────────────


_DIV_ETFS = ["VYM", "SCHD", "DVY", "HDV", "NOBL", "DGRO", "SPHD"]


def backtest_dividend_growth(window_days: int) -> BacktestSummary:
    """Top-2 dividend ETFs by 90d total return; rebalance monthly.
    Mirrors src/strategies/dividend_growth.py: TOP_N=2, LOOKBACK=90."""
    lookback, top_n = 90, 2
    rebalance_every = 21
    per_leg_usd = 3000.0

    histories = _load_universe(_DIV_ETFS, window_days + lookback + 10)
    if len(histories) < 3:
        return BacktestSummary(
            strategy="dividend_growth", window_days=window_days,
            note="Insufficient Yahoo data",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, lookback + 1)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        scored: list[tuple[str, float, float]] = []
        for sym, candles in histories.items():
            if i >= len(candles) or i - lookback < 0:
                continue
            start = float(candles[i - lookback, 4])
            end = float(candles[i, 4])
            if start <= 0:
                continue
            ret = (end - start) / start
            scored.append((sym, ret, end))
        if not scored:
            continue
        scored.sort(key=lambda r: r[1], reverse=True)
        winners = {s for s, _, _ in scored[:top_n]}

        for sym in list(positions.keys()):
            if sym in winners or i >= len(histories[sym]):
                continue
            pos = positions.pop(sym)
            close = float(histories[sym][i, 4])
            gross = pos["qty"] * (close - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * close) * _FEE_RATE
            trades.append({
                "strategy": "dividend_growth", "side": "SELL",
                "product_id": sym, "amount_usd": pos["qty"] * close,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": close, "open_time": pos["entry_time"],
                "close_time": bar_time, "pnl_usd": gross - fees,
                "exit_reason": "out of top-2",
            })
        for sym, _, last in scored[:top_n]:
            if sym in positions:
                continue
            qty = per_leg_usd / last
            positions[sym] = {
                "qty": qty, "entry_price": last, "entry_time": bar_time,
            }
            entry_volume += per_leg_usd
            trades.append({
                "strategy": "dividend_growth", "side": "BUY",
                "product_id": sym, "amount_usd": per_leg_usd,
                "quantity": qty, "entry_price": last,
                "open_time": bar_time, "reason": "top-2 by 90d return",
            })

    return _equity_curve_to_summary(
        "dividend_growth", window_days, trades, entry_volume,
    )


# ─── 9) internationals_rotation ───────────────────────────────────────


_INTL_ETFS = ["EFA", "EEM", "EWJ", "EWG", "EWU", "INDA", "EWZ", "FXI"]
_US_BASELINE = "SPY"


def backtest_internationals_rotation(window_days: int) -> BacktestSummary:
    """Long top-2 country ETFs only when their 90d return beats SPY.
    Mirrors src/strategies/internationals_rotation.py."""
    lookback, top_n = 90, 2
    rebalance_every = 21
    per_leg_usd = 3000.0

    universe = _INTL_ETFS + [_US_BASELINE]
    histories = _load_universe(universe, window_days + lookback + 10)
    if _US_BASELINE not in histories or len(histories) < 4:
        return BacktestSummary(
            strategy="internationals_rotation", window_days=window_days,
            note="Insufficient Yahoo data (need SPY baseline + ≥3 country ETFs)",
        )

    n_bars = min(len(h) for h in histories.values())
    base_idx = max(n_bars - window_days, lookback + 1)

    trades: list[dict] = []
    positions: dict[str, dict] = {}
    entry_volume = 0.0

    for i in range(base_idx, n_bars):
        if (i - base_idx) % rebalance_every != 0:
            continue
        bar_time = datetime.fromtimestamp(
            next(iter(histories.values()))[i, 0], tz=UTC,
        ).isoformat()

        # SPY baseline return
        spy = histories[_US_BASELINE]
        if i >= len(spy):
            continue
        spy_start = float(spy[i - lookback, 4])
        spy_end = float(spy[i, 4])
        if spy_start <= 0:
            continue
        spy_ret = (spy_end - spy_start) / spy_start

        # Country ETFs that BEAT SPY
        scored: list[tuple[str, float, float]] = []
        for sym in _INTL_ETFS:
            if sym not in histories:
                continue
            candles = histories[sym]
            if i >= len(candles):
                continue
            start = float(candles[i - lookback, 4])
            end = float(candles[i, 4])
            if start <= 0:
                continue
            ret = (end - start) / start
            if ret > spy_ret:
                scored.append((sym, ret, end))
        scored.sort(key=lambda r: r[1], reverse=True)
        winners = {s for s, _, _ in scored[:top_n]}

        for sym in list(positions.keys()):
            if sym in winners or i >= len(histories[sym]):
                continue
            pos = positions.pop(sym)
            close = float(histories[sym][i, 4])
            gross = pos["qty"] * (close - pos["entry_price"])
            fees = (pos["qty"] * pos["entry_price"]
                     + pos["qty"] * close) * _FEE_RATE
            trades.append({
                "strategy": "internationals_rotation", "side": "SELL",
                "product_id": sym, "amount_usd": pos["qty"] * close,
                "quantity": pos["qty"], "entry_price": pos["entry_price"],
                "exit_price": close, "open_time": pos["entry_time"],
                "close_time": bar_time, "pnl_usd": gross - fees,
                "exit_reason": "no longer beating SPY (top-2)",
            })
        for sym, _, last in scored[:top_n]:
            if sym in positions:
                continue
            qty = per_leg_usd / last
            positions[sym] = {
                "qty": qty, "entry_price": last, "entry_time": bar_time,
            }
            entry_volume += per_leg_usd
            trades.append({
                "strategy": "internationals_rotation", "side": "BUY",
                "product_id": sym, "amount_usd": per_leg_usd,
                "quantity": qty, "entry_price": last,
                "open_time": bar_time, "reason": "beats SPY 90d",
            })

    return _equity_curve_to_summary(
        "internationals_rotation", window_days, trades, entry_volume,
    )


# ─── Helpers ──────────────────────────────────────────────────────────


def _load_universe(symbols: list[str], days: int) -> dict[str, np.ndarray]:
    histories = {sym: _yahoo_history(sym, days) for sym in symbols}
    return {s: h for s, h in histories.items() if len(h) >= 30}


def _rsi(closes: np.ndarray, period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    diffs = np.diff(closes[-(period + 1):])
    gains = np.maximum(diffs, 0)
    losses = np.abs(np.minimum(diffs, 0))
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _sma(closes: np.ndarray, period: int) -> float | None:
    if len(closes) < period:
        return None
    return float(closes[-period:].mean())


def _is_turn_of_month(bar_dt: datetime, candles: np.ndarray,
                        i: int) -> bool:
    """True if `bar_dt` is in the last 3 OR first 2 trading days of
    its month. Looks at neighbouring bars to count "trading days"
    correctly even across weekends + holidays."""
    if i < 2 or i >= len(candles) - 2:
        # Edge of the window — be conservative and skip
        return False
    # Last-3-of-month: today's month differs from bar 3 days hence
    future_3 = datetime.fromtimestamp(candles[min(i + 3, len(candles) - 1), 0],
                                        tz=UTC)
    if future_3.month != bar_dt.month:
        return True
    # First-2-of-month: today's month differs from bar 2 days back
    past_2 = datetime.fromtimestamp(candles[max(i - 2, 0), 0], tz=UTC)
    if past_2.month != bar_dt.month:
        return True
    return False
