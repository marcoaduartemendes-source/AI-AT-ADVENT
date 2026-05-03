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
