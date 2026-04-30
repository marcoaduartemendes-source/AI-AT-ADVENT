"""Backtest engine.

Replays a strategy over historical OHLCV bars using the same risk rules
as the live bot (stop-loss, take-profit, cooldown, max-trade size).

Historical candles come from Binance's public klines endpoint — no auth,
no rate-limit headaches. Prices differ from Coinbase by <0.05% on majors.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import requests

from trading.market_data import GRANULARITY_SECONDS, _BINANCE_INTERVALS
from trading.strategies.base import BaseStrategy, SignalType


# Coinbase Advanced Trade taker fee for retail accounts (worst-case 60bps).
DEFAULT_FEE_BPS = 60.0
# Slippage: difference between expected and filled price (5bps each side).
DEFAULT_SLIPPAGE_BPS = 5.0


@dataclass
class BacktestTrade:
    strategy: str
    product_id: str
    open_time: datetime
    close_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    amount_usd: float
    pnl_usd: Optional[float]
    return_pct: Optional[float]
    exit_reason: Optional[str]  # 'signal' | 'stop_loss' | 'take_profit' | 'open'


@dataclass
class BacktestResult:
    strategy: str
    product_id: str
    window_days: int
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)  # [{t, pnl_cumulative}]

    def summary(self) -> Dict:
        closed = [t for t in self.trades if t.pnl_usd is not None]
        wins = sum(1 for t in closed if t.pnl_usd > 0)
        losses = len(closed) - wins
        total_pnl = sum(t.pnl_usd for t in closed) if closed else 0.0
        total_volume = sum(t.amount_usd for t in self.trades) * 2  # entry + exit notional
        # Use entry-only volume for "trade volume" denominator (industry convention)
        entry_volume = sum(t.amount_usd for t in self.trades)
        return {
            "strategy": self.strategy,
            "product_id": self.product_id,
            "window_days": self.window_days,
            "n_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / len(closed)) if closed else 0.0,
            "total_pnl_usd": total_pnl,
            "entry_volume_usd": entry_volume,
            "round_trip_volume_usd": total_volume,
            "return_on_volume_pct": (total_pnl / entry_volume * 100) if entry_volume > 0 else 0.0,
            "avg_pnl_usd": (total_pnl / len(closed)) if closed else 0.0,
        }


# ─── Historical data ──────────────────────────────────────────────────────────


def fetch_binance_history(
    product_id: str, granularity: str, days: int
) -> np.ndarray:
    """Paginated klines fetch. Returns OHLCV array sorted ascending by time."""
    base, _, _ = product_id.partition("-")
    symbol = f"{base}USDT"
    interval = _BINANCE_INTERVALS.get(granularity, "5m")
    interval_sec = GRANULARITY_SECONDS[granularity]

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    out: List[List[float]] = []

    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=20,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        for k in batch:
            # [open_time_ms, open, high, low, close, volume, ...]
            out.append([
                float(k[0]) / 1000.0,
                float(k[3]),  # low
                float(k[2]),  # high
                float(k[1]),  # open
                float(k[4]),  # close
                float(k[5]),  # volume
            ])
        last_open_ms = int(batch[-1][0])
        next_cursor = last_open_ms + interval_sec * 1000
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < 1000:
            break

    arr = np.array(out, dtype=float) if out else np.empty((0, 6))
    if arr.size:
        arr = arr[arr[:, 0].argsort()]
    return arr


# ─── Backtest core ────────────────────────────────────────────────────────────


def backtest_strategy(
    strategy: BaseStrategy,
    product_id: str,
    candles: np.ndarray,
    *,
    window_days: int,
    max_trade_usd: float = 20.0,
    stop_loss_pct: float = 0.02,
    take_profit_pct: float = 0.04,
    cooldown_seconds: int = 900,
    min_confidence: float = 0.6,
    fee_bps: float = DEFAULT_FEE_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    lookback: Optional[int] = None,
) -> BacktestResult:
    result = BacktestResult(
        strategy=strategy.name, product_id=product_id, window_days=window_days
    )
    if len(candles) < (lookback or strategy.lookback) + 5:
        return result

    look = lookback or strategy.lookback
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    open_pos: Optional[BacktestTrade] = None
    last_trade_ts: float = 0.0
    cum_pnl = 0.0

    for i in range(look, len(candles)):
        bar = candles[i]
        ts, low, high, open_, close, vol = bar
        bar_time = datetime.fromtimestamp(ts, tz=timezone.utc)
        window = candles[i - look + 1: i + 1]

        # ─ Risk checks against open position FIRST (use intra-bar high/low) ─
        if open_pos is not None:
            stop_price = open_pos.entry_price * (1 - stop_loss_pct)
            take_price = open_pos.entry_price * (1 + take_profit_pct)
            exit_price: Optional[float] = None
            exit_reason: Optional[str] = None

            # Conservative: if both stop and take fired within same bar, assume stop fired first
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop_loss"
            elif high >= take_price:
                exit_price = take_price
                exit_reason = "take_profit"

            if exit_price is not None:
                # Apply slippage on exit (sell fills slightly worse)
                fill = exit_price * (1 - slip_rate)
                gross = open_pos.quantity * (fill - open_pos.entry_price)
                fees = (open_pos.amount_usd + open_pos.quantity * fill) * fee_rate
                pnl = gross - fees
                open_pos.exit_price = fill
                open_pos.close_time = bar_time
                open_pos.pnl_usd = pnl
                open_pos.return_pct = pnl / open_pos.amount_usd * 100
                open_pos.exit_reason = exit_reason
                cum_pnl += pnl
                result.equity_curve.append(
                    {"t": bar_time.isoformat(), "pnl_cumulative": cum_pnl}
                )
                last_trade_ts = ts
                open_pos = None
                # Don't take a new signal in the same bar — be conservative
                continue

        # ─ Run strategy on window
        try:
            signal = strategy.analyze(product_id, window)
        except Exception:
            continue

        # ─ Signal-driven exit (close on opposing signal)
        if (
            open_pos is not None
            and signal.signal == SignalType.SELL
            and signal.confidence >= min_confidence
        ):
            fill = close * (1 - slip_rate)
            gross = open_pos.quantity * (fill - open_pos.entry_price)
            fees = (open_pos.amount_usd + open_pos.quantity * fill) * fee_rate
            pnl = gross - fees
            open_pos.exit_price = fill
            open_pos.close_time = bar_time
            open_pos.pnl_usd = pnl
            open_pos.return_pct = pnl / open_pos.amount_usd * 100
            open_pos.exit_reason = "signal"
            cum_pnl += pnl
            result.equity_curve.append(
                {"t": bar_time.isoformat(), "pnl_cumulative": cum_pnl}
            )
            last_trade_ts = ts
            open_pos = None
            continue

        # ─ Signal-driven entry
        if (
            open_pos is None
            and signal.signal == SignalType.BUY
            and signal.confidence >= min_confidence
            and (ts - last_trade_ts) >= cooldown_seconds
        ):
            entry_fill = close * (1 + slip_rate)  # buy slips slightly higher
            qty = max_trade_usd / entry_fill
            open_pos = BacktestTrade(
                strategy=strategy.name,
                product_id=product_id,
                open_time=bar_time,
                close_time=None,
                entry_price=entry_fill,
                exit_price=None,
                quantity=qty,
                amount_usd=max_trade_usd,
                pnl_usd=None,
                return_pct=None,
                exit_reason=None,
            )
            result.trades.append(open_pos)
            last_trade_ts = ts

    return result


def trade_to_dict(t: BacktestTrade) -> Dict:
    d = asdict(t)
    d["open_time"] = t.open_time.isoformat() if t.open_time else None
    d["close_time"] = t.close_time.isoformat() if t.close_time else None
    return d
