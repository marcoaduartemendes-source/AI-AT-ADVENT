"""Backtest engine.

Replays a strategy over historical OHLCV bars using the same risk rules
as the live bot (stop-loss, take-profit, cooldown, max-trade size).

Historical candles come from Coinbase's public market endpoint
(no auth, geo-friendly). Returns up to 350 candles per request.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC

import numpy as np
import requests

from trading.market_data import GRANULARITY_SECONDS
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
    close_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    amount_usd: float
    pnl_usd: float | None
    return_pct: float | None
    exit_reason: str | None  # 'signal' | 'stop_loss' | 'take_profit' | 'open'


@dataclass
class BacktestResult:
    strategy: str
    product_id: str
    window_days: int
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)  # [{t, pnl_cumulative}]

    def summary(self) -> dict:
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


def fetch_coinbase_public_history(
    product_id: str, granularity: str, days: int
) -> np.ndarray:
    """Paginated public Coinbase candles fetch. Returns OHLCV (N,6).

    Coinbase Advanced Trade public market endpoint accepts up to ~350
    candles per request. We page backwards from now, taking 300/window
    to stay safely under the cap.
    """
    interval_sec = GRANULARITY_SECONDS[granularity]
    per_page = 300
    page_seconds = per_page * interval_sec

    end = int(time.time())
    earliest = end - days * 86400
    out: list[list[float]] = []

    cursor_end = end
    while cursor_end > earliest:
        cursor_start = max(earliest, cursor_end - page_seconds)
        resp = requests.get(
            f"https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles",
            params={
                "start": cursor_start,
                "end": cursor_end,
                "granularity": granularity,
            },
            timeout=20,
        )
        if resp.status_code != 200:
            # Surface details for the first failure but don't keep retrying forever
            raise RuntimeError(
                f"Coinbase public candles {product_id} {resp.status_code}: {resp.text[:160]}"
            )
        batch = resp.json().get("candles", [])
        if not batch:
            break
        for c in batch:
            out.append([
                float(c["start"]),
                float(c["low"]),
                float(c["high"]),
                float(c["open"]),
                float(c["close"]),
                float(c["volume"]),
            ])
        # Coinbase returns most recent first; advance cursor to the earliest in batch
        oldest_ts = min(float(c["start"]) for c in batch)
        next_end = int(oldest_ts)
        if next_end >= cursor_end:
            break
        cursor_end = next_end
        # Politeness: stay well under the 10 req/sec public limit
        time.sleep(0.1)

    arr = np.array(out, dtype=float) if out else np.empty((0, 6))
    if arr.size:
        # De-dupe in case of overlap, then sort
        _, unique_idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[np.sort(unique_idx)]
        arr = arr[arr[:, 0].argsort()]
    return arr


# Backward-compat alias kept for the dashboard import name
fetch_binance_history = fetch_coinbase_public_history


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
    lookback: int | None = None,
) -> BacktestResult:
    result = BacktestResult(
        strategy=strategy.name, product_id=product_id, window_days=window_days
    )
    if len(candles) < (lookback or strategy.lookback) + 5:
        return result

    look = lookback or strategy.lookback
    fee_rate = fee_bps / 10000.0
    slip_rate = slippage_bps / 10000.0
    open_pos: BacktestTrade | None = None
    last_trade_ts: float = 0.0
    cum_pnl = 0.0

    for i in range(look, len(candles)):
        bar = candles[i]
        ts, low, high, open_, close, vol = bar
        bar_time = datetime.fromtimestamp(ts, tz=UTC)
        window = candles[i - look + 1: i + 1]

        # ─ Risk checks against open position FIRST (use intra-bar high/low) ─
        if open_pos is not None:
            stop_price = open_pos.entry_price * (1 - stop_loss_pct)
            take_price = open_pos.entry_price * (1 + take_profit_pct)
            exit_price: float | None = None
            exit_reason: str | None = None

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


def trade_to_dict(t: BacktestTrade) -> dict:
    d = asdict(t)
    d["open_time"] = t.open_time.isoformat() if t.open_time else None
    d["close_time"] = t.close_time.isoformat() if t.close_time else None
    return d
