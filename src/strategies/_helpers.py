"""Shared helpers for strategies.

Lifts `_lookback_return_pct` and `_past_cooldown` out of the half-dozen
near-identical copies that lived in sector_rotation, internationals_rotation,
dividend_growth, low_vol_anomaly, gap_trading, etc. The 12 copies were the
proximate cause of the "1Day" Alpaca-granularity bug going unfixed for so
long — a single fix would have caught all of them.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


def lookback_return_pct(broker, name: str, symbol: str, days: int,
                          *, granularity: str = "1Day",
                          extra_candles: int = 5) -> float | None:
    """Percent change between candle[-days] and candle[-1].

    Returns None if the broker call fails or fewer than `days` candles
    are available. `name` is the caller's strategy name for log scoping.
    """
    try:
        candles = broker.get_candles(symbol, granularity,
                                       num_candles=days + extra_candles)
    except Exception as e:
        logger.debug(f"[{name}] {symbol} candles failed: {e}")
        return None
    if len(candles) < days:
        return None
    start = candles[-days].close
    end = candles[-1].close
    if start <= 0:
        return None
    return (end - start) / start * 100


def vol_scaler(ctx, asset_class: str = "equity_momentum",
                 default: float = 1.0) -> float:
    """Return the vol-managed-overlay scaler for `asset_class`.

    Moreira-Muir (2017): scaling momentum exposure inversely to realized
    vol adds ~0.2-0.4 Sharpe at near-zero cost. The vol_managed_overlay
    strategy publishes scalers to the signal bus every cycle; this
    helper reads them with a graceful default of 1.0 (no scaling) when
    the signal is missing or stale, so consumers can safely multiply
    their target_alloc_usd by the return value unconditionally.

    Usage in a momentum strategy:
        size_usd = ctx.target_alloc_usd * vol_scaler(ctx)
    """
    try:
        sig = (ctx.scout_signals or {}).get("vol_scaler") if ctx else None
    except AttributeError:
        return default
    if not isinstance(sig, dict):
        return default
    val = sig.get(asset_class)
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def past_cooldown(pos: dict, cooldown_days: int) -> bool:
    """True if the position's entry_time is older than `cooldown_days`.
    Missing or unparseable entry_time → treated as past cooldown so a
    legitimate rebalance isn't blocked by stale metadata.
    """
    et = pos.get("entry_time")
    if not et:
        return True
    try:
        dt = (datetime.fromisoformat(et) if isinstance(et, str) else et)
        return datetime.now(UTC) - dt > timedelta(days=cooldown_days)
    except (ValueError, TypeError):
        return True
