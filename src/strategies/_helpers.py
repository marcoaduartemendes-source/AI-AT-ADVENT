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

    The returned scalar already folds in the regime multiplier
    (HIGH / EXTREME vol regimes halve / quarter the equity sleeve)
    so callers don't need to read it separately. Use `equity_regime()`
    if you want to log or branch on the named regime.

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
        scaler = float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
    # Apply equity-regime multiplier on top for equity-class consumers.
    if asset_class == "equity_momentum":
        try:
            regime_mult = float(sig.get("equity_regime_multiplier", 1.0))
        except (TypeError, ValueError):
            regime_mult = 1.0
        scaler *= regime_mult
    return scaler


def equity_regime(ctx) -> str:
    """Return 'NORMAL' | 'HIGH' | 'EXTREME' from vol_managed_overlay.
    Defaults to 'NORMAL' when the signal is missing or stale.
    """
    try:
        sig = (ctx.scout_signals or {}).get("vol_scaler") if ctx else None
    except AttributeError:
        return "NORMAL"
    if not isinstance(sig, dict):
        return "NORMAL"
    return sig.get("equity_regime", "NORMAL")


def crypto_vol_scaler(ctx, default: float = 1.0) -> float:
    """Return the crypto vol-regime scaler from
    `crypto_vol_regime_overlay`. Folds in the regime multiplier
    (LOW=1.0, MEDIUM=0.7, HIGH=0.3) so callers can multiply
    target_alloc_usd unconditionally.

    Usage in a crypto momentum / breakout strategy:
        size_usd = ctx.target_alloc_usd * crypto_vol_scaler(ctx)
    """
    try:
        sig = (ctx.scout_signals or {}).get("crypto_vol_scaler") if ctx else None
    except AttributeError:
        return default
    if not isinstance(sig, dict):
        return default
    val = sig.get("crypto_momentum")
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def crypto_regime(ctx) -> str:
    """Return 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN' from the crypto
    vol-regime overlay."""
    try:
        sig = (ctx.scout_signals or {}).get("crypto_vol_scaler") if ctx else None
    except AttributeError:
        return "UNKNOWN"
    if not isinstance(sig, dict):
        return "UNKNOWN"
    return sig.get("crypto_regime", "UNKNOWN")


def net_qty_from_ledger(strategy: str, venue: str | None = None
                          ) -> dict[str, float]:
    """Reconstruct net position quantities for a strategy from the
    trades ledger. Returns {symbol: net_qty} where positive = long,
    negative = short.

    Why: the spot-only `CoinbaseAdapter.get_positions()` doesn't
    surface perp/future legs, so `ctx.open_positions` misses them
    even though the orchestrator placed (and recorded) the SELL legs
    of carry/basis trades. KILL-switch overrides need this fallback
    to close shorts they can't see in the broker snapshot.

    Audit-fix F1 (2026-05-07).
    """
    import os
    import sqlite3
    from pathlib import Path
    db_path = os.environ.get(
        "TRADING_DB_PATH", "data/trading_performance.db"
    )
    if not Path(db_path).exists():
        return {}
    out: dict[str, float] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            where = ["strategy = ?", "fill_status = 'FILLED'"]
            params: list = [strategy]
            if venue:
                where.append("(venue = ? OR venue IS NULL)")
                params.append(venue)
            sql = (
                "SELECT product_id, side, COALESCE(quantity, 0) AS qty "
                "  FROM trades "
                f" WHERE {' AND '.join(where)}"
            )
            for row in conn.execute(sql, params).fetchall():
                sym = row["product_id"]
                qty = float(row["qty"] or 0)
                if not sym or qty == 0:
                    continue
                delta = qty if row["side"] == "BUY" else -qty
                out[sym] = out.get(sym, 0.0) + delta
    except sqlite3.Error:
        return {}
    # Drop zero-net symbols (closed positions)
    return {s: q for s, q in out.items() if abs(q) > 1e-9}


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
