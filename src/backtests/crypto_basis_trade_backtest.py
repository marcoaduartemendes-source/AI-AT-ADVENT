"""Crypto basis trade backtest using Bybit perp + spot data.

Strategy logic (mirrors src/strategies/crypto_basis_trade.py):

    Long spot, short dated future. Capture the basis as the future
    converges to spot at expiry. Open when annualized basis APR
    > 8%, close when it decays to < 2%.

The live strategy uses Coinbase dated futures (BTC-29MAY26-CDE etc).
This backtest uses **Bybit perp basis** as a proxy: instead of
spot-vs-quarterly-future, we use spot-vs-perp price discrepancy.
The perp tracks spot via funding (different mechanism), but the
basis between perp and spot mid-quotes is a measurable signal.

Honest about limitations:
  - Perpetuals don't have a fixed expiry; basis is held in check
    by funding payments (the funding-carry strategy already
    captures this). So perp-spot basis is mostly < 50 bps in
    normal markets — far below the 8% threshold of the live
    strategy.
  - This backtest will RARELY trigger trades. Result: a clean
    "strategy was infrequently active during this window" report
    rather than fake P&L numbers.
  - Real Coinbase quarterly-futures basis can hit 5-15% APR
    during contango — that data isn't free. Treat the live
    strategy as separately validated by the documented edge,
    not by this proxy backtest.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .data.bybit import BybitClient
from .runner import BacktestSummary

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Match live thresholds (src/strategies/crypto_basis_trade.py)
ENTRY_BASIS_BPS = 800        # 8% APR — open
EXIT_BASIS_BPS = 200         # 2% APR — close
TRADE_SIZE_USD = 5000.0
ROUND_TRIP_FEE_BPS = 6.0
HOLD_DAYS_MAX = 30           # safety: force close after 30d


CARRY_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD"]


# ─── Trade representation ─────────────────────────────────────────────


@dataclass
class _BasisTrade:
    symbol: str
    entry_date: date
    entry_basis_bps: float
    exit_date: date
    exit_basis_bps: float
    notional_usd: float
    realized_basis_bps: float    # actual basis convergence captured

    @property
    def gross_pnl_usd(self) -> float:
        # Captured basis × notional / 10000
        return self.realized_basis_bps * self.notional_usd / 10_000

    @property
    def net_pnl_usd(self) -> float:
        fee = self.notional_usd * (ROUND_TRIP_FEE_BPS / 10_000)
        return self.gross_pnl_usd - fee


# ─── Core ─────────────────────────────────────────────────────────────


def backtest_crypto_basis_trade(
    window_days: int,
    *,
    universe: list[str] | None = None,
    bybit: BybitClient | None = None,
) -> BacktestSummary:
    client = bybit or BybitClient()
    universe = universe or CARRY_UNIVERSE

    today = date.today()

    trades: list[_BasisTrade] = []
    skipped = {"no_data": 0, "no_basis_signals": 0}

    for symbol in universe:
        # Fetch perp + spot daily bars over the window + buffer
        spot_bars = client.daily_bars(symbol, kind="spot",
                                       days=window_days + 30)
        perp_bars = client.daily_bars(symbol, kind="linear",
                                       days=window_days + 30)
        if not spot_bars or not perp_bars:
            skipped["no_data"] += 1
            continue

        # Index by date for daily basis computation
        spot_idx = {b.timestamp.date(): b for b in spot_bars}
        perp_idx = {b.timestamp.date(): b for b in perp_bars}
        common_dates = sorted(set(spot_idx.keys()) & set(perp_idx.keys()))
        if len(common_dates) < 30:
            skipped["no_data"] += 1
            continue

        # Walk daily; compute basis bps; open on entry threshold,
        # close on exit threshold or after HOLD_DAYS_MAX.
        in_position = False
        entry_date: date | None = None
        entry_basis_bps = 0.0

        window_start = today - timedelta(days=window_days)
        for d in common_dates:
            if d < window_start:
                continue
            spot_close = spot_idx[d].close
            perp_close = perp_idx[d].close
            if spot_close <= 0:
                continue
            basis_bps = (perp_close - spot_close) / spot_close * 10_000

            if not in_position and basis_bps >= ENTRY_BASIS_BPS:
                in_position = True
                entry_date = d
                entry_basis_bps = basis_bps
            elif in_position and entry_date is not None:
                hold_days = (d - entry_date).days
                if (basis_bps <= EXIT_BASIS_BPS
                        or hold_days >= HOLD_DAYS_MAX):
                    realized = entry_basis_bps - basis_bps
                    trades.append(_BasisTrade(
                        symbol=symbol,
                        entry_date=entry_date,
                        entry_basis_bps=entry_basis_bps,
                        exit_date=d,
                        exit_basis_bps=basis_bps,
                        notional_usd=TRADE_SIZE_USD,
                        realized_basis_bps=realized,
                    ))
                    in_position = False
                    entry_date = None

        if not trades:
            skipped["no_basis_signals"] += 1

    if skipped:
        logger.info(f"crypto_basis_trade {window_days}d: {skipped}")

    return _summarize(trades, window_days)


def _summarize(trades: list[_BasisTrade], window_days: int) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            strategy="crypto_basis_trade",
            window_days=window_days,
            note=("No qualifying basis signals (perp-spot basis rarely "
                  "reaches 8% APR threshold; live Coinbase quarterly "
                  "futures have higher contango — that data not in "
                  "free public APIs)"),
        )

    pnls = [t.net_pnl_usd for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    entry_vol = sum(t.notional_usd for t in trades)

    return BacktestSummary(
        strategy="crypto_basis_trade",
        window_days=window_days,
        n_trades=n,
        n_wins=wins,
        n_losses=n - wins,
        win_rate=wins / n if n else 0.0,
        total_pnl_usd=total_pnl,
        entry_volume_usd=entry_vol,
        return_on_volume_pct=(total_pnl / entry_vol * 100) if entry_vol > 0 else 0.0,
        avg_pnl_usd=total_pnl / n if n else 0.0,
        sharpe=None,
        max_drawdown_usd=0.0,
        trades=[
            {
                "open_time": t.entry_date.isoformat(),
                "close_time": t.exit_date.isoformat(),
                "product_id": t.symbol,
                "side": "BUY",
                "entry_price": t.entry_basis_bps,
                "exit_price": t.exit_basis_bps,
                "quantity": 1.0,
                "amount_usd": t.notional_usd,
                "pnl_usd": t.net_pnl_usd,
                "reason": f"Basis {t.entry_basis_bps:.0f} bps → open",
                "exit_reason": f"Basis decayed to {t.exit_basis_bps:.0f} bps",
            }
            for t in trades
        ],
        note="Bybit perp-vs-spot basis proxy. Live Coinbase quarterly "
             "futures basis is structurally larger (annualized contango).",
    )
