"""Crypto funding-carry backtest using Bybit historical funding rates.

Strategy logic mirrors src/strategies/crypto_funding_carry.py:

    Long spot, short perp; capture the funding rate paid every 8h
    when funding is positive (perp trades premium to spot, longs
    pay shorts). Open when annualized funding-rate APR > entry
    threshold; close when APR drops below exit threshold.

The live strategy uses Coinbase International Exchange perp data,
which requires authenticated access for historical funding history.
This backtest uses Bybit (free public API) — funding rates are highly
correlated across major venues (Coinbase / Bybit / Binance / OKX
arbitrage them within ~1bp), so Bybit's series is a faithful
directional proxy.

Honest about limitations:
  - Live strategy receives the actual Coinbase funding the bot is
    paid; backtest assumes Bybit's published rate (≈ same).
  - Spot leg is approximated via Bybit USDT-quoted spot prices;
    Coinbase USD prices differ by basis points (USDT depeg risk).
  - Roll-cost between perp and spot legs is modelled as 6 bps
    round-trip, which is generous given Coinbase's 0.6% retail
    fee tier — adjust UP for live-trading projections.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC

from .data.bybit import BybitClient, FundingPoint
from .runner import BacktestSummary

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Match the live strategy's thresholds (src/strategies/crypto_funding_carry.py)
ENTRY_FUNDING_APR_PCT = 5.0     # 5% APR — open the carry
EXIT_FUNDING_APR_PCT = 1.0      # 1% APR — close, funding decayed
ANNUALIZATION_FACTOR = 365 * 3  # 8h funding × 3/day × 365 days
TRADE_SIZE_USD = 5000.0         # per-trade notional (one leg)
ROUND_TRIP_FEE_BPS = 6.0        # 6 bps generous fee assumption


CARRY_UNIVERSE = ["BTC-USD", "ETH-USD", "SOL-USD"]


# ─── Trade representation ─────────────────────────────────────────────


@dataclass
class _CarryTrade:
    symbol: str
    entry_date: date
    entry_funding_apr: float     # APR % at entry
    exit_date: date
    exit_funding_apr: float
    notional_usd: float
    funding_collected_usd: float    # accumulated funding over hold period

    @property
    def hold_days(self) -> int:
        return (self.exit_date - self.entry_date).days

    @property
    def gross_pnl_usd(self) -> float:
        # The funding collected IS the P&L (long spot / short perp
        # is delta-neutral; only the funding rate flows). Spot/perp
        # price drift cancels in a true delta-neutral leg.
        return self.funding_collected_usd

    @property
    def net_pnl_usd(self) -> float:
        # 6 bps round-trip on the notional (entering + exiting both
        # legs). Live numbers will be 2-3× this on Coinbase retail.
        fee = self.notional_usd * (ROUND_TRIP_FEE_BPS / 10_000)
        return self.gross_pnl_usd - fee


# ─── Core backtest ────────────────────────────────────────────────────


def backtest_crypto_funding_carry(
    window_days: int,
    *,
    universe: list[str] | None = None,
    bybit: BybitClient | None = None,
) -> BacktestSummary:
    """Run the funding-carry backtest over the last `window_days`."""
    client = bybit or BybitClient()
    universe = universe or CARRY_UNIVERSE

    today = date.today()
    window_start = today - timedelta(days=window_days)

    trades: list[_CarryTrade] = []
    skipped = {"no_data": 0, "still_holding": 0}

    for symbol in universe:
        # Pull funding history covering the backtest window plus
        # ~10 days of buffer for entry-condition lookback.
        from_dt = datetime.combine(
            window_start - timedelta(days=10),
            datetime.min.time(),
            tzinfo=UTC,
        )
        history = client.funding_history(
            symbol,
            limit=200,
            from_ms=int(from_dt.timestamp() * 1000),
        )
        if not history:
            skipped["no_data"] += 1
            continue

        # Walk the funding-rate timeline, opening a position whenever
        # APR > entry threshold and not currently in a position;
        # closing when APR drops below exit threshold.
        in_position = False
        entry_pt: FundingPoint | None = None
        funding_collected = 0.0

        for pt in history:
            apr_pct = pt.funding_rate * ANNUALIZATION_FACTOR * 100
            d = pt.timestamp.date()
            if d < window_start or d > today:
                continue

            if not in_position and apr_pct >= ENTRY_FUNDING_APR_PCT:
                # Open the carry
                in_position = True
                entry_pt = pt
                funding_collected = 0.0
            elif in_position:
                # Each 8h tick, accumulate funding × notional
                funding_collected += pt.funding_rate * TRADE_SIZE_USD
                if apr_pct < EXIT_FUNDING_APR_PCT:
                    # Close the carry
                    if entry_pt is None:
                        in_position = False
                        continue
                    trades.append(_CarryTrade(
                        symbol=symbol,
                        entry_date=entry_pt.timestamp.date(),
                        entry_funding_apr=entry_pt.funding_rate * ANNUALIZATION_FACTOR * 100,
                        exit_date=d,
                        exit_funding_apr=apr_pct,
                        notional_usd=TRADE_SIZE_USD,
                        funding_collected_usd=funding_collected,
                    ))
                    in_position = False
                    entry_pt = None
                    funding_collected = 0.0

        # If still in position at end of window, close at the last tick
        if in_position and entry_pt is not None:
            last = history[-1]
            trades.append(_CarryTrade(
                symbol=symbol,
                entry_date=entry_pt.timestamp.date(),
                entry_funding_apr=entry_pt.funding_rate * ANNUALIZATION_FACTOR * 100,
                exit_date=last.timestamp.date(),
                exit_funding_apr=last.funding_rate * ANNUALIZATION_FACTOR * 100,
                notional_usd=TRADE_SIZE_USD,
                funding_collected_usd=funding_collected,
            ))
            skipped["still_holding"] += 1

    if skipped:
        logger.info(f"crypto_funding_carry {window_days}d: {skipped}")

    return _summarize(trades, window_days)


def _summarize(trades: list[_CarryTrade], window_days: int) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            strategy="crypto_funding_carry",
            window_days=window_days,
            note="no qualifying funding spikes in window "
                 "(or BYBIT API returned empty)",
        )

    pnls = [t.net_pnl_usd for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    entry_vol = sum(t.notional_usd for t in trades)

    sharpe = _sharpe(pnls)
    max_dd = _max_drawdown(pnls)

    trade_dicts = [
        {
            "open_time": t.entry_date.isoformat(),
            "close_time": t.exit_date.isoformat(),
            "product_id": t.symbol,
            "side": "BUY",   # spot leg long
            "entry_price": t.entry_funding_apr,    # APR %, not price
            "exit_price": t.exit_funding_apr,
            "quantity": 1.0,
            "amount_usd": t.notional_usd,
            "pnl_usd": t.net_pnl_usd,
            "reason": (f"Funding {t.entry_funding_apr:.1f}% APR — "
                       f"long spot/short perp"),
            "exit_reason": (f"Funding decayed to {t.exit_funding_apr:.1f}% APR "
                            f"after {t.hold_days}d"),
        }
        for t in trades
    ]

    eq_curve: list[dict] = []
    cum = 0.0
    for t, pnl in zip(sorted(trades, key=lambda x: x.exit_date), pnls,
                       strict=False):
        cum += pnl
        eq_curve.append({"t": t.exit_date.isoformat(), "pnl_cumulative": cum})

    return BacktestSummary(
        strategy="crypto_funding_carry",
        window_days=window_days,
        n_trades=n,
        n_wins=wins,
        n_losses=n - wins,
        win_rate=wins / n if n else 0.0,
        total_pnl_usd=total_pnl,
        entry_volume_usd=entry_vol,
        return_on_volume_pct=(total_pnl / entry_vol * 100) if entry_vol > 0 else 0.0,
        avg_pnl_usd=total_pnl / n if n else 0.0,
        sharpe=sharpe,
        max_drawdown_usd=max_dd,
        trades=trade_dicts,
        equity_curve=eq_curve,
        note=("Bybit-funding proxy backtest. Coinbase live numbers will "
              "differ by venue spread + retail fee drag (~3× the modelled "
              "6 bps round-trip). Directional signal preserved."),
    )


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    sd = var ** 0.5
    return None if sd == 0 else mean / sd * (len(pnls) ** 0.5)


def _max_drawdown(pnls: list[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        if peak - cum > max_dd:
            max_dd = peak - cum
    return max_dd
