"""Kalshi calibration-arb backtest using settled-market history.

Strategy logic (mirrors src/strategies/kalshi_calibration_arb.py):

    Kalshi market prices are not unbiased probability estimates.
    Longshots (5–15% YES) tend to be systematically overpriced; heavy
    favorites (85–95%) systematically underpriced. The strategy applies
    a per-bucket recalibration shift, computes edge = fair_value −
    market_price, and bets when |edge| ≥ entry threshold.

For the backtest we walk the universe of *resolved* Kalshi markets in
the lookback window. For each one:

    1. Take the YES close-of-day price (`yes_close`) — i.e. the last
       quoted market price before settlement.
    2. Apply the live strategy's recalibration table to get a fair
       value, hence an edge in cents.
    3. If |edge| ≥ ENTRY_EDGE_CENTS, simulate the trade Kelly-sized
       per the live strategy.
    4. Settlement value (0 or 1) determines payoff. P&L is the
       payoff minus the cost (or vice versa for SELL/NO bets).

Edge cases honestly handled:
  - settled_value == 0.5 (rare void/refund) → skip
  - missing yes_close → skip
  - fees on Kalshi: ~5% on profits per the live broker; modelled at
    same rate so net P&L matches what we'd actually pocket.

Auth: requires KalshiAdapter to be configured (live broker keys).
On a CI box without keys, returns an empty BacktestSummary with a
note rather than crashing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .data.kalshi_history import KalshiHistoryClient
from .runner import BacktestSummary

logger = logging.getLogger(__name__)


# ─── Tunables (mirror live strategy) ──────────────────────────────────


# Same recalibration table as src/strategies/kalshi_calibration_arb.py
DEFAULT_RECALIBRATION = [
    (0.02, 0.10, -0.020),
    (0.10, 0.20, -0.015),
    (0.20, 0.40, -0.005),
    (0.40, 0.60,  0.000),
    (0.60, 0.80, +0.010),
    (0.80, 0.90, +0.015),
    (0.90, 0.98, +0.020),
]

ENTRY_EDGE_CENTS = 3.0
KELLY_FRACTION = 0.25
MAX_PER_TRADE_USD = 50.0       # per-market position cap
STRATEGY_ALLOC_USD = 500.0     # nominal book size for sizing
KALSHI_FEE_RATE = 0.05         # 5% on winnings (Kalshi standard)


# ─── Trade representation ─────────────────────────────────────────────


@dataclass
class _ArbTrade:
    ticker: str
    settle_date: date
    side: str                  # "BUY_YES" or "BUY_NO"
    market_price: float        # YES price (or 1-yes for NO bets)
    fair_value: float
    edge_cents: float
    n_contracts: int
    cost_per_contract: float   # what we paid per contract (0..1)
    settled_yes: bool          # True if YES won

    @property
    def total_cost(self) -> float:
        return self.n_contracts * self.cost_per_contract

    @property
    def gross_payoff(self) -> float:
        # Each winning contract pays $1; losers pay $0.
        won = (self.side == "BUY_YES" and self.settled_yes) or (
              self.side == "BUY_NO" and not self.settled_yes)
        return self.n_contracts * 1.0 if won else 0.0

    @property
    def gross_pnl(self) -> float:
        return self.gross_payoff - self.total_cost

    @property
    def net_pnl(self) -> float:
        # Kalshi takes 5% off gross winnings
        if self.gross_pnl > 0:
            return self.gross_pnl * (1.0 - KALSHI_FEE_RATE)
        return self.gross_pnl


# ─── Core backtest ────────────────────────────────────────────────────


def backtest_kalshi_calibration_arb(
    window_days: int,
    *,
    client: KalshiHistoryClient | None = None,
    recalibration: list[tuple[float, float, float]] | None = None,
) -> BacktestSummary:
    """Walk every settled Kalshi market in the window and simulate the
    live calibration-arb strategy on each."""
    client = client or KalshiHistoryClient()
    table = recalibration or DEFAULT_RECALIBRATION

    if not client.is_configured():
        return BacktestSummary(
            strategy="kalshi_calibration_arb",
            window_days=window_days,
            note=("Kalshi adapter not configured (KALSHI_API_KEY_ID / "
                  "KALSHI_PRIVATE_KEY_PATH missing) — backtest skipped"),
        )

    today = date.today()
    from_date = today - timedelta(days=window_days)
    settled = client.settled_markets(from_date=from_date, to_date=today,
                                       limit=1000)
    if not settled:
        return BacktestSummary(
            strategy="kalshi_calibration_arb",
            window_days=window_days,
            note=("No settled Kalshi markets returned in window "
                  "(check API status or widen window)"),
        )

    trades: list[_ArbTrade] = []
    skipped = {"no_yes_close": 0, "no_settlement": 0,
                "below_threshold": 0, "tied_or_void": 0}

    for m in settled:
        if m.yes_close_price <= 0 or m.yes_close_price >= 1.0:
            skipped["no_yes_close"] += 1
            continue
        if m.settlement_value not in (0.0, 1.0):
            skipped["tied_or_void"] += 1
            continue

        market_price = m.yes_close_price
        fair_value = _fair_value(market_price, table)
        edge = fair_value - market_price
        edge_cents = edge * 100

        if abs(edge_cents) < ENTRY_EDGE_CENTS:
            skipped["below_threshold"] += 1
            continue

        # Replicate live Kelly sizing
        if edge > 0:
            # BUY YES at market_price, payoff $1 if YES
            p = fair_value
            cost = market_price
            side = "BUY_YES"
        else:
            # BUY NO at (1 - market_price), payoff $1 if NO
            p = 1 - fair_value
            cost = 1 - market_price
            side = "BUY_NO"

        if cost <= 0:
            continue
        b = (1 - cost) / cost
        full_kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
        kelly = full_kelly * KELLY_FRACTION
        position_usd = min(STRATEGY_ALLOC_USD * kelly, MAX_PER_TRADE_USD)
        if position_usd < 1.0:
            continue
        n_contracts = int(position_usd / cost)
        if n_contracts < 1:
            continue

        trades.append(_ArbTrade(
            ticker=m.ticker,
            settle_date=m.close_ts.date(),
            side=side,
            market_price=market_price,
            fair_value=fair_value,
            edge_cents=edge_cents,
            n_contracts=n_contracts,
            cost_per_contract=cost,
            settled_yes=(m.settlement_value == 1.0),
        ))

    if skipped:
        logger.info(f"kalshi_calibration_arb {window_days}d filter: {skipped}")

    return _summarize(trades, window_days)


def _fair_value(market_price: float,
                 table: list[tuple[float, float, float]]) -> float:
    for lo, hi, shift in table:
        if lo <= market_price < hi:
            return max(0.0, min(1.0, market_price + shift))
    return market_price


def _summarize(trades: list[_ArbTrade], window_days: int) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            strategy="kalshi_calibration_arb",
            window_days=window_days,
            note=("No Kalshi markets crossed the 3¢ edge threshold "
                  "in this window (markets well-calibrated, or yes_close "
                  "data missing)"),
        )

    pnls = [t.net_pnl for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    entry_vol = sum(t.total_cost for t in trades)

    sharpe = _sharpe(pnls)
    max_dd = _max_drawdown(sorted(trades, key=lambda x: x.settle_date))

    eq_curve: list[dict] = []
    cum = 0.0
    for t in sorted(trades, key=lambda x: x.settle_date):
        cum += t.net_pnl
        eq_curve.append({"t": t.settle_date.isoformat(), "pnl_cumulative": cum})

    trade_dicts = [
        {
            "open_time": t.settle_date.isoformat(),  # we know exit only
            "close_time": t.settle_date.isoformat(),
            "product_id": t.ticker,
            "side": t.side,
            "entry_price": t.market_price,
            "exit_price": 1.0 if t.settled_yes else 0.0,
            "quantity": float(t.n_contracts),
            "amount_usd": t.total_cost,
            "pnl_usd": t.net_pnl,
            "reason": (f"market={t.market_price:.3f}, fair={t.fair_value:.3f}, "
                       f"edge={t.edge_cents:+.1f}c"),
            "exit_reason": ("settled YES" if t.settled_yes else "settled NO"),
        }
        for t in trades
    ]

    return BacktestSummary(
        strategy="kalshi_calibration_arb",
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
        note=("Kalshi resolved-markets backtest. Used last-quoted "
              "yes_close as entry price (proxy for live entry). "
              "5% Kalshi profit fee applied."),
    )


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    sd = var ** 0.5
    return None if sd == 0 else round(mean / sd * (len(pnls) ** 0.5), 3)


def _max_drawdown(trades: list[_ArbTrade]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cum += t.net_pnl
        if cum > peak:
            peak = cum
        if peak - cum > max_dd:
            max_dd = peak - cum
    return max_dd
