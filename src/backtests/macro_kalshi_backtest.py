"""Macro Kalshi backtest — FOMC / CPI / NFP mispricing.

Strategy logic (mirrors src/strategies/macro_kalshi.py):

    Filter Kalshi resolved markets to macro-keyword tickers (Fed,
    CPI, NFP, unemployment, GDP, etc.). Apply the same recalibration
    table as kalshi_calibration_arb but with a tighter entry edge
    (2.5¢) and bigger size (Kelly × 0.30, 20% per-trade cap),
    reflecting that macro markets are more liquid and the divergence
    signal is statistically cleaner.

The backtest:

    1. Pulls all settled Kalshi markets in the lookback window.
    2. Filters by MACRO_KEYWORDS in ticker/title.
    3. For each, simulates the entry/exit per the strategy. Profit
       attribution is settlement-based (binary 0 or 1 payoff).

The FRED hook is **optional context**: we don't change the bet
based on FRED actuals (the live strategy doesn't either — it
trades on Kalshi-vs-CME divergence, which only Kalshi shows post-
resolution). FRED is logged in the trade `metadata` so a reviewer
can sanity-check that the macro events resolved consistently with
the published macro series. It also lets us flag any market where
the FRED truth strongly disagreed with the Kalshi settlement
(rare data-quality issue).

If FRED_API_KEY is unset, the backtest still runs — it just omits
the FRED context fields. If the Kalshi adapter is unconfigured,
the backtest returns an empty summary with a note (mirrors
kalshi_calibration_arb behaviour).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .data.fred import FREDClient
from .data.kalshi_history import KalshiHistoryClient, ResolvedMarket
from .runner import BacktestSummary

logger = logging.getLogger(__name__)


# ─── Tunables (mirror live strategy) ──────────────────────────────────


# Reuse the calibration arb's recalibration table — same bias source.
DEFAULT_RECALIBRATION = [
    (0.02, 0.10, -0.020),
    (0.10, 0.20, -0.015),
    (0.20, 0.40, -0.005),
    (0.40, 0.60,  0.000),
    (0.60, 0.80, +0.010),
    (0.80, 0.90, +0.015),
    (0.90, 0.98, +0.020),
]

# Tighter than calibration arb; macro markets are more liquid
ENTRY_EDGE_CENTS = 2.5
KELLY_FRACTION = 0.30
MAX_PER_TRADE_USD = 100.0
STRATEGY_ALLOC_USD = 500.0
KALSHI_FEE_RATE = 0.05

MACRO_KEYWORDS = ("fed", "fomc", "rate", "cpi", "inflation", "nfp",
                   "jobs", "unemployment", "gdp", "pce", "pmi", "ism")


# ─── Trade representation ─────────────────────────────────────────────


@dataclass
class _MacroTrade:
    ticker: str
    title: str
    settle_date: date
    side: str
    market_price: float
    fair_value: float
    edge_cents: float
    n_contracts: int
    cost_per_contract: float
    settled_yes: bool
    fred_context: dict      # snapshot of relevant FRED actuals (optional)

    @property
    def total_cost(self) -> float:
        return self.n_contracts * self.cost_per_contract

    @property
    def gross_pnl(self) -> float:
        won = (self.side == "BUY_YES" and self.settled_yes) or (
              self.side == "BUY_NO" and not self.settled_yes)
        payoff = self.n_contracts * 1.0 if won else 0.0
        return payoff - self.total_cost

    @property
    def net_pnl(self) -> float:
        if self.gross_pnl > 0:
            return self.gross_pnl * (1.0 - KALSHI_FEE_RATE)
        return self.gross_pnl


# ─── Core backtest ────────────────────────────────────────────────────


def backtest_macro_kalshi(
    window_days: int,
    *,
    kalshi: KalshiHistoryClient | None = None,
    fred: FREDClient | None = None,
) -> BacktestSummary:
    kalshi = kalshi or KalshiHistoryClient()
    fred = fred or FREDClient()

    if not kalshi.is_configured():
        return BacktestSummary(
            strategy="macro_kalshi",
            window_days=window_days,
            note=("Kalshi adapter not configured — backtest skipped"),
        )

    today = date.today()
    from_date = today - timedelta(days=window_days)
    settled = kalshi.settled_markets(from_date=from_date, to_date=today,
                                       limit=1000)

    if not settled:
        return BacktestSummary(
            strategy="macro_kalshi",
            window_days=window_days,
            note="No settled Kalshi markets returned in window",
        )

    # Filter to macro-relevant markets only
    macro_markets = [m for m in settled if _is_macro_market(m)]
    if not macro_markets:
        return BacktestSummary(
            strategy="macro_kalshi",
            window_days=window_days,
            note=(f"No macro-keyword markets among {len(settled)} settled "
                  f"Kalshi markets in window"),
        )

    # Pre-fetch FRED context series (one call each, disk-cached).
    # If FRED isn't configured this returns empty dicts and we just
    # skip the per-trade context attachment.
    fred_context = _fetch_fred_context(fred, from_date, today)

    trades: list[_MacroTrade] = []
    skipped = {"no_yes_close": 0, "tied_or_void": 0,
                "below_threshold": 0}

    for m in macro_markets:
        if m.yes_close_price <= 0 or m.yes_close_price >= 1.0:
            skipped["no_yes_close"] += 1
            continue
        if m.settlement_value not in (0.0, 1.0):
            skipped["tied_or_void"] += 1
            continue

        market_price = m.yes_close_price
        fair_value = _fair_value(market_price)
        edge = fair_value - market_price
        edge_cents = edge * 100

        if abs(edge_cents) < ENTRY_EDGE_CENTS:
            skipped["below_threshold"] += 1
            continue

        if edge > 0:
            p, cost, side = fair_value, market_price, "BUY_YES"
        else:
            p, cost, side = 1 - fair_value, 1 - market_price, "BUY_NO"

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

        trades.append(_MacroTrade(
            ticker=m.ticker,
            title=m.title,
            settle_date=m.close_ts.date(),
            side=side,
            market_price=market_price,
            fair_value=fair_value,
            edge_cents=edge_cents,
            n_contracts=n_contracts,
            cost_per_contract=cost,
            settled_yes=(m.settlement_value == 1.0),
            fred_context=_match_fred_context(m, fred_context),
        ))

    if skipped:
        logger.info(f"macro_kalshi {window_days}d filter: {skipped}")

    return _summarize(trades, window_days,
                       n_macro_markets=len(macro_markets))


def _is_macro_market(m: ResolvedMarket) -> bool:
    haystack = f"{m.ticker} {m.title}".lower()
    return any(k in haystack for k in MACRO_KEYWORDS)


def _fair_value(market_price: float) -> float:
    for lo, hi, shift in DEFAULT_RECALIBRATION:
        if lo <= market_price < hi:
            return max(0.0, min(1.0, market_price + shift))
    return market_price


def _fetch_fred_context(client: FREDClient,
                         from_date: date, to_date: date) -> dict:
    """Pre-pull a few common macro series for trade-level context."""
    if not client.is_configured():
        return {}
    out = {}
    for series in ("FEDFUNDS", "CPIAUCSL", "UNRATE", "PAYEMS"):
        try:
            obs = client.series_observations(series, from_date=from_date,
                                               to_date=to_date)
            out[series] = obs
        except Exception as e:
            logger.debug(f"FRED {series} fetch failed: {e}")
    return out


def _match_fred_context(market: ResolvedMarket, ctx: dict) -> dict:
    """Attach the most-relevant FRED data point (last observation
    before market close_ts) to each trade for sanity-checking."""
    if not ctx:
        return {}
    out: dict = {}
    cutoff = market.close_ts.date()
    for series, obs_list in ctx.items():
        latest = None
        for o in obs_list:
            if o.date <= cutoff and o.value is not None:
                latest = o
        if latest is not None:
            out[series] = {"date": latest.date.isoformat(),
                           "value": latest.value}
    return out


def _summarize(trades: list[_MacroTrade], window_days: int,
                *, n_macro_markets: int) -> BacktestSummary:
    if not trades:
        return BacktestSummary(
            strategy="macro_kalshi",
            window_days=window_days,
            note=(f"No macro Kalshi markets crossed 2.5¢ edge threshold "
                  f"({n_macro_markets} macro markets in window)"),
        )

    pnls = [t.net_pnl for t in trades]
    total_pnl = sum(pnls)
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    entry_vol = sum(t.total_cost for t in trades)

    sharpe = _sharpe(pnls)
    max_dd = _max_drawdown(sorted(trades, key=lambda x: x.settle_date))

    eq_curve = []
    cum = 0.0
    for t in sorted(trades, key=lambda x: x.settle_date):
        cum += t.net_pnl
        eq_curve.append({"t": t.settle_date.isoformat(), "pnl_cumulative": cum})

    trade_dicts = [
        {
            "open_time": t.settle_date.isoformat(),
            "close_time": t.settle_date.isoformat(),
            "product_id": t.ticker,
            "side": t.side,
            "entry_price": t.market_price,
            "exit_price": 1.0 if t.settled_yes else 0.0,
            "quantity": float(t.n_contracts),
            "amount_usd": t.total_cost,
            "pnl_usd": t.net_pnl,
            "reason": (f"{t.title[:50]} | market={t.market_price:.3f} "
                        f"fair={t.fair_value:.3f} edge={t.edge_cents:+.1f}c"),
            "exit_reason": ("settled YES" if t.settled_yes else "settled NO"),
            "fred_context": t.fred_context,
        }
        for t in trades
    ]

    return BacktestSummary(
        strategy="macro_kalshi",
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
        note=(f"Macro Kalshi resolved-markets backtest. {n_macro_markets} "
              f"macro-keyword markets in window, {n} crossed entry edge. "
              f"FRED context attached when available."),
    )


def _sharpe(pnls: list[float]) -> float | None:
    if len(pnls) < 3:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    sd = var ** 0.5
    return None if sd == 0 else round(mean / sd * (len(pnls) ** 0.5), 3)


def _max_drawdown(trades: list[_MacroTrade]) -> float:
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
