"""Per-strategy backtests for the new orchestrator strategies.

Backtestable from candles alone (no scout signals required):
  • tsmom_etf            (Alpaca daily ETF candles)
  • risk_parity_etf      (Alpaca daily ETF candles)
  • crypto_xsmom         (Coinbase daily candles, top-15 alts)
  • vol_managed_overlay  (returns the vol-scaler series for SPY/BTC)

Strategies that need scout-fed data (funding rates, mispriced markets,
earnings surprises) are NOT backtestable here — flagged in
`UNBACKTESTABLE`.

Each backtest returns a `BacktestSummary` matching the existing dashboard
shape so it can plug straight into the 7/15/30-day tabs.
"""
from .runner import (
    BacktestSummary,
    UNBACKTESTABLE,
    backtest_all,
    backtest_strategy_by_name,
)

__all__ = [
    "BacktestSummary",
    "UNBACKTESTABLE",
    "backtest_all",
    "backtest_strategy_by_name",
]
