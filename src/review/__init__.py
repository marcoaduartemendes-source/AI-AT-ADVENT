"""Strategic review — LLM-driven weekly meta-review.

A senior PM-style agent that reads:
  • per-strategy rolling metrics (Sharpe, win rate, drawdown)
  • recent allocator decisions
  • lifecycle events (transitions in the past N days)
  • risk-state history (drawdowns, kill-switch events)
  • scout signal volume per asset class

Outputs a structured recommendation:
  • freezes / unfreezes / retirements
  • allocation tilts (which pods deserve more capital, which less)
  • risk-multiplier adjustment
  • specific issues to investigate

Recommendations are PERSISTED to a `review_recommendations` SQLite table.
They are NEVER auto-applied. The user reviews them on the dashboard and
applies via repo variables / dashboard manual controls.
"""
from .reviewer import StrategicReviewer

__all__ = ["StrategicReviewer"]
