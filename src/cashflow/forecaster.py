"""
13-week cash flow forecasting engine.

Phases:
  1. Deduplicate transactions across sources.
  2. Detect recurring items (weekly / bi-weekly / monthly) via gap analysis.
  3. Compute variable-category weekly averages for non-recurring spend.
  4. Project both layers forward for forecast_weeks (default 13).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import re
import statistics

from .connectors.base import Transaction


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRANSFER_KEYWORDS = frozenset({
    "transfer", "zelle", "venmo", "paypal", "cash app", "square cash",
    "wire", "ach transfer",
})

# How many days off a payment can be and still count as recurring
_GAP_TOLERANCE = 5

# Minimum occurrences before we call something recurring
_MIN_OCCURRENCES = 2

# Variable averages computed over this many weeks of history
_VARIABLE_LOOKBACK_WEEKS = 8

# Alert threshold for low-balance weeks
LOW_BALANCE_THRESHOLD = 500.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class RecurringItem:
    description: str
    amount: float           # avg; negative = expense, positive = income
    frequency_days: int     # 7, 14, 30, or 365
    category: str
    next_expected: date     # first future occurrence
    confidence: float       # 0–1


@dataclass
class WeekForecast:
    week_number: int
    week_start: date
    week_end: date
    projected_income: float
    projected_expenses: float
    projected_net: float
    opening_balance: float
    closing_balance: float
    line_items: List[Tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(desc: str) -> str:
    """Normalize a description for grouping: uppercase, strip digits/symbols."""
    s = (desc or "").upper()
    s = re.sub(r"\d{4,}", "", s)          # long numbers (card #, order #)
    s = re.sub(r"[*#@&/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:40]


def _is_transfer(t: Transaction) -> bool:
    desc = (t.description or "").lower()
    return t.is_transfer or any(kw in desc for kw in _TRANSFER_KEYWORDS)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(transactions: List[Transaction]) -> List[Transaction]:
    """
    Remove cross-source duplicates.

    Key: (date, amount rounded to 2dp, first 20 chars of normalised description).
    When the same transaction appears in multiple sources (e.g. a Chase account
    aggregated by Monarch AND fetched directly via Plaid), keep the first one
    seen, preferring Monarch → Rocket → Chase order.
    """
    source_order = {"monarch": 0, "rocket": 1, "chase": 2}
    sorted_txns = sorted(transactions, key=lambda t: source_order.get(t.source, 9))

    seen: set = set()
    result: List[Transaction] = []
    for t in sorted_txns:
        key = (t.date, round(t.amount, 2), _norm(t.description)[:20])
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Recurring item detection
# ---------------------------------------------------------------------------

def identify_recurring(transactions: List[Transaction]) -> List[RecurringItem]:
    """
    Group transactions by normalised description and detect those that repeat
    on a consistent schedule (weekly / bi-weekly / monthly / annual).
    """
    groups: Dict[str, List[Transaction]] = defaultdict(list)
    for t in transactions:
        if _is_transfer(t) or t.is_pending:
            continue
        groups[_norm(t.description)].append(t)

    today = date.today()
    recurring: List[RecurringItem] = []

    for norm_desc, txns in groups.items():
        if len(txns) < _MIN_OCCURRENCES:
            continue

        txns_s = sorted(txns, key=lambda t: t.date)
        gaps = [
            (txns_s[i + 1].date - txns_s[i].date).days
            for i in range(len(txns_s) - 1)
        ]
        if not gaps:
            continue

        avg_gap = statistics.mean(gaps)
        stdev = statistics.stdev(gaps) if len(gaps) > 1 else 0.0

        # Map to a canonical frequency
        if abs(avg_gap - 7) <= _GAP_TOLERANCE:
            freq = 7
        elif abs(avg_gap - 14) <= _GAP_TOLERANCE:
            freq = 14
        elif abs(avg_gap - 30) <= _GAP_TOLERANCE * 2:
            freq = 30
        elif abs(avg_gap - 365) <= 30:
            freq = 365
        else:
            continue

        # Confidence: penalise high variance relative to the period
        confidence = max(0.0, 1.0 - (stdev / max(avg_gap, 1)))
        if confidence < 0.3:
            continue

        avg_amount = statistics.mean(t.amount for t in txns_s)
        last_date = txns_s[-1].date

        # Walk next_expected forward until it is in the future
        next_exp = last_date + timedelta(days=freq)
        while next_exp < today:
            next_exp += timedelta(days=freq)

        recurring.append(RecurringItem(
            description=txns_s[-1].description or norm_desc.title(),
            amount=round(avg_amount, 2),
            frequency_days=freq,
            category=txns_s[-1].category,
            next_expected=next_exp,
            confidence=confidence,
        ))

    return recurring


# ---------------------------------------------------------------------------
# Variable category averages
# ---------------------------------------------------------------------------

def _variable_weekly_avg(
    transactions: List[Transaction],
    recurring_descriptions: frozenset,
) -> Dict[str, float]:
    """
    Compute average weekly spend per category, excluding:
    - Transfers
    - Pending transactions
    - Transactions whose description is already captured as recurring
    - Income (positive amounts)
    """
    cutoff = date.today() - timedelta(weeks=_VARIABLE_LOOKBACK_WEEKS)
    category_totals: Dict[str, float] = defaultdict(float)

    for t in transactions:
        if t.date < cutoff or t.is_pending or _is_transfer(t):
            continue
        if t.amount >= 0:   # income — handled by recurring detection
            continue
        if _norm(t.description) in recurring_descriptions:
            continue
        category_totals[t.category or "Uncategorized"] += t.amount  # negative

    return {
        cat: round(total / _VARIABLE_LOOKBACK_WEEKS, 2)
        for cat, total in category_totals.items()
    }


# ---------------------------------------------------------------------------
# Forecast builder
# ---------------------------------------------------------------------------

def build_forecast(
    transactions: List[Transaction],
    current_balance: float = 0.0,
    forecast_weeks: int = 13,
) -> List[WeekForecast]:
    """
    Build a week-by-week cash flow forecast.

    Args:
        transactions:    Deduplicated historical transactions.
        current_balance: Today’s known cash balance (from env or Monarch API).
        forecast_weeks:  Number of weeks to project (default 13).

    Returns:
        List of WeekForecast, one per week, in order.
    """
    clean = [t for t in transactions if not t.is_pending]
    recurring = identify_recurring(clean)
    recurring_norms = frozenset(_norm(r.description) for r in recurring)
    variable_avg = _variable_weekly_avg(clean, recurring_norms)

    today = date.today()
    forecast_end = today + timedelta(weeks=forecast_weeks)

    # Pre-compute all recurring occurrences in the forecast window.
    # Map week_number (1-based) → list of (description, amount)
    weekly_recurring: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for r in recurring:
        cursor = r.next_expected
        while cursor <= forecast_end:
            days_ahead = (cursor - today).days
            if days_ahead >= 0:
                week_num = days_ahead // 7 + 1
                if 1 <= week_num <= forecast_weeks:
                    weekly_recurring[week_num].append((r.description, r.amount))
            cursor += timedelta(days=r.frequency_days)

    weeks: List[WeekForecast] = []
    for w in range(1, forecast_weeks + 1):
        week_start = today + timedelta(weeks=w - 1)
        week_end = week_start + timedelta(days=6)

        line_items: List[Tuple[str, float]] = list(weekly_recurring.get(w, []))

        # Add variable category estimates (expenses only)
        for cat, weekly_spend in variable_avg.items():
            line_items.append((f"{cat} (variable avg)", weekly_spend))

        projected_income = sum(amt for _, amt in line_items if amt > 0)
        projected_expenses = sum(amt for _, amt in line_items if amt < 0)
        projected_net = projected_income + projected_expenses

        opening = current_balance if w == 1 else weeks[-1].closing_balance
        closing = opening + projected_net

        weeks.append(WeekForecast(
            week_number=w,
            week_start=week_start,
            week_end=week_end,
            projected_income=round(projected_income, 2),
            projected_expenses=round(projected_expenses, 2),
            projected_net=round(projected_net, 2),
            opening_balance=round(opening, 2),
            closing_balance=round(closing, 2),
            line_items=sorted(line_items, key=lambda x: x[1]),
        ))

    return weeks
