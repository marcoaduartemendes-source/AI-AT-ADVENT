from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class Transaction:
    date: date
    amount: float          # positive = income/inflow, negative = expense/outflow
    description: str
    category: str
    account: str
    source: str            # "monarch" | "rocket" | "chase"
    merchant: Optional[str] = None
    transaction_id: Optional[str] = None
    is_transfer: bool = False
    is_pending: bool = False
