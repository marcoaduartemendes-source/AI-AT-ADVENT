import csv
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .base import Transaction

logger = logging.getLogger(__name__)

# Rocket Money exports dates as MM/DD/YYYY; some exports use YYYY-MM-DD
_DATE_FMTS = ["%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y"]


def _parse_date(s: str) -> date:
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognized date format: {s!r}")


def _parse_amount(s: str) -> float:
    return float(s.strip().replace("$", "").replace(",", ""))


def fetch_rocket_transactions(
    csv_path: Optional[str] = None,
    lookback_days: int = 90,
) -> List[Transaction]:
    """
    Rocket Money has no public API.  Export your transactions from the app:
    Settings → Export Data → Transactions CSV.
    Set ROCKET_MONEY_CSV_PATH to the downloaded file path.
    """
    csv_path = csv_path or os.getenv("ROCKET_MONEY_CSV_PATH", "")
    if not csv_path:
        logger.warning(
            "ROCKET_MONEY_CSV_PATH not set — skipping Rocket Money.\n"
            "  Export from the Rocket Money app: Settings → Export Data → Transactions CSV"
        )
        return []

    p = Path(csv_path)
    if not p.exists():
        logger.error("Rocket Money CSV not found: %s", csv_path)
        return []

    cutoff = date.today() - timedelta(days=lookback_days)
    transactions: List[Transaction] = []

    with p.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                txn_date = _parse_date(row.get("Date", ""))
                if txn_date < cutoff:
                    continue

                raw = _parse_amount(row.get("Amount", "0"))

                # Determine sign from Transaction Type column when present.
                # Rocket Money exports expenses as positive amounts.
                txn_type = row.get("Transaction Type", "").strip().lower()
                if txn_type in ("debit", "expense", "withdrawal"):
                    amount = -abs(raw)
                elif txn_type in ("credit", "income", "deposit"):
                    amount = abs(raw)
                else:
                    # Absent/unknown type: Rocket Money typically exports positive = expense
                    amount = -raw

                category = row.get("Category", "Uncategorized")
                transactions.append(Transaction(
                    date=txn_date,
                    amount=amount,
                    description=row.get("Description", ""),
                    category=category,
                    account=row.get("Account Name", "Unknown"),
                    source="rocket",
                    merchant=row.get("Merchant") or None,
                    transaction_id=None,
                    is_transfer="transfer" in category.lower(),
                    is_pending=False,
                ))
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping Rocket Money row %s: %s", row, exc)

    logger.info("Rocket Money: loaded %d transactions from %s", len(transactions), csv_path)
    return transactions
