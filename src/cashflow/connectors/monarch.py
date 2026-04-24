import asyncio
import logging
import os
from datetime import date, timedelta
from typing import List, Optional

from .base import Transaction

logger = logging.getLogger(__name__)

_PAGE_SIZE = 500


async def _fetch(email: str, password: str, mfa_secret: Optional[str], lookback_days: int) -> List[Transaction]:
    try:
        from monarch_money import MonarchMoney
    except ImportError:
        raise ImportError("pip install monarch-money")

    mm = MonarchMoney()

    if mfa_secret:
        try:
            import pyotp
            mfa_code = pyotp.TOTP(mfa_secret).now()
        except ImportError:
            raise ImportError("pip install pyotp  # required for Monarch MFA")
        await mm.login(email, password, mfa_code=mfa_code)
    else:
        await mm.login(email, password)

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()

    transactions: List[Transaction] = []
    offset = 0

    while True:
        resp = await mm.get_transactions(
            limit=_PAGE_SIZE,
            offset=offset,
            start_date=start,
            end_date=end,
        )
        batch = resp.get("allTransactions", {}).get("results", [])
        if not batch:
            break

        for t in batch:
            # Monarch: negative amount = expense, positive = income
            transactions.append(Transaction(
                date=date.fromisoformat(t["date"]),
                amount=float(t.get("amount", 0)),
                description=(
                    (t.get("merchant") or {}).get("name")
                    or t.get("originalName", "")
                ),
                category=(t.get("category") or {}).get("name", "Uncategorized"),
                account=(t.get("account") or {}).get("displayName", "Unknown"),
                source="monarch",
                merchant=(t.get("merchant") or {}).get("name"),
                transaction_id=t.get("id"),
                is_transfer=t.get("isTransfer", False),
                is_pending=t.get("pending", False),
            ))

        offset += len(batch)
        total = resp.get("allTransactions", {}).get("totalCount", 0)
        if offset >= total:
            break

    logger.info("Monarch Money: fetched %d transactions", len(transactions))
    return transactions


async def _get_monarch_balance(email: str, password: str, mfa_secret: Optional[str]) -> float:
    """Return sum of all depository account balances from Monarch."""
    try:
        from monarch_money import MonarchMoney
    except ImportError:
        return 0.0

    mm = MonarchMoney()
    if mfa_secret:
        import pyotp
        await mm.login(email, password, mfa_code=pyotp.TOTP(mfa_secret).now())
    else:
        await mm.login(email, password)

    accounts = await mm.get_accounts()
    total = 0.0
    for acct in accounts.get("accounts", []):
        acct_type = (acct.get("type") or {}).get("name", "").lower()
        if acct_type in ("depository", "checking", "savings", "cash"):
            total += float(acct.get("currentBalance") or 0)
    return total


def fetch_monarch_transactions(
    lookback_days: int = 90,
    email: Optional[str] = None,
    password: Optional[str] = None,
    mfa_secret: Optional[str] = None,
) -> List[Transaction]:
    email = email or os.getenv("MONARCH_EMAIL", "")
    password = password or os.getenv("MONARCH_PASSWORD", "")
    mfa_secret = mfa_secret or os.getenv("MONARCH_MFA_SECRET") or None

    if not email or not password:
        logger.warning("MONARCH_EMAIL/MONARCH_PASSWORD not set — skipping Monarch Money")
        return []

    return asyncio.run(_fetch(email, password, mfa_secret, lookback_days))


def get_monarch_balance(
    email: Optional[str] = None,
    password: Optional[str] = None,
    mfa_secret: Optional[str] = None,
) -> Optional[float]:
    email = email or os.getenv("MONARCH_EMAIL", "")
    password = password or os.getenv("MONARCH_PASSWORD", "")
    mfa_secret = mfa_secret or os.getenv("MONARCH_MFA_SECRET") or None
    if not email or not password:
        return None
    try:
        return asyncio.run(_get_monarch_balance(email, password, mfa_secret))
    except Exception as exc:
        logger.warning("Could not fetch Monarch balance: %s", exc)
        return None
