import logging
import os
from datetime import date, timedelta
from typing import List, Optional

from .base import Transaction

logger = logging.getLogger(__name__)


def fetch_chase_transactions(
    lookback_days: int = 90,
    client_id: Optional[str] = None,
    secret: Optional[str] = None,
    access_token: Optional[str] = None,
    plaid_env: Optional[str] = None,
) -> List[Transaction]:
    """
    Fetch Chase transactions via the Plaid API.

    Obtain an access token once by running:  python scripts/plaid_link_setup.py
    Then set PLAID_ACCESS_TOKEN in your .env / GitHub secrets.
    """
    client_id = client_id or os.getenv("PLAID_CLIENT_ID", "")
    secret = secret or os.getenv("PLAID_SECRET", "")
    access_token = access_token or os.getenv("PLAID_ACCESS_TOKEN", "")
    plaid_env = plaid_env or os.getenv("PLAID_ENV", "production")

    if not all([client_id, secret, access_token]):
        logger.warning(
            "PLAID_CLIENT_ID / PLAID_SECRET / PLAID_ACCESS_TOKEN not set — skipping Chase.\n"
            "  Run python scripts/plaid_link_setup.py to link your Chase account."
        )
        return []

    try:
        import plaid
        from plaid.api import plaid_api
        from plaid.model.transactions_get_request import TransactionsGetRequest
        from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions
    except ImportError:
        raise ImportError("pip install plaid-python")

    env_map = {
        "sandbox": plaid.Environment.Sandbox,
        "development": plaid.Environment.Development,
        "production": plaid.Environment.Production,
    }
    cfg = plaid.Configuration(
        host=env_map.get(plaid_env, plaid.Environment.Production),
        api_key={"clientId": client_id, "secret": secret},
    )
    client = plaid_api.PlaidApi(plaid.ApiClient(cfg))

    start = date.today() - timedelta(days=lookback_days)
    end = date.today()
    all_raw: list = []
    offset = 0

    while True:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start,
            end_date=end,
            options=TransactionsGetRequestOptions(offset=offset, count=500),
        )
        resp = client.transactions_get(request)
        batch = resp["transactions"]
        all_raw.extend(batch)
        offset += len(batch)
        if offset >= resp["total_transactions"]:
            break

    transactions: List[Transaction] = []
    for t in all_raw:
        # Plaid sign convention: positive = money out (debit), negative = money in (credit)
        amount = -float(t["amount"])

        pfc = (t.get("personal_finance_category") or {}).get("primary", "")
        legacy_cat = (t.get("category") or ["Uncategorized"])[0]
        category = pfc or legacy_cat

        transactions.append(Transaction(
            date=t["date"],
            amount=amount,
            description=t.get("name", ""),
            category=category,
            account=t.get("account_id", "chase"),
            source="chase",
            merchant=t.get("merchant_name"),
            transaction_id=t.get("transaction_id"),
            is_transfer="transfer" in [c.lower() for c in (t.get("category") or [])],
            is_pending=t.get("pending", False),
        ))

    logger.info("Chase/Plaid: fetched %d transactions", len(transactions))
    return transactions
