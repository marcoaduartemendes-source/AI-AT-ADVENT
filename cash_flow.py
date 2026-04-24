#!/usr/bin/env python3
"""
Cash Flow Forecast

Reads from Monarch Money, Rocket Money (CSV), and Chase (Plaid),
then emails a styled 13-week cash flow projection.

Usage:
    python cash_flow.py

Requires secrets in .env (see .env.example for all options).
At least one data source must be configured.
"""
import logging
import os
import smtplib
import socket
import ssl
import sys
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Force IPv4 (same pattern as the rest of the project)
_orig_gai = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, *a, **k: _orig_gai(h, p, socket.AF_INET, *a, **k)


def _send_email(html: str, plain: str, subject: str) -> None:
    from_email = os.getenv("EMAIL_FROM", "").strip()
    to_email = os.getenv("EMAIL_TO", "").strip()
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME", "").strip()
    smtp_pass = os.getenv("SMTP_PASSWORD", "").strip()

    if not all([from_email, to_email, smtp_pass]):
        logger.warning("Email not configured — printing report to stdout instead.")
        print(plain)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    def _starttls():
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_email, [to_email], msg.as_string())

    def _ssl():
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_host, 465, context=ctx, timeout=30) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(from_email, [to_email], msg.as_string())

    try:
        _starttls()
        logger.info("Cash flow report sent to %s (STARTTLS)", to_email)
    except Exception as exc:
        logger.warning("STARTTLS failed (%s), retrying with SSL port 465…", exc)
        try:
            _ssl()
            logger.info("Cash flow report sent to %s (SSL)", to_email)
        except Exception as exc2:
            logger.error("Both SMTP methods failed: %s", exc2)
            print(plain)
            raise


def main() -> int:
    from src.cashflow.connectors.monarch import fetch_monarch_transactions, get_monarch_balance
    from src.cashflow.connectors.rocket import fetch_rocket_transactions
    from src.cashflow.connectors.chase import fetch_chase_transactions
    from src.cashflow.forecaster import deduplicate, build_forecast
    from src.cashflow.report import build_report_html, build_report_text

    lookback_days = int(os.getenv("CASHFLOW_LOOKBACK_DAYS", "90"))
    forecast_weeks = int(os.getenv("CASHFLOW_FORECAST_WEEKS", "13"))

    logger.info("=" * 60)
    logger.info("Cash Flow Forecast  —  %d-week projection", forecast_weeks)
    logger.info("=" * 60)

    # ── Step 1: Collect transactions ──────────────────────────────
    all_transactions = []
    sources_used = []

    logger.info("[1/4] Fetching Monarch Money transactions…")
    monarch_txns = fetch_monarch_transactions(lookback_days=lookback_days)
    if monarch_txns:
        all_transactions.extend(monarch_txns)
        sources_used.append("monarch")

    logger.info("[2/4] Loading Rocket Money transactions…")
    rocket_txns = fetch_rocket_transactions(lookback_days=lookback_days)
    if rocket_txns:
        all_transactions.extend(rocket_txns)
        sources_used.append("rocket money")

    logger.info("[3/4] Fetching Chase transactions via Plaid…")
    chase_txns = fetch_chase_transactions(lookback_days=lookback_days)
    if chase_txns:
        all_transactions.extend(chase_txns)
        sources_used.append("chase")

    if not all_transactions:
        logger.error(
            "No transactions found. Configure at least one source in .env:\n"
            "  Monarch Money : MONARCH_EMAIL + MONARCH_PASSWORD\n"
            "  Rocket Money  : ROCKET_MONEY_CSV_PATH (CSV export from the app)\n"
            "  Chase         : PLAID_CLIENT_ID + PLAID_SECRET + PLAID_ACCESS_TOKEN\n"
            "                  (run python scripts/plaid_link_setup.py first)"
        )
        return 1

    logger.info("Collected %d transactions; deduplicating…", len(all_transactions))
    transactions = deduplicate(all_transactions)
    logger.info("After dedup: %d transactions from sources: %s",
                len(transactions), ", ".join(sources_used))

    # ── Step 2: Resolve opening balance ───────────────────────────
    current_balance = float(os.getenv("CURRENT_BALANCE", "0"))
    if current_balance == 0 and "monarch" in sources_used:
        logger.info("CURRENT_BALANCE not set — fetching live balance from Monarch…")
        monarch_bal = get_monarch_balance()
        if monarch_bal is not None:
            current_balance = monarch_bal
            logger.info("Monarch balance: $%,.2f", current_balance)

    # ── Step 3: Build forecast ────────────────────────────────────
    logger.info("[4/4] Building %d-week cash flow forecast…", forecast_weeks)
    weeks = build_forecast(
        transactions,
        current_balance=current_balance,
        forecast_weeks=forecast_weeks,
    )

    # ── Step 4: Send report ───────────────────────────────────────
    html = build_report_html(weeks, sources_used)
    plain = build_report_text(weeks)
    subject = f"13-Week Cash Flow Forecast — {date.today().strftime('%B %d, %Y')}"

    _send_email(html, plain, subject)

    logger.info("=" * 60)
    logger.info("Done.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
