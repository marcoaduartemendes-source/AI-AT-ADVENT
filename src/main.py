"""
AI-AT-ADVENT Daily AI News Digest
Entry point — fetches news, generates digest with Claude, sends email.
"""
import logging
import os
import sys
import traceback
from dotenv import load_dotenv

from .news_fetcher import fetch_all_news
from .summarizer import generate_digest, _fallback_digest
from .email_sender import send_email

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _diagnose_env() -> dict:
    """Report presence/absence of every secret without leaking values."""
    keys = [
        "EMAIL_FROM", "EMAIL_TO",
        "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD",
    ]
    report = {}
    for k in keys:
        v = os.getenv(k, "").strip()
        report[k] = f"SET ({len(v)} chars)" if v else "MISSING"
    return report


def main() -> int:
    logger.info("=" * 60)
    logger.info("AI-AT-ADVENT Daily Digest starting")
    logger.info("=" * 60)

    env = _diagnose_env()
    for k, v in env.items():
        logger.info("  %s: %s", k, v)

    missing = [k for k, v in env.items() if v == "MISSING"]
    if missing:
        logger.error("Cannot send email — missing secrets: %s", ", ".join(missing))
        logger.error("Add these at: GitHub → Settings → Secrets → Actions")
        return 1

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    email_from        = os.getenv("EMAIL_FROM", "").strip()
    email_to          = os.getenv("EMAIL_TO", "").strip()
    smtp_host         = os.getenv("SMTP_HOST", "").strip()
    smtp_port         = int(os.getenv("SMTP_PORT", "587"))
    smtp_username     = os.getenv("SMTP_USERNAME", "").strip()
    smtp_password     = os.getenv("SMTP_PASSWORD", "").strip()
    hours_back        = int(os.getenv("HOURS_BACK", "24"))

    # ── Step 1: Fetch news (never raises) ────────────────────────
    logger.info("Step 1/3 — Fetching AI news from the past %d hours...", hours_back)
    try:
        articles = fetch_all_news(hours_back=hours_back)
    except Exception:
        logger.error("Article fetch crashed:\n%s", traceback.format_exc())
        articles = []
    logger.info("→ Collected %d article(s)", len(articles))

    # ── Step 2: Generate digest (falls back on error) ────────────
    logger.info("Step 2/3 — Generating digest...")
    try:
        digest = generate_digest(articles, anthropic_api_key)
    except Exception:
        logger.error("Digest generation crashed — using fallback:\n%s", traceback.format_exc())
        digest = _fallback_digest(articles)

    # ── Step 3: Send email (retries once, then reports clearly) ──
    logger.info("Step 3/3 — Sending email...")
    try:
        send_email(
            digest_content=digest,
            from_email=email_from,
            to_email=email_to,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_username=smtp_username,
            smtp_password=smtp_password,
        )
    except Exception:
        logger.error("Email send failed:\n%s", traceback.format_exc())
        logger.error("Common causes: wrong SMTP_PASSWORD (must be Gmail App Password,")
        logger.error("no spaces), SMTP_USERNAME mismatch, or 2FA not enabled.")
        return 2

    logger.info("=" * 60)
    logger.info("Daily digest completed successfully!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
