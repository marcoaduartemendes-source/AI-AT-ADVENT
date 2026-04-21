"""
AI-AT-ADVENT Daily AI News Digest
Entry point — fetches news, generates digest with Claude, sends email.
"""
import logging
import os
import sys
from dotenv import load_dotenv

from .news_fetcher import fetch_all_news
from .summarizer import generate_digest
from .email_sender import send_email

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


def main() -> None:
    logger.info("=" * 60)
    logger.info("AI-AT-ADVENT Daily Digest starting")
    logger.info("=" * 60)

    # ── Load configuration ──────────────────────────────────────
    anthropic_api_key = _require("ANTHROPIC_API_KEY")
    email_from       = _require("EMAIL_FROM")
    email_to         = _require("EMAIL_TO")
    smtp_host        = _require("SMTP_HOST")
    smtp_port        = int(os.getenv("SMTP_PORT", "587"))
    smtp_username    = _require("SMTP_USERNAME")
    smtp_password    = _require("SMTP_PASSWORD")
    hours_back       = int(os.getenv("HOURS_BACK", "24"))

    # ── Step 1: Fetch news ──────────────────────────────────────
    logger.info("Step 1/3 — Fetching AI news from the past %d hours...", hours_back)
    articles = fetch_all_news(hours_back=hours_back)

    if not articles:
        logger.warning("No articles found — the digest will note a quiet news day.")

    # ── Step 2: Generate digest with Claude ─────────────────────
    logger.info("Step 2/3 — Generating digest with Claude claude-sonnet-4-6...")
    digest = generate_digest(articles, anthropic_api_key)

    # ── Step 3: Send email ──────────────────────────────────────
    logger.info("Step 3/3 — Sending email...")
    send_email(
        digest_content=digest,
        from_email=email_from,
        to_email=email_to,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_username=smtp_username,
        smtp_password=smtp_password,
    )

    logger.info("=" * 60)
    logger.info("Daily digest completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
