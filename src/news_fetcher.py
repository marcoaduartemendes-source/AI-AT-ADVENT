import feedparser
import requests
import time
import logging
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class Article:
    title: str
    url: str
    summary: str
    published: Optional[datetime]
    source: str
    category: str

# News sources with RSS feeds — add/remove as needed
NEWS_SOURCES = [
    # === Core AI Providers ===
    {
        "name": "Anthropic",
        "feed_url": "https://www.anthropic.com/news/rss",
        "category": "Anthropic & Claude",
    },
    {
        "name": "OpenAI",
        "feed_url": "https://openai.com/blog/rss.xml",
        "category": "OpenAI & ChatGPT",
    },
    {
        "name": "Google DeepMind",
        "feed_url": "https://blog.google/technology/ai/rss/",
        "category": "Google AI & Gemini",
    },
    {
        "name": "Google AI Blog",
        "feed_url": "https://ai.googleblog.com/feeds/posts/default?alt=rss",
        "category": "Google AI & Gemini",
    },
    {
        "name": "Meta AI",
        "feed_url": "https://engineering.fb.com/feed/",
        "category": "Meta AI",
    },
    {
        "name": "Mistral AI",
        "feed_url": "https://mistral.ai/news/feed.xml",
        "category": "Other AI Providers",
    },
    {
        "name": "Hugging Face",
        "feed_url": "https://huggingface.co/blog/feed.xml",
        "category": "Other AI Providers",
    },
    # === Industry News ===
    {
        "name": "The Verge AI",
        "feed_url": "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
        "category": "Industry News",
    },
    {
        "name": "TechCrunch AI",
        "feed_url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "Industry News",
    },
    {
        "name": "VentureBeat AI",
        "feed_url": "https://venturebeat.com/category/ai/feed/",
        "category": "Industry News",
    },
    {
        "name": "Ars Technica",
        "feed_url": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "category": "Industry News",
    },
    {
        "name": "MIT Technology Review AI",
        "feed_url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "category": "Industry News",
    },
]

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0; "
        "+https://github.com/marcoaduartemendes-source/AI-AT-ADVENT)"
    )
}


def _parse_published(entry) -> Optional[datetime]:
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return None


def _strip_html(raw: str, max_chars: int = 500) -> str:
    try:
        return BeautifulSoup(raw, "lxml").get_text(separator=" ", strip=True)[:max_chars]
    except Exception:
        return raw[:max_chars]


def _fetch_feed(source: dict, cutoff: datetime, max_per_source: int = 8) -> list[Article]:
    articles: list[Article] = []
    try:
        resp = requests.get(source["feed_url"], headers=_REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)

        for entry in feed.entries[:max_per_source]:
            pub = _parse_published(entry)
            if pub and pub < cutoff:
                continue  # too old

            raw_summary = (
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or ""
            )
            articles.append(
                Article(
                    title=entry.get("title", "Untitled").strip(),
                    url=entry.get("link", ""),
                    summary=_strip_html(raw_summary),
                    published=pub,
                    source=source["name"],
                    category=source["category"],
                )
            )
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", source["name"], exc)
    return articles


def fetch_all_news(hours_back: int = 24) -> list[Article]:
    """Fetch articles from all sources published within the last N hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    all_articles: list[Article] = []

    for source in NEWS_SOURCES:
        logger.info("Fetching: %s", source["name"])
        all_articles.extend(_fetch_feed(source, cutoff))
        time.sleep(0.5)  # polite crawling delay

    # Sort newest first; undated articles go last
    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    all_articles.sort(key=lambda a: a.published or _epoch, reverse=True)

    logger.info("Fetched %d articles total across %d sources", len(all_articles), len(NEWS_SOURCES))
    return all_articles
