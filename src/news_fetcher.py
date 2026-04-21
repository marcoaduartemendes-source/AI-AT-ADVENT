import requests
import time
import logging
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
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
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
}


def _strip_html(raw: str, max_chars: int = 500) -> str:
    try:
        return BeautifulSoup(raw, "html.parser").get_text(separator=" ", strip=True)[:max_chars]
    except Exception:
        return raw[:max_chars]


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse RFC 2822 (RSS) or ISO 8601 (Atom) date strings."""
    if not date_str:
        return None
    date_str = date_str.strip()
    # RFC 2822 — used by RSS 2.0
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # ISO 8601 variants — used by Atom
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:25], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


# XML namespaces used by Atom feeds
_ATOM_NS = "http://www.w3.org/2005/Atom"
_MEDIA_NS = "http://search.yahoo.com/mrss/"


def _parse_rss_or_atom(content: bytes, source_name: str, category: str,
                        cutoff: datetime, max_items: int) -> list[Article]:
    """Parse RSS 2.0 or Atom 1.0 feed bytes into Article objects."""
    articles: list[Article] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        logger.warning("XML parse error for %s: %s", source_name, exc)
        return articles

    tag = root.tag.lower()

    # ── Atom feed ──────────────────────────────────────────────
    if "atom" in tag or root.tag == f"{{{_ATOM_NS}}}feed":
        ns = {"a": _ATOM_NS}
        entries = root.findall("a:entry", ns) or root.findall("entry")
        for entry in entries[:max_items]:
            title_el  = entry.find("a:title", ns) or entry.find("title")
            link_el   = entry.find("a:link[@rel='alternate']", ns) or \
                        entry.find("a:link", ns) or entry.find("link")
            summ_el   = entry.find("a:summary", ns) or entry.find("summary") or \
                        entry.find("a:content", ns) or entry.find("content")
            date_el   = entry.find("a:updated", ns) or entry.find("updated") or \
                        entry.find("a:published", ns) or entry.find("published")

            title = (title_el.text or "").strip() if title_el is not None else "Untitled"
            url   = (link_el.get("href") or link_el.text or "").strip() \
                    if link_el is not None else ""
            raw   = (summ_el.text or "") if summ_el is not None else ""
            pub   = _parse_date(date_el.text) if date_el is not None else None

            if pub and pub < cutoff:
                continue
            articles.append(Article(
                title=title, url=url, summary=_strip_html(raw),
                published=pub, source=source_name, category=category,
            ))

    # ── RSS 2.0 / RSS 1.0 ──────────────────────────────────────
    else:
        channel = root.find("channel") or root
        items = channel.findall("item") or root.findall("item")
        for item in items[:max_items]:
            def _text(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""

            title = _text("title") or "Untitled"
            url   = _text("link") or _text("guid")
            raw   = _text("description") or _text("summary")
            pub   = _parse_date(_text("pubDate") or _text("dc:date") or _text("date"))

            if pub and pub < cutoff:
                continue
            articles.append(Article(
                title=title, url=url, summary=_strip_html(raw),
                published=pub, source=source_name, category=category,
            ))

    return articles


def _fetch_feed(source: dict, cutoff: datetime, max_per_source: int = 8) -> list[Article]:
    try:
        resp = requests.get(source["feed_url"], headers=_REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        return _parse_rss_or_atom(
            resp.content, source["name"], source["category"], cutoff, max_per_source
        )
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", source["name"], exc)
        return []


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
