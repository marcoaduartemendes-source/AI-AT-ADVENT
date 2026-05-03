"""Multi-source news RSS aggregator with ticker-keyword matching.

Why this matters:
    Several strategies are sensitive to news catalysts:
      - earnings_momentum: post-earnings drift turbo-charged by
        sentiment-aligned headlines.
      - sector_rotation: rotation triggers (Fed-speak, sector
        downgrades) often telegraphed in news before price moves.
      - pead: real-time validation that an earnings beat/miss has
        actually been reported — guards against firing on stale data.

Free-tier sufficiency vs paid alternatives:
    FREE (this file):
      - Reuters business RSS — 24h headline lag, free, ~100 posts/day
      - Yahoo Finance RSS per-ticker — auto-tagged headlines
      - MarketWatch top-stories RSS — generalist
      - SEC EDGAR filings RSS — 8-K materials within 4h of filing

    Free is GOOD ENOUGH for our 30-min scout cadence and our trade-
    time horizons (intraday to several-day holds).

    PAID — only worth it if we go HFT:
      - Benzinga Pro (~$199/mo): sub-second institutional headlines,
        structured tickers + sentiment.
      - RavenPack (~$3000/mo): NLP-scored sentiment + analytics
        ready-to-feed.
      - Bloomberg / Refinitiv terminals (~$24k/yr): the institutional
        gold standard.

    Recommendation: stay free until we're trading on intra-second
    headline leakage, which our retail Coinbase + Alpaca paper stack
    cannot exploit anyway.

Failure mode: RSS feeds occasionally lag, return stale data, or rate-
limit. We dedupe by URL + headline hash; missing data is no-info
not stale-info.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, UTC
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


DEFAULT_CACHE_TTL = 5 * 60   # 5 min — RSS publishers update at most that often


# Headline sources we aggregate. Each entry: (label, RSS URL).
DEFAULT_SOURCES = [
    ("reuters_business", "https://feeds.reuters.com/reuters/businessNews"),
    ("marketwatch_top",
        "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("yahoo_finance",
        "https://finance.yahoo.com/news/rssindex"),
    ("sec_8k_filings",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&"
        "type=8-K&output=atom"),
]


@dataclass
class NewsItem:
    """One headline from one RSS feed."""
    source: str          # e.g. "reuters_business"
    title: str
    url: str
    published_at: datetime | None
    summary: str         # first 500 chars of description
    raw_id: str          # stable hash for dedup


class NewsRSSClient:
    """Disk-cached multi-source RSS aggregator with simple dedup."""

    def __init__(
        self,
        sources: list[tuple[str, str]] | None = None,
        cache_dir: str | None = None,
        timeout_seconds: float = 15.0,
    ):
        self.sources = sources or list(DEFAULT_SOURCES)
        self.cache_dir = Path(
            cache_dir or os.environ.get("NEWS_CACHE_DIR", "data/cache/news")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """Always True — public RSS needs no key."""
        return True

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_key(self, url: str) -> str:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return f"rss_{h}"

    def _read_cache(self, key: str, ttl_seconds: int):
        p = self.cache_dir / f"{key}.xml"
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _write_cache(self, key: str, body: str) -> None:
        p = self.cache_dir / f"{key}.xml"
        try:
            p.write_text(body, encoding="utf-8")
        except OSError as e:
            logger.debug(f"news cache write {key} failed: {e}")

    # ── Public API ────────────────────────────────────────────────────

    def fetch_all(
        self,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL,
        max_items_per_source: int = 50,
    ) -> list[NewsItem]:
        """Fetch every configured source; return flat deduped list.

        Order: newest first across all sources.
        """
        out: list[NewsItem] = []
        seen_ids: set[str] = set()
        for label, url in self.sources:
            try:
                items = self._fetch_one(label, url, cache_ttl_seconds,
                                          max_items_per_source)
            except Exception as e:
                logger.warning(f"news source {label} failed: {e}")
                continue
            for it in items:
                if it.raw_id in seen_ids:
                    continue
                seen_ids.add(it.raw_id)
                out.append(it)
        out.sort(
            key=lambda x: x.published_at or datetime(1970, 1, 1, tzinfo=UTC),
            reverse=True,
        )
        return out

    def _fetch_one(
        self, label: str, url: str, ttl_seconds: int, max_items: int,
    ) -> list[NewsItem]:
        cache_key = self._cache_key(url)
        body = self._read_cache(cache_key, ttl_seconds)
        if body is None:
            try:
                resp = requests.get(
                    url,
                    headers={
                        "User-Agent":
                            "Mozilla/5.0 (compatible; trading-bot/1.0)",
                    },
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"RSS {label} HTTP {resp.status_code}"
                    )
                    return []
                body = resp.text
                self._write_cache(cache_key, body)
            except requests.RequestException as e:
                logger.warning(f"RSS {label} request failed: {e}")
                return []
        return _parse_feed(label, body, max_items)

    def search_tickers(
        self,
        tickers: list[str],
        items: list[NewsItem] | None = None,
        within_hours: int = 24,
    ) -> dict[str, list[NewsItem]]:
        """Filter aggregated headlines to ones mentioning given tickers
        (case-insensitive whole-word match) within the last `within_hours`.

        Returns {ticker: [NewsItem, …]} for tickers that had any match.
        """
        items = items or self.fetch_all()
        cutoff = datetime.now(UTC).timestamp() - within_hours * 3600
        out: dict[str, list[NewsItem]] = {}
        for t in tickers:
            tu = t.upper()
            matches: list[NewsItem] = []
            for it in items:
                if it.published_at and it.published_at.timestamp() < cutoff:
                    continue
                hay = f"{it.title} {it.summary}".upper()
                # Whole-word check: the ticker must appear delimited
                if (f" {tu} " in f" {hay} " or
                        f"({tu})" in hay or
                        f"${tu}" in hay):
                    matches.append(it)
            if matches:
                out[t] = matches
        return out


def _parse_feed(label: str, body: str, max_items: int) -> list[NewsItem]:
    """Parse RSS 2.0 or Atom XML; return up to max_items NewsItems."""
    if not body:
        return []
    try:
        # Strip BOM if present
        if body.startswith("﻿"):
            body = body[1:]
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.warning(f"RSS parse {label} failed: {e}")
        return []

    out: list[NewsItem] = []
    # RSS 2.0: <rss><channel><item>
    for item in root.iter("item"):
        if len(out) >= max_items:
            break
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub = item.findtext("pubDate")
        out.append(_make_item(label, title, link, desc, pub))

    # Atom: <feed><entry>
    if not out:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
            if len(out) >= max_items:
                break
            title = (entry.findtext("atom:title", default="", namespaces=ns)
                     or "").strip()
            link_el = entry.find("atom:link", namespaces=ns)
            link = link_el.get("href") if link_el is not None else ""
            desc = (entry.findtext("atom:summary", default="",
                                    namespaces=ns) or "").strip()
            pub = entry.findtext("atom:updated", default="", namespaces=ns)
            out.append(_make_item(label, title, link, desc, pub))
    return out


def _make_item(label: str, title: str, link: str, desc: str,
                pub_str: str | None) -> NewsItem:
    pub_dt = _parse_pub_date(pub_str)
    raw_id = hashlib.sha256(
        f"{link}|{title}".encode("utf-8", errors="replace")
    ).hexdigest()[:24]
    return NewsItem(
        source=label,
        title=title,
        url=link,
        published_at=pub_dt,
        summary=desc[:500],
        raw_id=raw_id,
    )


def _parse_pub_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        # RFC 822 format common in RSS 2.0
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    except (TypeError, ValueError):
        pass
    # Atom uses ISO 8601
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(
            UTC
        )
    except (TypeError, ValueError):
        return None
