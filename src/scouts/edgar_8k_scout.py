"""EDGAR 8-K scout — an LLM reads material corporate events in real time.

THE CREATIVE EDGE (2025-26 state of the art):
    ~3,000 8-K filings hit EDGAR weekly. Each is a regulator-mandated
    disclosure of a MATERIAL event (M&A agreement, guidance change,
    CEO departure, restructuring) and the price drift after the filing
    runs hours-to-days — far too slow to need HFT infrastructure, far
    too high-volume for human analysts to cover the tail. That gap is
    exactly the shape of an LLM: this scout has Claude (Haiku — cheap,
    fast) read each fresh 8-K headline + item list and score direction.

    Academic support: LLM-scored news/filings tone predicts returns
    (Lopez-Lira & Tang 2023 "Can ChatGPT Forecast Stock Price
    Movements?"; a fast-growing 2024-26 literature). The repo already
    carries the `anthropic` dependency and the droplet an API key — a
    structural asset most retail stacks don't have.

COST CONTROL (hard bounds):
    • Haiku only, ≤ ~400 tokens/filing → ≈ $0.0005 per filing.
    • ≤ MAX_LLM_CALLS_PER_SCAN per scan (default 25).
    • Accession-level cache — a filing is scored exactly once, ever.
    • No ANTHROPIC_API_KEY → scout emits nothing (the LLM IS the
      signal; a keyless heuristic would just be noise).

DATA:
    EDGAR "current events" Atom feed (free, near-real-time) + the
    company_tickers.json CIK→ticker map (cached 24h).
"""
from __future__ import annotations

import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)

_UA = ("ai-at-advent 8-K research bot "
       "(contact: marcoaduartemendes@gmail.com)")

_FEED_URL = ("https://www.sec.gov/cgi-bin/browse-edgar?"
             "action=getcurrent&type=8-K&company=&dateb=&owner=include"
             "&count=60&output=atom")
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

_CACHE_DIR = Path("data/cache/edgar8k")
_SCORED_CACHE = _CACHE_DIR / "scored.json"
_TICKER_CACHE = _CACHE_DIR / "tickers.json"
_TICKER_TTL_S = 24 * 3600

MAX_LLM_CALLS_PER_SCAN = 25
LLM_MODEL = "claude-haiku-4-5-20251001"

_PROMPT = """You are a buy-side event analyst. An SEC Form 8-K was just \
filed. Based ONLY on the headline and item codes, classify the likely \
1-3 day stock price impact.

Company: {company}
Headline: {title}
8-K items: {items}

Item code cheat-sheet: 1.01 material agreement, 1.03 bankruptcy, \
2.01 completed acquisition/disposal, 2.02 results of operations, \
2.05 exit/disposal costs, 3.01 delisting notice, 4.01 auditor change, \
4.02 non-reliance on prior financials (restatement), 5.02 officer/\
director departure or appointment, 7.01 Reg FD, 8.01 other.

Respond with ONLY a JSON object, no other text:
{{"direction": "VERY_BULLISH"|"BULLISH"|"NEUTRAL"|"BEARISH"|"VERY_BEARISH", \
"confidence": 0.0-1.0, "rationale": "<one short sentence>"}}"""


def _fetch(url: str, timeout: int = 12) -> bytes | None:
    try:
        req = Request(url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (URLError, TimeoutError, OSError) as e:
        logger.warning(f"[edgar_8k] fetch failed {url[:60]}: {e}")
        return None


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_json(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[edgar_8k] cache write failed: {e}")


def _cik_to_ticker_map() -> dict[str, str]:
    """CIK (int-string, no leading zeros) → ticker. Cached 24h."""
    cached = _load_json(_TICKER_CACHE)
    if cached:
        try:
            age = (datetime.now(UTC).timestamp()
                   - float(cached.get("_fetched_at", 0)))
            if age < _TICKER_TTL_S and cached.get("map"):
                return cached["map"]
        except (TypeError, ValueError):
            pass
    raw = _fetch(_TICKER_MAP_URL)
    if not raw:
        return cached.get("map", {}) if cached else {}
    try:
        data = json.loads(raw)
        # Shape: {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
        out = {str(int(v["cik_str"])): str(v["ticker"]).upper()
               for v in data.values()
               if v.get("cik_str") and v.get("ticker")}
        _save_json(_TICKER_CACHE, {
            "_fetched_at": datetime.now(UTC).timestamp(), "map": out})
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[edgar_8k] ticker map parse failed: {e}")
        return cached.get("map", {}) if cached else {}


def _parse_feed(raw: bytes) -> list[dict]:
    """Atom feed → [{accession, cik, company, title, items, filed}]."""
    out: list[dict] = []
    try:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = (entry.findtext("a:title", "", ns) or "").strip()
            summary = (entry.findtext("a:summary", "", ns) or "")
            link_el = entry.find("a:link", ns)
            href = link_el.get("href", "") if link_el is not None else ""
            updated = (entry.findtext("a:updated", "", ns) or "").strip()
            # CIK from the index URL; accession from the same.
            cik_m = re.search(r"/data/(\d+)/", href)
            acc_m = re.search(r"/(\d{10}-?\d{2}-?\d{6})", href)
            items_m = re.findall(r"Item\s+(\d+\.\d+)", summary or title)
            if not cik_m:
                continue
            out.append({
                "cik": str(int(cik_m.group(1))),
                "accession": (acc_m.group(1).replace("-", "")
                              if acc_m else href[-24:]),
                "company": re.sub(r"^8-K\s*-\s*", "", title)[:120],
                "title": title[:200],
                "items": items_m,
                "filed": updated,
            })
    except ET.ParseError as e:
        logger.warning(f"[edgar_8k] feed parse failed: {e}")
    return out


def _llm_score(company: str, title: str, items: list[str]) -> dict | None:
    """One Haiku call → {"direction", "confidence", "rationale"}."""
    try:
        import anthropic
        client = anthropic.Anthropic()   # key from ANTHROPIC_API_KEY
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": _PROMPT.format(
                    company=company, title=title,
                    items=", ".join(items) or "unknown"),
            }],
        )
        text = "".join(
            b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        if parsed.get("direction") not in (
                "VERY_BULLISH", "BULLISH", "NEUTRAL",
                "BEARISH", "VERY_BEARISH"):
            return None
        parsed["confidence"] = max(0.0, min(1.0,
                                            float(parsed.get("confidence", 0))))
        return parsed
    except Exception as e:  # noqa: BLE001 — LLM failures must not break scouts
        logger.warning(f"[edgar_8k] LLM scoring failed: {e}")
        return None


class Edgar8KScout(ScoutAgent):
    """LLM-scores fresh 8-K filings; publishes directional event signals."""

    name = "edgar_8k"

    def scan(self) -> list[ScoutSignal]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.debug("[edgar_8k] no ANTHROPIC_API_KEY — scout idle")
            return []
        raw = _fetch(_FEED_URL)
        if not raw:
            return []
        filings = _parse_feed(raw)
        if not filings:
            return []
        tickers = _cik_to_ticker_map()
        scored_cache = _load_json(_SCORED_CACHE)

        signals: list[ScoutSignal] = []
        llm_calls = 0
        for f in filings:
            acc = f["accession"]
            tic = tickers.get(f["cik"])
            if not tic:
                continue            # private/foreign filer — untradeable
            if acc in scored_cache:
                score = scored_cache[acc]
            else:
                if llm_calls >= MAX_LLM_CALLS_PER_SCAN:
                    continue
                score = _llm_score(f["company"], f["title"], f["items"])
                llm_calls += 1
                if score is None:
                    continue
                scored_cache[acc] = score
            direction = score.get("direction", "NEUTRAL")
            if direction == "NEUTRAL":
                continue
            signals.append(ScoutSignal(
                venue="alpaca",
                signal_type="llm_8k_event",
                payload={
                    "ticker": tic,
                    "direction": direction,
                    "confidence": score.get("confidence", 0.0),
                    "rationale": score.get("rationale", "")[:200],
                    "items": f["items"],
                    "accession": acc,
                    "filed": f["filed"],
                },
                # The 8-K drift horizon is short — 2 trading days.
                ttl_seconds=2 * 24 * 3600,
            ))
        # Bound the cache (keep the freshest ~2000 accessions).
        if len(scored_cache) > 2000:
            scored_cache = dict(list(scored_cache.items())[-1500:])
        _save_json(_SCORED_CACHE, scored_cache)
        if signals:
            logger.info(
                f"[edgar_8k] {len(signals)} directional 8-K signals "
                f"({llm_calls} new LLM scores)")
        return signals
