"""Polymarket public API — second prediction-market venue.

Why this matters:
    The kalshi_calibration_arb strategy bets on favorite-longshot bias
    *within* Kalshi. But cross-venue divergence (Kalshi YES @ 0.65 vs
    Polymarket YES @ 0.55 on the same underlying event) is a much
    cleaner arbitrage — when two liquid venues disagree, one of them
    is wrong.

    Adding Polymarket gives us:
      1. An independent reference price for sanity-checking Kalshi.
      2. A direct cross-venue arb signal (when liquidity supports it).
      3. Coverage of events Kalshi doesn't list (US elections,
         crypto/sports/world events).

Free-tier sufficiency:
    Polymarket exposes their full Gamma API + CLOB API publicly with
    no auth and no rate limits at our cadence. There is no paid tier
    that adds value for binary signal extraction — the public endpoints
    ARE the source of truth for the market they make.

Limitations:
    Polymarket is on-chain (Polygon network). Trading there from this
    bot requires wallet integration we don't have today, so this
    wrapper is read-only — we use Polymarket prices as a *reference*
    against which Kalshi mispricings can be sized more confidently.
    A future iteration could add on-chain execution via the CLOB
    relayer.

Endpoints (free, no auth):
    GET https://gamma-api.polymarket.com/markets?active=true&closed=false
    GET https://gamma-api.polymarket.com/markets/{id}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


GAMMA_BASE = "https://gamma-api.polymarket.com"
DEFAULT_CACHE_TTL = 5 * 60   # 5 minutes — prices move fast, but our
                              # cadence is 30 min so 5 min is fine


@dataclass
class PolymarketContract:
    """One open Polymarket binary contract (YES side)."""
    market_id: str               # Polymarket internal ID
    question: str                # human-readable title
    end_date: datetime | None    # contract resolution date
    yes_price: float             # 0-1 last YES price
    no_price: float              # 0-1
    volume_24h_usd: float
    liquidity_usd: float         # on-chain liquidity proxy
    category: str                # "Politics", "Crypto", "Sports", etc.
    raw: dict


class PolymarketClient:
    """Free public Polymarket Gamma client. No auth required."""

    def __init__(
        self,
        cache_dir: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.cache_dir = Path(
            cache_dir or os.environ.get(
                "POLYMARKET_CACHE_DIR", "data/cache/polymarket"
            )
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        """Always True — Polymarket public API needs no key."""
        return True

    # ── Cache helpers ─────────────────────────────────────────────────

    def _cache_key(self, path: str, params: dict) -> str:
        scrubbed = dict(sorted(params.items()))
        h = hashlib.sha256(
            f"{path}|{json.dumps(scrubbed, sort_keys=True)}".encode()
        ).hexdigest()[:16]
        return f"{path.strip('/').replace('/', '_')}_{h}"

    def _read_cache(self, key: str, ttl_seconds: int):
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > ttl_seconds:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, key: str, data) -> None:
        p = self.cache_dir / f"{key}.json"
        try:
            p.write_text(json.dumps(data), encoding="utf-8")
        except OSError as e:
            logger.debug(f"Polymarket cache write {key} failed: {e}")

    def _get(self, path: str, params: dict | None = None,
              cache_ttl_seconds: int = DEFAULT_CACHE_TTL) -> list | dict:
        params = dict(params or {})
        cache_key = self._cache_key(path, params)
        cached = self._read_cache(cache_key, cache_ttl_seconds)
        if cached is not None:
            return cached
        url = f"{GAMMA_BASE}{path}"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(
                    f"Polymarket {path} HTTP {resp.status_code}: "
                    f"{resp.text[:200]}"
                )
                return [] if path.endswith("markets") else {}
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Polymarket GET {path} failed: {e}")
            return [] if path.endswith("markets") else {}
        self._write_cache(cache_key, data)
        return data

    # ── Public API ────────────────────────────────────────────────────

    def active_markets(
        self,
        limit: int = 200,
        category: str | None = None,
        min_volume_usd: float = 0.0,
    ) -> list[PolymarketContract]:
        """Active, non-closed binary markets. Returns up to `limit` rows."""
        params: dict = {"active": "true", "closed": "false",
                         "limit": min(limit, 500)}
        if category:
            params["category"] = category
        data = self._get("/markets", params)
        rows = data if isinstance(data, list) else (data.get("markets") or [])
        out: list[PolymarketContract] = []
        for m in rows:
            try:
                vol = _coerce_float(m.get("volume24hr") or m.get("volume24h")
                                     or m.get("volume", 0)) or 0.0
                if vol < min_volume_usd:
                    continue
                yes_px = _coerce_float(m.get("lastTradePrice") or
                                        m.get("yesPrice"))
                if yes_px is None:
                    # outcomes might be an array
                    outcomes = m.get("outcomes") or []
                    if (isinstance(outcomes, list) and len(outcomes) >= 2
                            and isinstance(outcomes[0], dict)):
                        yes_px = _coerce_float(outcomes[0].get("price"))
                if yes_px is None:
                    continue
                if yes_px > 1:
                    yes_px = yes_px / 100
                no_px = 1.0 - yes_px

                end_iso = (m.get("endDate") or m.get("end_date") or
                           m.get("resolutionDate"))
                end_dt = _parse_iso(end_iso)

                out.append(PolymarketContract(
                    market_id=str(m.get("id") or m.get("conditionId") or
                                   m.get("slug") or ""),
                    question=m.get("question") or m.get("title") or "",
                    end_date=end_dt,
                    yes_price=yes_px,
                    no_price=no_px,
                    volume_24h_usd=vol,
                    liquidity_usd=_coerce_float(
                        m.get("liquidity") or m.get("liquidityUsd")
                    ) or 0.0,
                    category=m.get("category") or "",
                    raw=m,
                ))
            except (TypeError, ValueError, KeyError):
                continue
        return out

    def find_kalshi_match(
        self,
        kalshi_title: str,
        polymarket_markets: list[PolymarketContract] | None = None,
    ) -> PolymarketContract | None:
        """Heuristic: find a Polymarket contract that likely covers the
        same underlying event as a Kalshi market title.

        Token-overlap match — if ≥ 50% of the meaningful tokens in the
        Kalshi title appear in a Polymarket question, it's a candidate.
        Returns the highest-volume match, or None.
        """
        if polymarket_markets is None:
            polymarket_markets = self.active_markets(limit=300)
        if not polymarket_markets:
            return None
        kalshi_tokens = _meaningful_tokens(kalshi_title)
        if len(kalshi_tokens) < 2:
            return None
        best: PolymarketContract | None = None
        best_score = 0.0
        for pm in polymarket_markets:
            pm_tokens = _meaningful_tokens(pm.question)
            if not pm_tokens:
                continue
            overlap = len(kalshi_tokens & pm_tokens)
            score = overlap / len(kalshi_tokens)
            if score >= 0.5 and pm.volume_24h_usd > (
                    best.volume_24h_usd if best else 0):
                best = pm
                best_score = score
        if best is not None:
            logger.debug(
                f"Polymarket match for '{kalshi_title[:40]}': "
                f"'{best.question[:40]}' (overlap {best_score:.0%})"
            )
        return best


_STOPWORDS = {"the", "a", "an", "of", "in", "on", "at", "for", "to",
              "by", "be", "is", "are", "was", "were", "will", "would",
              "and", "or", "but", "if", "which", "this", "that", "with",
              "as", "from", "than", "then", "any", "all"}


def _meaningful_tokens(s: str) -> set[str]:
    if not s:
        return set()
    raw = s.lower().replace("?", "").replace(",", "").replace(".", "")
    return {t for t in raw.split() if len(t) > 2 and t not in _STOPWORDS}


def _coerce_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_iso(s) -> datetime | None:
    if not s:
        return None
    try:
        # Handle trailing Z
        s2 = s.replace("Z", "+00:00") if isinstance(s, str) else s
        return datetime.fromisoformat(s2).astimezone(UTC)
    except (TypeError, ValueError):
        return None
