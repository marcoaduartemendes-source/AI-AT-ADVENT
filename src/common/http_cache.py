"""Tiny TTL cache for idempotent GET requests.

Several scouts and strategies poll the same public endpoints (Coinbase
/products listing, Yahoo VIX, etc.) within the same cycle. Without
caching, each cycle generates 5–10 redundant HTTP calls. With a 30-second
in-process TTL we de-dupe within a cycle but stay fresh across cycles.

Not thread-safe; not persisted across restarts. Used only for read-only
public market data — never for authenticated calls.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests

# (url, frozenset(params.items())) -> (expires_at, response_json)
_CACHE: Dict[Tuple[str, frozenset], Tuple[float, Any]] = {}
_DEFAULT_TTL = 30.0


def cached_get(
    url: str,
    *,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: float = 15.0,
    ttl_seconds: float = _DEFAULT_TTL,
) -> Optional[Dict]:
    """Cached GET returning parsed JSON. Returns None on HTTP error.

    Bypasses cache (still issues the request) when ttl_seconds <= 0.
    """
    key = (url, frozenset((params or {}).items()))
    now = time.time()
    if ttl_seconds > 0:
        cached = _CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    if ttl_seconds > 0:
        _CACHE[key] = (now + ttl_seconds, data)
        # Cap cache size at 100 entries — drop oldest
        if len(_CACHE) > 100:
            oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[:len(_CACHE) - 100]
            for k, _ in oldest:
                _CACHE.pop(k, None)
    return data


def clear_cache() -> None:
    _CACHE.clear()
