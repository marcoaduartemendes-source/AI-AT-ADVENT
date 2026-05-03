"""Tests for src/common/http_cache.py — the in-process TTL cache used
by every scout and many strategies for public-API GETs.

Coverage gap: this 60-line helper carries cache invalidation logic
that, if broken, causes silent staleness. A failing TTL means
strategies act on hour-old prices; a failing cache key means we
hammer the same endpoint dozens of times per cycle.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import requests

from common.http_cache import cached_get, clear_cache


def _setup_response(text="ok", code=200, json_body=None):
    r = MagicMock()
    r.status_code = code
    r.raise_for_status.side_effect = (
        None if 200 <= code < 300
        else requests.HTTPError(f"HTTP {code}")
    )
    r.json.return_value = json_body if json_body is not None else {"ok": 1}
    return r


def setup_function():
    clear_cache()


def test_first_call_hits_network(monkeypatch):
    with patch("common.http_cache.requests.get",
                return_value=_setup_response()) as mock_get:
        out = cached_get("https://example.com/x", params={"a": 1})
    assert out == {"ok": 1}
    assert mock_get.call_count == 1


def test_cached_call_does_not_hit_network():
    with patch("common.http_cache.requests.get",
                return_value=_setup_response(json_body={"v": 1})) as mock_get:
        cached_get("https://x.com/a", ttl_seconds=60)
        cached_get("https://x.com/a", ttl_seconds=60)
        cached_get("https://x.com/a", ttl_seconds=60)
    # Only one network call across 3 invocations
    assert mock_get.call_count == 1


def test_different_params_different_cache_keys():
    with patch("common.http_cache.requests.get",
                side_effect=[_setup_response(json_body={"v": 1}),
                              _setup_response(json_body={"v": 2})]) as mock_get:
        a = cached_get("https://x.com/a", params={"page": 1})
        b = cached_get("https://x.com/a", params={"page": 2})
    assert a == {"v": 1}
    assert b == {"v": 2}
    assert mock_get.call_count == 2


def test_zero_ttl_bypasses_cache():
    with patch("common.http_cache.requests.get",
                return_value=_setup_response()) as mock_get:
        cached_get("https://x.com/a", ttl_seconds=0)
        cached_get("https://x.com/a", ttl_seconds=0)
    # Both calls hit the network — no caching when ttl <= 0
    assert mock_get.call_count == 2


def test_expired_entry_refetches():
    with patch("common.http_cache.requests.get",
                side_effect=[_setup_response(json_body={"v": 1}),
                              _setup_response(json_body={"v": 2})]) as mock_get:
        out1 = cached_get("https://x.com/a", ttl_seconds=1)
        time.sleep(1.1)
        out2 = cached_get("https://x.com/a", ttl_seconds=1)
    assert out1 == {"v": 1}
    assert out2 == {"v": 2}
    assert mock_get.call_count == 2


def test_http_error_returns_none_no_cache():
    """5xx response should return None and NOT poison the cache."""
    with patch("common.http_cache.requests.get",
                side_effect=requests.ConnectionError("boom")):
        out = cached_get("https://x.com/a", ttl_seconds=60)
    assert out is None
    # Subsequent successful call should not be blocked
    with patch("common.http_cache.requests.get",
                return_value=_setup_response()):
        ok = cached_get("https://x.com/a", ttl_seconds=60)
    assert ok == {"ok": 1}


def test_clear_cache_clears():
    with patch("common.http_cache.requests.get",
                return_value=_setup_response()):
        cached_get("https://x.com/a", ttl_seconds=60)
    clear_cache()
    # After clear, the next call must hit network again
    with patch("common.http_cache.requests.get",
                return_value=_setup_response(json_body={"v": "fresh"})) as mock_get:
        out = cached_get("https://x.com/a", ttl_seconds=60)
    assert out == {"v": "fresh"}
    assert mock_get.call_count == 1


def test_headers_pass_through():
    """User-Agent / Accept must reach the underlying request."""
    captured = {}
    def _fake_get(url, params=None, headers=None, timeout=None):
        captured.update(headers=headers, params=params)
        return _setup_response()
    with patch("common.http_cache.requests.get",
                side_effect=_fake_get):
        cached_get(
            "https://x.com/a",
            headers={"User-Agent": "test/1.0"},
        )
    assert captured["headers"]["User-Agent"] == "test/1.0"
