"""Tests for the Supabase REST client.

Stubs requests.{post,patch,get} so we test the wire-format payloads
without touching a real Supabase instance. Confirms that:

  - Without env vars, every method returns False (graceful no-op).
  - insert_trade posts the right table + JSON body + auth headers.
  - update_trade_fill patches by order_id with the right payload.
  - upsert_strategy_state uses the merge-duplicates header.
  - Network failures never raise.
"""
from __future__ import annotations


import requests

from common.supabase_store import SupabaseStore


# ─── Fake response helpers ────────────────────────────────────────────


class _FakeResp:
    def __init__(self, status_code=201, body=None, text="ok"):
        self.status_code = status_code
        self._body = body or []
        self.text = text

    def json(self):
        return self._body


def test_unconfigured_store_returns_false_everywhere():
    """No URL/key → every method returns False, never calls HTTP."""
    store = SupabaseStore(url="", service_key="")
    assert not store.is_configured()
    assert store.insert_trade({"strategy": "x"}) is False
    assert store.update_trade_fill(
        order_id="o1", price=10, quantity=1,
        amount_usd=10, pnl_usd=None,
    ) is False
    assert store.insert_equity_snapshot(equity_usd=100, timestamp="t") is False


def test_insert_trade_posts_correct_table_and_payload(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResp(status_code=201)

    monkeypatch.setattr(requests, "post", _fake_post)
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    row = {"strategy": "tsmom_etf", "side": "BUY"}
    assert store.insert_trade(row) is True
    assert captured["url"] == "https://x.supabase.co/rest/v1/trades"
    assert captured["headers"]["apikey"] == "k"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["json"] == row


def test_update_trade_fill_uses_order_id_query(monkeypatch):
    captured = {}

    def _fake_patch(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResp(status_code=204)

    monkeypatch.setattr(requests, "patch", _fake_patch)
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    ok = store.update_trade_fill(
        order_id="abc-123",
        price=725.0, quantity=10, amount_usd=7250, pnl_usd=50.0,
        fill_status="FILLED",
    )
    assert ok is True
    assert "order_id=eq.abc-123" in captured["url"]
    assert captured["json"]["price"] == 725.0
    assert captured["json"]["pnl_usd"] == 50.0
    assert captured["json"]["fill_status"] == "FILLED"


def test_upsert_strategy_state_uses_merge_duplicates_header(monkeypatch):
    captured = {}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResp(status_code=204)

    monkeypatch.setattr(requests, "post", _fake_post)
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    ok = store.upsert_strategy_state({
        "name": "tsmom_etf", "state": "ACTIVE",
        "updated_at": "2026-05-01T00:00:00Z",
    })
    assert ok is True
    # Verify the upsert header is present
    prefer = captured["headers"]["Prefer"]
    assert "resolution=merge-duplicates" in prefer


def test_network_failure_returns_false(monkeypatch):
    """ConnectionError must not propagate. Trading depends on this."""
    def _exploding_post(*a, **kw):
        raise requests.ConnectionError("supabase unreachable")
    monkeypatch.setattr(requests, "post", _exploding_post)
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    # No exception, just False
    assert store.insert_trade({"strategy": "x"}) is False


def test_5xx_returns_false_and_does_not_raise(monkeypatch):
    """Supabase returning HTTP 500 → False, no exception."""
    monkeypatch.setattr(requests, "post",
                         lambda *a, **kw: _FakeResp(status_code=500, text="server down"))
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    assert store.insert_trade({"strategy": "x"}) is False


def test_ensure_schema_detects_missing_table(monkeypatch):
    """If `trades` table doesn't exist (schema not loaded), the
    probe returns False and logs a clear hint."""
    monkeypatch.setattr(requests, "get",
                         lambda *a, **kw: _FakeResp(status_code=404, text="not found"))
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    assert store.ensure_schema() is False


def test_ensure_schema_succeeds_when_table_exists(monkeypatch):
    monkeypatch.setattr(requests, "get",
                         lambda *a, **kw: _FakeResp(status_code=200, body=[]))
    store = SupabaseStore(url="https://x.supabase.co", service_key="k")
    assert store.ensure_schema() is True
