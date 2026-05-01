"""Tests for common.alerts.

Webhook is fire-and-forget — must NEVER raise, even on misconfigured
URLs, network failures, or unknown vendors. Trading depends on this.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

import requests

from common import alerts as alerts_module


def test_no_webhook_returns_false_and_does_not_raise(monkeypatch):
    """Without ALERT_WEBHOOK_URL, alert() must return False and log
    only — must NOT raise. Trading code must not block on alerting."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    result = alerts_module.alert("test message", severity="info")
    assert result is False


def test_slack_webhook_posts_attachment(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
    monkeypatch.delenv("ALERT_WEBHOOK_TYPE", raising=False)
    captured = {}

    class _FakeResp:
        status_code = 200
        text = "ok"

    def _fake_post(url, data, headers, timeout):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    # Reset the per-process rate limiter so this test isn't affected
    # by other tests
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("KILL switch fired", severity="critical")
    assert result is True
    assert captured["url"] == "https://hooks.slack.com/services/T/B/X"
    body = json.loads(captured["data"])
    # Slack payload has `attachments` with color + title
    assert "attachments" in body
    att = body["attachments"][0]
    assert att["color"] == "#cc3333"
    assert "critical" in att["title"]
    assert att["text"] == "KILL switch fired"


def test_discord_webhook_posts_embed(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://discord.com/api/webhooks/123/abc")
    monkeypatch.delenv("ALERT_WEBHOOK_TYPE", raising=False)
    captured = {}

    class _FakeResp:
        status_code = 204
        text = ""

    def _fake_post(url, data, headers, timeout):
        captured["data"] = data
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("Drift detected", severity="warning")
    assert result is True
    body = json.loads(captured["data"])
    # Discord payload has `embeds` with color
    assert "embeds" in body
    emb = body["embeds"][0]
    assert emb["color"] == 0xdaa520
    assert emb["description"] == "Drift detected"


def test_webhook_failure_does_not_raise(monkeypatch):
    """If the webhook server is down, alert() returns False — never
    raises. Trading must keep going even if Slack is down."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.slack.com/T/B/X")

    def _exploding_post(*a, **kw):
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", _exploding_post)
    alerts_module._recent_calls.clear()
    result = alerts_module.alert("anything", severity="info")
    assert result is False    # explicit failure signal
    # Most importantly: did not raise


def test_rate_limiter_drops_excess_calls(monkeypatch):
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.slack.com/T/B/X")

    class _FakeResp:
        status_code = 200
        text = "ok"

    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp())
    alerts_module._recent_calls.clear()

    # First N succeed
    for _ in range(alerts_module._RATE_LIMIT_PER_60S):
        assert alerts_module.alert("flood", severity="info") is True
    # The next one should be dropped silently
    assert alerts_module.alert("flood", severity="info") is False


def test_unknown_webhook_uses_minimal_payload(monkeypatch):
    """A webhook URL we don't recognize falls back to {"text": "..."}
    — most generic webhook receivers (n8n, Zapier, etc.) accept this."""
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/hook")
    monkeypatch.delenv("ALERT_WEBHOOK_TYPE", raising=False)

    captured = {}

    class _FakeResp:
        status_code = 200
        text = "ok"

    def _fake_post(url, data, headers, timeout):
        captured["data"] = data
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    alerts_module._recent_calls.clear()

    alerts_module.alert("hi", severity="info")
    body = json.loads(captured["data"])
    assert body == {"text": "[info] hi"}
