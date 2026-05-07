"""Tests for common.alerts.

Webhook is fire-and-forget — must NEVER raise, even on misconfigured
URLs, network failures, or unknown vendors. Trading depends on this.
"""
from __future__ import annotations

import json


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
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)

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


# ─── Email sink ───────────────────────────────────────────────────────


class _FakeSMTP:
    """Captures sendmail calls so we can assert on them without
    actually sending email."""

    instances: list = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.starttls_called = False
        self.login_args = None
        self.sendmail_args = None
        _FakeSMTP.instances.append(self)

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def starttls(self): self.starttls_called = True
    def login(self, user, pw): self.login_args = (user, pw)
    def sendmail(self, from_, to_, msg): self.sendmail_args = (from_, to_, msg)


def _decode_body(raw: str) -> str:
    """Parse the wire-format MIME message and return the decoded plain
    body. MIMEText base64-encodes by default, so a substring match on
    `raw` doesn't work — we have to decode."""
    import email
    msg = email.message_from_string(raw)
    return msg.get_payload(decode=True).decode("utf-8", errors="replace")


def test_email_sink_sends_to_configured_recipient(monkeypatch):
    """ALERT_EMAIL_TO + SMTP_PASSWORD + SMTP_FROM → send via Gmail SMTP."""
    import smtplib
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "you@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "test-app-password")
    monkeypatch.setenv("SMTP_FROM", "alerts@example.com")

    _FakeSMTP.instances.clear()
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("KILL switch fired", severity="critical")
    assert result is True
    assert len(_FakeSMTP.instances) == 1
    inst = _FakeSMTP.instances[0]
    assert inst.host == "smtp.gmail.com"
    assert inst.port == 587
    assert inst.starttls_called
    sender, to_list, raw_msg = inst.sendmail_args
    assert "alerts@example.com" in sender
    assert to_list == ["you@gmail.com"]
    assert _decode_body(raw_msg) == "KILL switch fired"
    assert "[CRITICAL]" in raw_msg     # severity badge in subject header


def test_email_sink_supports_carrier_sms_gateway(monkeypatch):
    """Phone number @ carrier gateway routes through Gmail to a text
    message on your phone. e.g. 5551234567@vtext.com (Verizon)."""
    import smtplib
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "5551234567@vtext.com,you@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    monkeypatch.setenv("SMTP_FROM", "alerts@example.com")

    _FakeSMTP.instances.clear()
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    alerts_module._recent_calls.clear()

    alerts_module.alert("Drift $42", severity="warning")
    inst = _FakeSMTP.instances[0]
    sender, to_list, raw_msg = inst.sendmail_args
    assert to_list == ["5551234567@vtext.com", "you@gmail.com"]
    assert _decode_body(raw_msg) == "Drift $42"
    assert "[WARNING]" in raw_msg


def test_email_sink_skipped_without_smtp_from(monkeypatch):
    """SMTP_FROM unset → graceful skip (no hardcoded default sender)."""
    import smtplib
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "you@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    monkeypatch.delenv("SMTP_FROM", raising=False)
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    _FakeSMTP.instances.clear()
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("hi")
    assert result is False
    assert len(_FakeSMTP.instances) == 0


def test_email_sink_skipped_without_smtp_password(monkeypatch):
    """ALERT_EMAIL_TO set but no SMTP_PASSWORD → graceful skip."""
    import smtplib
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "you@gmail.com")
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    _FakeSMTP.instances.clear()
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("hi")
    assert result is False
    assert len(_FakeSMTP.instances) == 0


def test_both_sinks_fire_when_both_configured(monkeypatch):
    """Webhook AND email both fire when both configured. Returns True
    if at least one delivered."""
    import smtplib
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://hooks.slack.com/T/B/X")
    monkeypatch.setenv("ALERT_EMAIL_TO", "you@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    monkeypatch.setenv("SMTP_FROM", "alerts@example.com")

    class _FakeResp:
        status_code = 200
        text = "ok"
    monkeypatch.setattr(requests, "post", lambda *a, **kw: _FakeResp())

    _FakeSMTP.instances.clear()
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("dual sink test", severity="info")
    assert result is True
    # Email actually sent (webhook also did but no easy assertion)
    assert len(_FakeSMTP.instances) == 1


# ─── Pushover sink ────────────────────────────────────────────────────


def test_pushover_sink_sends_with_correct_priority(monkeypatch):
    """info → priority 0, warning → 1, critical → 2 (emergency)."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u-test")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "a-test")

    captured = {}

    class _FakeResp:
        status_code = 200
        text = '{"status": 1}'
        def json(self): return {"status": 1, "request": "abc"}

    def _fake_post(url, data, timeout):
        captured["url"] = url
        captured["data"] = data
        return _FakeResp()

    monkeypatch.setattr(requests, "post", _fake_post)
    alerts_module._recent_calls.clear()

    # Critical → priority 2 (emergency, requires retry/expire)
    result = alerts_module.alert("KILL switch fired", severity="critical")
    assert result is True
    assert captured["url"] == "https://api.pushover.net/1/messages.json"
    assert captured["data"]["priority"] == 2
    assert captured["data"]["retry"] == 60
    assert captured["data"]["expire"] == 3600
    assert captured["data"]["message"] == "KILL switch fired"
    assert "[CRITICAL]" in captured["data"]["title"]

    alerts_module._recent_calls.clear()
    captured.clear()
    alerts_module.alert("normal info", severity="info")
    assert captured["data"]["priority"] == 0
    assert "retry" not in captured["data"]   # only set for emergency

    alerts_module._recent_calls.clear()
    captured.clear()
    alerts_module.alert("warn", severity="warning")
    assert captured["data"]["priority"] == 1


def test_pushover_optional_device_targeting(monkeypatch):
    """If PUSHOVER_DEVICE is set, alert goes only to that device."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "a")
    monkeypatch.setenv("PUSHOVER_DEVICE", "iphone-marco")

    captured = {}

    class _FakeResp:
        status_code = 200
        def json(self): return {"status": 1}

    monkeypatch.setattr(requests, "post",
        lambda url, data, timeout: (captured.update({"data": data}) or _FakeResp()))
    alerts_module._recent_calls.clear()

    alerts_module.alert("hi", severity="info")
    assert captured["data"]["device"] == "iphone-marco"


def test_pushover_skipped_without_both_keys(monkeypatch):
    """Need BOTH user_key AND app_token; one alone is invalid."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.delenv("PUSHOVER_APP_TOKEN", raising=False)

    called = {"n": 0}

    def _fake_post(*a, **kw):
        called["n"] += 1
        raise AssertionError("Pushover should not have been called")

    monkeypatch.setattr(requests, "post", _fake_post)
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("hi")
    assert result is False
    assert called["n"] == 0


def test_pushover_failure_does_not_raise(monkeypatch):
    """Pushover API down → returns False, never raises."""
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.setenv("PUSHOVER_USER_KEY", "u")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "a")

    def _exploding_post(*a, **kw):
        raise requests.ConnectionError("pushover down")

    monkeypatch.setattr(requests, "post", _exploding_post)
    alerts_module._recent_calls.clear()
    result = alerts_module.alert("anything", severity="critical")
    assert result is False    # graceful


def test_email_sink_failure_does_not_raise(monkeypatch):
    """Gmail down → return False, never raise. Trading must not block."""
    import smtplib

    class _ExplodingSMTP(_FakeSMTP):
        def login(self, user, pw):
            raise smtplib.SMTPAuthenticationError(535, b"auth failed")

    class _ExplodingSSL(_FakeSMTP):
        def login(self, user, pw):
            raise smtplib.SMTPException("ssl too broke")

    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("ALERT_EMAIL_TO", "you@gmail.com")
    monkeypatch.setenv("SMTP_PASSWORD", "x")
    monkeypatch.setattr(smtplib, "SMTP", _ExplodingSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _ExplodingSSL)
    alerts_module._recent_calls.clear()

    result = alerts_module.alert("hi")
    assert result is False    # both transports failed, but no exception
