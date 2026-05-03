"""Tests for the heartbeat / dead-man's-switch helper.

Sprint A4 audit fix: ensures the orchestrator emits success+fail
pings to healthchecks.io after each cycle so silent process death
is alertable. These tests stub requests.post so no network calls
happen.
"""
from __future__ import annotations

from unittest.mock import patch

from common import heartbeat


def test_ping_no_op_when_env_var_unset(monkeypatch):
    """No HEALTHCHECKS_PING_URL_ORCHESTRATOR → silently skip,
    return False, no HTTP call."""
    monkeypatch.delenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR", raising=False)
    with patch("common.heartbeat.requests.post") as mock_post:
        result = heartbeat.ping_success("orchestrator")
    assert result is False
    assert mock_post.call_count == 0


def test_ping_success_calls_bare_url(monkeypatch):
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123")
    with patch("common.heartbeat.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        result = heartbeat.ping_success("orchestrator", message="ok")
    assert result is True
    args, kwargs = mock_post.call_args
    assert args[0] == "https://hc.io/abc-123"
    assert kwargs["data"] == b"ok"


def test_ping_fail_appends_fail_path(monkeypatch):
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123")
    with patch("common.heartbeat.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        heartbeat.ping_fail("orchestrator", message="cycle errored")
    assert mock_post.call_args[0][0] == "https://hc.io/abc-123/fail"


def test_ping_start_appends_start_path(monkeypatch):
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123")
    with patch("common.heartbeat.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        heartbeat.ping_start("orchestrator")
    assert mock_post.call_args[0][0] == "https://hc.io/abc-123/start"


def test_ping_strips_trailing_slash(monkeypatch):
    """Common pasted-URL footgun — strip trailing /."""
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123/")
    with patch("common.heartbeat.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        heartbeat.ping_fail("orchestrator")
    assert mock_post.call_args[0][0] == "https://hc.io/abc-123/fail"


def test_ping_unknown_component_skips(monkeypatch):
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc")
    with patch("common.heartbeat.requests.post") as mock_post:
        result = heartbeat.ping_success("not_a_component")
    assert result is False
    assert mock_post.call_count == 0


def test_ping_swallows_request_failure(monkeypatch):
    """Heartbeat MUST NOT raise — that would crash the cycle. Network
    errors return False and let the cycle continue."""
    import requests
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123")
    with patch("common.heartbeat.requests.post",
                side_effect=requests.ConnectionError("DNS fail")):
        result = heartbeat.ping_success("orchestrator")
    assert result is False    # didn't raise


def test_ping_swallows_non_2xx(monkeypatch):
    monkeypatch.setenv("HEALTHCHECKS_PING_URL_ORCHESTRATOR",
                        "https://hc.io/abc-123")
    with patch("common.heartbeat.requests.post") as mock_post:
        mock_post.return_value.status_code = 503
        result = heartbeat.ping_success("orchestrator")
    assert result is False


def test_ping_url_env_includes_all_components():
    """Defensive: every timer we ship must have a heartbeat slot."""
    expected = {"orchestrator", "scouts", "dashboard", "db_backup"}
    assert set(heartbeat.PING_URL_ENV.keys()) == expected
