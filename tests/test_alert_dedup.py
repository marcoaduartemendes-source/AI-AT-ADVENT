"""Tests for the cross-cycle alert dedup + ALERTS_MUTE panic switch.

Production bug fixed by this commit: the orchestrator's PnL-drift
sanity check fires `alert()` every cycle when drift > $1. The
in-memory rate limit (10/60s) is per-process; every systemd
oneshot is a NEW process, so the limit reset every 5 min and
the user got Pushover spam ~12×/hour.

Fix: persistent SQLite-backed dedup keyed by SHA256(text). Same
text within ALERT_DEDUP_COOLDOWN_SECONDS (default 1h) is
suppressed. Critical alerts always pass.
"""
from __future__ import annotations

from unittest.mock import patch

from common.alerts import _should_suppress, alert


def test_first_call_passes_through(monkeypatch):
    """Conftest sets cooldown=0 by default; this test enables dedup
    explicitly to verify the first-call-passes path."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "60")
    suppress, count = _should_suppress("hello world", "warning")
    assert suppress is False
    assert count == 1


def test_second_identical_call_suppressed(monkeypatch):
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "3600")
    _should_suppress("identical message", "warning")
    suppress, count = _should_suppress("identical message", "warning")
    assert suppress is True
    assert count == 2


def test_different_messages_not_suppressed(monkeypatch):
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "3600")
    s1, _ = _should_suppress("message A", "warning")
    s2, _ = _should_suppress("message B", "warning")
    assert s1 is False
    assert s2 is False


def test_critical_severity_always_passes(monkeypatch):
    """KILL switch alerts must NEVER be suppressed even if dedup says
    we just sent the same text. Safety > spam."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "3600")
    # Send a critical alert twice in rapid succession
    s1, _ = _should_suppress("KILL!!!", "critical")
    s2, _ = _should_suppress("KILL!!!", "critical")
    assert s1 is False
    assert s2 is False


def test_cooldown_expires(monkeypatch):
    """After cooldown, the same message fires again."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "1")
    import time
    _should_suppress("recurring drift $5", "warning")
    time.sleep(1.1)
    suppress, _ = _should_suppress("recurring drift $5", "warning")
    assert suppress is False


def test_cooldown_zero_disables_dedup(monkeypatch):
    """ALERT_DEDUP_COOLDOWN_SECONDS=0 → dedup is a no-op.
    Useful for the legacy alert tests that exercise the rate limiter
    independently."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "0")
    s1, _ = _should_suppress("same", "warning")
    s2, _ = _should_suppress("same", "warning")
    s3, _ = _should_suppress("same", "warning")
    assert s1 is False
    assert s2 is False
    assert s3 is False


def test_alerts_mute_silences_non_critical(monkeypatch):
    """ALERTS_MUTE=1 → alert() returns False for warnings/infos and
    never reaches any sink. Critical still fires."""
    monkeypatch.setenv("ALERTS_MUTE", "1")
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ALERT_EMAIL_TO", raising=False)
    monkeypatch.delenv("PUSHOVER_USER_KEY", raising=False)
    # Mock all sinks so we can detect calls
    with patch("common.alerts._send_pushover") as mock_pushover:
        with patch("common.alerts._send_webhook") as mock_webhook:
            with patch("common.alerts._send_email") as mock_email:
                # Warning is muted
                ok = alert("warning text", severity="warning")
                assert ok is False
                assert mock_pushover.call_count == 0
                assert mock_webhook.call_count == 0
                assert mock_email.call_count == 0


def test_alerts_mute_does_not_silence_critical(monkeypatch):
    """Even with ALERTS_MUTE=1, KILL switch alerts must fire."""
    monkeypatch.setenv("ALERTS_MUTE", "1")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "test")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test")
    with patch("common.alerts._send_pushover", return_value=True) as mock:
        ok = alert("KILL!!!", severity="critical")
    # Critical bypasses MUTE
    assert ok is True
    assert mock.call_count == 1


def test_dedup_count_appended_when_eventually_firing(monkeypatch):
    """When a previously-suppressed message eventually fires (after
    cooldown), the alert text includes a hint about how many were
    suppressed in between."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "1")
    monkeypatch.setenv("PUSHOVER_USER_KEY", "test")
    monkeypatch.setenv("PUSHOVER_APP_TOKEN", "test")

    with patch("common.alerts._send_pushover", return_value=True) as mock:
        # First fire — passes through
        alert("recurring drift", severity="warning")
        # Second + third — suppressed (within cooldown)
        alert("recurring drift", severity="warning")
        alert("recurring drift", severity="warning")
        import time
        time.sleep(1.1)
        # Fourth — cooldown expired → fires WITH the suppressed hint
        alert("recurring drift", severity="warning")

    # Pushover called exactly twice (first + post-cooldown)
    assert mock.call_count == 2
    # The second call's text should mention suppression
    second_call_text = mock.call_args_list[1].args[0]
    assert "suppressed" in second_call_text


def test_dedup_db_unwritable_falls_through(monkeypatch, tmp_path):
    """If the dedup DB can't be opened, alert() must STILL work
    (over-alert, not crash). Critical safety property."""
    monkeypatch.setenv("ALERT_DEDUP_COOLDOWN_SECONDS", "60")
    # Point at a path under a file (not a dir) — will fail to mkdir
    (tmp_path / "blocker").write_text("blocking file")
    monkeypatch.setenv(
        "ALERT_DEDUP_DB", str(tmp_path / "blocker" / "dedup.db"),
    )
    # Should not raise; simply returns (False, 0) and alert proceeds.
    suppress, count = _should_suppress("test", "warning")
    assert suppress is False
