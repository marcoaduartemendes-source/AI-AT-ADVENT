"""Out-of-band alerting via webhook OR email.

Simple, fire-and-forget. Not a circuit breaker — this is for human
notification, not automated remediation. Designed to never raise: if
delivery is misconfigured or the network is down, log and move on.
Trading must NOT block on alerting.

Each enabled sink fires independently — you can have all of them
configured and an alert will go out via every channel that's wired up.

Configure via env vars (any/all):

  Webhook (Slack/Discord/generic):
    ALERT_WEBHOOK_URL    — full webhook URL
    ALERT_WEBHOOK_TYPE   — "slack" | "discord" (auto-detected from URL)

  Email (Gmail SMTP — reuses the existing daily-digest creds):
    ALERT_EMAIL_TO       — comma-separated recipients. Examples:
                           "you@gmail.com"
                           "you@gmail.com,5551234567@vtext.com"
                           "5551234567@tmomail.net"
                           (carrier SMS gateways turn email → text)
    SMTP_PASSWORD        — Gmail App Password (already in your secrets)
    SMTP_FROM            — sender address (default: Marcoaduartemendes@gmail.com)

  Pushover (phone push notifications, $5 one-time):
    PUSHOVER_USER_KEY    — your User Key from https://pushover.net dashboard
    PUSHOVER_APP_TOKEN   — application token from a Pushover app you create
    PUSHOVER_DEVICE      — optional device name to target (default: all devices)

Usage:
    from common.alerts import alert
    alert("Phase 0 fill polling: drift > $50 detected", severity="warning")
    alert("KILL switch triggered — DD 16%", severity="critical")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import smtplib
import socket
import sqlite3
import ssl
import time
from email.mime.text import MIMEText
from pathlib import Path
from threading import Lock

import requests

logger = logging.getLogger(__name__)

# Per-process rate limit so a runaway loop doesn't spam the channel.
# Allow at most N alerts per ~minute; drop the rest with a warning.
_RATE_LIMIT_PER_60S = 10
_recent_calls: list[float] = []
_lock = Lock()


# ── Cross-cycle dedup ────────────────────────────────────────────────
# The orchestrator runs as systemd oneshot — every cycle is a NEW
# process, so the in-memory rate limit above resets every 5 min.
# That meant a persistent drift alert (or any recurring warning)
# fired ~12 times/hour to Pushover. The dedup cache below records
# (message_hash, last_sent_at) on disk and suppresses re-fires
# within ALERT_DEDUP_COOLDOWN_SECONDS (default 1 hour).
#
# Critical alerts (severity="critical") bypass dedup so a real
# kill-switch can't be muted by a recent drift warning.

def _cooldown_seconds() -> int:
    """Read ALERT_DEDUP_COOLDOWN_SECONDS at call time (not import
    time) so tests + runtime config changes take effect immediately."""
    try:
        return int(os.environ.get("ALERT_DEDUP_COOLDOWN_SECONDS", "3600"))
    except ValueError:
        return 3600


def _dedup_db_path() -> str:
    return os.environ.get("ALERT_DEDUP_DB", "data/alert_dedup.db")


def _dedup_db_conn() -> sqlite3.Connection | None:
    """Open the dedup DB; create the table on first call. Returns None
    when the path is unwritable (silently disables dedup — better to
    over-alert than to crash the alert path itself)."""
    try:
        path = _dedup_db_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(path)
        c.execute("""
            CREATE TABLE IF NOT EXISTS alert_dedup (
                msg_hash TEXT PRIMARY KEY,
                first_sent_at REAL NOT NULL,
                last_sent_at  REAL NOT NULL,
                count         INTEGER NOT NULL DEFAULT 1,
                last_text     TEXT
            )
        """)
        return c
    except (sqlite3.Error, OSError):
        return None


def _should_suppress(text: str, severity: str) -> tuple[bool, int]:
    """Returns (suppress, dedup_count). Critical alerts always pass.

    Cooldown is keyed by SHA256(text) so identical recurring messages
    are suppressed; a different message body bypasses dedup. Strategy
    name + drift amount changing → new hash → new alert.

    Set ALERT_DEDUP_COOLDOWN_SECONDS=0 to disable dedup entirely
    (every alert fires regardless of recent history). Tests use this
    to exercise the rate-limiter independently."""
    if severity == "critical":
        return (False, 0)
    cooldown = _cooldown_seconds()
    if cooldown <= 0:
        return (False, 0)    # dedup disabled
    msg_hash = hashlib.sha256(text.encode()).hexdigest()[:24]
    now = time.time()
    conn = _dedup_db_conn()
    if conn is None:
        return (False, 0)    # dedup unavailable → fall through
    try:
        with conn:
            row = conn.execute(
                "SELECT last_sent_at, count FROM alert_dedup "
                "WHERE msg_hash = ?",
                (msg_hash,),
            ).fetchone()
            if row:
                last_sent, count = row
                if now - last_sent < _cooldown_seconds():
                    # Suppress — bump the counter so we know how many
                    # we silenced when we eventually do fire.
                    conn.execute(
                        "UPDATE alert_dedup SET count = count + 1 "
                        "WHERE msg_hash = ?",
                        (msg_hash,),
                    )
                    return (True, count + 1)
                # Cooldown expired — reset and fire
                conn.execute(
                    "UPDATE alert_dedup SET last_sent_at = ?, "
                    "  count = 1, last_text = ? "
                    "WHERE msg_hash = ?",
                    (now, text[:500], msg_hash),
                )
                return (False, count + 1)
            # First time seeing this message
            conn.execute(
                "INSERT INTO alert_dedup "
                "(msg_hash, first_sent_at, last_sent_at, count, last_text) "
                "VALUES (?, ?, ?, 1, ?)",
                (msg_hash, now, now, text[:500]),
            )
            return (False, 1)
    except sqlite3.Error as e:
        logger.debug(f"alert dedup failed: {e}")
        return (False, 0)
    finally:
        conn.close()


def _classify_webhook(url: str) -> str:
    if "discord.com/api/webhooks" in url or "discordapp.com/api/webhooks" in url:
        return "discord"
    if "hooks.slack.com" in url:
        return "slack"
    return "unknown"


def _slack_payload(text: str, severity: str) -> dict:
    color = {
        "info":     "#7d8590",
        "warning":  "#daa520",
        "critical": "#cc3333",
    }.get(severity, "#7d8590")
    return {
        "attachments": [{
            "color": color,
            "fallback": text,
            "title": f"AI-AT-ADVENT [{severity}]",
            "text": text,
            "ts": int(time.time()),
        }],
    }


def _discord_payload(text: str, severity: str) -> dict:
    color = {
        "info":     0x7d8590,
        "warning":  0xdaa520,
        "critical": 0xcc3333,
    }.get(severity, 0x7d8590)
    return {
        "embeds": [{
            "title": f"AI-AT-ADVENT [{severity}]",
            "description": text[:2000],
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }],
    }


def alert(text: str, severity: str = "info", *, timeout: float = 5.0) -> bool:
    """Fan out an alert to every configured sink.

    Sinks are independent: webhook (Slack/Discord) and email both fire
    if both are configured; either is enough; neither just logs.

    Parameters
    ----------
    text : str
        Plain-text message.
    severity : "info" | "warning" | "critical"
        Color/badge in the rendered message. Crash-safe — unknown
        severities default to grey.
    timeout : float
        Per-sink timeout in seconds. Trading must not block waiting
        on Slack/SMTP — keep this short.

    Returns
    -------
    bool
        True if AT LEAST ONE sink delivered. False if none did
        (or none were configured). Caller should NOT raise on
        False — alerting is best-effort.
    """
    # ── Panic switch ─────────────────────────────────────────────
    # ALERTS_MUTE=1 silences EVERYTHING except critical kill-switch
    # alerts. Set this when the user is getting alert spam and needs
    # peace right now while we debug. Critical alerts still fire
    # because the kill switch is a safety mechanism we can't mute.
    if os.environ.get("ALERTS_MUTE", "").lower() in ("1", "true", "yes"):
        if severity != "critical":
            logger.info(f"[alert/{severity}] MUTED via ALERTS_MUTE: {text[:120]}")
            return False

    # ── Cross-cycle dedup ────────────────────────────────────────
    # Suppress identical messages within ALERT_DEDUP_COOLDOWN_SECONDS.
    # The orchestrator's PnL-drift sanity check fires the same alert
    # every 5 min when drift persists; without dedup that's 12/hour
    # of identical Pushover spam.
    suppress, dedup_count = _should_suppress(text, severity)
    cooldown = _cooldown_seconds()
    if suppress:
        logger.info(
            f"[alert/{severity}] DEDUPED (#{dedup_count}, cooldown "
            f"{cooldown}s): {text[:120]}"
        )
        return False
    # If we're firing a previously-suppressed message, append a hint
    # so the user knows how many were suppressed
    if dedup_count > 1:
        text = (f"{text}\n\n[suppressed {dedup_count - 1} identical "
                f"alert(s) in last {cooldown // 60}m]")

    # Rate limit at the entry point so a runaway loop can't spam
    # multiple sinks in parallel.
    now = time.time()
    with _lock:
        global _recent_calls
        _recent_calls = [t for t in _recent_calls if now - t < 60]
        if len(_recent_calls) >= _RATE_LIMIT_PER_60S:
            logger.warning(f"alert dropped (rate limit): {text[:80]}")
            return False
        _recent_calls.append(now)

    delivered = False

    # Sink 1: webhook
    if os.environ.get("ALERT_WEBHOOK_URL"):
        if _send_webhook(text, severity, timeout=timeout):
            delivered = True

    # Sink 2: email (Gmail SMTP, reusing existing daily-digest creds)
    if os.environ.get("ALERT_EMAIL_TO"):
        if _send_email(text, severity, timeout=timeout):
            delivered = True

    # Sink 3: Pushover (phone push)
    if (os.environ.get("PUSHOVER_USER_KEY")
            and os.environ.get("PUSHOVER_APP_TOKEN")):
        if _send_pushover(text, severity, timeout=timeout):
            delivered = True

    if not delivered:
        # Either nothing was configured, or everything failed. Either
        # way the message must not get lost — log it.
        logger.info(f"[alert/{severity}] {text}")
        # Last-resort escalation for critical alerts: ping the
        # Healthchecks dead-man's-switch /fail endpoint so its
        # alerting (separate auth + delivery path) takes over.
        # This is the right response when every in-process sink is
        # down (Pushover throttled + webhook stale + SMTP misconfig).
        if severity in ("critical", "kill"):
            try:
                from common.heartbeat import ping_fail
                ping_fail(
                    "orchestrator",
                    message=f"alert delivery failed: {text[:120]}",
                )
            except Exception as e:
                logger.warning(f"healthchecks fallback failed: {e}")
    return delivered


def _send_pushover(text: str, severity: str, *, timeout: float) -> bool:
    """Push notification via Pushover (https://pushover.net).

    Pushover priorities map cleanly to our severity levels:
      info     → 0 (normal)
      warning  → 1 (high — bypasses quiet hours, always notifies)
      critical → 2 (emergency — vibrates/sounds until ack'd)

    Priority 2 (emergency) requires `retry` and `expire` parameters
    by Pushover's API; we use 60s retry / 1h expire so a missed KILL
    switch alert keeps buzzing until you ack from the app.
    """
    user_key = os.environ.get("PUSHOVER_USER_KEY")
    app_token = os.environ.get("PUSHOVER_APP_TOKEN")
    if not user_key or not app_token:
        return False

    severity_to_priority = {"info": 0, "warning": 1, "critical": 2}
    priority = severity_to_priority.get(severity, 0)

    payload = {
        "token": app_token,
        "user": user_key,
        "title": f"AI-AT-ADVENT [{severity.upper()}]",
        "message": text[:1024],   # Pushover hard-caps at ~1024 chars
        "priority": priority,
    }
    if priority == 2:
        # Required by Pushover for emergency priority. The phone will
        # repeat the alert sound every 60s for up to 1h until the
        # user opens the app and acknowledges.
        payload["retry"] = 60
        payload["expire"] = 3600
    device = os.environ.get("PUSHOVER_DEVICE")
    if device:
        payload["device"] = device

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data=payload, timeout=timeout,
        )
        if resp.status_code != 200:
            logger.warning(f"Pushover HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        body = resp.json()
        if body.get("status") != 1:
            logger.warning(f"Pushover error: {body}")
            return False
        return True
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Pushover send failed: {type(e).__name__}: {e}")
        return False


def _send_webhook(text: str, severity: str, *, timeout: float) -> bool:
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return False
    kind = os.environ.get("ALERT_WEBHOOK_TYPE", "").lower() or _classify_webhook(url)
    if kind == "slack":
        payload = _slack_payload(text, severity)
    elif kind == "discord":
        payload = _discord_payload(text, severity)
    else:
        payload = {"text": f"[{severity}] {text}"}
        logger.info(f"alert: unknown webhook vendor {kind!r}; "
                    f"using minimal payload")
    try:
        resp = requests.post(
            url, data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code >= 300:
            logger.warning(f"alert webhook returned HTTP {resp.status_code}: "
                            f"{resp.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        logger.warning(f"alert webhook failed: {type(e).__name__}: {e}")
        return False


def _send_email(text: str, severity: str, *, timeout: float) -> bool:
    """Send via Gmail SMTP, reusing the SMTP_PASSWORD App Password
    that's already in the repo for the daily news digest.

    Recipients: comma-separated list in ALERT_EMAIL_TO. Each can be
    a regular email (you@gmail.com) OR a carrier SMS gateway address
    (5551234567@vtext.com etc) — Gmail won't care, and your phone
    will get a text.

    Subject line carries severity for filterable inboxes.
    Body is plain text — keep it short so SMS gateways don't truncate.
    """
    to_raw = os.environ.get("ALERT_EMAIL_TO", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    if not to_raw or not password:
        return False

    sender = os.environ.get("SMTP_FROM", "").strip()
    if not sender:
        # No default sender — forks of this repo shouldn't accidentally
        # try to send mail with someone else's From: header. Caller
        # must set SMTP_FROM explicitly. Auth would have rejected the
        # mismatched address anyway; this just makes the error clear.
        logger.debug("SMTP_FROM not set; skipping email alert")
        return False
    recipients = [r.strip() for r in to_raw.split(",") if r.strip()]
    if not recipients:
        return False

    # Carrier gateways drop attachments; keep it plain text and short.
    badge = {"critical": "[CRITICAL]", "warning": "[WARNING]"}.get(severity, "")
    subject = f"AI-AT-ADVENT {badge}".strip()
    body = text[:1400]   # SMS gateways often cap at ~160 chars; emails get more

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    # Try STARTTLS on port 587 first (Gmail standard). Fall back to
    # SSL/465 to handle the rare network where 587 is blocked.
    # Force IPv4 the same way the daily digest does — some hosts
    # don't support IPv6 to smtp.gmail.com.
    _real_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
        return _real_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

    socket.getaddrinfo = _ipv4_getaddrinfo
    try:
        try:
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=timeout) as s:
                s.starttls()
                s.login(sender, password)
                s.sendmail(sender, recipients, msg.as_string())
            return True
        except (smtplib.SMTPException, OSError) as e:
            logger.info(f"alert email STARTTLS failed: {e}; retrying SSL/465")
            try:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    "smtp.gmail.com", 465, context=ctx, timeout=timeout,
                ) as s:
                    s.login(sender, password)
                    s.sendmail(sender, recipients, msg.as_string())
                return True
            except Exception as e2:
                logger.warning(f"alert email failed both transports: {e2}")
                return False
    finally:
        socket.getaddrinfo = _real_getaddrinfo
