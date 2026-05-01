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

import json
import logging
import os
import smtplib
import socket
import ssl
import time
from email.mime.text import MIMEText
from threading import Lock

import requests

logger = logging.getLogger(__name__)

# Per-process rate limit so a runaway loop doesn't spam the channel.
# Allow at most N alerts per ~minute; drop the rest with a warning.
_RATE_LIMIT_PER_60S = 10
_recent_calls: list[float] = []
_lock = Lock()


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

    sender = os.environ.get("SMTP_FROM", "Marcoaduartemendes@gmail.com")
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
