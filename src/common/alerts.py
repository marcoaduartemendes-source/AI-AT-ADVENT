"""Out-of-band alerting via Slack or Discord webhooks.

Simple, batched, fire-and-forget. Not a circuit breaker — this is for
human notification, not automated remediation. Designed to never
raise: if the webhook is misconfigured or the network is down,
log and move on. Trading must NOT block on alerting.

Configure via env vars:
  ALERT_WEBHOOK_URL   — full webhook URL
  ALERT_WEBHOOK_TYPE  — "slack" | "discord" (default: auto-detect from URL)

Usage:
    from common.alerts import alert
    alert("Phase 0 fill polling: drift > $50 detected", severity="warning")
    alert("KILL switch triggered — DD 16%", severity="critical")
"""
from __future__ import annotations

import json
import logging
import os
import time
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
    """Post a one-shot alert to the configured webhook.

    Parameters
    ----------
    text : str
        Plain-text message. Will be wrapped in a Slack attachment or
        Discord embed depending on which webhook is configured.
    severity : "info" | "warning" | "critical"
        Color/badge in the rendered message. Crash-safe — unknown
        severities default to grey.
    timeout : float
        HTTP timeout in seconds. Trading must not block waiting on
        Slack — keep this short.

    Returns
    -------
    bool
        True on successful delivery, False on any failure (rate
        limit, network, mis-config). Caller should NOT raise on
        False — alerting is best-effort.
    """
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        # No webhook configured — log only. This is the dev-loop
        # default and explicitly NOT an error.
        logger.info(f"[alert/{severity}] {text}")
        return False

    # Per-process rate limit
    now = time.time()
    with _lock:
        # Drop calls older than 60s
        global _recent_calls
        _recent_calls = [t for t in _recent_calls if now - t < 60]
        if len(_recent_calls) >= _RATE_LIMIT_PER_60S:
            logger.warning(f"alert dropped (rate limit): {text[:80]}")
            return False
        _recent_calls.append(now)

    kind = os.environ.get("ALERT_WEBHOOK_TYPE", "").lower() or _classify_webhook(url)
    if kind == "slack":
        payload = _slack_payload(text, severity)
    elif kind == "discord":
        payload = _discord_payload(text, severity)
    else:
        # Unknown vendor — ship a minimal {"text": "..."} blob and
        # hope. Most webhook receivers accept this fallback.
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
