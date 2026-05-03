"""Heartbeat ping → healthchecks.io for the dead-man's switch.

Sprint A4 audit fix: pre-fix the orchestrator could die silently
(systemd failure, OOM kill, network partition) and the user had no
alert path — they'd notice via dashboard staleness ~15 minutes
after the next missed cycle. healthchecks.io gives us a true dead-
man's switch: every successful cycle pings a unique URL, and if no
ping arrives within the configured window, healthchecks pages us
via Pushover/email/webhook (all free at our scale).

Usage:
    1. Sign up at https://healthchecks.io (free for 20 checks).
    2. Create one check per timer:
         orchestrator-cycle  — grace 15m, period 5m
         scouts-cycle        — grace 60m, period 30m
         dashboard-build     — grace 30m, period 15m
    3. Add HEALTHCHECKS_PING_URL_ORCHESTRATOR (and _SCOUTS, _DASHBOARD)
       to /etc/aaa.env on the VPS.
    4. The orchestrator's main() calls ping_success() on a clean
       finish; ping_fail() on errors.

Failure mode: if the env var isn't set, the ping is a no-op. We
NEVER block the cycle on a heartbeat ping.
"""
from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


# Map of logical-name → env-var that holds the ping URL. Adding a
# new timer just needs a new entry here + the env var on the VPS.
PING_URL_ENV = {
    "orchestrator": "HEALTHCHECKS_PING_URL_ORCHESTRATOR",
    "scouts":       "HEALTHCHECKS_PING_URL_SCOUTS",
    "dashboard":    "HEALTHCHECKS_PING_URL_DASHBOARD",
    "db_backup":    "HEALTHCHECKS_PING_URL_DB_BACKUP",
}


def ping(component: str, *, status: str = "success",
          message: str | None = None, timeout_seconds: float = 8.0) -> bool:
    """Send a heartbeat. `component` selects which env var holds the
    ping URL; `status` is one of:

        "success"  — append nothing (default healthchecks "success" ping)
        "fail"     — append /fail (escalates to alerting)
        "start"    — append /start (records cycle start, useful for
                     long-running jobs; pair with a later success ping)

    Returns True on HTTP 2xx, False on any failure. Never raises;
    callers must not depend on this method to detect HC.io outages.
    """
    env = PING_URL_ENV.get(component)
    if not env:
        logger.debug(f"heartbeat: unknown component {component}")
        return False
    base = os.environ.get(env, "").strip()
    if not base:
        # Heartbeat not configured for this component — that's fine,
        # silently skip. The user can add the env var later.
        return False

    url = base.rstrip("/")
    if status == "fail":
        url += "/fail"
    elif status == "start":
        url += "/start"
    # else: bare URL = success ping

    try:
        # Healthchecks accepts an optional log message in the body
        # (up to 100 KB). Keep it short — these get rate-limited.
        body = (message or "")[:1000]
        resp = requests.post(url, data=body.encode("utf-8"),
                              timeout=timeout_seconds)
        if 200 <= resp.status_code < 300:
            return True
        logger.warning(
            f"heartbeat {component}/{status} HTTP {resp.status_code}"
        )
        return False
    except requests.RequestException as e:
        logger.warning(f"heartbeat {component}/{status} failed: {e}")
        return False


def ping_success(component: str, message: str | None = None) -> bool:
    return ping(component, status="success", message=message)


def ping_fail(component: str, message: str | None = None) -> bool:
    return ping(component, status="fail", message=message)


def ping_start(component: str, message: str | None = None) -> bool:
    return ping(component, status="start", message=message)
