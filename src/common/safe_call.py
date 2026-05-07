"""`safe_call` — standardized exception-swallowing with WARNING-level logs.

Why this exists. The audit history of this repo reads as a recurring
pattern: a `try/except Exception: logger.debug(...)` swallowed a real
bug for weeks before it surfaced (the phantom-loss PnL, the alert-
pipeline failure, the broker-outage-falls-back-to-{}-pending, the
strategy-granularity-string-mismatch). Each of those was caught at
debug, hidden in journalctl noise, and only found when a human
correlated symptom→cause manually.

The fix is not "stop using broad excepts" — that's neither realistic
nor desirable in a hot loop where one failed strategy must not kill
the cycle. The fix is: when you swallow, do it loudly enough that the
next regression of the same shape is visible without forensics.

Usage:

    from common.safe_call import safe_call

    # Swap out:
    #     try: do_it()
    #     except Exception as e: logger.debug(f"do_it failed: {e}")
    # for:
    safe_call("do_it", do_it)

    # With args + a default fallback value:
    n = safe_call("get_open_orders", adapter.get_open_orders,
                    default=[], context={"venue": vname})

The contract:

  - Default log level is WARNING (visible in journalctl, in CI logs, in
    the dashboard's error feed). Override with `level=logging.DEBUG`
    only when you've established the call is intentionally noisy.
  - Includes the call site label, the exception type + truncated str,
    and any caller-provided `context` dict.
  - Never re-raises. Returns `default` on failure, the call's result
    on success.

This module is dependency-free so it can be imported from anywhere
in the hot path.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def safe_call(
    label: str,
    fn: Callable[..., T],
    *args: Any,
    default: T | None = None,
    level: int = logging.WARNING,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> T | None:
    """Call `fn(*args, **kwargs)` and swallow any exception, logging at
    `level` (default WARNING). Returns `default` on failure.

    `label` is a short, stable identifier shown in the log message.
    `context` is an optional dict that gets stringified into the log
    line — useful for adding venue, symbol, strategy_name etc. without
    string-formatting at the call site.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:    # noqa: BLE001 — this is the standardization point
        ctx = ""
        if context:
            ctx = " [" + ", ".join(f"{k}={v}" for k, v in context.items()) + "]"
        logger.log(
            level,
            f"safe_call({label}) raised: {type(e).__name__}: "
            f"{str(e)[:200]}{ctx}",
        )
        return default
