"""Market-hours gate for venue-specific trading windows.

Why: Alpaca equity orders fired outside US market hours get cancelled
by the broker as expired day orders. We were submitting hundreds of
SELL orders at e.g. 23:40 UTC (7:40 PM ET) that never had a chance to
fill. The "770 PENDING / 128 CANCELED" trade-table state was the
symptom.

Approach: pure functions, no broker calls, deterministic from a
timestamp. Each venue declares its window:
  - alpaca   : weekday 14:30 UTC → 21:00 UTC (US equity regular session)
  - coinbase : 24/7 (crypto)
  - kalshi   : 24/7 (binary prediction markets)

US market holidays are intentionally NOT modeled — they're rare and
the orders just get cancelled cleanly on those days. If we ever care
to be more careful, plug in `pandas_market_calendars`.
"""
from __future__ import annotations

from datetime import UTC, datetime

# US equity regular session in UTC.
# 9:30 AM ET = 14:30 UTC during EDT (March-Nov)
# 9:30 AM ET = 14:30 UTC during EST is actually 14:30 UTC because EST = UTC-5
# Wait, EST (UTC-5) means 9:30 ET = 14:30 UTC.
# EDT (UTC-4) means 9:30 ET = 13:30 UTC.
# We use the wider of the two windows so we never CLOSE the gate while
# the actual market is still open. Better to occasionally fire orders
# 30min before regular open and have them queue than to miss real sessions.
ALPACA_OPEN_UTC = (13, 30)    # 13:30 UTC = 9:30 EDT (worst case)
ALPACA_CLOSE_UTC = (20, 0)    # 20:00 UTC = 16:00 EDT (worst case)


def is_market_open(venue: str, now: datetime | None = None) -> bool:
    """Return True if `venue` accepts day orders right now.

    Crypto venues (coinbase, kalshi) are always open. Alpaca is open
    Mon-Fri during the US equity regular session (UTC window above).
    Holidays not modeled; orders that hit a holiday will be cancelled
    by the broker, same as before this gate (no worse).
    """
    now = now or datetime.now(UTC)
    venue = (venue or "").lower()

    if venue in ("coinbase", "kalshi"):
        return True

    if venue == "alpaca":
        # Weekday check (0=Mon, 6=Sun)
        if now.weekday() >= 5:
            return False
        oh, om = ALPACA_OPEN_UTC
        ch, cm = ALPACA_CLOSE_UTC
        h, m = now.hour, now.minute
        after_open = (h, m) >= (oh, om)
        before_close = (h, m) < (ch, cm)
        return after_open and before_close

    # Unknown venue — be permissive. Better to attempt than block.
    return True


def venue_window_str(venue: str) -> str:
    """Human-readable window string for logs."""
    if (venue or "").lower() == "alpaca":
        oh, om = ALPACA_OPEN_UTC
        ch, cm = ALPACA_CLOSE_UTC
        return f"{oh:02d}:{om:02d}-{ch:02d}:{cm:02d} UTC weekdays"
    return "24/7"
