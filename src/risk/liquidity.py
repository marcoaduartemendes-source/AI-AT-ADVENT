"""Average-daily-volume (ADV) estimation for participation limits.

2026-08-05 CAPITAL-SCALE REVIEW. Every sizing cap in `RiskConfig` is a
fraction of *equity* — `max_position_pct`, `max_asset_class_pct`,
`leverage_cap`. Not one of them refers to the *instrument*. That is
harmless at $100k paper and dangerous at $1M: the caps scale up linearly
with the account while the market on the other side does not.

Concretely, at $1M equity with `max_position_pct=0.30` the risk layer
would authorise a $300,000 order in any symbol a strategy names,
including a small-cap that trades $2M/day. That order is ~15% of a day's
volume; it moves the price against itself, and the fill bears no
resemblance to the backtested close-to-close return that justified the
trade. Backtests in this repo assume infinite liquidity — they multiply a
signal by a close price. The gap between that assumption and reality is
pure, unmodelled loss, and it grows with size.

This module supplies the missing denominator: dollar ADV, from the same
daily candles the strategies already consume, so it needs no new data
vendor and no new dependency.

Design notes:

  * MEDIAN, not mean. A single earnings-day volume spike can be 10x
    normal; a mean lets that one day license an oversized position for
    the next month.
  * Returns None rather than a guess when it cannot measure. The caller
    decides what "unknown" means — see `RiskConfig.adv_unknown_max_usd`.
    Silently returning an optimistic default is the exact failure that
    produced a 10.0/10 alpha grade on $0.50 of profit (AUDIT_INCEPTION
    §4); an unmeasurable quantity must never be reported as a measured
    one.
  * Cached with a TTL. `check_order` runs per proposal per cycle, and
    ADV moves on a timescale of days, not minutes.
"""
from __future__ import annotations

import logging
import time
from statistics import median

logger = logging.getLogger(__name__)

# (venue, symbol) -> (expires_at_epoch, adv_usd_or_None)
_ADV_CACHE: dict[tuple[str, str], tuple[float, float | None]] = {}

_CACHE_TTL_SECONDS = 6 * 3600.0

# Below this many usable bars we decline to estimate. A handful of bars
# is not a distribution, and a too-small sample is how an illiquid name
# gets waved through on one busy day.
_MIN_USABLE_BARS = 5


def clear_adv_cache() -> None:
    """Drop the ADV cache. Used by tests and by long-lived processes that
    want to force a refresh."""
    _ADV_CACHE.clear()


def adv_usd(broker, symbol: str, lookback_days: int = 20) -> float | None:
    """Median daily dollar volume for `symbol`, or None if unmeasurable.

    Args:
        broker: a BrokerAdapter. Only `get_candles` and `venue` are used.
        symbol: venue-native symbol, as the strategy proposed it.
        lookback_days: number of daily bars to summarise.

    Returns:
        Median of (volume x close) over the window in USD, or None when
        the venue reports no usable volume (Kalshi event contracts, a
        broker error, a brand-new listing). None means "unknown", NOT
        "unlimited" — callers must treat it conservatively.

    Volume units work out to USD on every venue we trade: equities report
    shares against a USD close, crypto reports base units against a USD
    close. Venues that report no volume at all fall through to None.
    """
    if broker is None or not symbol:
        return None
    venue = str(getattr(broker, "venue", "?"))
    key = (venue, symbol)

    cached = _ADV_CACHE.get(key)
    if cached is not None and cached[0] > time.time():
        return cached[1]

    value: float | None = None
    try:
        # A few extra bars so holidays/halts don't shrink the sample
        # below the usable-bar floor.
        candles = broker.get_candles(
            symbol, "ONE_DAY", num_candles=lookback_days + 5)
    except Exception as exc:      # broker error, bad symbol, rate limit
        logger.debug(f"adv_usd({venue}:{symbol}): candles unavailable: {exc}")
        candles = None

    if candles:
        dollars = []
        for c in candles[-lookback_days:]:
            vol = getattr(c, "volume", None) or 0.0
            close = getattr(c, "close", None) or 0.0
            if vol > 0 and close > 0:
                dollars.append(float(vol) * float(close))
        if len(dollars) >= _MIN_USABLE_BARS:
            value = float(median(dollars))

    _ADV_CACHE[key] = (time.time() + _CACHE_TTL_SECONDS, value)
    return value


def participation_cap_usd(
    broker, symbol: str, *,
    participation_pct: float,
    lookback_days: int = 20,
    unknown_max_usd: float = 0.0,
) -> float | None:
    """Largest order in `symbol` that stays within `participation_pct`
    of a normal day's dollar volume.

    Returns None when no cap applies (feature disabled, or ADV unknown
    and no unknown-fallback configured). Returns `unknown_max_usd` when
    ADV could not be measured and a fallback IS configured — the
    fail-safe branch: we cannot certify the name is liquid, so we permit
    only a small order rather than either blocking the book (the May-4
    freeze failure mode) or waving it through (the current one).
    """
    if participation_pct <= 0:
        return None
    adv = adv_usd(broker, symbol, lookback_days)
    if adv is None:
        return unknown_max_usd if unknown_max_usd > 0 else None
    return adv * participation_pct
