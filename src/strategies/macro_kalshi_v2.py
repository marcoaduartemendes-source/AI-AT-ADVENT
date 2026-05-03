"""macro_kalshi_v2 — Kalshi-vs-CME divergence on Fed rate markets.

Difference from macro_kalshi (v1):
    v1 trades intra-Kalshi mispricing via the favorite-longshot fade.
        Edge: ~2.5–3 cents from the Snowberg-Wolfers prior. Statistical
        but small.

    v2 trades CROSS-venue divergence between Kalshi-implied probability
    of a specific FOMC outcome and the CME Fed Funds futures-implied
    probability for the same outcome.
        Edge: 5–15 cents on the days when Kalshi traders disagree with
        the broader macro futures market — i.e. the days when there's
        a real informational gap.

    Both strategies bet on Kalshi reverting toward fair value, but v2's
    "fair value" anchor is the deepest, most-liquid macro futures market
    on the planet, not a fixed table. When the two strategies agree
    they reinforce; when they disagree (e.g. Kalshi at 30%, CME at 40%,
    calibration table says CME is right, v1 says fade) v2 takes
    precedence because its anchor is empirical.

Algorithm:
    1. Pull Kalshi open markets via the prediction_scout's "mispriced"
       feed (already filtered to liquid contracts).
    2. Filter to Fed-rate-decision markets (ticker / title contains
       "FOMC", "FED-RATE", "RATE-DECISION", etc).
    3. For each, parse out:
         - the FOMC meeting date
         - the target rate range (e.g. "425-450" bps = 4.25-4.50%)
       This is encoded in Kalshi tickers in fairly stable formats but
       we tolerate parse failures (skip, don't crash).
    4. Look up the CME-implied probability for the same (date, range)
       via CMEFedWatchClient.
    5. If |kalshi_prob - cme_prob| >= ENTRY_DIVERGENCE → trade.
       Direction: buy whichever side Kalshi is undervaluing (i.e. if
       Kalshi YES @ 0.30 and CME implies 0.45, BUY YES).

Sizing: Kelly × 0.30 (same as v1 — these are correlated bets).
Per-trade cap: 5% of strategy alloc.

Failure modes:
    - CME endpoint down → strategy emits no proposals (safer than
      firing on stale data).
    - No matching CME meeting for a Kalshi market → skip that market;
      let v1 handle it via the calibration fade if it qualifies.
    - Kalshi adapter unconfigured → no proposals (same as v1).

This strategy is the first one to consume the audit-fix-#3 CME data
feed. It's the empirical proof that the new feed adds edge.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from brokers.base import OrderSide, OrderType
from strategy_engine.base import Strategy, StrategyContext, TradeProposal

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Minimum probability divergence (absolute) to trigger a trade.
# 5% is conservative — at 5¢ on a $1 contract the edge after Kalshi's
# 5% profit fee still covers a 1c spread + slippage with room to spare.
ENTRY_DIVERGENCE = 0.05

# Kelly fraction (matches macro_kalshi v1 — correlated risk, conservative)
KELLY_FRACTION = 0.30

# Per-trade cap as fraction of strategy alloc
MAX_PER_TRADE_PCT = 0.05

# Markets we consider "Fed rate decision". Matched against the
# combined (ticker, title) string after normalizing hyphens and
# underscores to spaces — so "FED-26JUN-T425" + "Fed cut rate" both
# hit "FED" cleanly. Keep keywords broad enough to catch real Kalshi
# market titles ("Will the Fed cut rates?") without false-positiving
# generic "rate" markets (mortgage rate, exchange rate, etc).
RATE_DECISION_KEYWORDS = (
    "FOMC",
    "FED",            # any Fed reference — Kalshi titles consistent
    "FEDFUNDS",
    "RATE DECISION",  # post-normalization "RATE-DECISION" hits this
    "FED FUNDS",
)


# ─── Strategy ─────────────────────────────────────────────────────────


class MacroKalshiV2(Strategy):
    name = "macro_kalshi_v2"
    venue = "kalshi"

    def __init__(self, broker, *, cme_client=None,
                  entry_divergence: float = ENTRY_DIVERGENCE,
                  kelly_fraction: float = KELLY_FRACTION,
                  max_per_trade_pct: float = MAX_PER_TRADE_PCT):
        super().__init__(broker)
        self.entry_divergence = entry_divergence
        self.kelly_fraction = kelly_fraction
        self.max_per_trade_pct = max_per_trade_pct
        # Lazy-init CME client. Tests inject a stub.
        if cme_client is None:
            try:
                from backtests.data.cme_fedwatch import CMEFedWatchClient
                cme_client = CMEFedWatchClient()
            except Exception as e:    # noqa: BLE001
                logger.info(f"[{self.name}] CME client init failed: {e}")
                cme_client = None
        self._cme = cme_client

    # ── Public ──────────────────────────────────────────────────────

    def compute(self, ctx: StrategyContext) -> list[TradeProposal]:
        if ctx.target_alloc_usd <= 0:
            return []
        if self._cme is None:
            return []

        candidates: list[dict] = ctx.scout_signals.get("mispriced", []) or []
        if not candidates:
            return []

        # Filter to Fed-rate-decision markets
        rate_markets = [c for c in candidates
                        if self._is_rate_decision(c.get("ticker", ""),
                                                     c.get("title", ""))]
        if not rate_markets:
            return []

        # Pull CME probabilities once per cycle (the wrapper caches 30 min)
        try:
            cme_meetings = self._cme.upcoming_meetings()
        except Exception as e:    # noqa: BLE001
            logger.warning(f"[{self.name}] CME fetch failed: {e}")
            return []
        if not cme_meetings:
            return []

        proposals: list[TradeProposal] = []
        per_trade_cap_usd = ctx.target_alloc_usd * self.max_per_trade_pct

        for m in rate_markets:
            ticker = m["ticker"]
            yes_price = float(m["yes_price"])
            if yes_price <= 0 or yes_price >= 1:
                continue

            parsed = self._parse_rate_ticker(ticker)
            if parsed is None:
                continue
            meeting_date, lo_bps, hi_bps = parsed

            cme_prob = _cme_prob_for(meeting_date, lo_bps, hi_bps,
                                       cme_meetings)
            if cme_prob is None:
                # Kalshi market doesn't match any CME contract → skip;
                # v1 will handle it via calibration fade if appropriate.
                continue

            divergence = cme_prob - yes_price    # +ve → Kalshi too cheap
            if abs(divergence) < self.entry_divergence:
                continue

            # Kelly sizing on a binary contract
            if divergence > 0:
                # Kalshi YES underpricing → BUY YES
                p, cost, side = cme_prob, yes_price, OrderSide.BUY
            else:
                # Kalshi YES overpricing → BUY NO (= sell YES)
                p, cost, side = 1 - cme_prob, 1 - yes_price, OrderSide.SELL
            if cost <= 0:
                continue
            b = (1 - cost) / cost
            full_kelly = max(0.0, p - (1 - p) / b) if b > 0 else 0.0
            kelly = full_kelly * self.kelly_fraction
            position_usd = min(ctx.target_alloc_usd * kelly, per_trade_cap_usd)
            if position_usd < 1.0:
                continue
            n_contracts = int(position_usd / cost)
            if n_contracts < 1:
                continue

            proposals.append(TradeProposal(
                strategy=self.name, venue=self.venue, symbol=ticker,
                side=side, order_type=OrderType.LIMIT,
                quantity=float(n_contracts), limit_price=yes_price,
                confidence=min(0.95, abs(divergence) * 5),
                reason=(f"kalshi {yes_price:.2f} vs CME {cme_prob:.2f} "
                        f"(Δ={divergence:+.2f}); meeting {meeting_date.isoformat()}"),
                metadata={
                    "meeting_date": meeting_date.isoformat(),
                    "rate_lo_bps": lo_bps,
                    "rate_hi_bps": hi_bps,
                    "kalshi_yes": yes_price,
                    "cme_prob": cme_prob,
                    "divergence": divergence,
                },
            ))
        return proposals

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _is_rate_decision(ticker: str, title: str) -> bool:
        # Normalize hyphens / underscores to spaces so multi-word
        # keywords match either form ("FED-CUT" / "Fed cut").
        haystack = (f"{ticker} {title}".upper()
                    .replace("-", " ").replace("_", " "))
        return any(k in haystack for k in RATE_DECISION_KEYWORDS)

    @staticmethod
    def _parse_rate_ticker(ticker: str) -> tuple[date, int, int] | None:
        """Try to extract (meeting_date, lo_bps, hi_bps) from a Kalshi ticker.

        Tolerates several common Kalshi ticker shapes:
          FED-25JUN-T425         meeting Jun 2025, target 4.25% (single-rate)
          FED-25JUN-B425         band starting at 4.25%
          FOMC-25JUN-CUT-T425    cut to 4.25% target
          FED-RATE-25JUN-425-450 explicit lo-hi range

        Returns None if the format doesn't match — that market is just
        skipped rather than crashing the strategy.
        """
        s = ticker.upper()
        # Find a YYMMM-style date token: e.g. 25JUN, 26FEB
        date_match = re.search(
            r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)",
            s,
        )
        if date_match is None:
            return None
        try:
            yy, mon = date_match.group(1), date_match.group(2)
            year = 2000 + int(yy)
            month_idx = ("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC"
                          .split().index(mon) + 1)
            # Approximate the meeting day to mid-month — CME's
            # "next meeting" lookup is by month, not exact day.
            meeting_date = date(year, month_idx, 15)
        except (ValueError, IndexError):
            return None

        # Look for explicit "T<bps>" token (single target rate)
        t_match = re.search(r"T(\d{3,4})", s)
        if t_match:
            lo = int(t_match.group(1))
            hi = lo + 25    # Fed targets in 25bp bands
            return (meeting_date, lo, hi)
        # Look for explicit "<lo>-<hi>" range
        rng_match = re.search(r"(\d{3,4})[-_](\d{3,4})", s)
        if rng_match:
            lo = int(rng_match.group(1))
            hi = int(rng_match.group(2))
            return (meeting_date, lo, hi)
        return None


def _cme_prob_for(meeting_date: date, lo_bps: int, hi_bps: int,
                   cme_meetings: list) -> float | None:
    """Find the CME-implied probability for this (meeting, range).

    `cme_meetings` is a list of FedMeetingProb (from CMEFedWatchClient).
    Returns None if the CME data doesn't cover this meeting / range.
    """
    # Match by month (forgiving — CME publishes the actual meeting date,
    # we approximated to mid-month, so allow ±10 day window).
    target = None
    for m in cme_meetings:
        delta = abs((m.meeting_date - meeting_date).days)
        if delta <= 10:
            target = m
            break
    if target is None:
        return None
    # Sum probability for any prob row whose lo-hi overlaps ours
    total = 0.0
    matched = False
    for prob_lo, prob_hi, prob in target.target_rate_probs:
        if prob_lo <= lo_bps < prob_hi or prob_lo < hi_bps <= prob_hi:
            total += prob
            matched = True
    return total if matched else None
