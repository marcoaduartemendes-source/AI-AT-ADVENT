"""Equities scout — earnings calendar + simple sentiment heuristics.

Outputs:
  signal_type="earnings_upcoming" payload=[{symbol, date, time}, …]
  signal_type="cross_sectional_momentum" payload=[{symbol, return_30d,
                                                     percentile}, …]

V1 fetches:
  • Upcoming earnings via Nasdaq's public calendar (no auth)
  • Cross-sectional momentum on a curated S&P-500-ETF basket via Alpaca

The earnings list will trigger PEAD strategy work in W3.
The momentum table feeds Phase-2 cross-sectional momentum strategy.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

from brokers.registry import get_broker

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Curated equity universe for cross-sectional momentum (large/mid-cap, liquid)
DEFAULT_UNIVERSE = [
    # Mag-7 + adjacent megacaps
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "BRK.B", "AVGO", "JPM", "LLY", "UNH", "V", "MA",
    # Sector leaders
    "XOM", "CVX", "JNJ", "PG", "HD", "WMT", "COST", "BAC",
    "ABBV", "PEP", "KO", "ADBE", "CRM", "NFLX", "AMD", "INTC",
]

# Lookback for the momentum signal
MOMENTUM_LOOKBACK_DAYS = 30


class EquitiesScout(ScoutAgent):
    name = "equities_scout"

    def __init__(self, bus=None, universe: Optional[List[str]] = None):
        super().__init__(bus=bus)
        self.universe = universe or DEFAULT_UNIVERSE

    def scan(self) -> List[ScoutSignal]:
        signals: List[ScoutSignal] = []

        # Earnings calendar (next 7 days)
        earnings = self._fetch_earnings_calendar(days=7)
        if earnings:
            signals.append(ScoutSignal(
                venue="alpaca", signal_type="earnings_upcoming",
                payload=earnings, ttl_seconds=12 * 3600,
            ))

        # Cross-sectional momentum
        xsmom = self._fetch_cross_sectional_momentum()
        if xsmom:
            signals.append(ScoutSignal(
                venue="alpaca", signal_type="cross_sectional_momentum",
                payload=xsmom, ttl_seconds=24 * 3600,
            ))

        return signals

    # ── Helpers ────────────────────────────────────────────────────────

    def _fetch_earnings_calendar(self, days: int = 7) -> List[Dict]:
        """Pull from Nasdaq's public earnings calendar."""
        out: List[Dict] = []
        today = datetime.now(timezone.utc).date()
        for offset in range(days):
            d = (today + timedelta(days=offset)).isoformat()
            try:
                r = requests.get(
                    "https://api.nasdaq.com/api/calendar/earnings",
                    params={"date": d},
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; EquitiesScout/1.0)",
                        "Accept": "application/json",
                    },
                    timeout=15,
                )
                if r.status_code != 200:
                    continue
                data = r.json()
                rows = data.get("data", {}).get("rows", []) or []
                for row in rows:
                    sym = row.get("symbol")
                    if not sym or sym not in self.universe:
                        continue
                    out.append({
                        "symbol": sym, "date": d,
                        "time": row.get("time", ""),
                        "estimate_eps": row.get("epsForecast", ""),
                    })
            except Exception as e:
                logger.warning(f"[{self.name}] earnings {d} failed: {e}")
        return out

    def _fetch_cross_sectional_momentum(self) -> List[Dict]:
        """Compute 30-day return percentile rank across the universe."""
        broker = get_broker("alpaca")
        if broker is None:
            logger.warning(f"[{self.name}] Alpaca adapter not configured")
            return []

        returns: List[Dict] = []
        for sym in self.universe:
            try:
                candles = broker.get_candles(
                    sym, "ONE_DAY", num_candles=MOMENTUM_LOOKBACK_DAYS + 5)
            except Exception as e:
                logger.debug(f"[{self.name}] candles {sym}: {e}")
                continue
            if len(candles) < MOMENTUM_LOOKBACK_DAYS:
                continue
            start_close = candles[-MOMENTUM_LOOKBACK_DAYS].close
            end_close = candles[-1].close
            if start_close <= 0:
                continue
            ret = (end_close - start_close) / start_close
            returns.append({"symbol": sym, "return_30d": ret,
                              "price": end_close})

        if not returns:
            return []
        returns.sort(key=lambda r: r["return_30d"])
        n = len(returns)
        for idx, row in enumerate(returns):
            row["percentile"] = round((idx + 1) / n * 100, 1)
        # Highest momentum first in payload
        returns.sort(key=lambda r: r["return_30d"], reverse=True)
        return returns
