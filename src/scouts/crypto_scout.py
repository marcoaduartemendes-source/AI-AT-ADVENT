"""Crypto scout — funding rates + perp/spot dislocations on Coinbase.

Outputs:
  signal_type="funding_rates"   payload={<perp_id>: {apr_bps, raw_rate}}
  signal_type="spot_change_24h" payload={<spot_id>: pct_change}

The funding-rates signal is the trigger for crypto_funding_carry strategy.
We only flag perps whose annualized funding exceeds the strategy's entry
threshold; the strategy applies its own logic.

Data: Coinbase public market endpoints — no auth needed for funding info,
spot quotes use the same public catalog we used for the dashboard backtest.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

from common import cached_get

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Pairs we monitor (must match strategies/crypto_funding_carry.py)
PAIRS = [
    ("BTC-USD", "BIT-PERP-INTX"),
    ("ETH-USD", "ETP-PERP-INTX"),
    ("SOL-USD", "SLP-PERP-INTX"),
]

COINBASE_PUBLIC = "https://api.coinbase.com/api/v3/brokerage/market"


class CryptoScout(ScoutAgent):
    name = "crypto_scout"

    def scan(self) -> List[ScoutSignal]:
        signals: List[ScoutSignal] = []
        funding = self._fetch_funding_rates()
        if funding:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="funding_rates",
                payload=funding,
                ttl_seconds=2 * 3600,  # funding pays every 8h, refresh hourly
            ))
        spot = self._fetch_spot_changes()
        if spot:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="spot_change_24h",
                payload=spot,
                ttl_seconds=4 * 3600,
            ))
        return signals

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fetch_funding_rates(self) -> Dict:
        """Fetch each perp's product detail; product payload includes
        future_product_details with the latest funding-rate fields when the
        instrument is a perpetual."""
        out: Dict = {}
        for spot_sym, perp_sym in PAIRS:
            data = cached_get(f"{COINBASE_PUBLIC}/products/{perp_sym}",
                                ttl_seconds=60)
            if not data:
                continue
            try:
                # Newer Coinbase responses include `perpetual_details` with
                # `funding_rate` (per-period) and `funding_rate_24h_avg`.
                perp = (data.get("future_product_details", {}) or {}) \
                       .get("perpetual_details", {}) or {}
                rate = perp.get("funding_rate")
                if rate is None:
                    # Older shape: top-level funding_rate
                    rate = data.get("funding_rate")
                if rate is None:
                    continue
                rate = float(rate)
                # Coinbase US perps fund every 60 minutes (24x/day = 8760/yr).
                # Some venues fund every 8h (3x/day). Use 24/day to match
                # Coinbase US per docs.
                apr = rate * 24 * 365
                out[perp_sym] = {
                    "perp_id": perp_sym, "spot_id": spot_sym,
                    "raw_rate": rate, "apr": apr,
                    "apr_bps": round(apr * 10000, 1),
                    "as_of": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                logger.warning(f"[{self.name}] {perp_sym} fetch failed: {e}")
        return out

    def _fetch_spot_changes(self) -> Dict:
        out: Dict = {}
        for spot_sym, _ in PAIRS:
            d = cached_get(f"{COINBASE_PUBLIC}/products/{spot_sym}",
                            ttl_seconds=60)
            if not d:
                continue
            pct = d.get("price_percentage_change_24h")
            if pct is None:
                continue
            try:
                out[spot_sym] = {
                    "pct_change_24h": float(pct),
                    "price": float(d.get("price", 0) or 0),
                    "volume_24h": float(d.get("volume_24h", 0) or 0),
                }
            except (TypeError, ValueError):
                continue
        return out
