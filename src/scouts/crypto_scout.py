"""Crypto scout — funding rates, spot changes, multi-venue funding consensus.

Outputs:
  signal_type="funding_rates"        payload={<perp_id>: {apr_bps,
                                       raw_rate, binance_apr_bps,
                                       venues_agree}}
  signal_type="spot_change_24h"      payload={<spot_id>: pct_change}
  signal_type="cross_venue_funding"  payload=[{symbol, coinbase_apr,
                                       bybit_apr, binance_apr,
                                       agree}, …]

The funding-rates signal is the trigger for crypto_funding_carry strategy.
We only flag perps whose annualized funding exceeds the strategy's entry
threshold; the strategy applies its own logic.

Data: Coinbase public market endpoints (primary source) plus Binance
public futures funding rates (cross-venue consensus). When Coinbase
and Binance both show elevated funding, the signal is more reliable;
when they diverge materially, treat as no-info — the funding spike
might be a Coinbase-only quirk (low liquidity, technical glitch).
"""
from __future__ import annotations

import logging
from datetime import datetime, UTC

from backtests.data.binance import BinanceClient
from common import cached_get

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


# Pairs we monitor (must match strategies/crypto_funding_carry.py).
# Sprint B1: use real Coinbase Intl perp tickers (was placeholders).
PAIRS = [
    ("BTC-USD", "BTC-PERP-INTX"),
    ("ETH-USD", "ETH-PERP-INTX"),
    ("SOL-USD", "SOL-PERP-INTX"),
]

COINBASE_PUBLIC = "https://api.coinbase.com/api/v3/brokerage/market"


class CryptoScout(ScoutAgent):
    name = "crypto_scout"

    def __init__(self, bus=None, binance: BinanceClient | None = None):
        super().__init__(bus=bus)
        self._binance = binance or BinanceClient()

    def scan(self) -> list[ScoutSignal]:
        signals: list[ScoutSignal] = []
        funding = self._fetch_funding_rates()
        # Enrich with Binance cross-venue confirmation
        cross_venue = self._fetch_binance_funding(funding)
        if funding:
            # Annotate the per-perp funding rows with binance_apr_bps +
            # venues_agree so downstream strategies can require consensus.
            for _perp_id, row in funding.items():
                spot = row.get("spot_id")
                bn = cross_venue.get(spot)
                if bn is not None:
                    row["binance_apr_bps"] = bn["binance_apr_bps"]
                    row["venues_agree"] = bn["agree"]
                else:
                    row["binance_apr_bps"] = None
                    row["venues_agree"] = False
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="funding_rates",
                payload=funding,
                ttl_seconds=2 * 3600,  # funding pays every 8h, refresh hourly
            ))
        if cross_venue:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="cross_venue_funding",
                payload=list(cross_venue.values()),
                ttl_seconds=2 * 3600,
            ))
        spot = self._fetch_spot_changes()
        if spot:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="spot_change_24h",
                payload=spot,
                ttl_seconds=4 * 3600,
            ))
        return signals

    def _fetch_binance_funding(self, coinbase_funding: dict) -> dict:
        """Pull Binance funding for each Coinbase symbol; return a
        keyed-by-spot-symbol map of cross-venue consensus rows.

        Returns empty dict if Binance is unreachable — strategies
        treat absence as no-info, not stale-info.
        """
        out: dict = {}
        # Iterate the SPOT side of every PAIR so we cover symbols even
        # if Coinbase didn't return funding for the matching perp.
        for spot_sym, _ in PAIRS:
            try:
                bn = self._binance.funding_history(spot_sym, limit=1)
            except Exception as e:
                logger.info(f"[{self.name}] binance funding {spot_sym} "
                             f"failed (skipping): {e}")
                continue
            if not bn:
                continue
            # 8h funding × 365 × 3 = annual rate
            binance_apr = bn[-1].funding_rate * 365 * 3
            binance_apr_bps = round(binance_apr * 10000, 1)
            cb_row = coinbase_funding.get(
                next((p for s, p in PAIRS if s == spot_sym), None) or "", {}
            )
            cb_apr_bps = cb_row.get("apr_bps")
            agree = (
                cb_apr_bps is not None
                and abs(binance_apr_bps - cb_apr_bps) <= 200    # 2% APR window
                and cb_apr_bps > 100   # both ≥ 1% APR (i.e. genuinely elevated)
                and binance_apr_bps > 100
            )
            out[spot_sym] = {
                "symbol": spot_sym,
                "coinbase_apr_bps": cb_apr_bps,
                "binance_apr_bps": binance_apr_bps,
                "agree": agree,
            }
        return out

    # ── Helpers ──────────────────────────────────────────────────────────

    def _fetch_funding_rates(self) -> dict:
        """Fetch each perp's product detail; product payload includes
        future_product_details with the latest funding-rate fields when the
        instrument is a perpetual."""
        out: dict = {}
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
                    "as_of": datetime.now(UTC).isoformat(),
                }
            except Exception as e:
                logger.warning(f"[{self.name}] {perp_sym} fetch failed: {e}")
        return out

    def _fetch_spot_changes(self) -> dict:
        out: dict = {}
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
