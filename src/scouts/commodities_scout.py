"""Commodities scout — futures term structure + cross-commodity ratios.

Outputs:
  signal_type="term_structure"  payload={<root>: {front, second, third,
                                                    backwardation_pct,
                                                    annualized_carry_pct}}
  signal_type="cross_ratios"    payload={"gold_silver": ratio,
                                          "platinum_gold": ratio, …}

Data: Coinbase public market endpoints (the FUTURE products we already
mapped in W1). The carry signal feeds Phase-2 commodity carry strategy;
the gold/silver ratio specifically powers the cross-commodity reversion
trade flagged in research.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List

import requests

from .base import ScoutAgent, ScoutSignal

logger = logging.getLogger(__name__)


COINBASE_PUBLIC = "https://api.coinbase.com/api/v3/brokerage/market"

# Commodity roots we care about (must match Coinbase product_id prefixes)
COMMODITIES = {
    "GOL": "gold",
    "SLR": "silver",
    "PT":  "platinum",
    "CU":  "copper",
    "NOL": "crude_oil",
    "NGS": "natural_gas",
}


class CommoditiesScout(ScoutAgent):
    name = "commodities_scout"

    def scan(self) -> List[ScoutSignal]:
        # Pull all FUTURE products and group by root
        try:
            r = requests.get(
                f"{COINBASE_PUBLIC}/products",
                params={"product_type": "FUTURE", "limit": 300},
                timeout=20,
            )
            if r.status_code != 200:
                logger.warning(f"[{self.name}] futures fetch HTTP {r.status_code}")
                return []
            products = r.json().get("products", [])
        except Exception as e:
            logger.warning(f"[{self.name}] futures fetch failed: {e}")
            return []

        # group by root + sort by expiry
        by_root: Dict[str, List[Dict]] = {}
        for p in products:
            pid = p.get("product_id", "")
            m = re.match(r"^([A-Z0-9]+)-(\d+[A-Z]+\d+)-([A-Z]+)$", pid)
            if not m:
                continue
            root, expiry, _ = m.groups()
            if root not in COMMODITIES:
                continue
            try:
                exp_dt = datetime.strptime(expiry, "%d%b%y")
            except Exception:
                continue
            by_root.setdefault(root, []).append({
                "product_id": pid,
                "expiry": exp_dt,
                "price": float(p.get("price") or 0),
                "display_name": p.get("display_name", ""),
            })

        # Build term structure
        term_struct: Dict = {}
        spot_prices: Dict[str, float] = {}
        for root, contracts in by_root.items():
            contracts.sort(key=lambda c: c["expiry"])
            if len(contracts) < 2:
                continue
            front = contracts[0]
            second = contracts[1]
            third = contracts[2] if len(contracts) >= 3 else None
            front_p = front["price"]
            second_p = second["price"]
            if front_p <= 0 or second_p <= 0:
                continue
            spot_prices[root] = front_p

            # Backwardation = front > deferred. Express as % of front.
            backwardation_pct = (front_p - second_p) / front_p * 100
            # Annualize: percent over time-to-second / 365
            days_between = max(1, (second["expiry"] - front["expiry"]).days)
            annualized_carry = backwardation_pct * (365 / days_between)

            term_struct[root] = {
                "name": COMMODITIES[root],
                "front":  {"id": front["product_id"], "price": front_p,
                            "expiry": front["expiry"].isoformat()[:10]},
                "second": {"id": second["product_id"], "price": second_p,
                            "expiry": second["expiry"].isoformat()[:10]},
                "third": ({"id": third["product_id"], "price": third["price"],
                            "expiry": third["expiry"].isoformat()[:10]}
                          if third else None),
                "backwardation_pct": round(backwardation_pct, 3),
                "annualized_carry_pct": round(annualized_carry, 2),
            }

        # Cross-commodity ratios
        ratios: Dict = {}
        if "GOL" in spot_prices and "SLR" in spot_prices and spot_prices["SLR"] > 0:
            ratios["gold_silver"] = round(spot_prices["GOL"] / spot_prices["SLR"], 3)
        if "PT" in spot_prices and "GOL" in spot_prices and spot_prices["GOL"] > 0:
            ratios["platinum_gold"] = round(spot_prices["PT"] / spot_prices["GOL"], 4)

        signals: List[ScoutSignal] = []
        if term_struct:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="term_structure",
                payload=term_struct, ttl_seconds=6 * 3600,
            ))
        if ratios:
            signals.append(ScoutSignal(
                venue="coinbase", signal_type="cross_ratios",
                payload=ratios, ttl_seconds=6 * 3600,
            ))
        return signals
