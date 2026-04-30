"""ScoutAgent base class.

Each concrete scout implements `scan()` which returns a list of
`ScoutSignal` rows; the base class persists them to the SignalBus.

Scouts are intentionally thin: they fetch external data, structure it, and
write to the bus. They do NOT make trading decisions. Strategies consume
the published signals and decide what to do.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .signal_bus import SignalBus

logger = logging.getLogger(__name__)


@dataclass
class ScoutSignal:
    """One signal a scout wants to publish."""

    venue: str          # "coinbase" | "alpaca" | "kalshi" | "macro"
    signal_type: str    # arbitrary kebab-cased type identifier
    payload: Dict
    ttl_seconds: Optional[int] = None   # None = use bus default

    def __post_init__(self):
        if not self.venue or not self.signal_type:
            raise ValueError("ScoutSignal needs venue + signal_type")


class ScoutAgent(ABC):
    """Implement once per asset class / data domain."""

    name: str           # short identifier; e.g. "crypto_scout"

    def __init__(self, bus: Optional[SignalBus] = None):
        self.bus = bus or SignalBus()

    @abstractmethod
    def scan(self) -> List[ScoutSignal]:
        """Pull external data, return signals to publish.

        Should be idempotent — if called twice in succession, both calls
        produce equivalent output (the bus dedupes via TTL, not identity).
        """

    def run_once(self) -> Dict:
        """Called by the scout workflow. Wraps scan() with persistence and
        returns a summary dict for logging/dashboards."""
        published = 0
        errors: List[str] = []
        signals: List[ScoutSignal] = []
        try:
            signals = self.scan()
        except Exception as e:
            errors.append(f"scan failed: {e}")
            logger.exception(f"[{self.name}] scan raised")
            return {"scout": self.name, "published": 0,
                    "errors": errors, "signals": []}

        for s in signals:
            try:
                self.bus.publish(
                    scout=self.name,
                    venue=s.venue,
                    signal_type=s.signal_type,
                    payload=s.payload,
                    ttl_seconds=s.ttl_seconds,
                )
                published += 1
            except Exception as e:
                errors.append(f"publish failed for {s.signal_type}: {e}")
                logger.exception(f"[{self.name}] publish raised")

        logger.info(f"[{self.name}] published {published}/{len(signals)} signals")
        return {
            "scout": self.name,
            "published": published,
            "total_proposed": len(signals),
            "errors": errors,
            "signal_types": [s.signal_type for s in signals],
        }
