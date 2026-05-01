"""Broker registry — instantiate adapters from env config in one call."""
from __future__ import annotations

import logging
import os

from .alpaca import AlpacaAdapter
from .base import BrokerAdapter
from .coinbase import CoinbaseAdapter
from .kalshi import KalshiAdapter

logger = logging.getLogger(__name__)


def build_brokers() -> dict[str, BrokerAdapter]:
    """Construct every broker we have credentials for. Adapters with missing
    creds are still included (they raise on use) so the dashboard can report
    which venues are not yet wired up."""
    brokers: dict[str, BrokerAdapter] = {}

    cb_key = os.environ.get("COINBASE_API_KEY", "")
    cb_sec = os.environ.get("COINBASE_API_SECRET", "")
    if cb_key and cb_sec:
        brokers["coinbase"] = CoinbaseAdapter(cb_key, cb_sec)
    else:
        logger.info("Coinbase: credentials missing — adapter not constructed")

    al_key = os.environ.get("ALPACA_API_KEY_ID", "")
    al_sec = os.environ.get("ALPACA_SECRET_KEY", "")
    if al_key and al_sec:
        brokers["alpaca"] = AlpacaAdapter(al_key, al_sec,
                                           os.environ.get("ALPACA_ENDPOINT", ""))
    else:
        logger.info("Alpaca: credentials missing — adapter not constructed")

    ks_key = os.environ.get("KALSHI_KEY_ID", "")
    ks_pem = os.environ.get("KALSHI_PRIVATE_KEY", "")
    # Always include Kalshi so the dashboard surfaces "configured: false"
    # rather than silent omission when credentials are absent.
    brokers["kalshi"] = KalshiAdapter(ks_key, ks_pem,
                                       os.environ.get("KALSHI_ENDPOINT", ""))

    return brokers


def get_broker(name: str) -> BrokerAdapter | None:
    return build_brokers().get(name)
