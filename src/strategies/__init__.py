"""Production strategies.

10 strategies covering 4 asset classes (crypto, equities/ETFs, commodities
futures, prediction markets), per the W1 research synthesis.

Phase 1 (built W1):
    crypto_funding_carry      — Coinbase perp funding
    risk_parity_etf           — Alpaca ETF book (vol-balanced)
    kalshi_calibration_arb    — Kalshi favorite-longshot fader

Phase 2 (built W2):
    crypto_basis_trade        — Coinbase dated futures vs spot
    tsmom_etf                 — Time-series momentum on 7-ETF basket
    commodity_carry           — Top-N backwardated commodity futures

Phase 3 (built W2):
    pead                      — Post-earnings drift on Alpaca equities
    macro_kalshi              — Kalshi macro events vs implied probabilities
    crypto_xsmom              — Cross-sectional momentum on top-15 alts
    vol_managed_overlay       — Vol-target multiplier (publishes scaler
                                signal; emits no orders directly)

The orchestrator wires every strategy to its broker; the meta-allocator
sizes each pod based on rolling Sharpe + lifecycle state.
"""
from .commodity_carry import CommodityCarry
from .crypto_basis_trade import CryptoBasisTrade
from .crypto_funding_carry import CryptoFundingCarry
from .crypto_xsmom import CryptoXSMom
from .kalshi_calibration_arb import KalshiCalibrationArb
from .macro_kalshi import MacroKalshi
from .pead import PEAD
from .risk_parity_etf import RiskParityETF
from .tsmom_etf import TSMomETF
from .vol_managed_overlay import VolManagedOverlay

__all__ = [
    "CommodityCarry",
    "CryptoBasisTrade",
    "CryptoFundingCarry",
    "CryptoXSMom",
    "KalshiCalibrationArb",
    "MacroKalshi",
    "PEAD",
    "RiskParityETF",
    "TSMomETF",
    "VolManagedOverlay",
]
