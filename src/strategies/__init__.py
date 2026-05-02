"""Production strategies.

15 strategies (after Phase-4 expansion) covering 4 asset classes
(crypto, equities/ETFs, commodities futures, prediction markets).

Phase 1 (W1):
    crypto_funding_carry      — Coinbase perp funding
    risk_parity_etf           — Alpaca ETF book (vol-balanced)
    kalshi_calibration_arb    — Kalshi favorite-longshot fader

Phase 2 (W2):
    crypto_basis_trade        — Coinbase dated futures vs spot
    tsmom_etf                 — Time-series momentum on 7-ETF basket
    commodity_carry           — Top-N backwardated commodity futures

Phase 3 (W2):
    pead                      — Post-earnings drift (scout-fed)
    macro_kalshi              — Kalshi macro events
    crypto_xsmom              — Cross-sectional momentum top-15 alts
    vol_managed_overlay       — Vol-target multiplier (signals only)

Phase 4 (W3, experimental — small initial allocations):
    rsi_mean_reversion        — Connors-style 2-day RSI reversal
    sector_rotation           — Top-N SPDR sector ETFs by 90d return
    pairs_trading             — Stat-arb on classic correlated pairs
    bollinger_breakout        — Momentum continuation via 20d BB upper
    earnings_momentum         — LIVE PEAD using FMP earnings calendar
    dividend_growth           — Quality-dividend ETF rotation

The Phase-4 strategies all run on Alpaca paper. Allocator gives each
a 5% baseline; the auto-Sharpe-tilt mechanism reallocates capital to
winners over time. Strategies that earn a 30d Sharpe > 1.0 enter the
"champion" tier and get 1.5× their baseline.
"""
from .bollinger_breakout import BollingerBreakout
from .commodity_carry import CommodityCarry
from .crypto_basis_trade import CryptoBasisTrade
from .crypto_funding_carry import CryptoFundingCarry
from .crypto_xsmom import CryptoXSMom
from .dividend_growth import DividendGrowth
from .earnings_momentum import EarningsMomentum
from .kalshi_calibration_arb import KalshiCalibrationArb
from .macro_kalshi import MacroKalshi
from .pairs_trading import PairsTrading
from .pead import PEAD
from .risk_parity_etf import RiskParityETF
from .rsi_mean_reversion import RSIMeanReversion
from .sector_rotation import SectorRotation
from .tsmom_etf import TSMomETF
from .vol_managed_overlay import VolManagedOverlay

__all__ = [
    # Phase 1-3
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
    # Phase 4 — experimental
    "BollingerBreakout",
    "DividendGrowth",
    "EarningsMomentum",
    "PairsTrading",
    "RSIMeanReversion",
    "SectorRotation",
]
