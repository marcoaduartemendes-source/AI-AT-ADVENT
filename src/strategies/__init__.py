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

Phase 4b (this session — adds 4 more, all on Alpaca paper):
    gap_trading               — Overnight-gap reversion (long-only fades down-gaps)
    turn_of_month             — Seasonal 4-7-day SPY long around month boundaries
    low_vol_anomaly           — Long lowest realized-vol ETFs + stocks (positive trend)
    internationals_rotation   — Country-ETF momentum vs SPY baseline

The Phase-4 strategies all run on Alpaca paper. Allocator gives each
a 4% baseline; the auto-Sharpe-tilt + champion-tier (≥1.0 Sharpe → 1.5×
boost) reallocate capital to winners.
"""
from .bollinger_breakout import BollingerBreakout
from .commodity_carry import CommodityCarry
from .cross_venue_arb import CrossVenueArb
from .crypto_basis_trade import CryptoBasisTrade
from .crypto_funding_carry import CryptoFundingCarry
from .crypto_funding_carry_v2 import CryptoFundingCarryV2
from .crypto_xsmom import CryptoXSMom
from .dividend_growth import DividendGrowth
from .earnings_momentum import EarningsMomentum
from .earnings_news_pead import EarningsNewsPEAD
from .gap_trading import GapTrading
from .internationals_rotation import InternationalsRotation
from .kalshi_calibration_arb import KalshiCalibrationArb
from .low_vol_anomaly import LowVolAnomaly
from .macro_kalshi import MacroKalshi
from .macro_kalshi_v2 import MacroKalshiV2
from .pairs_trading import PairsTrading
from .pead import PEAD
from .risk_parity_etf import RiskParityETF
from .rsi_mean_reversion import RSIMeanReversion
from .sector_rotation import SectorRotation
from .tsmom_etf import TSMomETF
from .turn_of_month import TurnOfMonth
from .vol_managed_overlay import VolManagedOverlay

__all__ = [
    # Phase 1-3
    "CommodityCarry",
    "CryptoBasisTrade",
    "CryptoFundingCarry",
    "CryptoXSMom",
    "KalshiCalibrationArb",
    "MacroKalshi",
    "MacroKalshiV2",
    "PEAD",
    "RiskParityETF",
    "TSMomETF",
    "VolManagedOverlay",
    # Phase 4 — experimental (Alpaca paper)
    "BollingerBreakout",
    "DividendGrowth",
    "EarningsMomentum",
    "PairsTrading",
    "RSIMeanReversion",
    "SectorRotation",
    # Phase 4b — more experimental sleeve (Alpaca paper)
    "GapTrading",
    "InternationalsRotation",
    "LowVolAnomaly",
    "TurnOfMonth",
    # Phase 5 — strategies consuming the new Sprint-3 data feeds
    "CrossVenueArb",         # Kalshi vs Polymarket
    "CryptoFundingCarryV2",  # Coinbase + Binance consensus
    "EarningsNewsPEAD",      # PEAD × news corroboration
]
