"""Phase 1 production strategies.

Three strategies that survive the friction filter at $1k–$25k capital:

    1. crypto_funding_carry     — Coinbase perp funding (top-Sharpe sleeve)
    2. risk_parity_etf          — Boring backbone via Alpaca ETFs
    3. kalshi_calibration_arb   — Favorite-longshot bias on Kalshi

Each strategy is an importable class implementing the Strategy ABC. The
orchestrator wires them up in W1 Day 6.
"""
from .crypto_funding_carry import CryptoFundingCarry
from .kalshi_calibration_arb import KalshiCalibrationArb
from .risk_parity_etf import RiskParityETF

__all__ = ["CryptoFundingCarry", "KalshiCalibrationArb", "RiskParityETF"]
