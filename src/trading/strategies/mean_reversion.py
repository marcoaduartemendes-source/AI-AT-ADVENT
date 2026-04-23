import numpy as np

from ..market_data import bollinger_bands, get_close_prices, rsi, zscore
from .base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    Statistical mean-reversion strategy using Z-score analysis.

    Inspired by Renaissance Technologies' Medallion Fund statistical
    arbitrage approach: identify when a price has deviated significantly
    from its statistical mean and bet on reversion.

    BUY  when Z-score < -z_entry  (price statistically oversold).
    SELL when Z-score >  z_entry  (price statistically overbought).

    Bollinger Bands confirm the deviation; RSI filters momentum extremes.
    """

    def __init__(
        self,
        products: list,
        window: int = 20,
        z_entry: float = 2.0,
        granularity: str = "ONE_HOUR",
    ):
        super().__init__(
            "MeanReversion", products, granularity, lookback=max(window * 6, 150)
        )
        self.window = window
        self.z_entry = z_entry

    def analyze(self, product_id: str, candles: np.ndarray) -> Signal:
        closes = get_close_prices(candles)
        current_price = closes[-1]

        def hold(reason: str) -> Signal:
            return Signal(self.name, product_id, SignalType.HOLD, 0.0, current_price, reason)

        if len(closes) < self.window + 20:
            return hold("Insufficient data")

        z = zscore(closes, self.window)
        mid, upper, lower = bollinger_bands(closes, self.window)
        rsi_vals = rsi(closes)

        z_c = z[-1]
        rsi_c = rsi_vals[-1]
        mid_c = mid[-1]

        if np.isnan(z_c) or np.isnan(rsi_c) or np.isnan(mid_c):
            return hold("Indicators not ready")

        dist_pct = (current_price - mid_c) / mid_c * 100

        if z_c < -self.z_entry and rsi_c < 40:
            confidence = min(abs(z_c) / (self.z_entry * 2), 0.95)
            return Signal(
                self.name, product_id, SignalType.BUY, confidence, current_price,
                f"Oversold: Z={z_c:.2f}<-{self.z_entry}, RSI={rsi_c:.1f}, "
                f"{dist_pct:.1f}% below mean={mid_c:.2f}",
                {"zscore": z_c, "rsi": rsi_c, "mean": mid_c, "dist_pct": dist_pct},
            )

        if z_c > self.z_entry and rsi_c > 60:
            confidence = min(z_c / (self.z_entry * 2), 0.95)
            return Signal(
                self.name, product_id, SignalType.SELL, confidence, current_price,
                f"Overbought: Z={z_c:.2f}>{self.z_entry}, RSI={rsi_c:.1f}, "
                f"{dist_pct:.1f}% above mean={mid_c:.2f}",
                {"zscore": z_c, "rsi": rsi_c, "mean": mid_c, "dist_pct": dist_pct},
            )

        return hold(f"Within normal range: Z={z_c:.2f}, RSI={rsi_c:.1f}")
