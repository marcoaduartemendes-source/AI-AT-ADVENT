import numpy as np

from ..market_data import ema, get_close_prices, macd, rsi
from .base import BaseStrategy, Signal, SignalType


class MomentumStrategy(BaseStrategy):
    """
    Trend-following strategy using EMA crossovers + MACD confirmation.

    Based on systematic trend-following principles used by quant macro funds
    (Simons / Medallion, Man AHL, Winton).

    BUY  when fast EMA > slow EMA, MACD histogram positive & growing, RSI 40–70.
    SELL when fast EMA < slow EMA, MACD histogram negative & falling, RSI 30–60.
    """

    def __init__(
        self,
        products: list,
        fast_period: int = 10,
        slow_period: int = 30,
        rsi_period: int = 14,
        granularity: str = "ONE_HOUR",
    ):
        super().__init__(
            "Momentum", products, granularity, lookback=max(slow_period * 4, 120)
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period

    def analyze(self, product_id: str, candles: np.ndarray) -> Signal:
        closes = get_close_prices(candles)
        current_price = closes[-1]

        def hold(reason: str) -> Signal:
            return Signal(self.name, product_id, SignalType.HOLD, 0.0, current_price, reason)

        if len(closes) < self.slow_period + self.rsi_period + 5:
            return hold("Insufficient data")

        ema_fast = ema(closes, self.fast_period)
        ema_slow = ema(closes, self.slow_period)
        _, _, histogram = macd(closes)
        rsi_vals = rsi(closes, self.rsi_period)

        f_c, f_p = ema_fast[-1], ema_fast[-2]
        s_c, s_p = ema_slow[-1], ema_slow[-2]
        hist_c, hist_p = histogram[-1], histogram[-2]
        rsi_c = rsi_vals[-1]

        if any(np.isnan(v) for v in [f_c, s_c, hist_c, rsi_c]):
            return hold("Indicators not ready")

        bullish_trend = f_c > s_c
        bearish_trend = f_c < s_c
        golden_cross = bullish_trend and f_p <= s_p   # cross happened this bar
        death_cross = bearish_trend and f_p >= s_p

        macd_bullish = hist_c > 0 and hist_c > hist_p
        macd_bearish = hist_c < 0 and hist_c < hist_p

        if bullish_trend and macd_bullish and 40 <= rsi_c <= 70:
            confidence = 0.8 if golden_cross else (0.7 if hist_c > hist_p * 1.5 else 0.55)
            return Signal(
                self.name, product_id, SignalType.BUY, confidence, current_price,
                f"Bullish: EMA{self.fast_period}={f_c:.2f}>EMA{self.slow_period}={s_c:.2f}, "
                f"MACD_hist={hist_c:.4f}, RSI={rsi_c:.1f}",
                {"ema_fast": f_c, "ema_slow": s_c, "macd_hist": hist_c, "rsi": rsi_c},
            )

        if bearish_trend and macd_bearish and 30 <= rsi_c <= 60:
            confidence = 0.8 if death_cross else (0.7 if hist_c < hist_p * 1.5 else 0.55)
            return Signal(
                self.name, product_id, SignalType.SELL, confidence, current_price,
                f"Bearish: EMA{self.fast_period}={f_c:.2f}<EMA{self.slow_period}={s_c:.2f}, "
                f"MACD_hist={hist_c:.4f}, RSI={rsi_c:.1f}",
                {"ema_fast": f_c, "ema_slow": s_c, "macd_hist": hist_c, "rsi": rsi_c},
            )

        return hold(
            f"No trend alignment: EMA_diff={f_c - s_c:.2f}, MACD_hist={hist_c:.4f}, RSI={rsi_c:.1f}"
        )
