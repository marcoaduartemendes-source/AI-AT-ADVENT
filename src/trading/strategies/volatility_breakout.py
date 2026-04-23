import numpy as np

from ..market_data import bandwidth, bollinger_bands, ema, get_close_prices, get_volumes, rsi
from .base import BaseStrategy, Signal, SignalType


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility breakout via Bollinger Band squeeze detection.

    Rationale (used by CTAs and volatility-focused quant funds):
      Volatility is mean-reverting — periods of compression (squeeze) are
      reliably followed by explosive directional moves.

    1. Detect squeeze: current bandwidth < historical_avg * squeeze_threshold.
    2. When bandwidth expands after a squeeze, identify the direction via
       short EMA vs long EMA and RSI.
    3. Volume surge acts as a confidence booster.

    BUY  on upward breakout (momentum_up + RSI > 50).
    SELL on downward breakout (momentum_down + RSI < 50).
    """

    def __init__(
        self,
        products: list,
        bb_window: int = 20,
        squeeze_threshold: float = 0.5,
        history_window: int = 50,
        granularity: str = "ONE_HOUR",
    ):
        super().__init__(
            "VolatilityBreakout",
            products,
            granularity,
            lookback=max(bb_window * 6, 150),
        )
        self.bb_window = bb_window
        self.squeeze_threshold = squeeze_threshold
        self.history_window = history_window

    def analyze(self, product_id: str, candles: np.ndarray) -> Signal:
        closes = get_close_prices(candles)
        volumes = get_volumes(candles)
        current_price = closes[-1]

        def hold(reason: str) -> Signal:
            return Signal(self.name, product_id, SignalType.HOLD, 0.0, current_price, reason)

        if len(closes) < self.bb_window * 4:
            return hold("Insufficient data")

        mid, upper, lower = bollinger_bands(closes, self.bb_window)
        bw = bandwidth(upper, lower, mid)
        rsi_vals = rsi(closes)
        ema_s = ema(closes, 5)
        ema_l = ema(closes, 20)

        bw_c = bw[-1]
        rsi_c = rsi_vals[-1]
        es_c, el_c = ema_s[-1], ema_l[-1]

        if any(np.isnan(v) for v in [bw_c, rsi_c, es_c, el_c]):
            return hold("Indicators not ready")

        # Historical bandwidth reference (exclude current bar)
        hist_bw = bw[-(self.history_window + 1):-1]
        valid_hist = hist_bw[~np.isnan(hist_bw)]
        if len(valid_hist) < 10:
            return hold("Insufficient bandwidth history")

        avg_bw = np.mean(valid_hist)
        in_squeeze = bw_c < avg_bw * self.squeeze_threshold

        # Check whether we just exited a squeeze (previous 5 bars were squeezed)
        recent_bw = bw[-6:-1]
        was_squeezed = any(
            (not np.isnan(b)) and b < avg_bw * self.squeeze_threshold
            for b in recent_bw
        )

        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_surge = volumes[-1] > avg_vol * 1.5

        if not in_squeeze and was_squeezed:
            bw_ratio = bw_c / avg_bw
            confidence = min(bw_ratio * 0.4, 0.88)
            if volume_surge:
                confidence = min(confidence + 0.1, 0.95)

            if es_c > el_c and rsi_c > 50:
                return Signal(
                    self.name, product_id, SignalType.BUY, confidence, current_price,
                    f"Upward breakout post-squeeze: BW={bw_c:.4f} (avg={avg_bw:.4f}), "
                    f"RSI={rsi_c:.1f}, vol_surge={volume_surge}",
                    {"bw": bw_c, "avg_bw": avg_bw, "rsi": rsi_c, "volume_surge": volume_surge},
                )

            if es_c < el_c and rsi_c < 50:
                return Signal(
                    self.name, product_id, SignalType.SELL, confidence, current_price,
                    f"Downward breakout post-squeeze: BW={bw_c:.4f} (avg={avg_bw:.4f}), "
                    f"RSI={rsi_c:.1f}, vol_surge={volume_surge}",
                    {"bw": bw_c, "avg_bw": avg_bw, "rsi": rsi_c, "volume_surge": volume_surge},
                )

        status = "SQUEEZED" if in_squeeze else "NORMAL"
        return hold(
            f"No breakout ({status}): BW={bw_c:.4f}/{avg_bw:.4f}, RSI={rsi_c:.1f}"
        )
