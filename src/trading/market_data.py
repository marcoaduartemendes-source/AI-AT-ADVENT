import time
from typing import Optional

import numpy as np

from .coinbase_client import CoinbaseClient

GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}


def fetch_candles(
    client: CoinbaseClient,
    product_id: str,
    granularity: str = "ONE_HOUR",
    num_candles: int = 100,
) -> np.ndarray:
    """
    Returns OHLCV array of shape (N, 6):
    [timestamp, low, high, open, close, volume], sorted oldest-first.
    """
    interval = GRANULARITY_SECONDS[granularity]
    end = int(time.time())
    start = end - interval * num_candles

    raw = client.get_candles(product_id, granularity, start, end)
    if not raw:
        return np.array([])

    candles = []
    for c in raw:
        candles.append([
            float(c["start"]),
            float(c["low"]),
            float(c["high"]),
            float(c["open"]),
            float(c["close"]),
            float(c["volume"]),
        ])

    arr = np.array(candles, dtype=float)
    arr = arr[arr[:, 0].argsort()]  # sort ascending by timestamp
    return arr


def get_close_prices(candles: np.ndarray) -> np.ndarray:
    return candles[:, 4]


def get_volumes(candles: np.ndarray) -> np.ndarray:
    return candles[:, 5]


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    if len(prices) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)
    return result


def macd(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9
):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = ema_fast - ema_slow  # NaN for first slow-1 indices

    signal_line = np.full(len(prices), np.nan)
    first_valid = slow - 1
    valid_macd = macd_line[first_valid:]

    if len(valid_macd) >= signal_period:
        sig = ema(valid_macd, signal_period)
        signal_line[first_valid:] = sig

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's smoothed RSI."""
    result = np.full(len(prices), np.nan)
    if len(prices) < period + 1:
        return result

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    result[period] = 100 - 100 / (1 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
        result[i + 1] = 100 - 100 / (1 + rs)

    return result


def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0):
    """Returns (mid, upper, lower) arrays."""
    n = len(prices)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(window - 1, n):
        chunk = prices[i - window + 1: i + 1]
        m = np.mean(chunk)
        s = np.std(chunk, ddof=1)
        mid[i] = m
        upper[i] = m + num_std * s
        lower[i] = m - num_std * s
    return mid, upper, lower


def zscore(prices: np.ndarray, window: int = 20) -> np.ndarray:
    result = np.full(len(prices), np.nan)
    for i in range(window - 1, len(prices)):
        chunk = prices[i - window + 1: i + 1]
        m = np.mean(chunk)
        s = np.std(chunk, ddof=1)
        if s > 0:
            result[i] = (prices[i] - m) / s
    return result


def bandwidth(upper: np.ndarray, lower: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Bollinger Band Width — normalised measure of volatility."""
    return np.where(mid > 0, (upper - lower) / mid, np.nan)


def get_current_price(client: CoinbaseClient, product_id: str) -> Optional[float]:
    try:
        data = client.get_best_bid_ask([product_id])
        for pb in data.get("pricebooks", []):
            if pb["product_id"] == product_id:
                bid = float(pb["bids"][0]["price"]) if pb.get("bids") else None
                ask = float(pb["asks"][0]["price"]) if pb.get("asks") else None
                if bid and ask:
                    return (bid + ask) / 2
    except Exception:
        pass
    return None
