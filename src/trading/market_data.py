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


_BINANCE_INTERVALS = {
    "ONE_MINUTE": "1m",
    "FIVE_MINUTE": "5m",
    "FIFTEEN_MINUTE": "15m",
    "THIRTY_MINUTE": "30m",
    "ONE_HOUR": "1h",
    "TWO_HOUR": "2h",
    "SIX_HOUR": "6h",
    "ONE_DAY": "1d",
}


def _fetch_binance_public(product_id: str, granularity: str, num_candles: int) -> np.ndarray:
    """Public OHLCV fallback via Binance — used when the Coinbase endpoint is
    inaccessible (e.g. from a restricted network)."""
    import requests

    # BTC-USD → BTCUSDT (Binance uses USDT stablecoin pairs)
    base, _, _ = product_id.partition("-")
    symbol = f"{base}USDT"
    interval = _BINANCE_INTERVALS.get(granularity, "1h")

    resp = requests.get(
        "https://api.binance.com/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": num_candles},
        timeout=15,
    )
    resp.raise_for_status()
    raw = resp.json()
    # Kline fields: [open_time, open, high, low, close, volume, close_time, ...]
    candles = [
        [
            float(k[0]) / 1000.0,  # ms → s
            float(k[3]),           # low
            float(k[2]),           # high
            float(k[1]),           # open
            float(k[4]),           # close
            float(k[5]),           # volume
        ]
        for k in raw
    ]
    arr = np.array(candles, dtype=float)
    return arr[arr[:, 0].argsort()]


_SYNTHETIC_SEEDS = {"BTC-USD": 65000.0, "ETH-USD": 3400.0, "SOL-USD": 150.0}


def _generate_synthetic_candles(product_id: str, num_candles: int, granularity: str) -> np.ndarray:
    """Produce realistic-looking OHLCV for offline demos (no network)."""
    import os
    seed_price = _SYNTHETIC_SEEDS.get(product_id, 100.0)
    rng = np.random.default_rng(hash((product_id, os.environ.get("SYNTH_SEED", ""))) & 0xFFFFFFFF)

    # Geometric Brownian motion + occasional regime shifts for variety
    n = num_candles
    returns = rng.normal(0, 0.012, n)
    # Inject a trending regime and a mean-reverting regime so all 3 strategies fire
    returns[n // 4 : n // 4 + 15] += 0.008       # upward trend
    returns[n // 2 : n // 2 + 10] -= 0.015       # sharp sell-off (mean-reversion setup)
    returns[int(n * 0.8) :] += rng.normal(0, 0.025, n - int(n * 0.8))  # vol expansion

    closes = seed_price * np.exp(np.cumsum(returns))
    opens = np.concatenate([[seed_price], closes[:-1]])
    intrabar = np.abs(rng.normal(0, 0.006, n)) * closes
    highs = np.maximum(opens, closes) + intrabar
    lows = np.minimum(opens, closes) - intrabar
    volumes = np.abs(rng.normal(1000, 300, n))

    interval = GRANULARITY_SECONDS[granularity]
    end = int(time.time())
    timestamps = np.array([end - interval * (n - 1 - i) for i in range(n)], dtype=float)

    return np.column_stack([timestamps, lows, highs, opens, closes, volumes])


def fetch_candles(
    client: CoinbaseClient,
    product_id: str,
    granularity: str = "ONE_HOUR",
    num_candles: int = 100,
) -> np.ndarray:
    """Returns OHLCV array (N,6): [timestamp, low, high, open, close, volume]."""
    import os

    # Offline demo mode — for sandboxes / CI with no internet access
    if os.environ.get("SYNTHETIC_DATA", "").lower() == "true":
        return _generate_synthetic_candles(product_id, num_candles, granularity)

    # Public fallback (Binance) when running without Coinbase auth
    if getattr(client, "auth_mode", "") == "public":
        try:
            return _fetch_binance_public(product_id, granularity, num_candles)
        except Exception:
            pass

    interval = GRANULARITY_SECONDS[granularity]
    end = int(time.time())
    start = end - interval * num_candles

    raw = client.get_candles(product_id, granularity, start, end)
    if not raw:
        return np.array([])

    candles = [
        [float(c["start"]), float(c["low"]), float(c["high"]),
         float(c["open"]),  float(c["close"]), float(c["volume"])]
        for c in raw
    ]
    arr = np.array(candles, dtype=float)
    return arr[arr[:, 0].argsort()]


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
