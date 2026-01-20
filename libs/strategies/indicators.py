"""
Technical indicators.
"""
from typing import Union, List

import numpy as np
import pandas as pd


def MA(data: Union[List[float], np.ndarray, pd.Series], period: int) -> float:
    """
    Simple Moving Average.

    Args:
        data: Price data (latest is last).
        period: Lookback period.

    Returns:
        MA value.
    """
    arr = np.array(data)
    if len(arr) < period:
        return float(arr[-1]) if len(arr) > 0 else 0.0
    return float(np.mean(arr[-period:]))


def KAMA(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int = 5,
    fast_sc: int = 2,
    slow_sc: int = 30,
) -> float:
    """
    Kaufman's Adaptive Moving Average.

    NOTE: Uses KAMA_series for standard time-varying ER/SC calculation.

    Args:
        data: Price data (latest is last).
        period: ER window (default 5).
        fast_sc: Fast smoothing constant (default 2).
        slow_sc: Slow smoothing constant (default 30).

    Returns:
        KAMA value.
    """
    arr = np.array(data, dtype=float)

    if len(arr) < period + 1:
        return float(arr[-1]) if len(arr) > 0 else 0.0

    series = KAMA_series(arr, period=period, fast_sc=fast_sc, slow_sc=slow_sc)
    return float(series[-1]) if len(series) > 0 else 0.0


def KAMA_series(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int = 5,
    fast_sc: int = 2,
    slow_sc: int = 30,
) -> np.ndarray:
    """
    Full KAMA series.

    Returns:
        np.ndarray of KAMA values.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)

    if n < period + 1:
        return arr.copy()

    kama_arr = np.zeros(n)
    kama_arr[:period] = arr[:period]

    fast_alpha = 2 / (fast_sc + 1)
    slow_alpha = 2 / (slow_sc + 1)

    for i in range(period, n):
        change = abs(arr[i] - arr[i - period])
        volatility = np.sum(np.abs(np.diff(arr[i - period:i + 1])))

        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility

        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama_arr[i] = kama_arr[i - 1] + sc * (arr[i] - kama_arr[i - 1])

    return kama_arr


def TSMOM(
    data: Union[List[float], np.ndarray, pd.Series],
    lookback: int = 90,
) -> float:
    """
    Time-Series Momentum.

    Args:
        data: Price data (latest is last).
        lookback: Lookback period (default 90).

    Returns:
        Momentum value (return).
    """
    arr = np.array(data, dtype=float)

    if len(arr) <= lookback:
        if len(arr) > 1:
            return (arr[-1] - arr[0]) / arr[0] if arr[0] != 0 else 0.0
        return 0.0

    past_price = arr[-(lookback + 1)]
    current_price = arr[-1]

    if past_price == 0:
        return 0.0

    return (current_price - past_price) / past_price


def TSMOM_signal(
    data: Union[List[float], np.ndarray, pd.Series],
    lookback: int = 90,
) -> bool:
    """
    TSMOM signal (positive momentum).

    Returns:
        True if momentum > 0.
    """
    return TSMOM(data, lookback) > 0


def RSI(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int = 14,
) -> float:
    """
    Relative Strength Index.

    Args:
        data: Price data (latest is last).
        period: RSI period (default 14).

    Returns:
        RSI value (0-100).
    """
    arr = np.array(data, dtype=float)

    if len(arr) < period + 1:
        return 50.0  # Neutral when insufficient data

    # Calculate price changes
    deltas = np.diff(arr)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use last 'period' changes
    recent_gains = gains[-(period):]
    recent_losses = losses[-(period):]

    avg_gain = np.mean(recent_gains)
    avg_loss = np.mean(recent_losses)

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi)


def RSI_series(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int = 14,
) -> np.ndarray:
    """
    Full RSI series using Wilder's smoothing method.

    Args:
        data: Price data.
        period: RSI period.

    Returns:
        np.ndarray of RSI values.
    """
    arr = np.array(data, dtype=float)
    n = len(arr)

    if n < period + 1:
        return np.full(n, 50.0)

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi_values = np.full(n, 50.0)

    # Initial averages (SMA)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        # Wilder's smoothing
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values[i + 1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i + 1] = 100 - (100 / (1 + rs))

    return rsi_values


def EMA(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int,
) -> float:
    """
    Exponential Moving Average.

    Args:
        data: Price data.
        period: EMA period.

    Returns:
        EMA value.
    """
    arr = np.array(data, dtype=float)

    if len(arr) < period:
        return float(arr[-1]) if len(arr) > 0 else 0.0

    multiplier = 2 / (period + 1)
    ema = arr[0]

    for price in arr[1:]:
        ema = (price - ema) * multiplier + ema

    return float(ema)


def EMA_series(
    data: Union[List[float], np.ndarray, pd.Series],
    period: int,
) -> np.ndarray:
    """Full EMA series."""
    arr = np.array(data, dtype=float)
    n = len(arr)

    if n == 0:
        return np.array([])

    ema_arr = np.zeros(n)
    ema_arr[0] = arr[0]

    multiplier = 2 / (period + 1)

    for i in range(1, n):
        ema_arr[i] = (arr[i] - ema_arr[i - 1]) * multiplier + ema_arr[i - 1]

    return ema_arr


def MACD(
    data: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[float, float, float]:
    """
    Moving Average Convergence Divergence.

    Args:
        data: Price data.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line period (default 9).

    Returns:
        Tuple of (MACD line, Signal line, Histogram).
    """
    arr = np.array(data, dtype=float)

    if len(arr) < slow_period:
        return (0.0, 0.0, 0.0)

    fast_ema = EMA_series(arr, fast_period)
    slow_ema = EMA_series(arr, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = EMA_series(macd_line, signal_period)
    histogram = macd_line - signal_line

    return (float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1]))


def MACD_series(
    data: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full MACD series.

    Returns:
        Tuple of (MACD line array, Signal line array, Histogram array).
    """
    arr = np.array(data, dtype=float)

    fast_ema = EMA_series(arr, fast_period)
    slow_ema = EMA_series(arr, slow_period)

    macd_line = fast_ema - slow_ema
    signal_line = EMA_series(macd_line, signal_period)
    histogram = macd_line - signal_line

    return (macd_line, signal_line, histogram)


def ATR(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> float:
    """
    Average True Range.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ATR period (default 14).

    Returns:
        ATR value.
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)

    n = len(c)
    if n < 2:
        return 0.0

    # True Range calculation
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]

    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )

    if n <= period:
        return float(np.mean(tr))

    return float(np.mean(tr[-period:]))
