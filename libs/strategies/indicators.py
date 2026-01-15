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
