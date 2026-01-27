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


def ATR_series(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> np.ndarray:
    """
    Full ATR series using Wilder's smoothing.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ATR period (default 14).

    Returns:
        np.ndarray of ATR values.
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)

    n = len(c)
    if n < 2:
        return np.zeros(n)

    # True Range calculation
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]

    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )

    # ATR using Wilder's smoothing (EMA with alpha = 1/period)
    atr = np.zeros(n)
    atr[:period] = np.cumsum(tr[:period]) / np.arange(1, period + 1)

    alpha = 1 / period
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    return atr


def ADX(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> float:
    """
    Average Directional Index.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ADX period (default 14).

    Returns:
        ADX value (0-100).
    """
    series = ADX_series(high, low, close, period)
    return float(series[-1]) if len(series) > 0 else 0.0


def ADX_series(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> np.ndarray:
    """
    Full ADX series.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: ADX period (default 14).

    Returns:
        np.ndarray of ADX values.
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)

    n = len(c)
    if n < period + 1:
        return np.full(n, 0.0)

    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # ATR for normalization
    atr = ATR_series(h, l, c, period)

    # Smoothed +DI and -DI
    alpha = 1 / period
    smoothed_plus_dm = np.zeros(n)
    smoothed_minus_dm = np.zeros(n)

    smoothed_plus_dm[period] = np.sum(plus_dm[1:period + 1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - (smoothed_plus_dm[i - 1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - (smoothed_minus_dm[i - 1] / period) + minus_dm[i]

    # +DI and -DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    for i in range(period, n):
        if atr[i] != 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / (atr[i] * period)
            minus_di[i] = 100 * smoothed_minus_dm[i] / (atr[i] * period)

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # Smooth DX to get ADX
    adx = np.zeros(n)
    if n > 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def Supertrend(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    atr_period: int = 10,
    factor: float = 3.0,
) -> tuple[float, int]:
    """
    Supertrend indicator.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        atr_period: ATR period (default 10).
        factor: ATR multiplier (default 3.0).

    Returns:
        Tuple of (supertrend value, direction: 1=up/long, -1=down/short).
    """
    st, direction = Supertrend_series(high, low, close, atr_period, factor)
    return (float(st[-1]), int(direction[-1])) if len(st) > 0 else (0.0, 0)


def Supertrend_series(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    atr_period: int = 10,
    factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full Supertrend series.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        atr_period: ATR period (default 10).
        factor: ATR multiplier (default 3.0).

    Returns:
        Tuple of (supertrend array, direction array: 1=up, -1=down).
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)

    n = len(c)
    if n < atr_period + 1:
        return (c.copy(), np.ones(n, dtype=int))

    atr = ATR_series(h, l, c, atr_period)
    hl2 = (h + l) / 2

    # Basic upper and lower bands
    basic_upper = hl2 + (factor * atr)
    basic_lower = hl2 - (factor * atr)

    # Final bands with trailing logic
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)

    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    supertrend[0] = basic_lower[0]

    for i in range(1, n):
        # Upper band
        if basic_upper[i] < final_upper[i - 1] or c[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Lower band
        if basic_lower[i] > final_lower[i - 1] or c[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Supertrend
        if supertrend[i - 1] == final_upper[i - 1]:
            # Previous was in downtrend
            if c[i] > final_upper[i]:
                supertrend[i] = final_lower[i]
                direction[i] = 1
            else:
                supertrend[i] = final_upper[i]
                direction[i] = -1
        else:
            # Previous was in uptrend
            if c[i] < final_lower[i]:
                supertrend[i] = final_upper[i]
                direction[i] = -1
            else:
                supertrend[i] = final_lower[i]
                direction[i] = 1

    return (supertrend, direction)


def OBV(
    close: Union[List[float], np.ndarray],
    volume: Union[List[float], np.ndarray],
) -> float:
    """
    On Balance Volume.

    Args:
        close: Close prices.
        volume: Volume data.

    Returns:
        OBV value.
    """
    series = OBV_series(close, volume)
    return float(series[-1]) if len(series) > 0 else 0.0


def OBV_series(
    close: Union[List[float], np.ndarray],
    volume: Union[List[float], np.ndarray],
) -> np.ndarray:
    """
    Full OBV series.

    Args:
        close: Close prices.
        volume: Volume data.

    Returns:
        np.ndarray of OBV values.
    """
    c = np.array(close, dtype=float)
    v = np.array(volume, dtype=float)

    n = len(c)
    if n == 0:
        return np.array([])

    obv = np.zeros(n)
    obv[0] = v[0]

    for i in range(1, n):
        if c[i] > c[i - 1]:
            obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            obv[i] = obv[i - 1] - v[i]
        else:
            obv[i] = obv[i - 1]

    return obv


def OBV_signal(
    close: Union[List[float], np.ndarray],
    volume: Union[List[float], np.ndarray],
    ema_period: int = 20,
) -> bool:
    """
    OBV signal (OBV above its EMA = bullish).

    Args:
        close: Close prices.
        volume: Volume data.
        ema_period: EMA period for OBV (default 20).

    Returns:
        True if OBV > OBV_EMA (bullish).
    """
    obv = OBV_series(close, volume)
    if len(obv) < ema_period:
        return False

    obv_ema = EMA_series(obv, ema_period)
    return obv[-1] > obv_ema[-1]


def Choppiness(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> float:
    """
    Choppiness Index.

    Values:
        - > 61.8: Choppy/ranging market
        - < 38.2: Trending market
        - Typical threshold: 55 for trending

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period (default 14).

    Returns:
        Choppiness Index value (0-100).
    """
    series = Choppiness_series(high, low, close, period)
    return float(series[-1]) if len(series) > 0 else 50.0


def Choppiness_series(
    high: Union[List[float], np.ndarray],
    low: Union[List[float], np.ndarray],
    close: Union[List[float], np.ndarray],
    period: int = 14,
) -> np.ndarray:
    """
    Full Choppiness Index series.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Lookback period (default 14).

    Returns:
        np.ndarray of Choppiness Index values.
    """
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)

    n = len(c)
    if n < period + 1:
        return np.full(n, 50.0)

    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]

    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )

    chop = np.full(n, 50.0)

    for i in range(period, n):
        tr_sum = np.sum(tr[i - period + 1:i + 1])
        high_max = np.max(h[i - period + 1:i + 1])
        low_min = np.min(l[i - period + 1:i + 1])

        hl_diff = high_max - low_min
        if hl_diff > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / hl_diff) / np.log10(period)

    return chop


def BBWP(
    close: Union[List[float], np.ndarray],
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 252,
) -> float:
    """
    Bollinger Band Width Percentile.

    Values:
        - > 80: High volatility (squeeze exit)
        - < 20: Low volatility (squeeze)

    Args:
        close: Close prices.
        bb_period: Bollinger Band period (default 20).
        bb_std: Standard deviation multiplier (default 2.0).
        lookback: Lookback for percentile calculation (default 252).

    Returns:
        BBWP value (0-100).
    """
    series = BBWP_series(close, bb_period, bb_std, lookback)
    return float(series[-1]) if len(series) > 0 else 50.0


def BBWP_series(
    close: Union[List[float], np.ndarray],
    bb_period: int = 20,
    bb_std: float = 2.0,
    lookback: int = 252,
) -> np.ndarray:
    """
    Full BBWP series.

    Args:
        close: Close prices.
        bb_period: Bollinger Band period (default 20).
        bb_std: Standard deviation multiplier (default 2.0).
        lookback: Lookback for percentile calculation (default 252).

    Returns:
        np.ndarray of BBWP values.
    """
    c = np.array(close, dtype=float)
    n = len(c)

    if n < bb_period:
        return np.full(n, 50.0)

    # Calculate Bollinger Band Width
    bb_width = np.zeros(n)

    for i in range(bb_period - 1, n):
        window = c[i - bb_period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window, ddof=0)

        if sma != 0:
            bb_width[i] = (2 * bb_std * std) / sma * 100  # Width as percentage

    # Calculate percentile
    bbwp = np.full(n, 50.0)

    for i in range(lookback, n):
        width_window = bb_width[i - lookback + 1:i + 1]
        current_width = bb_width[i]
        bbwp[i] = (np.sum(width_window < current_width) / lookback) * 100

    return bbwp


def TSMOM_volume_weighted(
    close: Union[List[float], np.ndarray],
    volume: Union[List[float], np.ndarray],
    period: int = 20,
) -> float:
    """
    Volume-Weighted Time-Series Momentum.

    Weights momentum by volume for more reliable signals.

    Args:
        close: Close prices.
        volume: Volume data.
        period: Lookback period (default 20).

    Returns:
        Volume-weighted momentum value.
    """
    c = np.array(close, dtype=float)
    v = np.array(volume, dtype=float)

    n = len(c)
    if n < period + 1:
        return 0.0

    # Calculate returns
    returns = np.diff(c) / c[:-1]

    # Weight by volume (use volume from the return period)
    weighted_returns = returns[-(period):] * v[-(period):]
    total_volume = np.sum(v[-(period):])

    if total_volume == 0:
        return 0.0

    return float(np.sum(weighted_returns) / total_volume)


def TSMOM_volume_weighted_series(
    close: Union[List[float], np.ndarray],
    volume: Union[List[float], np.ndarray],
    period: int = 20,
) -> np.ndarray:
    """
    Full Volume-Weighted TSMOM series.

    Args:
        close: Close prices.
        volume: Volume data.
        period: Lookback period (default 20).

    Returns:
        np.ndarray of volume-weighted momentum values.
    """
    c = np.array(close, dtype=float)
    v = np.array(volume, dtype=float)

    n = len(c)
    if n < period + 1:
        return np.zeros(n)

    returns = np.zeros(n)
    returns[1:] = np.diff(c) / c[:-1]

    tsmom_vw = np.zeros(n)

    for i in range(period, n):
        weighted_sum = np.sum(returns[i - period + 1:i + 1] * v[i - period + 1:i + 1])
        vol_sum = np.sum(v[i - period + 1:i + 1])

        if vol_sum > 0:
            tsmom_vw[i] = weighted_sum / vol_sum

    return tsmom_vw
