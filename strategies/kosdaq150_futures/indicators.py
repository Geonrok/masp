"""
KOSDAQ 150 선물 전략용 기술적 지표
==================================
"""

import pandas as pd


def chande_momentum(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Chande Momentum Oscillator (CMO)

    범위: -100 ~ +100
    - 과매도: < -50
    - 과매수: > +50
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).sum()
    loss = (-delta).where(delta < 0, 0).rolling(window=period).sum()
    cmo = 100 * (gain - loss) / (gain + loss + 1e-10)
    return cmo


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Williams %R

    범위: -100 ~ 0
    - 과매도: < -80
    - 과매수: > -20
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr


def bollinger_percent_b(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """
    Bollinger Band %B

    범위: 일반적으로 0 ~ 1
    - 과매도: < 0.1
    - 과매수: > 0.9
    """
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    percent_b = (close - lower) / (upper - lower + 1e-10)
    return percent_b


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range"""
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple:
    """
    Average Directional Index

    Returns: (ADX, +DI, -DI)
    - ADX > 25: 강한 추세
    """
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_val = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / (atr_val + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr_val + 1e-10)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_val = dx.rolling(period).mean()

    return adx_val, plus_di, minus_di
