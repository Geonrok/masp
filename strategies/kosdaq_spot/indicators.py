# -*- coding: utf-8 -*-
"""
코스닥 현물 전략 - 기술적 지표 모듈
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calc_moving_average(series: pd.Series, period: int) -> pd.Series:
    """단순 이동평균"""
    return series.rolling(period).mean()


def calc_foreign_wght_mas(
    df: pd.DataFrame, short: int, long: int
) -> Tuple[pd.Series, pd.Series]:
    """
    외국인 비중 이동평균 계산

    Args:
        df: wght 컬럼을 포함한 DataFrame
        short: 단기 이동평균 기간
        long: 장기 이동평균 기간

    Returns:
        (단기 MA, 장기 MA) 튜플
    """
    wght_short = df["wght"].rolling(short).mean()
    wght_long = df["wght"].rolling(long).mean()
    return wght_short, wght_long


def calc_price_ma(df: pd.DataFrame, period: int) -> pd.Series:
    """가격 이동평균 계산"""
    return df["close"].rolling(period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """MACD 계산"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal


def check_multi_tf_conditions(
    df: pd.DataFrame,
    wght_short1: int = 5,
    wght_long1: int = 20,
    wght_short2: int = 10,
    wght_long2: int = 40,
    price_ma: int = 50,
) -> pd.Series:
    """
    Multi TF Short 조건 확인

    Args:
        df: OHLCV + 외국인 데이터가 포함된 DataFrame
        wght_short1: 단기 외국인 비중 MA 기간
        wght_long1: 단기 비교 외국인 비중 MA 기간
        wght_short2: 중기 외국인 비중 MA 기간
        wght_long2: 중기 비교 외국인 비중 MA 기간
        price_ma: 가격 MA 기간

    Returns:
        Boolean Series (True = 진입 조건 충족)
    """
    # 단기 외국인 비중 상승
    wght_s1, wght_l1 = calc_foreign_wght_mas(df, wght_short1, wght_long1)
    cond1 = wght_s1 > wght_l1

    # 중기 외국인 비중 상승
    wght_s2, wght_l2 = calc_foreign_wght_mas(df, wght_short2, wght_long2)
    cond2 = wght_s2 > wght_l2

    # 가격 상승 추세
    ma = calc_price_ma(df, price_ma)
    cond3 = df["close"] > ma

    return cond1 & cond2 & cond3
