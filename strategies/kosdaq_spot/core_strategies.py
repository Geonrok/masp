# -*- coding: utf-8 -*-
"""
코스닥 현물 전략 - 핵심 전략 모듈
Multi_TF_Short 및 대안 전략 구현
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pandas as pd

from .indicators import calc_foreign_wght_mas, calc_price_ma, check_multi_tf_conditions


class SignalType(Enum):
    """신호 유형"""

    BUY = 1
    HOLD = 0
    SELL = -1


@dataclass
class TradingSignal:
    """거래 신호"""

    ticker: str
    signal_type: SignalType
    strength: float  # 0.0 ~ 1.0
    reason: str
    conditions: Dict[str, bool]
    price: float
    foreign_wght: float


@dataclass
class StrategyParams:
    """전략 파라미터"""

    wght_short1: int = 5
    wght_long1: int = 20
    wght_short2: int = 10
    wght_long2: int = 40
    price_ma: int = 50


class MultiTFShortStrategy:
    """
    Multi TF Short 전략

    진입 조건:
    1. 외국인 비중 5일 MA > 20일 MA (단기 상승)
    2. 외국인 비중 10일 MA > 40일 MA (중기 상승)
    3. 현재가 > 50일 MA (가격 상승 추세)

    청산 조건:
    - 진입 조건 중 하나라도 불만족
    """

    def __init__(self, params: Optional[StrategyParams] = None):
        self.params = params or StrategyParams()
        self.name = "Multi_TF_Short"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        전체 기간 신호 생성

        Args:
            df: OHLCV + 외국인 데이터 DataFrame

        Returns:
            신호 Series (1=매수, 0=홀드)
        """
        conditions = check_multi_tf_conditions(
            df,
            wght_short1=self.params.wght_short1,
            wght_long1=self.params.wght_long1,
            wght_short2=self.params.wght_short2,
            wght_long2=self.params.wght_long2,
            price_ma=self.params.price_ma,
        )

        signals = pd.Series(0, index=df.index)
        signals[conditions] = 1
        return signals

    def get_latest_signal(self, df: pd.DataFrame, ticker: str) -> TradingSignal:
        """
        최신 신호 생성

        Args:
            df: OHLCV + 외국인 데이터 DataFrame
            ticker: 종목 코드

        Returns:
            TradingSignal 객체
        """
        # 조건별 계산
        wght_s1, wght_l1 = calc_foreign_wght_mas(
            df, self.params.wght_short1, self.params.wght_long1
        )
        wght_s2, wght_l2 = calc_foreign_wght_mas(
            df, self.params.wght_short2, self.params.wght_long2
        )
        price_ma = calc_price_ma(df, self.params.price_ma)

        # 최신값
        df.index[-1]
        cond1 = wght_s1.iloc[-1] > wght_l1.iloc[-1]
        cond2 = wght_s2.iloc[-1] > wght_l2.iloc[-1]
        cond3 = df["close"].iloc[-1] > price_ma.iloc[-1]

        conditions = {
            f"wght_{self.params.wght_short1}>{self.params.wght_long1}": cond1,
            f"wght_{self.params.wght_short2}>{self.params.wght_long2}": cond2,
            f"price>ma{self.params.price_ma}": cond3,
        }

        # 신호 결정
        all_true = cond1 and cond2 and cond3
        signal_type = SignalType.BUY if all_true else SignalType.HOLD

        # 신호 강도 (충족 조건 수 / 전체 조건 수)
        strength = sum([cond1, cond2, cond3]) / 3.0

        # 이유 생성
        reasons = []
        if cond1:
            reasons.append(
                f"단기 외국인↑ ({wght_s1.iloc[-1]:.2f}>{wght_l1.iloc[-1]:.2f})"
            )
        if cond2:
            reasons.append(
                f"중기 외국인↑ ({wght_s2.iloc[-1]:.2f}>{wght_l2.iloc[-1]:.2f})"
            )
        if cond3:
            reasons.append(f"가격>MA{self.params.price_ma}")

        reason = ", ".join(reasons) if reasons else "조건 미충족"

        return TradingSignal(
            ticker=ticker,
            signal_type=signal_type,
            strength=strength,
            reason=reason,
            conditions=conditions,
            price=df["close"].iloc[-1],
            foreign_wght=df["wght"].iloc[-1],
        )


class ForeignScoreStrategy:
    """
    외국인 점수 기반 전략 (대안 전략)

    10개 조건 중 7개 이상 만족 시 진입
    """

    def __init__(self, threshold: int = 7):
        self.threshold = threshold
        self.name = "Foreign_Score_7"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """신호 생성"""
        df = df.copy()
        score = pd.Series(0.0, index=df.index)

        # 외국인 순매수 조건 (4점)
        for period in [3, 5, 10, 20]:
            df[f"f{period}"] = df["chg_qty"].rolling(period).sum()
            score += (df[f"f{period}"] > 0).astype(float)

        # 가격 정배열 (3점)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        score += (df["close"] > df["ma20"]).astype(float)
        score += (df["ma5"] > df["ma20"]).astype(float)
        score += (df["ma20"] > df["ma60"]).astype(float)

        # MACD (1점)
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        score += (macd > signal_line).astype(float)

        # 모멘텀 (1점)
        df["mom20"] = df["close"].pct_change(20)
        score += (df["mom20"] > 0).astype(float)

        # 거래량 (1점)
        df["vol_ma"] = df["volume"].rolling(20).mean()
        score += (df["volume"] > df["vol_ma"]).astype(float)

        signals = pd.Series(0, index=df.index)
        signals[score >= self.threshold] = 1
        return signals


def backtest_strategy(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000_000,
    fee_rate: float = 0.0015,
    slippage: float = 0.001,
) -> Dict:
    """
    단일 종목 백테스트

    Returns:
        결과 딕셔너리 (return_pct, trades, etc.)
    """
    capital = initial_capital
    position = 0
    shares = 0
    trades = 0

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0

        # 청산
        if position == 1 and signal == 0:
            capital += shares * price * (1 - slippage) * (1 - fee_rate * 1.5)
            position = 0
            shares = 0
            trades += 1

        # 진입
        if position == 0 and signal == 1:
            buy_price = price * (1 + slippage)
            shares = int(capital * 0.95 / buy_price)
            if shares > 0:
                capital -= shares * buy_price * (1 + fee_rate)
                position = 1
                trades += 1

    # 마지막 청산
    if position == 1 and shares > 0:
        capital += shares * df["close"].iloc[-1] * (1 - slippage) * (1 - fee_rate * 1.5)

    return_pct = (capital - initial_capital) / initial_capital * 100

    return {
        "return_pct": return_pct,
        "final_capital": capital,
        "trades": trades,
        "profitable": return_pct > 0,
    }
