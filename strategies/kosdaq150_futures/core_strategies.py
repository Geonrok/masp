"""
KOSDAQ 150 선물 핵심 전략
=========================

검증 완료된 3개 역추세(Mean Reversion) 전략

전략 특성:
- 모두 CMO + Williams %R + Bollinger %B 기반
- 크로스오버(돌파) 신호 사용
- 3개 조건 중 2개 이상 충족 시 신호 발생
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from .indicators import chande_momentum, williams_r, bollinger_percent_b


@dataclass
class TradingSignal:
    """거래 신호"""
    date: datetime
    direction: int      # 1: Long, -1: Short, 0: Exit
    strength: float     # 0.0 ~ 1.0
    strategy: str       # 전략 이름
    reason: str         # 신호 사유


class TripleV5Strategy:
    """
    Triple V5 전략 (기본형)

    신호 조건:
    - CMO, WR, %B 3개 지표의 크로스오버
    - 2개 이상 동시 발생 시 신호

    매수 (Long):
    - CMO가 -threshold를 상향 돌파
    - WR이 -threshold를 상향 돌파
    - %B가 0.1을 상향 돌파

    매도 (Short):
    - CMO가 +threshold를 하향 돌파
    - WR이 -(100-threshold)를 하향 돌파
    - %B가 0.9를 하향 돌파
    """

    def __init__(self, cmo_period: int = 14, cmo_threshold: int = 38,
                 wr_period: int = 14, wr_threshold: int = 78,
                 bb_period: int = 20):
        self.cmo_period = cmo_period
        self.cmo_threshold = cmo_threshold
        self.wr_period = wr_period
        self.wr_threshold = wr_threshold
        self.bb_period = bb_period
        self.name = f"TripleV5_{cmo_period}_{cmo_threshold}_{wr_period}_{wr_threshold}_{bb_period}"

    def generate_signals(self, df: pd.DataFrame) -> List[TradingSignal]:
        """신호 생성"""
        signals = []

        cmo = chande_momentum(df['Close'], self.cmo_period)
        wr = williams_r(df['High'], df['Low'], df['Close'], self.wr_period)
        pct_b = bollinger_percent_b(df['Close'], self.bb_period, 2)

        start = max(self.cmo_period, self.wr_period, self.bb_period) + 5

        for i in range(start, len(df)):
            date = df.index[i]

            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]) or pd.isna(pct_b.iloc[i-1]):
                continue

            # 매수 조건 (상향 돌파)
            bull_count = 0
            if cmo.iloc[i-2] < -self.cmo_threshold and cmo.iloc[i-1] >= -self.cmo_threshold:
                bull_count += 1
            if wr.iloc[i-2] < -self.wr_threshold and wr.iloc[i-1] >= -self.wr_threshold:
                bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1:
                bull_count += 1

            # 매도 조건 (하향 돌파)
            bear_count = 0
            if cmo.iloc[i-2] > self.cmo_threshold and cmo.iloc[i-1] <= self.cmo_threshold:
                bear_count += 1
            if wr.iloc[i-2] > -(100 - self.wr_threshold) and wr.iloc[i-1] <= -(100 - self.wr_threshold):
                bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9:
                bear_count += 1

            # 신호 생성
            if bull_count >= 2:
                signals.append(TradingSignal(
                    date=date,
                    direction=1,
                    strength=bull_count / 3,
                    strategy=self.name,
                    reason=f"Bull crossover ({bull_count}/3 conditions)"
                ))
            elif bear_count >= 2:
                signals.append(TradingSignal(
                    date=date,
                    direction=-1,
                    strength=bear_count / 3,
                    strategy=self.name,
                    reason=f"Bear crossover ({bear_count}/3 conditions)"
                ))

        return signals


class TripleVolStrategy:
    """
    Triple Vol 전략 (거래량 필터 추가)

    TripleV5 + 거래량 조건
    - 거래량이 20일 평균의 vol_mult배 이상일 때만 신호
    """

    def __init__(self, period: int = 14, cmo_threshold: int = 38,
                 wr_threshold: int = 78, vol_mult: float = 0.8):
        self.period = period
        self.cmo_threshold = cmo_threshold
        self.wr_threshold = wr_threshold
        self.vol_mult = vol_mult
        self.name = f"TripleVol_{period}_{cmo_threshold}_{wr_threshold}_{vol_mult}"

    def generate_signals(self, df: pd.DataFrame) -> List[TradingSignal]:
        """신호 생성"""
        signals = []

        cmo = chande_momentum(df['Close'], self.period)
        wr = williams_r(df['High'], df['Low'], df['Close'], self.period)
        pct_b = bollinger_percent_b(df['Close'], self.period * 2, 2)
        vol_ma = df['Volume'].rolling(window=self.period).mean()

        start = self.period * 2 + 5

        for i in range(start, len(df)):
            date = df.index[i]

            if pd.isna(cmo.iloc[i-1]) or pd.isna(wr.iloc[i-1]) or pd.isna(pct_b.iloc[i-1]):
                continue

            # 거래량 필터
            if df['Volume'].iloc[i] < vol_ma.iloc[i] * self.vol_mult:
                continue

            # 매수 조건
            bull_count = 0
            if cmo.iloc[i-2] < -self.cmo_threshold and cmo.iloc[i-1] >= -self.cmo_threshold:
                bull_count += 1
            if wr.iloc[i-2] < -self.wr_threshold and wr.iloc[i-1] >= -self.wr_threshold:
                bull_count += 1
            if pct_b.iloc[i-2] < 0.1 and pct_b.iloc[i-1] >= 0.1:
                bull_count += 1

            # 매도 조건
            bear_count = 0
            if cmo.iloc[i-2] > self.cmo_threshold and cmo.iloc[i-1] <= self.cmo_threshold:
                bear_count += 1
            if wr.iloc[i-2] > -(100 - self.wr_threshold) and wr.iloc[i-1] <= -(100 - self.wr_threshold):
                bear_count += 1
            if pct_b.iloc[i-2] > 0.9 and pct_b.iloc[i-1] <= 0.9:
                bear_count += 1

            if bull_count >= 2:
                signals.append(TradingSignal(
                    date=date,
                    direction=1,
                    strength=bull_count / 3,
                    strategy=self.name,
                    reason=f"Bull + Volume ({bull_count}/3)"
                ))
            elif bear_count >= 2:
                signals.append(TradingSignal(
                    date=date,
                    direction=-1,
                    strength=bear_count / 3,
                    strategy=self.name,
                    reason=f"Bear + Volume ({bear_count}/3)"
                ))

        return signals


# 검증된 전략 인스턴스 생성
def create_validated_strategies() -> Dict[str, object]:
    """검증된 전략 인스턴스 생성"""
    return {
        'TripleV5_38': TripleV5Strategy(14, 38, 14, 78, 20),
        'TripleV5_33': TripleV5Strategy(14, 33, 14, 73, 20),
        'TripleVol_38': TripleVolStrategy(14, 38, 78, 0.8),
    }
