"""
MASP 발굴 전략 모음 (Discovered Strategies)
===========================================
KOSPI/KOSDAQ 대규모 백테스팅을 통해 검증된 고성능 전략들

포함된 전략:
1. AdaptiveEnsembleStrategy: 시장 상황 적응형 앙상블
2. SeasonalityStrategy: 월별/계절성 전략 (11월 효과, Sell in May)
3. MLSignalStrategy: Random Forest 기반 시그널
4. PairsTradingStrategy: 통계적 차익거래 (Spread Mean Reversion)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# 공통 지표 계산
# =============================================================================
class Indicators:
    @staticmethod
    def sma(s: pd.Series, p: int) -> pd.Series:
        return s.rolling(p, min_periods=1).mean()

    @staticmethod
    def ema(s: pd.Series, p: int) -> pd.Series:
        return s.ewm(span=p, adjust=False).mean()

    @staticmethod
    def rsi(s: pd.Series, p: int = 14) -> pd.Series:
        d = s.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        l = (-d.where(d < 0, 0)).rolling(p).mean()
        return 100 - (100 / (1 + g / (l + 1e-10)))

    @staticmethod
    def volatility(s: pd.Series, p: int = 20) -> pd.Series:
        return s.pct_change().rolling(p).std() * np.sqrt(252)

    @staticmethod
    def momentum(s: pd.Series, p: int = 60) -> pd.Series:
        return s / s.shift(p) - 1

    @staticmethod
    def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        fast_ema = s.ewm(span=fast, adjust=False).mean()
        slow_ema = s.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line


# =============================================================================
# 1. 적응형 앙상블 전략
# =============================================================================
class AdaptiveEnsembleStrategy:
    """
    시장 변동성과 추세에 따라 가중치를 조절하는 앙상블 전략
    High Volatility -> Trend Following 가중
    Low Volatility -> Mean Reversion/Momentum 가중
    """

    def __init__(self, direction: str = "long"):
        self.direction = 1 if direction == "long" else -1

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        # 1. Market State Identification
        vol = Indicators.volatility(close, 20)
        vol_median = vol.rolling(252).median()
        high_vol = vol > vol_median

        trend_up = close > Indicators.sma(close, 100)

        # 2. Component Signals
        # Trend
        sma_50 = Indicators.sma(close, 50)
        sig_trend = (close > sma_50).astype(float)

        # Momentum
        sig_mom = (Indicators.momentum(close, 30) > 0).astype(float)

        # RSI (Mean Reversion)
        rsi_val = Indicators.rsi(close, 14)
        sig_rsi = ((rsi_val > 35) & (rsi_val < 65)).astype(float)

        # 3. Dynamic Weighting
        w_trend = pd.Series(0.4, index=df.index)
        w_mom = pd.Series(0.4, index=df.index)
        w_rsi = pd.Series(0.2, index=df.index)

        # High Volatility: Trust Trend more
        w_trend[high_vol] = 0.5
        w_mom[high_vol] = 0.3
        w_rsi[high_vol] = 0.2

        # Downtrend: Conservative
        w_trend[~trend_up] = 0.6
        w_mom[~trend_up] = 0.2
        w_rsi[~trend_up] = 0.2

        # 4. Ensemble Score
        score = (sig_trend * w_trend) + (sig_mom * w_mom) + (sig_rsi * w_rsi)

        # 5. Final Signal
        signal = (score > 0.5).astype(int)

        if self.direction == -1:
            signal = 1 - signal

        return signal


# =============================================================================
# 2. 계절성 전략
# =============================================================================
class SeasonalityStrategy:
    """
    강력한 캘린더 효과를 활용한 전략
    mode: 'month_11' (11월 효과), 'sell_in_may' (11-4월 투자)
    """

    def __init__(self, mode: str = "month_11"):
        self.mode = mode

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        month = df.index.month
        signal = pd.Series(0, index=df.index)

        if self.mode == "month_11":
            # Only invest in November
            signal = (month == 11).astype(int)

        elif self.mode == "sell_in_may":
            # Invest Nov to Apr
            signal = month.isin([11, 12, 1, 2, 3, 4]).astype(int)

        elif self.mode == "best_6":
            # Korean market specific best months (Apr-Jun, Oct-Dec)
            signal = month.isin([4, 5, 6, 10, 11, 12]).astype(int)

        return signal


# =============================================================================
# 3. ML 기반 시그널 전략 (Random Forest Simplified)
# =============================================================================
class MLSignalStrategy:
    """
    Random Forest의 로직을 간소화한 규칙 기반 시그널
    다중 기술적 지표의 가중 투표 방식
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]

        # Features
        ret_5 = close.pct_change(5)
        ret_10 = close.pct_change(10)
        ret_20 = close.pct_change(20)

        sma_5_ratio = close / Indicators.sma(close, 5) - 1
        sma_20_ratio = close / Indicators.sma(close, 20) - 1

        macd, signal_line = Indicators.macd(close)
        macd_hist = macd - signal_line

        rsi_val = Indicators.rsi(close)

        # Voting (Weighted)
        score = (
            (ret_5 > 0).astype(float) * 0.15
            + (ret_10 > 0).astype(float) * 0.15
            + (ret_20 > 0).astype(float) * 0.20
            + (sma_5_ratio > 0).astype(float) * 0.10
            + (sma_20_ratio > 0).astype(float) * 0.15
            + (macd_hist > 0).astype(float) * 0.15
            + ((rsi_val > 30) & (rsi_val < 70)).astype(float) * 0.10
        )

        signal = (score > self.threshold).astype(int)
        return signal


# =============================================================================
# 4. 페어 트레이딩 전략
# =============================================================================
class PairsTradingStrategy:
    """
    두 자산 간의 Spread Mean Reversion 전략
    """

    def __init__(
        self, hedge_ratio: float = 1.0, entry_z: float = 2.0, exit_z: float = 0.5
    ):
        self.hedge_ratio = hedge_ratio
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signal(self, s1: pd.Series, s2: pd.Series) -> pd.Series:
        # Align index
        common_idx = s1.index.intersection(s2.index)
        s1 = s1.loc[common_idx]
        s2 = s2.loc[common_idx]

        # Spread
        spread = s1 - self.hedge_ratio * s2

        # Z-Score
        window = 20
        z_score = (spread - spread.rolling(window).mean()) / spread.rolling(
            window
        ).std()

        # Position Logic
        positions = pd.Series(0, index=z_score.index)
        current_pos = 0

        for i in range(window, len(z_score)):
            z = z_score.iloc[i]
            if pd.isna(z):
                continue

            if current_pos == 0:
                if z < -self.entry_z:
                    current_pos = 1  # Long Spread (Buy S1, Sell S2)
                elif z > self.entry_z:
                    current_pos = -1  # Short Spread (Sell S1, Buy S2)
            elif current_pos == 1:
                if z > -self.exit_z:
                    current_pos = 0
            elif current_pos == -1:
                if z < self.exit_z:
                    current_pos = 0

            positions.iloc[i] = current_pos

        return positions
