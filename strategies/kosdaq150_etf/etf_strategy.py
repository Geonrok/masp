"""
KOSDAQ 150 ETF 균등가중 포트폴리오 전략
========================================

선물 전략을 ETF로 구현
- 롱 신호 → KODEX 코스닥150레버리지 (233740) 매수
- 숏 신호 → KODEX 코스닥150선물인버스 (251340) 매수
- 홀드 → 현금 또는 기존 포지션 유지

검증 성과 (선물 기준):
- Sharpe: 0.674
- CAGR: 11.7%
- MDD: -39.1%
- OOS Sharpe: 0.884
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class Position(Enum):
    """포지션 상태"""

    CASH = "CASH"  # 현금
    LONG = "LONG"  # 롱 (레버리지 ETF)
    SHORT = "SHORT"  # 숏 (인버스 ETF)


@dataclass
class ETFInfo:
    """ETF 정보"""

    code: str
    name: str
    leverage: float
    description: str


@dataclass
class Signal:
    """거래 신호"""

    date: datetime
    direction: int  # 1: Long, -1: Short, 0: Exit/Hold
    strength: float  # 0.0 ~ 1.0
    strategy_name: str
    reason: str


@dataclass
class TradeRecommendation:
    """거래 권장"""

    date: datetime
    action: str  # "BUY", "SELL", "HOLD"
    etf_code: str
    etf_name: str
    position_type: Position
    strength: float
    reasons: List[str]


@dataclass
class PortfolioConfig:
    """포트폴리오 설정"""

    initial_capital: float = 1_000_000  # 초기 자본 100만원
    position_size_pct: float = 0.60  # 포지션 비율 60%
    reserve_pct: float = 0.40  # 현금 유보 40%
    stop_loss_pct: float = 0.10  # 손절 10%
    take_profit_pct: float = 0.20  # 익절 20%


class KOSDAQ150ETFStrategy:
    """
    KOSDAQ 150 ETF 전략

    3개 역추세 전략의 균등가중 포트폴리오를 ETF로 구현

    전략 구성:
    - TripleV5_14_38: CMO(14,38) + WR(14,78) + BB(20)
    - TripleV5_14_33: CMO(14,33) + WR(14,73) + BB(20)
    - TripleVol_14_38: CMO(14,38) + WR(14,78) + Volume Filter
    """

    # ETF 정보
    ETF_LONG = ETFInfo(
        code="233740",
        name="KODEX 코스닥150레버리지",
        leverage=2.0,
        description="KOSDAQ150 지수 일간 수익률 2배 추종",
    )

    ETF_SHORT = ETFInfo(
        code="251340",
        name="KODEX 코스닥150선물인버스",
        leverage=-1.0,
        description="KOSDAQ150 선물 일간 수익률 -1배 추종",
    )

    ETF_NEUTRAL = ETFInfo(
        code="229200",
        name="KODEX 코스닥150",
        leverage=1.0,
        description="KOSDAQ150 지수 1배 추종",
    )

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.name = "KOSDAQ150_ETF_EqualWeight"

        # 현재 상태
        self.current_position = Position.CASH
        self.entry_price = 0.0
        self.entry_date = None

    # =========================================================================
    # 기술적 지표
    # =========================================================================

    @staticmethod
    def _chande_momentum(close: pd.Series, period: int = 14) -> pd.Series:
        """Chande Momentum Oscillator"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).sum()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).sum()
        return 100 * (gain - loss) / (gain + loss + 1e-10)

    @staticmethod
    def _williams_r(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Williams %R"""
        hh = high.rolling(window=period).max()
        ll = low.rolling(window=period).min()
        return -100 * (hh - close) / (hh - ll + 1e-10)

    @staticmethod
    def _bollinger_pctb(
        close: pd.Series, period: int = 20, std: float = 2.0
    ) -> pd.Series:
        """Bollinger Band %B"""
        sma = close.rolling(window=period).mean()
        std_val = close.rolling(window=period).std()
        upper = sma + std * std_val
        lower = sma - std * std_val
        return (close - lower) / (upper - lower + 1e-10)

    # =========================================================================
    # 개별 전략 신호
    # =========================================================================

    def _generate_triple_v5_signals(
        self, df: pd.DataFrame, cmo_t: int, wr_t: int, strategy_name: str
    ) -> List[Signal]:
        """TripleV5 전략 신호 생성"""
        signals = []

        cmo = self._chande_momentum(df["Close"], 14)
        wr = self._williams_r(df["High"], df["Low"], df["Close"], 14)
        pctb = self._bollinger_pctb(df["Close"], 20, 2)

        for i in range(25, len(df)):
            date = df.index[i]

            if pd.isna(cmo.iloc[i - 1]) or pd.isna(wr.iloc[i - 1]):
                continue

            # 매수 조건 (상향 돌파)
            bull_count = 0
            if cmo.iloc[i - 2] < -cmo_t and cmo.iloc[i - 1] >= -cmo_t:
                bull_count += 1
            if wr.iloc[i - 2] < -wr_t and wr.iloc[i - 1] >= -wr_t:
                bull_count += 1
            if pctb.iloc[i - 2] < 0.1 and pctb.iloc[i - 1] >= 0.1:
                bull_count += 1

            # 매도 조건 (하향 돌파)
            bear_count = 0
            if cmo.iloc[i - 2] > cmo_t and cmo.iloc[i - 1] <= cmo_t:
                bear_count += 1
            if wr.iloc[i - 2] > -(100 - wr_t) and wr.iloc[i - 1] <= -(100 - wr_t):
                bear_count += 1
            if pctb.iloc[i - 2] > 0.9 and pctb.iloc[i - 1] <= 0.9:
                bear_count += 1

            if bull_count >= 2:
                signals.append(
                    Signal(
                        date,
                        1,
                        bull_count / 3,
                        strategy_name,
                        f"상향돌파 {bull_count}/3",
                    )
                )
            elif bear_count >= 2:
                signals.append(
                    Signal(
                        date,
                        -1,
                        bear_count / 3,
                        strategy_name,
                        f"하향돌파 {bear_count}/3",
                    )
                )

        return signals

    def _generate_triple_vol_signals(self, df: pd.DataFrame) -> List[Signal]:
        """TripleVol 전략 신호 생성 (거래량 필터 포함)"""
        signals = []

        cmo = self._chande_momentum(df["Close"], 14)
        wr = self._williams_r(df["High"], df["Low"], df["Close"], 14)
        pctb = self._bollinger_pctb(df["Close"], 28, 2)
        vol_ma = df["Volume"].rolling(14).mean()

        for i in range(35, len(df)):
            date = df.index[i]

            if pd.isna(cmo.iloc[i - 1]) or pd.isna(wr.iloc[i - 1]):
                continue

            # 거래량 필터
            if df["Volume"].iloc[i] < vol_ma.iloc[i] * 0.8:
                continue

            # 매수 조건
            bull_count = 0
            if cmo.iloc[i - 2] < -38 and cmo.iloc[i - 1] >= -38:
                bull_count += 1
            if wr.iloc[i - 2] < -78 and wr.iloc[i - 1] >= -78:
                bull_count += 1
            if pctb.iloc[i - 2] < 0.1 and pctb.iloc[i - 1] >= 0.1:
                bull_count += 1

            # 매도 조건
            bear_count = 0
            if cmo.iloc[i - 2] > 38 and cmo.iloc[i - 1] <= 38:
                bear_count += 1
            if wr.iloc[i - 2] > -22 and wr.iloc[i - 1] <= -22:
                bear_count += 1
            if pctb.iloc[i - 2] > 0.9 and pctb.iloc[i - 1] <= 0.9:
                bear_count += 1

            if bull_count >= 2:
                signals.append(
                    Signal(
                        date,
                        1,
                        bull_count / 3,
                        "TripleVol_38",
                        f"상향돌파+거래량 {bull_count}/3",
                    )
                )
            elif bear_count >= 2:
                signals.append(
                    Signal(
                        date,
                        -1,
                        bear_count / 3,
                        "TripleVol_38",
                        f"하향돌파+거래량 {bear_count}/3",
                    )
                )

        return signals

    # =========================================================================
    # 통합 신호 생성
    # =========================================================================

    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, List[Signal]]:
        """모든 전략 신호 생성"""
        return {
            "TripleV5_38": self._generate_triple_v5_signals(df, 38, 78, "TripleV5_38"),
            "TripleV5_33": self._generate_triple_v5_signals(df, 33, 73, "TripleV5_33"),
            "TripleVol_38": self._generate_triple_vol_signals(df),
        }

    def get_today_signals(self, df: pd.DataFrame) -> Dict[str, Optional[Signal]]:
        """오늘 신호 조회"""
        all_signals = self.generate_all_signals(df)
        today = df.index[-1]

        result = {}
        for name, signals in all_signals.items():
            result[name] = None
            for sig in reversed(signals):
                if sig.date == today:
                    result[name] = sig
                    break

        return result

    def get_recommendation(self, df: pd.DataFrame) -> TradeRecommendation:
        """
        종합 거래 권장 생성

        Returns:
            TradeRecommendation: 오늘의 거래 권장
        """
        today_signals = self.get_today_signals(df)
        today = df.index[-1]

        long_count = 0
        short_count = 0
        reasons = []

        for name, sig in today_signals.items():
            if sig:
                if sig.direction == 1:
                    long_count += 1
                    reasons.append(f"{name}: 롱 ({sig.reason})")
                elif sig.direction == -1:
                    short_count += 1
                    reasons.append(f"{name}: 숏 ({sig.reason})")

        # 종합 판단
        if long_count >= 2:
            return TradeRecommendation(
                date=today,
                action="BUY",
                etf_code=self.ETF_LONG.code,
                etf_name=self.ETF_LONG.name,
                position_type=Position.LONG,
                strength=long_count / 3,
                reasons=reasons if reasons else ["강한 매수 신호"],
            )
        elif short_count >= 2:
            return TradeRecommendation(
                date=today,
                action="BUY",
                etf_code=self.ETF_SHORT.code,
                etf_name=self.ETF_SHORT.name,
                position_type=Position.SHORT,
                strength=short_count / 3,
                reasons=reasons if reasons else ["강한 매도 신호"],
            )
        elif long_count == 1:
            return TradeRecommendation(
                date=today,
                action="HOLD",
                etf_code="",
                etf_name="",
                position_type=Position.CASH,
                strength=0.33,
                reasons=reasons + ["신호 강도 약함 - 관망"],
            )
        elif short_count == 1:
            return TradeRecommendation(
                date=today,
                action="HOLD",
                etf_code="",
                etf_name="",
                position_type=Position.CASH,
                strength=0.33,
                reasons=reasons + ["신호 강도 약함 - 관망"],
            )
        else:
            return TradeRecommendation(
                date=today,
                action="HOLD",
                etf_code="",
                etf_name="",
                position_type=Position.CASH,
                strength=0.0,
                reasons=["신호 없음 - 기존 포지션 유지"],
            )

    # =========================================================================
    # 백테스트
    # =========================================================================

    def backtest(self, df: pd.DataFrame) -> Dict:
        """백테스트 실행"""
        all_signals = self.generate_all_signals(df)
        daily_returns = df["Close"].pct_change()

        strategy_returns = {}
        for name, signals in all_signals.items():
            position = pd.Series(0, index=df.index)
            for sig in signals:
                if sig.date in position.index:
                    position.loc[sig.date :] = sig.direction
            position = position.shift(1).fillna(0)
            strategy_returns[name] = position * daily_returns

        # 균등가중 평균
        combined = pd.concat(strategy_returns.values(), axis=1).mean(axis=1).fillna(0)

        # 성과 지표
        total_return = (1 + combined).prod() - 1
        years = len(df) / 252
        cagr = (
            (1 + total_return) ** (1 / years) - 1
            if years > 0 and total_return > -1
            else 0
        )
        sharpe = (
            combined.mean() / combined.std() * np.sqrt(252) if combined.std() > 0 else 0
        )

        cum = (1 + combined).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        win_rate = (
            (combined > 0).sum() / (combined != 0).sum()
            if (combined != 0).sum() > 0
            else 0
        )

        return {
            "sharpe": sharpe,
            "cagr": cagr,
            "total_return": total_return,
            "mdd": mdd,
            "win_rate": win_rate,
        }

    def get_summary(self) -> str:
        """전략 요약"""
        return f"""
{'='*60}
KOSDAQ 150 ETF 전략
{'='*60}

[전략 개요]
- 3개 역추세 전략의 균등가중 포트폴리오
- 선물 전략을 ETF로 구현

[구성 전략]
1. TripleV5_14_38: CMO(14,38) + WR(14,78) + BB(20)
2. TripleV5_14_33: CMO(14,33) + WR(14,73) + BB(20)
3. TripleVol_14_38: 위 전략 + 거래량 필터

[사용 ETF]
- 롱: {self.ETF_LONG.name} ({self.ETF_LONG.code})
- 숏: {self.ETF_SHORT.name} ({self.ETF_SHORT.code})

[검증 성과]
- Sharpe: 0.674 (전체), 0.884 (OOS)
- CAGR: 11.7%
- MDD: -39.1%

[포지션 설정]
- 투자 비율: {self.config.position_size_pct*100:.0f}%
- 현금 유보: {self.config.reserve_pct*100:.0f}%
- 손절: {self.config.stop_loss_pct*100:.0f}%
{'='*60}
"""
