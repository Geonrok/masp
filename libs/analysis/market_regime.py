"""
Market Regime Detector

시장 국면을 분석하여 Bull/Bear/Sideways 상태를 판단합니다.

국면 정의:
- BULL: Price > MA50 > MA200 (상승 추세)
- BEAR: Price < MA50 < MA200 (하락 추세)
- SIDEWAYS: 그 외 (횡보/전환)

추가 지표:
- 추세 강도 (ADX 기반)
- 변동성 레짐 (ATR 기반)
- 모멘텀 상태 (TSMOM 기반)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path("E:/data/crypto_ohlcv")


class MarketRegime(Enum):
    """시장 국면"""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class VolatilityRegime(Enum):
    """변동성 레짐"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class MomentumState(Enum):
    """모멘텀 상태"""

    STRONG_UP = "strong_up"
    WEAK_UP = "weak_up"
    NEUTRAL = "neutral"
    WEAK_DOWN = "weak_down"
    STRONG_DOWN = "strong_down"


@dataclass
class RegimeAnalysis:
    """시장 국면 분석 결과"""

    regime: MarketRegime
    volatility: VolatilityRegime
    momentum: MomentumState

    # 상세 지표
    price: float
    ma20: float
    ma50: float
    ma200: float
    atr_pct: float  # ATR as % of price
    tsmom_30d: float  # 30일 모멘텀
    tsmom_90d: float  # 90일 모멘텀

    # 추세 강도 (0-100)
    trend_strength: float

    # 메시지
    message: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "volatility": self.volatility.value,
            "momentum": self.momentum.value,
            "price": self.price,
            "ma20": self.ma20,
            "ma50": self.ma50,
            "ma200": self.ma200,
            "atr_pct": self.atr_pct,
            "tsmom_30d": self.tsmom_30d,
            "tsmom_90d": self.tsmom_90d,
            "trend_strength": self.trend_strength,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class MarketRegimeDetector:
    """
    시장 국면 감지기

    BTC 가격 데이터를 분석하여 현재 시장 상태를 판단합니다.

    사용 지표:
    - MA20/50/200 기반 추세
    - ATR 기반 변동성
    - TSMOM 기반 모멘텀
    - ADX 기반 추세 강도
    """

    def __init__(
        self,
        short_ma: int = 20,
        mid_ma: int = 50,
        long_ma: int = 200,
        atr_period: int = 14,
        exchange: str = "upbit",
    ):
        self.short_ma = short_ma
        self.mid_ma = mid_ma
        self.long_ma = long_ma
        self.atr_period = atr_period
        self.exchange = exchange

        logger.info(
            f"[MarketRegime] Initialized: MA{short_ma}/{mid_ma}/{long_ma}, ATR{atr_period}"
        )

    def _calc_sma(self, prices: np.ndarray, period: int) -> float:
        """SMA 계산"""
        if len(prices) < period:
            return np.nan
        return np.mean(prices[-period:])

    def _calc_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """ATR 계산"""
        n = len(close)
        if n < 2:
            return 0.0

        tr = np.zeros(n)
        tr[0] = high[0] - low[0]

        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        if n <= period:
            return np.mean(tr)

        return np.mean(tr[-period:])

    def _calc_adx(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """ADX 계산 (추세 강도)"""
        n = len(close)
        if n < period + 1:
            return 50.0  # Neutral

        # +DM, -DM 계산
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Smoothed values
        smooth_plus_dm = np.mean(plus_dm[-period:])
        smooth_minus_dm = np.mean(minus_dm[-period:])
        smooth_tr = np.mean(tr[-period:])

        if smooth_tr == 0:
            return 50.0

        plus_di = 100 * smooth_plus_dm / smooth_tr
        minus_di = 100 * smooth_minus_dm / smooth_tr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 50.0

        dx = 100 * abs(plus_di - minus_di) / di_sum

        return min(100, max(0, dx))

    def _calc_volatility_percentile(
        self, atr_values: np.ndarray, lookback: int = 252
    ) -> float:
        """변동성 백분위 계산"""
        if len(atr_values) < lookback:
            lookback = len(atr_values)

        if lookback < 20:
            return 50.0

        recent_atrs = atr_values[-lookback:]
        current_atr = atr_values[-1]

        percentile = np.sum(recent_atrs < current_atr) / len(recent_atrs) * 100
        return percentile

    def load_btc_data(self) -> Optional[pd.DataFrame]:
        """BTC 데이터 로드"""
        folder = DATA_ROOT / f"{self.exchange}_1d"
        if not folder.exists():
            logger.warning(f"[MarketRegime] Data folder not found: {folder}")
            return None

        btc_file = None
        for f in folder.glob("*.csv"):
            if "BTC" in f.stem.upper() and "DOWN" not in f.stem.upper():
                btc_file = f
                break

        if btc_file is None:
            logger.warning("[MarketRegime] BTC data not found")
            return None

        try:
            df = pd.read_csv(btc_file)
            date_col = [
                c for c in df.columns if "date" in c.lower() or "time" in c.lower()
            ]
            if not date_col:
                return None

            df["date"] = pd.to_datetime(df[date_col[0]]).dt.normalize()
            df = df.set_index("date").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                return None

            return df[required]
        except Exception as e:
            logger.error(f"[MarketRegime] Failed to load BTC data: {e}")
            return None

    def analyze(self, df: Optional[pd.DataFrame] = None) -> RegimeAnalysis:
        """
        시장 국면 분석

        Args:
            df: OHLCV 데이터프레임 (없으면 자동 로드)

        Returns:
            RegimeAnalysis 객체
        """
        if df is None:
            df = self.load_btc_data()

        if df is None or len(df) < self.long_ma + 10:
            return RegimeAnalysis(
                regime=MarketRegime.UNKNOWN,
                volatility=VolatilityRegime.NORMAL,
                momentum=MomentumState.NEUTRAL,
                price=0,
                ma20=0,
                ma50=0,
                ma200=0,
                atr_pct=0,
                tsmom_30d=0,
                tsmom_90d=0,
                trend_strength=50,
                message="Insufficient data for analysis",
            )

        # 가격 데이터
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        current_price = close[-1]

        # 이동평균 계산
        ma20 = self._calc_sma(close, self.short_ma)
        ma50 = self._calc_sma(close, self.mid_ma)
        ma200 = self._calc_sma(close, self.long_ma)

        # ATR 계산
        atr = self._calc_atr(high, low, close, self.atr_period)
        atr_pct = (atr / current_price) * 100 if current_price > 0 else 0

        # 모멘텀 계산
        tsmom_30d = (
            (current_price - close[-31]) / close[-31] * 100 if len(close) > 31 else 0
        )
        tsmom_90d = (
            (current_price - close[-91]) / close[-91] * 100 if len(close) > 91 else 0
        )

        # ADX (추세 강도) 계산
        adx = self._calc_adx(high, low, close, 14)

        # === 국면 판단 ===

        # 1. 추세 국면 판단
        if current_price > ma50 > ma200:
            regime = MarketRegime.BULL
        elif current_price < ma50 < ma200:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.SIDEWAYS

        # 2. 변동성 레짐 판단
        # 역사적 ATR% 기준
        if atr_pct < 2.0:
            volatility = VolatilityRegime.LOW
        elif atr_pct < 4.0:
            volatility = VolatilityRegime.NORMAL
        elif atr_pct < 7.0:
            volatility = VolatilityRegime.HIGH
        else:
            volatility = VolatilityRegime.EXTREME

        # 3. 모멘텀 상태 판단
        if tsmom_30d > 15:
            momentum = MomentumState.STRONG_UP
        elif tsmom_30d > 5:
            momentum = MomentumState.WEAK_UP
        elif tsmom_30d > -5:
            momentum = MomentumState.NEUTRAL
        elif tsmom_30d > -15:
            momentum = MomentumState.WEAK_DOWN
        else:
            momentum = MomentumState.STRONG_DOWN

        # 4. 메시지 생성
        message = self._generate_message(
            regime, volatility, momentum, adx, current_price, ma20, ma50, ma200
        )

        return RegimeAnalysis(
            regime=regime,
            volatility=volatility,
            momentum=momentum,
            price=current_price,
            ma20=ma20,
            ma50=ma50,
            ma200=ma200,
            atr_pct=atr_pct,
            tsmom_30d=tsmom_30d,
            tsmom_90d=tsmom_90d,
            trend_strength=adx,
            message=message,
        )

    def _generate_message(
        self,
        regime: MarketRegime,
        volatility: VolatilityRegime,
        momentum: MomentumState,
        adx: float,
        price: float,
        ma20: float,
        ma50: float,
        ma200: float,
    ) -> str:
        """분석 메시지 생성"""
        # 국면별 기본 메시지
        regime_msg = {
            MarketRegime.BULL: "Bullish trend confirmed",
            MarketRegime.BEAR: "Bearish trend confirmed",
            MarketRegime.SIDEWAYS: "Market in consolidation/transition",
            MarketRegime.UNKNOWN: "Unable to determine regime",
        }[regime]

        # 추세 강도
        if adx > 40:
            strength_msg = "Strong trend"
        elif adx > 25:
            strength_msg = "Moderate trend"
        else:
            strength_msg = "Weak/No trend"

        # 변동성 경고
        vol_msg = ""
        if volatility == VolatilityRegime.HIGH:
            vol_msg = " - High volatility!"
        elif volatility == VolatilityRegime.EXTREME:
            vol_msg = " - EXTREME volatility!"

        # MA 상태
        if price > ma20 > ma50 > ma200:
            ma_msg = "All MAs aligned bullish"
        elif price < ma20 < ma50 < ma200:
            ma_msg = "All MAs aligned bearish"
        elif price > ma200:
            ma_msg = "Above MA200 (long-term bullish)"
        elif price < ma200:
            ma_msg = "Below MA200 (long-term bearish)"
        else:
            ma_msg = "MAs mixed"

        return f"{regime_msg}. {strength_msg}. {ma_msg}.{vol_msg}"

    def get_trading_recommendation(self, analysis: RegimeAnalysis) -> dict:
        """
        국면에 따른 매매 권고

        Returns:
            권고 정보 딕셔너리
        """
        recommendations = {
            (MarketRegime.BULL, VolatilityRegime.LOW): {
                "action": "적극 매수",
                "position_size": 1.0,
                "message": "강한 상승장 + 낮은 변동성 - 최대 포지션 권장",
            },
            (MarketRegime.BULL, VolatilityRegime.NORMAL): {
                "action": "매수",
                "position_size": 0.8,
                "message": "상승장 - 일반적인 롱 포지션",
            },
            (MarketRegime.BULL, VolatilityRegime.HIGH): {
                "action": "신중한 매수",
                "position_size": 0.5,
                "message": "상승장이나 높은 변동성 - 포지션 축소 권장",
            },
            (MarketRegime.BULL, VolatilityRegime.EXTREME): {
                "action": "대기",
                "position_size": 0.2,
                "message": "극심한 변동성 - 안정화 대기",
            },
            (MarketRegime.BEAR, VolatilityRegime.LOW): {
                "action": "관망",
                "position_size": 0.0,
                "message": "하락장 - 롱 포지션 회피",
            },
            (MarketRegime.BEAR, VolatilityRegime.NORMAL): {
                "action": "관망",
                "position_size": 0.0,
                "message": "하락장 - 롱 포지션 회피",
            },
            (MarketRegime.BEAR, VolatilityRegime.HIGH): {
                "action": "관망",
                "position_size": 0.0,
                "message": "하락장 + 높은 변동성 - 현금 보유",
            },
            (MarketRegime.BEAR, VolatilityRegime.EXTREME): {
                "action": "관망",
                "position_size": 0.0,
                "message": "하락장 + 극심한 변동성 - 자본 보존 우선",
            },
            (MarketRegime.SIDEWAYS, VolatilityRegime.LOW): {
                "action": "대기",
                "position_size": 0.3,
                "message": "횡보장 - 돌파 대기",
            },
            (MarketRegime.SIDEWAYS, VolatilityRegime.NORMAL): {
                "action": "선별 매수",
                "position_size": 0.4,
                "message": "횡보장 - 선별적 진입만",
            },
            (MarketRegime.SIDEWAYS, VolatilityRegime.HIGH): {
                "action": "신중",
                "position_size": 0.2,
                "message": "횡보장 + 높은 변동성 - 주의 필요",
            },
            (MarketRegime.SIDEWAYS, VolatilityRegime.EXTREME): {
                "action": "대기",
                "position_size": 0.1,
                "message": "불확실성 높음 - 최소 노출",
            },
        }

        key = (analysis.regime, analysis.volatility)
        return recommendations.get(
            key, {"action": "대기", "position_size": 0.3, "message": "시장 상황 불확실"}
        )

    def format_telegram_message(self, analysis: RegimeAnalysis) -> str:
        """텔레그램 메시지 포맷팅 (한글)"""
        recommendation = self.get_trading_recommendation(analysis)

        # 국면 한글
        regime_kr = {
            MarketRegime.BULL: "상승장",
            MarketRegime.BEAR: "하락장",
            MarketRegime.SIDEWAYS: "횡보장",
            MarketRegime.UNKNOWN: "불명",
        }[analysis.regime]

        # 변동성 한글
        vol_kr = {
            VolatilityRegime.LOW: "낮음",
            VolatilityRegime.NORMAL: "보통",
            VolatilityRegime.HIGH: "높음",
            VolatilityRegime.EXTREME: "매우높음",
        }[analysis.volatility]

        # 모멘텀 한글
        mom_kr = {
            MomentumState.STRONG_UP: "강한 상승",
            MomentumState.WEAK_UP: "약한 상승",
            MomentumState.NEUTRAL: "중립",
            MomentumState.WEAK_DOWN: "약한 하락",
            MomentumState.STRONG_DOWN: "강한 하락",
        }[analysis.momentum]

        lines = [
            f"<b>[MASP] 시장 국면 분석</b>",
            f"시간: {analysis.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "",
            f"<b>국면: {regime_kr}</b>",
            f"변동성: {vol_kr} (ATR {analysis.atr_pct:.1f}%)",
            f"모멘텀: {mom_kr}",
            f"추세강도: {analysis.trend_strength:.0f}/100",
            "",
            f"<b>BTC: {analysis.price:,.0f}원</b>",
            f"  MA20: {analysis.ma20:,.0f}원 ({(analysis.price - analysis.ma20) / analysis.ma20 * 100:+.1f}%)",
            f"  MA50: {analysis.ma50:,.0f}원 ({(analysis.price - analysis.ma50) / analysis.ma50 * 100:+.1f}%)",
            f"  MA200: {analysis.ma200:,.0f}원 ({(analysis.price - analysis.ma200) / analysis.ma200 * 100:+.1f}%)",
            "",
            f"<b>모멘텀</b>",
            f"  30일: {analysis.tsmom_30d:+.1f}%",
            f"  90일: {analysis.tsmom_90d:+.1f}%",
            "",
            f"<b>권고: {recommendation['action']}</b>",
            f"포지션: {recommendation['position_size']*100:.0f}%",
            f"{recommendation['message']}",
        ]

        return "\n".join(lines)


def main():
    """CLI 실행"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("MASP Market Regime Detector")
    print("=" * 60)

    detector = MarketRegimeDetector()
    analysis = detector.analyze()

    print(f"\nTimestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M')}")
    print(f"\nRegime: {analysis.regime.value.upper()}")
    print(f"Volatility: {analysis.volatility.value}")
    print(f"Momentum: {analysis.momentum.value}")
    print(f"Trend Strength: {analysis.trend_strength:.0f}/100")

    print(f"\nBTC Price: {analysis.price:,.0f}")
    print(f"  MA20: {analysis.ma20:,.0f}")
    print(f"  MA50: {analysis.ma50:,.0f}")
    print(f"  MA200: {analysis.ma200:,.0f}")

    print(f"\nMomentum")
    print(f"  30D: {analysis.tsmom_30d:+.1f}%")
    print(f"  90D: {analysis.tsmom_90d:+.1f}%")
    print(f"  ATR: {analysis.atr_pct:.1f}%")

    print(f"\nMessage: {analysis.message}")

    recommendation = detector.get_trading_recommendation(analysis)
    print(f"\nRecommendation: {recommendation['action']}")
    print(f"Position Size: {recommendation['position_size']*100:.0f}%")
    print(f"{recommendation['message']}")

    print("\n" + "-" * 60)
    print("Telegram Message Preview:")
    print("-" * 60)
    print(
        detector.format_telegram_message(analysis)
        .replace("<b>", "")
        .replace("</b>", "")
    )


if __name__ == "__main__":
    main()
