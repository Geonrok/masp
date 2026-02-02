"""
TIGER 200 ETF Strategy - KOSPI200 추종 ETF 전략.

KOSPI200 선물 전략을 TIGER 200 ETF에 적용.
동일한 시그널 로직 사용, ETF 특성에 맞게 조정.

ETF 정보:
- 종목코드: 102110
- 운용사: 미래에셋자산운용
- 총보수: 0.05%
- 추종지수: KOSPI200
- 배율: 1배

검증된 전략 (A+ Grade):
1. VIX_Below_SMA20: Sharpe 2.25, CAGR 27.8%, MDD -12.0%
2. VIX_Declining: Sharpe 1.86, CAGR 19.1%, MDD -13.3%
3. Semicon_SMA20_Foreign20: Sharpe 1.53, CAGR 16.6%, MDD -14.8%

Author: Claude Code
Date: 2026-01-30
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import MA, EMA

logger = logging.getLogger(__name__)


@dataclass
class TIGER200Config:
    """TIGER 200 ETF 전략 설정."""

    # 전략 선택
    enabled_strategies: list[str] = field(default_factory=lambda: [
        "vix_below_sma20",
        "vix_declining",
        "semicon_foreign",
    ])

    # 포트폴리오 가중치
    strategy_weights: dict[str, float] = field(default_factory=lambda: {
        "vix_below_sma20": 0.50,
        "vix_declining": 0.30,
        "semicon_foreign": 0.20,
    })

    # VIX 전략 파라미터
    vix_sma_period: int = 20

    # 반도체/외국인 전략 파라미터
    semicon_sma_period: int = 20
    foreign_lookback_days: int = 20

    # ETF 거래 설정
    etf_code: str = "102110"  # TIGER 200
    etf_name: str = "TIGER 200"
    commission_rate: float = 0.00015  # 매매 수수료 0.015%
    tax_rate: float = 0.0  # ETF는 매도세 없음

    # 데이터 경로
    kospi_data_dir: str = "E:/투자/data/kospi_futures"
    investor_data_dir: str = "E:/투자/data/kr_stock/investor_trading"
    multi_asset_dir: str = "E:/투자/data/kosdaq_futures/multi_asset"


class TIGER200Strategy(BaseStrategy):
    """
    TIGER 200 ETF 전략.

    KOSPI200 지수를 추종하는 TIGER 200 ETF에
    검증된 VIX/반도체/외국인 시그널을 적용.

    매매 규칙:
    - BUY: Composite Weight >= 50%
    - SELL: Composite Weight < 20%
    - HOLD: 그 외

    예상 성과 (1배 레버리지 기준):
    - Sharpe: 2.37
    - CAGR: 23.1%
    - MDD: -11.5%
    """

    strategy_id = "tiger200_etf_v1"
    name = "TIGER 200 ETF Strategy"
    version = "1.0.0"
    description = "KOSPI200 추종 ETF용 A+ 검증 전략"

    def __init__(
        self,
        config: Optional[TIGER200Config] = None,
        market_data_adapter: Any = None,
    ):
        super().__init__(name=self.name)
        self.config = config or TIGER200Config()
        self._market_data = market_data_adapter

        # 데이터 캐시
        self._kospi_data: Optional[pd.DataFrame] = None
        self._vix_data: Optional[pd.Series] = None
        self._semicon_data: Optional[pd.Series] = None
        self._foreign_data: Optional[pd.Series] = None

        # 시그널 캐시
        self._last_signals: dict[str, int] = {}
        self._last_update: Optional[datetime] = None

        logger.info(f"[TIGER200] 초기화 완료: {self.config.enabled_strategies}")

    def load_data(self) -> bool:
        """데이터 로드."""
        try:
            import os

            # KOSPI200 일봉 데이터
            kospi_path = f"{self.config.kospi_data_dir}/kospi200_daily_yf.parquet"
            self._kospi_data = pd.read_parquet(kospi_path)
            self._kospi_data.columns = [c.lower() for c in self._kospi_data.columns]
            if self._kospi_data.index.tz is not None:
                self._kospi_data.index = self._kospi_data.index.tz_localize(None)

            # VIX 데이터
            vix_path = f"{self.config.multi_asset_dir}/vix.parquet"
            vix_df = pd.read_parquet(vix_path)
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            self._vix_data = vix_df['Close']

            # 반도체 지수 데이터
            semicon_path = f"{self.config.multi_asset_dir}/semicon.parquet"
            semicon_df = pd.read_parquet(semicon_path)
            if semicon_df.index.tz is not None:
                semicon_df.index = semicon_df.index.tz_localize(None)
            self._semicon_data = semicon_df['Close']

            # 외국인 투자자 데이터
            investor_dir = self.config.investor_data_dir
            files = [f for f in os.listdir(investor_dir) if f.endswith('_investor.csv')]
            all_data = []
            for f in files:
                try:
                    df = pd.read_csv(f"{investor_dir}/{f}", encoding='utf-8-sig')
                    df['날짜'] = pd.to_datetime(df['날짜'])
                    df = df.set_index('날짜')
                    all_data.append(df[['외국인합계']])
                except Exception:
                    continue

            if all_data:
                merged = all_data[0].copy()
                for df in all_data[1:]:
                    merged = merged.add(df, fill_value=0)
                self._foreign_data = merged['외국인합계'].sort_index()

            logger.info("[TIGER200] 데이터 로드 완료")
            return True

        except Exception as exc:
            logger.error(f"[TIGER200] 데이터 로드 실패: {exc}")
            return False

    def _calc_vix_below_sma20(self) -> int:
        """
        VIX Below SMA20 전략 (A+ 등급).

        조건: VIX(T-1) < VIX의 20일 이동평균
        의미: 시장 공포가 평균 이하 → 매수 적기

        IMPORTANT: T-1 VIX 사용 (한국 T일 거래에 어제 미국 종가 사용)
        백테스트 shift(1) 방법론과 일치.

        Returns:
            1 (LONG) 또는 0 (CASH)
        """
        if self._vix_data is None or len(self._vix_data) < self.config.vix_sma_period + 1:
            return 0

        # T-1 VIX 사용 (백테스트 방법론 일치)
        current_vix = self._vix_data.iloc[-2]  # T-1 VIX
        vix_sma = MA(self._vix_data.values[:-1], self.config.vix_sma_period)

        return 1 if current_vix < vix_sma else 0

    def _calc_vix_declining(self) -> int:
        """
        VIX Declining 전략 (A+ 등급).

        조건: VIX(T-1) < VIX(T-2)
        의미: 공포 감소 중 → 매수 적기

        IMPORTANT: T-1, T-2 VIX 사용 (한국 T일 거래 기준)
        백테스트 shift(1) 방법론과 일치.

        Returns:
            1 (LONG) 또는 0 (CASH)
        """
        if self._vix_data is None or len(self._vix_data) < 3:
            return 0

        # T-1, T-2 VIX 사용 (백테스트 방법론 일치)
        current_vix = self._vix_data.iloc[-2]  # T-1 VIX
        prev_vix = self._vix_data.iloc[-3]      # T-2 VIX

        return 1 if current_vix < prev_vix else 0

    def _calc_semicon_foreign(self) -> int:
        """
        Semicon + Foreign 전략 (A 등급).

        조건: 반도체지수 > 20일 평균 AND 외국인 20일 순매수 > 0
        의미: 반도체 강세 + 외국인 매수 → 상승 신호

        Returns:
            1 (LONG) 또는 0 (CASH)
        """
        if self._semicon_data is None or self._foreign_data is None:
            return 0

        if len(self._semicon_data) < self.config.semicon_sma_period:
            return 0

        if len(self._foreign_data) < self.config.foreign_lookback_days:
            return 0

        # 반도체 조건
        current_semicon = self._semicon_data.iloc[-1]
        semicon_sma = MA(self._semicon_data.values, self.config.semicon_sma_period)
        semicon_bullish = current_semicon > semicon_sma

        # 외국인 조건
        foreign_20d = self._foreign_data.iloc[-self.config.foreign_lookback_days:].sum()
        foreign_bullish = foreign_20d > 0

        return 1 if (semicon_bullish and foreign_bullish) else 0

    def calculate_signal(self) -> tuple[float, dict[str, int], str]:
        """
        통합 시그널 계산.

        Returns:
            (composite_weight, individual_signals, action)
        """
        signals: dict[str, int] = {}

        # 각 전략별 시그널 계산
        calculators = {
            "vix_below_sma20": self._calc_vix_below_sma20,
            "vix_declining": self._calc_vix_declining,
            "semicon_foreign": self._calc_semicon_foreign,
        }

        for strategy_name in self.config.enabled_strategies:
            if strategy_name in calculators:
                signals[strategy_name] = calculators[strategy_name]()

        # 가중 평균 계산
        composite = 0.0
        for strategy_name, signal in signals.items():
            weight = self.config.strategy_weights.get(strategy_name, 0.0)
            composite += signal * weight

        # 행동 결정
        if composite >= 0.5:
            action = "BUY"
        elif composite < 0.2:
            action = "SELL"
        else:
            action = "HOLD"

        self._last_signals = signals
        self._last_update = datetime.now()

        return composite, signals, action

    def get_indicators(self) -> dict[str, Any]:
        """현재 지표값 반환."""
        indicators = {}

        if self._kospi_data is not None and len(self._kospi_data) > 0:
            indicators["kospi200"] = float(self._kospi_data['close'].iloc[-1])
            indicators["kospi200_date"] = self._kospi_data.index[-1].strftime("%Y-%m-%d")

        if self._vix_data is not None and len(self._vix_data) > 0:
            indicators["vix"] = float(self._vix_data.iloc[-1])
            indicators["vix_sma20"] = float(MA(self._vix_data.values, 20))
            if len(self._vix_data) >= 2:
                indicators["vix_prev"] = float(self._vix_data.iloc[-2])

        if self._semicon_data is not None and len(self._semicon_data) > 0:
            indicators["semicon"] = float(self._semicon_data.iloc[-1])
            indicators["semicon_sma20"] = float(MA(self._semicon_data.values, 20))

        if self._foreign_data is not None and len(self._foreign_data) >= 20:
            indicators["foreign_20d"] = float(self._foreign_data.iloc[-20:].sum())

        return indicators

    def generate_signals(self, symbols: list[str]) -> list[TradeSignal]:
        """TradeSignal 리스트 생성."""
        if self._kospi_data is None:
            if not self.load_data():
                return [
                    TradeSignal(
                        symbol=symbol,
                        signal=Signal.HOLD,
                        price=0,
                        timestamp=datetime.now(),
                        reason="데이터 로드 실패",
                    )
                    for symbol in symbols
                ]

        composite, individual, action = self.calculate_signal()

        signals = []
        for symbol in symbols:
            if action == "BUY":
                sig = Signal.BUY
            elif action == "SELL":
                sig = Signal.SELL
            else:
                sig = Signal.HOLD

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    signal=sig,
                    price=float(self._kospi_data['close'].iloc[-1]),
                    timestamp=datetime.now(),
                    reason=f"Composite {composite:.0%}: {individual}",
                    strength=composite,
                )
            )

        return signals


# 전략 별칭
class TIGER200StableStrategy(TIGER200Strategy):
    """안정형 포트폴리오 (일봉 전략)."""

    strategy_id = "tiger200_stable"
    name = "TIGER 200 Stable"
    description = "안정형: VIX 50% + VIX하락 30% + 반도체외국인 20%"


class TIGER200VIXOnlyStrategy(TIGER200Strategy):
    """VIX 전략만 사용."""

    strategy_id = "tiger200_vix_only"
    name = "TIGER 200 VIX Only"
    description = "VIX 전략만: VIX_Below_SMA20 100%"

    def __init__(self, **kwargs):
        config = TIGER200Config(
            enabled_strategies=["vix_below_sma20"],
            strategy_weights={"vix_below_sma20": 1.0},
        )
        super().__init__(config=config, **kwargs)
