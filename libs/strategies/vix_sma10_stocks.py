"""
VIX < SMA10 KOSPI Individual Stock Strategy.

VIX가 10일 이동평균보다 낮을 때 매수 시그널 발생.
키움증권 소수점 거래 가능 종목에 최적화.

검증 결과 (Sharpe > 0.3):
- Tier 1 (Sharpe >= 1.0): 삼성전자, 삼성SDI, 카카오, SK하이닉스, LG화학
- Tier 2 (Sharpe 0.5-1.0): 현대차, SK이노베이션, 삼성전기 등
- Tier 3 (Sharpe 0.3-0.5): 네이버, LG, SK 등

타임존 검증:
- VIX[T]는 한국시간 T+1일 06:00에 확정
- 한국 장 개장 09:00 → 3시간 여유
- shift(1) on signal로 충분 (룩어헤드 바이어스 없음)

Author: Claude Code
Date: 2026-02-01
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from libs.strategies.base import BaseStrategy, Signal, TradeSignal
from libs.strategies.indicators import MA

logger = logging.getLogger(__name__)


# Kiwoom fractional trading available stocks with Sharpe > 0.3
VIX_VALID_STOCKS = {
    # Tier 1: Sharpe >= 1.0
    "005930": {"name": "삼성전자", "sharpe": 1.12, "cagr": "18.9%", "tier": 1},
    "006400": {"name": "삼성SDI", "sharpe": 1.12, "cagr": "27.9%", "tier": 1},
    "035720": {"name": "카카오", "sharpe": 1.11, "cagr": "25.0%", "tier": 1},
    "000660": {"name": "SK하이닉스", "sharpe": 1.10, "cagr": "26.3%", "tier": 1},
    "051910": {"name": "LG화학", "sharpe": 1.08, "cagr": "26.6%", "tier": 1},
    # Tier 2: Sharpe 0.5 - 1.0
    "009150": {"name": "삼성전기", "sharpe": 0.93, "cagr": "20.0%", "tier": 2},
    "005380": {"name": "현대차", "sharpe": 0.89, "cagr": "17.8%", "tier": 2},
    "010950": {"name": "S-Oil", "sharpe": 0.86, "cagr": "17.6%", "tier": 2},
    "096770": {"name": "SK이노베이션", "sharpe": 0.83, "cagr": "22.3%", "tier": 2},
    "086790": {"name": "하나금융지주", "sharpe": 0.72, "cagr": "13.0%", "tier": 2},
    "105560": {"name": "KB금융", "sharpe": 0.69, "cagr": "11.8%", "tier": 2},
    "005490": {"name": "POSCO홀딩스", "sharpe": 0.63, "cagr": "12.8%", "tier": 2},
    "012330": {"name": "현대모비스", "sharpe": 0.60, "cagr": "10.7%", "tier": 2},
    "036570": {"name": "엔씨소프트", "sharpe": 0.53, "cagr": "10.8%", "tier": 2},
    # Tier 3: Sharpe 0.3 - 0.5
    "032830": {"name": "삼성생명", "sharpe": 0.48, "cagr": "8.1%", "tier": 3},
    "034220": {"name": "LG디스플레이", "sharpe": 0.47, "cagr": "8.6%", "tier": 3},
    "055550": {"name": "신한지주", "sharpe": 0.46, "cagr": "6.4%", "tier": 3},
    "000270": {"name": "기아", "sharpe": 0.45, "cagr": "7.1%", "tier": 3},
    "003490": {"name": "대한항공", "sharpe": 0.44, "cagr": "7.3%", "tier": 3},
    "066570": {"name": "LG전자", "sharpe": 0.42, "cagr": "6.7%", "tier": 3},
    "028260": {"name": "삼성물산", "sharpe": 0.40, "cagr": "5.8%", "tier": 3},
    "034730": {"name": "SK", "sharpe": 0.39, "cagr": "6.2%", "tier": 3},
    "003550": {"name": "LG", "sharpe": 0.36, "cagr": "5.1%", "tier": 3},
    "035420": {"name": "네이버", "sharpe": 0.36, "cagr": "5.4%", "tier": 3},
    "018260": {"name": "삼성에스디에스", "sharpe": 0.31, "cagr": "4.3%", "tier": 3},
}


@dataclass
class VIXSMA10Config:
    """VIX < SMA10 전략 설정."""

    # SMA 기간
    vix_sma_period: int = 10

    # 티어별 필터링
    min_tier: int = 1  # 1=최상위만, 2=Tier1+2, 3=전체
    min_sharpe: float = 0.3

    # 데이터 경로
    vix_data_path: str = "E:/투자/data/global_indices/VIX.parquet"
    kospi_data_dir: str = "E:/투자/data/kr_stock/kospi_ohlcv"

    # yfinance 사용 (로컬 데이터 없을 때)
    use_yfinance: bool = True

    # 거래 비용
    commission_rate: float = 0.00015  # 매매 수수료 0.015%
    tax_rate: float = 0.0023  # 매도세 0.23%


class VIXSMA10StocksStrategy(BaseStrategy):
    """
    VIX < SMA10 KOSPI 개별 종목 전략.

    매매 규칙:
    - BUY: VIX(T-1) < VIX_SMA10(T-1)
    - SELL: VIX(T-1) >= VIX_SMA10(T-1)

    타임존 처리:
    - VIX는 미국 장 종가 기준 (한국 익일 06:00 확정)
    - 한국 장 개장 09:00에 3시간 여유
    - T-1 VIX를 사용하여 룩어헤드 바이어스 방지

    대상 종목:
    - 키움증권 소수점 거래 가능
    - VIX < SMA10 전략 Sharpe > 0.3 검증
    """

    strategy_id = "vix_sma10_stocks"
    name = "VIX SMA10 Individual Stocks"
    version = "1.0.0"
    description = "VIX < SMA10 전략 for KOSPI 개별 종목 (소수점 거래)"

    def __init__(
        self,
        config: Optional[VIXSMA10Config] = None,
        market_data_adapter: Any = None,
    ):
        super().__init__(name=self.name)
        self.config = config or VIXSMA10Config()
        self._market_data = market_data_adapter

        # 데이터 캐시
        self._vix_data: Optional[pd.Series] = None
        self._last_signal: Optional[int] = None
        self._last_update: Optional[datetime] = None

        # 유효 종목 필터링
        self._valid_symbols = self._filter_valid_symbols()

        logger.info(
            f"[VIXSMA10] 초기화: SMA{self.config.vix_sma_period}, "
            f"min_tier={self.config.min_tier}, 종목수={len(self._valid_symbols)}"
        )

    def _filter_valid_symbols(self) -> dict[str, dict]:
        """티어와 Sharpe 기준으로 유효 종목 필터링."""
        valid = {}
        for ticker, info in VIX_VALID_STOCKS.items():
            if info["tier"] <= self.config.min_tier:
                if info["sharpe"] >= self.config.min_sharpe:
                    valid[ticker] = info
        return valid

    def load_vix_data(self) -> bool:
        """VIX 데이터 로드."""
        try:
            from pathlib import Path

            vix_path = Path(self.config.vix_data_path)

            if vix_path.exists():
                # 로컬 parquet 파일 사용
                df = pd.read_parquet(vix_path)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                self._vix_data = df["Close"]
                logger.info(f"[VIXSMA10] VIX 로컬 데이터 로드: {len(self._vix_data)}일")
                return True

            elif self.config.use_yfinance:
                # yfinance로 다운로드
                import yfinance as yf

                logger.info("[VIXSMA10] VIX 데이터 yfinance에서 다운로드 중...")
                vix = yf.Ticker("^VIX")
                df = vix.history(period="2y")
                if len(df) > 0:
                    self._vix_data = df["Close"]
                    logger.info(
                        f"[VIXSMA10] VIX yfinance 로드: {len(self._vix_data)}일"
                    )
                    return True
                else:
                    logger.error("[VIXSMA10] yfinance VIX 데이터 없음")
                    return False

            else:
                logger.error(f"[VIXSMA10] VIX 데이터 없음: {vix_path}")
                return False

        except Exception as exc:
            logger.error(f"[VIXSMA10] VIX 로드 실패: {exc}")
            return False

    def calculate_signal(self) -> tuple[int, dict[str, Any]]:
        """
        VIX < SMA10 시그널 계산.

        Returns:
            (signal, metrics)
            signal: 1 (LONG) 또는 0 (CASH)
            metrics: VIX, SMA10, 판정 근거
        """
        if self._vix_data is None:
            if not self.load_vix_data():
                return 0, {"error": "VIX 데이터 로드 실패"}

        if len(self._vix_data) < self.config.vix_sma_period + 2:
            return 0, {"error": "VIX 데이터 부족"}

        # T-1 VIX 사용 (어제 미국 종가 = 오늘 한국 새벽 확정)
        # 백테스트에서 shift(1) 방법론과 일치
        vix_t1 = float(self._vix_data.iloc[-2])  # T-1 VIX
        vix_t2 = float(self._vix_data.iloc[-3])  # T-2 VIX (참고용)

        # SMA10 계산 (T-1까지의 데이터 사용)
        vix_values = self._vix_data.values[:-1]  # T-1까지
        vix_sma10 = MA(vix_values, self.config.vix_sma_period)

        # 시그널 판정
        signal = 1 if vix_t1 < vix_sma10 else 0

        metrics = {
            "vix_t1": round(vix_t1, 2),
            "vix_t2": round(vix_t2, 2),
            "vix_sma10": round(vix_sma10, 2),
            "vix_diff": round(vix_t1 - vix_sma10, 2),
            "vix_pct_diff": round((vix_t1 / vix_sma10 - 1) * 100, 2),
            "signal": "LONG" if signal else "CASH",
            "date": self._vix_data.index[-1].strftime("%Y-%m-%d"),
        }

        self._last_signal = signal
        self._last_update = datetime.now()

        return signal, metrics

    def get_valid_symbols(self) -> list[str]:
        """유효한 종목 코드 리스트 반환."""
        return list(self._valid_symbols.keys())

    def get_tier_symbols(self, tier: int) -> list[str]:
        """특정 티어 종목만 반환."""
        return [
            ticker
            for ticker, info in self._valid_symbols.items()
            if info["tier"] == tier
        ]

    def generate_signals(self, symbols: list[str]) -> list[TradeSignal]:
        """TradeSignal 리스트 생성."""
        # VIX 시그널 계산 (모든 종목 동일)
        signal_val, metrics = self.calculate_signal()

        if "error" in metrics:
            return [
                TradeSignal(
                    symbol=symbol,
                    signal=Signal.HOLD,
                    price=0,
                    timestamp=datetime.now(),
                    reason=metrics.get("error", "Unknown error"),
                )
                for symbol in symbols
            ]

        signals = []
        for symbol in symbols:
            # 유효 종목인지 확인
            if symbol in self._valid_symbols:
                stock_info = self._valid_symbols[symbol]
                if signal_val == 1:
                    sig = Signal.BUY
                    reason = (
                        f"VIX {metrics['vix_t1']:.1f} < SMA10 {metrics['vix_sma10']:.1f} | "
                        f"{stock_info['name']} Tier{stock_info['tier']} Sharpe {stock_info['sharpe']:.2f}"
                    )
                else:
                    sig = Signal.SELL
                    reason = (
                        f"VIX {metrics['vix_t1']:.1f} >= SMA10 {metrics['vix_sma10']:.1f} | "
                        f"CASH 전환"
                    )
            else:
                # 유효 종목 아님 → HOLD
                sig = Signal.HOLD
                reason = f"{symbol} is not in validated VIX stocks"

            signals.append(
                TradeSignal(
                    symbol=symbol,
                    signal=sig,
                    price=0,  # 실시간 가격은 adapter에서 제공
                    timestamp=datetime.now(),
                    reason=reason,
                    strength=float(signal_val),
                )
            )

        return signals

    def get_indicators(self) -> dict[str, Any]:
        """현재 지표값 반환."""
        if self._vix_data is None:
            self.load_vix_data()

        indicators = {}

        if self._vix_data is not None and len(self._vix_data) > 0:
            indicators["vix_current"] = float(self._vix_data.iloc[-1])
            indicators["vix_date"] = self._vix_data.index[-1].strftime("%Y-%m-%d")

            if len(self._vix_data) >= 2:
                indicators["vix_t1"] = float(self._vix_data.iloc[-2])

            if len(self._vix_data) >= self.config.vix_sma_period:
                indicators["vix_sma10"] = float(
                    MA(self._vix_data.values, self.config.vix_sma_period)
                )

        indicators["valid_stocks_count"] = len(self._valid_symbols)
        indicators["tier1_count"] = len(self.get_tier_symbols(1))
        indicators["tier2_count"] = len(self.get_tier_symbols(2))
        indicators["tier3_count"] = len(self.get_tier_symbols(3))

        return indicators


# Tier별 전략 별칭
class VIXSMA10Tier1Strategy(VIXSMA10StocksStrategy):
    """Tier 1 종목만 (Sharpe >= 1.0)."""

    strategy_id = "vix_sma10_tier1"
    name = "VIX SMA10 Tier 1 Only"
    description = "최상위 5종목: 삼성전자, 삼성SDI, 카카오, SK하이닉스, LG화학"

    def __init__(self, **kwargs):
        config = VIXSMA10Config(min_tier=1, min_sharpe=1.0)
        super().__init__(config=config, **kwargs)


class VIXSMA10Tier2Strategy(VIXSMA10StocksStrategy):
    """Tier 1+2 종목 (Sharpe >= 0.5)."""

    strategy_id = "vix_sma10_tier2"
    name = "VIX SMA10 Tier 1+2"
    description = "상위 14종목 (Sharpe >= 0.5)"

    def __init__(self, **kwargs):
        config = VIXSMA10Config(min_tier=2, min_sharpe=0.5)
        super().__init__(config=config, **kwargs)


class VIXSMA10AllTiersStrategy(VIXSMA10StocksStrategy):
    """전체 유효 종목 (Sharpe >= 0.3)."""

    strategy_id = "vix_sma10_all"
    name = "VIX SMA10 All Tiers"
    description = "전체 25종목 (Sharpe >= 0.3)"

    def __init__(self, **kwargs):
        config = VIXSMA10Config(min_tier=3, min_sharpe=0.3)
        super().__init__(config=config, **kwargs)
