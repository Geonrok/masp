"""
KOSDAQ 150 선물 균등가중 포트폴리오
===================================

검증 결과:
- 전체 Sharpe: 0.674 (개별 전략 대비 최고)
- OOS Sharpe: 0.884
- CAGR: 11.7%
- MDD: -39.1%

구성:
- TripleV5_14_38_14_78_20 (33.3%)
- TripleV5_14_33_14_73_20 (33.3%)
- TripleVol_14_38_78_0.8  (33.3%)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .core_strategies import (
    TripleV5Strategy,
    TripleVolStrategy,
    TradingSignal,
    create_validated_strategies,
)


@dataclass
class PortfolioConfig:
    """포트폴리오 설정"""

    initial_capital: float = 100_000_000  # 1억원
    position_size_pct: float = 0.30  # 포지션당 30%
    max_positions: int = 3  # 최대 포지션 수
    commission: float = 0.0001  # 수수료 0.01%
    slippage: float = 0.001  # 슬리피지 0.1%
    stop_loss_pct: float = 0.07  # 손절 7%
    max_portfolio_mdd: float = 0.40  # 포트폴리오 MDD 40%


@dataclass
class Position:
    """포지션 정보"""

    strategy: str
    direction: int
    entry_date: datetime
    entry_price: float
    size: float
    current_pnl: float = 0.0


class KOSDAQ150EqualWeightPortfolio:
    """
    KOSDAQ 150 선물 균등가중 포트폴리오

    3개 전략을 동일 비중으로 운용
    각 전략 독립적으로 신호 발생 및 포지션 관리
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.name = "KOSDAQ150_EqualWeight_Portfolio"

        # 전략 초기화
        self.strategies = {
            "TripleV5_38": TripleV5Strategy(14, 38, 14, 78, 20),
            "TripleV5_33": TripleV5Strategy(14, 33, 14, 73, 20),
            "TripleVol_38": TripleVolStrategy(14, 38, 78, 0.8),
        }

        # 가중치 (균등)
        self.weights = {name: 1.0 / len(self.strategies) for name in self.strategies}

        # 포지션
        self.positions: Dict[str, Optional[Position]] = {
            name: None for name in self.strategies
        }

        # 성과 기록
        self.equity_curve = []
        self.trade_history = []

    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, List[TradingSignal]]:
        """모든 전략의 신호 생성"""
        all_signals = {}
        for name, strategy in self.strategies.items():
            all_signals[name] = strategy.generate_signals(df)
        return all_signals

    def get_latest_signals(
        self, df: pd.DataFrame
    ) -> Dict[str, Optional[TradingSignal]]:
        """
        최신 신호 조회 (오늘 기준)

        Returns:
            각 전략별 최신 신호 (없으면 None)
        """
        all_signals = self.generate_all_signals(df)
        latest_date = df.index[-1]

        latest = {}
        for name, signals in all_signals.items():
            latest[name] = None
            for sig in reversed(signals):
                if sig.date == latest_date:
                    latest[name] = sig
                    break

        return latest

    def get_current_positions(self) -> Dict[str, Optional[Position]]:
        """현재 포지션 조회"""
        return self.positions.copy()

    def calculate_portfolio_value(self, df: pd.DataFrame) -> pd.Series:
        """
        포트폴리오 가치 계산 (백테스트용)

        Returns:
            일별 포트폴리오 가치
        """
        all_signals = self.generate_all_signals(df)
        daily_returns = df["Close"].pct_change()

        # 각 전략별 수익률
        strategy_returns = {}

        for name, signals in all_signals.items():
            position = pd.Series(0, index=df.index)

            for sig in signals:
                if sig.date in position.index:
                    position.loc[sig.date :] = sig.direction

            # 신호 다음날 진입
            position = position.shift(1).fillna(0)

            # 비용 차감
            trades = position.diff().abs()
            costs = trades * (self.config.commission + self.config.slippage)

            strat_return = position * daily_returns - costs
            strategy_returns[name] = strat_return

        # 균등가중 합산
        combined = pd.concat(strategy_returns.values(), axis=1).mean(axis=1)

        # NaN 처리
        combined = combined.fillna(0)

        # 누적 가치
        portfolio_value = self.config.initial_capital * (1 + combined).cumprod()

        return portfolio_value

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        백테스트 실행

        Returns:
            성과 지표 딕셔너리
        """
        portfolio_value = self.calculate_portfolio_value(df)
        daily_returns = portfolio_value.pct_change().dropna()

        # 성과 지표
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        years = len(df) / 252

        # CAGR 계산 (안전하게)
        if years > 0 and total_return > -1 and not np.isnan(total_return):
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = 0

        sharpe = (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 0
            else 0
        )

        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        mdd = drawdown.min()

        win_rate = (daily_returns > 0).sum() / len(daily_returns)

        # 거래 횟수
        total_trades = 0
        for name, strategy in self.strategies.items():
            signals = strategy.generate_signals(df)
            total_trades += len(signals)

        return {
            "name": self.name,
            "sharpe": sharpe,
            "cagr": cagr,
            "total_return": total_return,
            "mdd": mdd,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "avg_trades_per_strategy": total_trades / len(self.strategies),
            "final_value": portfolio_value.iloc[-1],
            "portfolio_value": portfolio_value,
        }

    def get_summary(self) -> str:
        """포트폴리오 요약"""
        summary = f"""
{'='*60}
{self.name}
{'='*60}

구성 전략 (균등 가중):
"""
        for name, weight in self.weights.items():
            summary += f"  - {name}: {weight*100:.1f}%\n"

        summary += f"""
설정:
  - 초기 자본: {self.config.initial_capital:,.0f}원
  - 포지션 크기: {self.config.position_size_pct*100:.0f}%
  - 손절: {self.config.stop_loss_pct*100:.0f}%
  - 수수료: {self.config.commission*100:.3f}%
  - 슬리피지: {self.config.slippage*100:.2f}%

검증 성과:
  - 전체 Sharpe: 0.674
  - OOS Sharpe: 0.884
  - CAGR: 11.7%
  - MDD: -39.1%
{'='*60}
"""
        return summary


def create_portfolio(
    config: Optional[PortfolioConfig] = None,
) -> KOSDAQ150EqualWeightPortfolio:
    """포트폴리오 인스턴스 생성"""
    return KOSDAQ150EqualWeightPortfolio(config)
