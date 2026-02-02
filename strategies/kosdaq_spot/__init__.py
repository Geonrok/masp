# -*- coding: utf-8 -*-
"""
코스닥 현물 외국인 멀티 타임프레임 전략
=====================================

전략 개요:
- 시총 상위 1/8 종목 대상 (유동성 상위 50% 내)
- 외국인 비중 단기/중기 이동평균 크로스 + 가격 추세 조합
- OOS 수익률: 78.6% 종목에서 수익

사용법:
    from strategies.kosdaq_spot import KOSDAQSpotSignalGenerator
    generator = KOSDAQSpotSignalGenerator()
    generator.run()

모듈 구성:
    - indicators: 기술적 지표 계산
    - universe_manager: 유니버스 관리 (시총 상위 1/8)
    - core_strategies: Multi_TF_Short 전략 구현
    - signal_generator: 일일 신호 생성기
"""

from .indicators import (
    calc_moving_average,
    calc_foreign_wght_mas,
    calc_price_ma,
    calc_rsi,
    calc_macd,
    check_multi_tf_conditions
)

from .universe_manager import (
    UniverseConfig,
    KOSDAQUniverseManager
)

from .core_strategies import (
    SignalType,
    TradingSignal,
    StrategyParams,
    MultiTFShortStrategy,
    ForeignScoreStrategy,
    backtest_strategy
)

from .signal_generator import (
    KOSDAQSpotSignalGenerator
)


__all__ = [
    # Indicators
    'calc_moving_average',
    'calc_foreign_wght_mas',
    'calc_price_ma',
    'calc_rsi',
    'calc_macd',
    'check_multi_tf_conditions',

    # Universe
    'UniverseConfig',
    'KOSDAQUniverseManager',

    # Strategies
    'SignalType',
    'TradingSignal',
    'StrategyParams',
    'MultiTFShortStrategy',
    'ForeignScoreStrategy',
    'backtest_strategy',

    # Signal Generator
    'KOSDAQSpotSignalGenerator',
]

__version__ = '1.0.0'
__strategy__ = 'Multi_TF_Short'
