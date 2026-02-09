"""
Strategy plugin loader.
Loads strategies by ID from registry or import path.
"""

import importlib
from typing import Optional, Type

from libs.strategies.atlas_futures import ATLASFuturesStrategy
from libs.strategies.base import BaseStrategy
from libs.strategies.binance_futures_v6 import BinanceFuturesV6Strategy
from libs.strategies.ma_crossover_strategy import MACrossoverStrategy
from libs.strategies.mock_strategy import MockStrategy, TrendFollowingMockStrategy
from libs.strategies.mr_adaptive_aggressive import MRAdaptiveAggressiveStrategy
from libs.strategies.vwap_breakout import VwapBreakoutStrategy

# KOSPI200 Futures - optional import (may not exist)
try:
    from libs.strategies.kospi200_futures import (
        KOSPI200AggressivePortfolioStrategy,
        KOSPI200FuturesStrategy,
        KOSPI200HourlyStrategy,
        KOSPI200StablePortfolioStrategy,
        SemiconForeignStrategy,
        VIXBelowSMA20Strategy,
        VIXDecliningStrategy,
    )

    _HAS_KOSPI200_FUTURES = True
except ImportError:
    _HAS_KOSPI200_FUTURES = False
    KOSPI200FuturesStrategy = None
    VIXBelowSMA20Strategy = None
    VIXDecliningStrategy = None
    SemiconForeignStrategy = None
    KOSPI200HourlyStrategy = None
    KOSPI200StablePortfolioStrategy = None
    KOSPI200AggressivePortfolioStrategy = None

from libs.strategies.foreign_trend_etf import (
    ForeignTrend1xStrategy,
    ForeignTrend2xStrategy,
    ForeignTrendStrategy,
)
from libs.strategies.sector_rotation_monthly import (
    SectorRotationAllStrategy,
    SectorRotationMonthlyStrategy,
    SectorRotationTier1Strategy,
    SectorRotationTier2Strategy,
)
from libs.strategies.ankle_buy_v2 import AnkleBuyV2Strategy
from libs.strategies.sept_v3_rsi50_gate import SeptV3Rsi50GateStrategy
from libs.strategies.trend_momentum_gate import TrendMomentumGateStrategy
from libs.strategies.tiger200_etf import (
    TIGER200StableStrategy,
    TIGER200Strategy,
    TIGER200VIXOnlyStrategy,
)
from libs.strategies.vix_sma10_stocks import (
    VIXSMA10AllTiersStrategy,
    VIXSMA10StocksStrategy,
    VIXSMA10Tier1Strategy,
    VIXSMA10Tier2Strategy,
)

# Strategy registry - maps strategy_id to class
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "mock_strategy": MockStrategy,
    "trend_following_mock": TrendFollowingMockStrategy,
    "ma_crossover_v1": MACrossoverStrategy,
    "atlas_futures_p04": ATLASFuturesStrategy,
    "binance_futures_v6": BinanceFuturesV6Strategy,
    "mr_adaptive_aggressive": MRAdaptiveAggressiveStrategy,
    "vwap_breakout": VwapBreakoutStrategy,
    # TIGER 200 ETF strategies
    "tiger200_etf_v1": TIGER200Strategy,
    "tiger200_stable": TIGER200StableStrategy,
    "tiger200_vix_only": TIGER200VIXOnlyStrategy,
    # Foreign Trend ETF strategies (Look-Ahead Bias Free)
    "foreign_trend_etf_v1": ForeignTrendStrategy,
    "foreign_trend_1x": ForeignTrend1xStrategy,
    "foreign_trend_2x": ForeignTrend2xStrategy,
    # VIX SMA10 Individual Stocks (Fractional Trading)
    "vix_sma10_stocks": VIXSMA10StocksStrategy,
    "vix_sma10_tier1": VIXSMA10Tier1Strategy,
    "vix_sma10_tier2": VIXSMA10Tier2Strategy,
    "vix_sma10_all": VIXSMA10AllTiersStrategy,
    # Crypto Spot - Ankle Buy v2.0 (SMA breakout + BTC gate, Codex PASS)
    "ankle_buy_v2": AnkleBuyV2Strategy,
    # Crypto Spot - Sept v3 RSI50 Gate (7-Signal OR Ensemble)
    "sept_v3_rsi50_gate": SeptV3Rsi50GateStrategy,
    # Crypto Spot - Trend-Momentum-Gate (Cross-sectional Momentum + BTC Gate)
    "trend_momentum_gate": TrendMomentumGateStrategy,
    # Sector Rotation Monthly (KOSPI Spot - Sharpe 1.5-3.78)
    "sector_rotation_m": SectorRotationMonthlyStrategy,
    "sector_rotation_m_tier1": SectorRotationTier1Strategy,
    "sector_rotation_m_tier2": SectorRotationTier2Strategy,
    "sector_rotation_m_all": SectorRotationAllStrategy,
}

# KOSPI200 Futures strategies (A+ Grade validated) - conditional
if _HAS_KOSPI200_FUTURES:
    STRATEGY_REGISTRY.update(
        {
            "kospi200_futures_v1": KOSPI200FuturesStrategy,
            "kospi200_vix_below_sma20": VIXBelowSMA20Strategy,
            "kospi200_vix_declining": VIXDecliningStrategy,
            "kospi200_semicon_foreign": SemiconForeignStrategy,
            "kospi200_hourly_ma": KOSPI200HourlyStrategy,
            "kospi200_stable_portfolio": KOSPI200StablePortfolioStrategy,
            "kospi200_aggressive_portfolio": KOSPI200AggressivePortfolioStrategy,
        }
    )

# Strategy metadata registry (not necessarily loadable)
AVAILABLE_STRATEGIES: list[dict] = []

# ATLAS-Futures registration (fully integrated)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "atlas_futures_p04",
        "id": "atlas_futures_p04",
        "name": "P0-4 Squeeze-Surge",
        "version": "v2.6.2-r1",
        "description": "ATLAS-Futures volatility squeeze + surge strategy (6x 3AI PASS)",
        "module": "libs.strategies.atlas_futures",
        "class_name": "ATLASFuturesStrategy",
        "config_class": "ATLASFuturesConfig",
        "markets": ["futures"],
        "exchanges": ["binance_futures"],
        "status": "phase_4_ready",
    }
)

# KAMA-TSMOM-Gate registration (metadata only)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kama_tsmom_gate",
        "id": "kama_tsmom_gate",
        "name": "KAMA-TSMOM-Gate",
        "version": "v1.0",
        "description": "KAMA-TSMOM-Gate strategy (BTC gate + KAMA/TSMOM)",
        "module": "libs.strategies.kama_tsmom_gate",
        "class_name": "KamaTsmomGateStrategy",
        "markets": ["spot"],
        "exchanges": ["upbit_spot", "paper"],
        "status": "phase_3a_ready",
    }
)

# Binance Futures v6 - AI Consensus Strategy
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "binance_futures_v6",
        "id": "binance_futures_v6",
        "name": "Binance Futures v6 - AI Consensus",
        "version": "6.0.0",
        "description": "Multi-AI consensus strategy (6 rounds, 10 AIs) for Binance USDT-M Futures",
        "module": "libs.strategies.binance_futures_v6",
        "class_name": "BinanceFuturesV6Strategy",
        "config_class": "BinanceFuturesV6Config",
        "markets": ["futures"],
        "exchanges": ["binance_futures"],
        "status": "backtest_pending",
        "features": [
            "market_regime_detection",
            "multi_timeframe_analysis",
            "supertrend_kama_tsmom",
            "quality_filters",
            "btc_gate",
            "regime_adaptive_sizing",
        ],
        "expected_performance": {
            "win_rate": "48-52%",
            "annual_return": "25-45%",
            "max_mdd": "25%",
        },
    }
)


# MR_ADAPTIVE_AGGRESSIVE - Mean Reversion with Trend Filter (R&D Validated)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "mr_adaptive_aggressive",
        "id": "mr_adaptive_aggressive",
        "name": "MR Adaptive Aggressive",
        "version": "1.0.0",
        "description": (
            "Mean reversion strategy with adaptive trend filter. "
            "Long-only, buys oversold (RSI<35, below BB lower), "
            "exits at RSI>55 or above BB mid. "
            "Position scales by trend: 100% uptrend, 30% downtrend. "
            "Validated via R&D: Sharpe 0.312 (+31.2% vs baseline), MDD -19.9%."
        ),
        "module": "libs.strategies.mr_adaptive_aggressive",
        "class_name": "MRAdaptiveAggressiveStrategy",
        "config_class": "MRAdaptiveConfig",
        "markets": ["futures"],
        "exchanges": ["binance_futures", "paper"],
        "status": "production_ready",
        "features": [
            "bollinger_bands_oversold",
            "rsi_mean_reversion",
            "trend_adaptive_sizing",
            "long_only",
            "daily_timeframe",
            "parameter_robustness_validated",
            "slippage_sensitivity_tested",
        ],
        "validated_performance": {
            "sharpe": 0.312,
            "vs_baseline": "+31.2%",
            "max_drawdown": "-19.9%",
            "coverage": "99%",
            "parameter_cv": 0.21,
            "slippage_3x_sharpe": 0.328,
        },
        "parameters": {
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_low": 35,
            "rsi_exit": 55,
            "trend_ma": 50,
            "trend_scale": 0.3,
        },
        "recommendation": {
            "capital_range": "300K-500K KRW",
            "leverage": "2x (isolated)",
            "max_positions": 30,
            "paper_trading_period": "2 weeks",
            "mdd_limit": "-35%",
        },
    }
)

# VWAP Breakout - KAMA+EMA Hybrid (Phase 18-19 validated)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "vwap_breakout",
        "id": "vwap_breakout",
        "name": "KAMA+EMA Hybrid VWAP Breakout",
        "version": "2.0.0",
        "description": (
            "Long-only trend-following breakout strategy for Binance USDT-M Futures (1h). "
            "Validated via TRUE OOS: Sharpe 1.41, MDD -4.9%, all regimes positive."
        ),
        "module": "libs.strategies.vwap_breakout",
        "class_name": "VwapBreakoutStrategy",
        "markets": ["futures"],
        "exchanges": ["binance_futures"],
        "status": "production_ready",
        "features": [
            "donchian_breakout",
            "vwap_filter",
            "ema_trend_filter",
            "kama_adaptive_filter",
            "atr_stop_loss",
            "atr_take_profit",
            "time_based_exit",
            "vol_targeting",
        ],
        "validated_performance": {
            "sharpe": 1.41,
            "total_return_5x": "+59.6%",
            "max_drawdown": "-4.9%",
            "bull_return": "+41.8%",
            "bear_return": "+2.2%",
            "sideways_return": "+10.1%",
        },
    }
)


# KOSPI200 Futures - Stable Portfolio (A+ Grade)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kospi200_stable_portfolio",
        "id": "kospi200_stable_portfolio",
        "name": "KOSPI200 Stable Portfolio",
        "version": "1.0.0",
        "description": (
            "Optimal daily portfolio combining VIX + Factor strategies. "
            "Validated: Sharpe 2.37, CAGR 23.1%, MDD -11.5%. "
            "Weights: VIX_Below_SMA20 50%, VIX_Declining 30%, Semicon_Foreign 20%."
        ),
        "module": "libs.strategies.kospi200_futures",
        "class_name": "KOSPI200StablePortfolioStrategy",
        "markets": ["futures"],
        "exchanges": ["kr_futures", "paper"],
        "status": "production_ready",
        "features": [
            "vix_regime_filter",
            "semicon_momentum",
            "foreign_flow_signal",
            "multi_strategy_composite",
            "walk_forward_validated",
            "monte_carlo_validated",
            "stress_tested",
        ],
        "validated_performance": {
            "sharpe": 2.37,
            "cagr": "23.1%",
            "max_drawdown": "-11.5%",
            "covid_2020_03": "+1.8%",
            "rate_hike_2022": "+8.5%",
            "win_rate_annual": "100%",
        },
    }
)

# KOSPI200 Futures - VIX Below SMA20 (A+ Grade)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kospi200_vix_below_sma20",
        "id": "kospi200_vix_below_sma20",
        "name": "KOSPI200 VIX Below SMA20",
        "version": "1.0.0",
        "description": (
            "A+ Grade VIX-based strategy. LONG when VIX < VIX_SMA20. "
            "Sharpe 2.25, CAGR 27.8%, MDD -12.0%. Best overall performance."
        ),
        "module": "libs.strategies.kospi200_futures",
        "class_name": "VIXBelowSMA20Strategy",
        "markets": ["futures"],
        "exchanges": ["kr_futures", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "sharpe": 2.25,
            "cagr": "27.8%",
            "max_drawdown": "-12.0%",
            "covid_2020_03": "+1.8%",
            "rate_hike_2022": "+8.5%",
        },
    }
)

# KOSPI200 Futures - VIX Declining (A+ Grade)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kospi200_vix_declining",
        "id": "kospi200_vix_declining",
        "name": "KOSPI200 VIX Declining",
        "version": "1.0.0",
        "description": (
            "A+ Grade VIX momentum strategy. LONG when VIX declining. "
            "Sharpe 1.86, CAGR 19.1%, MDD -13.3%. Best crisis performance."
        ),
        "module": "libs.strategies.kospi200_futures",
        "class_name": "VIXDecliningStrategy",
        "markets": ["futures"],
        "exchanges": ["kr_futures", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "sharpe": 1.86,
            "cagr": "19.1%",
            "max_drawdown": "-13.3%",
            "covid_2020_03": "+17.7%",
            "rate_hike_2022": "+20.6%",
        },
    }
)

# KOSPI200 Futures - Aggressive Portfolio (A Grade)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kospi200_aggressive_portfolio",
        "id": "kospi200_aggressive_portfolio",
        "name": "KOSPI200 Aggressive Portfolio",
        "version": "1.0.0",
        "description": (
            "Aggressive portfolio combining hourly MA + daily VIX strategies. "
            "EMA_15_20 40%, SMA_20_30 30%, VIX_Below_SMA20 30%. "
            "Higher returns, higher risk."
        ),
        "module": "libs.strategies.kospi200_futures",
        "class_name": "KOSPI200AggressivePortfolioStrategy",
        "markets": ["futures"],
        "exchanges": ["kr_futures", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "sharpe": 2.5,
            "cagr": "30%+",
            "max_drawdown": "-15%",
        },
    }
)

# KOSPI200 Futures - Hourly MA (A Grade)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "kospi200_hourly_ma",
        "id": "kospi200_hourly_ma",
        "name": "KOSPI200 Hourly MA",
        "version": "1.0.0",
        "description": (
            "Combined hourly MA crossover strategies. "
            "SMA_15_30, EMA_15_20, SMA_20_30. "
            "Sharpe 2.6+, CAGR 36%+, MDD -10%."
        ),
        "module": "libs.strategies.kospi200_futures",
        "class_name": "KOSPI200HourlyStrategy",
        "markets": ["futures"],
        "exchanges": ["kr_futures", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "sharpe": 2.65,
            "cagr": "36.5%",
            "max_drawdown": "-10.3%",
            "walk_forward_sharpe": 7.0,
        },
    }
)

# TIGER 200 ETF Strategy
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "tiger200_etf_v1",
        "id": "tiger200_etf_v1",
        "name": "TIGER 200 ETF Strategy",
        "version": "1.0.0",
        "description": (
            "KOSPI200 추종 TIGER 200 ETF용 전략. "
            "VIX_Below_SMA20 50%, VIX_Declining 30%, Semicon_Foreign 20%. "
            "소액 투자 가능 (4만원~), 레버리지 1배."
        ),
        "module": "libs.strategies.tiger200_etf",
        "class_name": "TIGER200Strategy",
        "markets": ["etf", "spot"],
        "exchanges": ["kr_stock", "paper"],
        "status": "production_ready",
        "etf_info": {
            "code": "102110",
            "name": "TIGER 200",
            "expense_ratio": "0.05%",
            "min_investment": "~40,000 KRW",
        },
        "validated_performance": {
            "sharpe": 2.37,
            "cagr": "23.1%",
            "max_drawdown": "-11.5%",
            "leverage": "1x",
        },
    }
)


# Foreign Trend ETF Strategy (Look-Ahead Bias Free, Validated)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "foreign_trend_etf_v1",
        "id": "foreign_trend_etf_v1",
        "name": "Foreign Trend ETF Strategy",
        "version": "1.0.0",
        "description": (
            "외국인+추세 기반 ETF 전략 (Look-Ahead Bias 제거). "
            "한국 데이터만 사용하여 미래 정보 참조 불가능. "
            "조건: 외국인 30일 순매수 > 0 AND 종가 > SMA100. "
            "Sharpe 1.225, CAGR 13.8%, MDD -16.9%."
        ),
        "module": "libs.strategies.foreign_trend_etf",
        "class_name": "ForeignTrendStrategy",
        "config_class": "ForeignTrendConfig",
        "markets": ["etf", "spot"],
        "exchanges": ["kr_stock", "paper"],
        "status": "production_ready",
        "etf_info": {
            "code_1x": "102110",
            "name_1x": "TIGER 200",
            "code_2x": "233160",
            "name_2x": "TIGER 200선물레버리지",
        },
        "validated_performance": {
            "sharpe": 1.225,
            "cagr": "13.8%",
            "max_drawdown": "-16.9%",
            "wf_ratio": 0.86,
            "wf_test_sharpe": 2.44,
            "positive_years": "6/10",
            "covid_defense": "+11.6%p vs B&H",
            "2022_defense": "+22.6%p vs B&H",
        },
        "features": [
            "no_lookahead_bias",
            "korean_data_only",
            "foreign_flow_signal",
            "trend_following",
            "walk_forward_validated",
            "crisis_defense",
        ],
        "recommendation": {
            "1x_etf": "권장 (MDD -16.9%)",
            "2x_etf": "조건부 (MDD -29.5%, 경험자만)",
        },
    }
)

# Foreign Trend 1x ETF (TIGER 200)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "foreign_trend_1x",
        "id": "foreign_trend_1x",
        "name": "Foreign Trend 1x ETF (TIGER 200)",
        "version": "1.0.0",
        "description": (
            "외국인+추세 전략 1배 ETF 버전. TIGER 200 (102110) 전용. "
            "안정적 수익, 위기 방어력 우수. 초보자 권장."
        ),
        "module": "libs.strategies.foreign_trend_etf",
        "class_name": "ForeignTrend1xStrategy",
        "markets": ["etf", "spot"],
        "exchanges": ["kr_stock", "paper"],
        "status": "production_ready",
        "etf_info": {
            "code": "102110",
            "name": "TIGER 200",
            "leverage": 1.0,
            "min_investment": "~40,000 KRW",
        },
        "validated_performance": {
            "sharpe": 1.225,
            "cagr": "13.8%",
            "max_drawdown": "-16.9%",
        },
    }
)

# Foreign Trend 2x ETF (TIGER 200선물레버리지)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "foreign_trend_2x",
        "id": "foreign_trend_2x",
        "name": "Foreign Trend 2x ETF (TIGER 200선물레버리지)",
        "version": "1.0.0",
        "description": (
            "외국인+추세 전략 2배 ETF 버전. TIGER 200선물레버리지 (233160) 전용. "
            "고수익 고위험. MDD -29.5% 감내 가능 시에만 사용. "
            "경험자/중급자 이상 권장."
        ),
        "module": "libs.strategies.foreign_trend_etf",
        "class_name": "ForeignTrend2xStrategy",
        "markets": ["etf", "spot"],
        "exchanges": ["kr_stock", "paper"],
        "status": "production_ready",
        "etf_info": {
            "code": "233160",
            "name": "TIGER 200선물레버리지",
            "leverage": 2.0,
            "min_investment": "~15,000 KRW",
            "warning": "MDD -29.5%, 횡보장 손실 누적",
        },
        "validated_performance": {
            "sharpe": 1.280,
            "cagr": "24.0%",
            "max_drawdown": "-29.5%",
        },
    }
)

# VIX SMA10 Individual Stocks Strategy (Fractional Trading Compatible)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "vix_sma10_stocks",
        "id": "vix_sma10_stocks",
        "name": "VIX SMA10 Individual Stocks",
        "version": "1.0.0",
        "description": (
            "VIX < SMA10 전략 for KOSPI 개별 종목. "
            "키움증권 소수점 거래 가능 종목 25개 대상. "
            "Tier 1 (Sharpe >= 1.0): 삼성전자, 삼성SDI, 카카오, SK하이닉스, LG화학."
        ),
        "module": "libs.strategies.vix_sma10_stocks",
        "class_name": "VIXSMA10StocksStrategy",
        "config_class": "VIXSMA10Config",
        "markets": ["spot"],
        "exchanges": ["kr_stock", "kiwoom", "paper"],
        "status": "production_ready",
        "features": [
            "vix_timing",
            "sma10_filter",
            "fractional_trading",
            "timezone_validated",
            "no_lookahead_bias",
        ],
        "validated_performance": {
            "tier1_avg_sharpe": 1.11,
            "tier1_avg_cagr": "24.7%",
            "tier2_avg_sharpe": 0.72,
            "tier3_avg_sharpe": 0.42,
            "stocks_count": 25,
        },
    }
)

# VIX SMA10 Tier 1 Only (Top 5 Stocks)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "vix_sma10_tier1",
        "id": "vix_sma10_tier1",
        "name": "VIX SMA10 Tier 1 Only",
        "version": "1.0.0",
        "description": (
            "VIX < SMA10 최상위 5종목만. "
            "삼성전자, 삼성SDI, 카카오, SK하이닉스, LG화학. "
            "평균 Sharpe 1.11, 평균 CAGR 24.7%."
        ),
        "module": "libs.strategies.vix_sma10_stocks",
        "class_name": "VIXSMA10Tier1Strategy",
        "markets": ["spot"],
        "exchanges": ["kr_stock", "kiwoom", "paper"],
        "status": "production_ready",
        "stocks": ["005930", "006400", "035720", "000660", "051910"],
        "validated_performance": {
            "avg_sharpe": 1.11,
            "avg_cagr": "24.7%",
            "stocks_count": 5,
        },
    }
)

# VIX SMA10 Tier 1+2 (Top 14 Stocks)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "vix_sma10_tier2",
        "id": "vix_sma10_tier2",
        "name": "VIX SMA10 Tier 1+2",
        "version": "1.0.0",
        "description": (
            "VIX < SMA10 상위 14종목 (Sharpe >= 0.5). "
            "Tier 1 + Tier 2 종목 포함. "
            "더 넓은 분산 투자."
        ),
        "module": "libs.strategies.vix_sma10_stocks",
        "class_name": "VIXSMA10Tier2Strategy",
        "markets": ["spot"],
        "exchanges": ["kr_stock", "kiwoom", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "min_sharpe": 0.5,
            "stocks_count": 14,
        },
    }
)

# VIX SMA10 All Tiers (All 25 Stocks)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "vix_sma10_all",
        "id": "vix_sma10_all",
        "name": "VIX SMA10 All Tiers",
        "version": "1.0.0",
        "description": (
            "VIX < SMA10 전체 25종목 (Sharpe >= 0.3). "
            "최대 분산 투자. "
            "소액 투자에 적합 (소수점 거래)."
        ),
        "module": "libs.strategies.vix_sma10_stocks",
        "class_name": "VIXSMA10AllTiersStrategy",
        "markets": ["spot"],
        "exchanges": ["kr_stock", "kiwoom", "paper"],
        "status": "production_ready",
        "validated_performance": {
            "min_sharpe": 0.3,
            "stocks_count": 25,
        },
    }
)

# Sept v3 RSI50 Gate v3 - 7중 OR + 거래량 상위 30% 필터
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "sept_v3_rsi50_gate",
        "id": "sept_v3_rsi50_gate",
        "name": "Sept-v3-RSI50-Gate",
        "version": "3.0.0",
        "description": (
            "7중 OR 신호 + 거래량 상위 30% 필터 전략. "
            "신호: KAMA, TSMOM, EMA Cross, Momentum, SMA Cross, RSI>50, Higher Low (7개 중 1개 이상). "
            "리스크 관리: 포트폴리오 -15% 손절 + 3일 쿨다운. "
            "OOS Sharpe 2.27, MDD -37.0%, Return 11,763%."
        ),
        "module": "libs.strategies.sept_v3_rsi50_gate",
        "class_name": "SeptV3Rsi50GateStrategy",
        "markets": ["spot"],
        "exchanges": ["upbit_spot", "bithumb_spot", "binance_spot", "paper"],
        "status": "production_ready",
        "features": [
            "7_signal_or",
            "volume_filter_top_30pct",
            "btc_gate_entry",
            "btc_gate_exit",
            "portfolio_stop_loss",
            "cooldown_period",
            "unlimited_positions",
            "oos_validated",
        ],
        "validated_performance": {
            "oos_sharpe": 2.27,
            "oos_mdd": "-37.0%",
            "oos_return": "11763%",
            "avg_positions": 4.1,
        },
        "risk_management": {
            "btc_gate_exit": True,
            "portfolio_stop": "-15%",
            "cooldown_days": 3,
            "volume_filter": "top 30%",
            "max_positions": "unlimited",
        },
        "recommendation": {
            "capital_allocation": "30-50%",
            "expected_mdd": "-37%",
            "live_trading_ready": True,
            "notes": [
                "소액으로 3개월 테스트 후 증액 권장",
                "백테스트 성과의 70-80% 수준 기대",
                "거래량 상위 30%만 선별하여 유동성 확보",
                "평균 4.1개 보유 (7중 OR + 상위 30%)",
            ],
        },
    }
)

# Sector Rotation Monthly - KOSPI Spot (Sharpe 1.5-3.78)
AVAILABLE_STRATEGIES.append(
    {
        "strategy_id": "sector_rotation_m",
        "id": "sector_rotation_m",
        "name": "Monthly Sector Rotation",
        "version": "1.0.0",
        "description": (
            "월별 모멘텀 기반 섹터 로테이션. "
            "월간 수익률 > 0 → LONG, else → CASH. "
            "월말 종가 단일가 리밸런싱 (15:20-15:30). "
            "Real Sharpe 1.5-3.78, CAGR 30-80%."
        ),
        "module": "libs.strategies.sector_rotation_monthly",
        "class_name": "SectorRotationMonthlyStrategy",
        "markets": ["spot"],
        "exchanges": ["kiwoom_spot", "paper"],
        "status": "production_ready",
        "features": [
            "monthly_momentum",
            "fractional_trading",
            "closing_auction_execution",
            "walk_forward_validated",
            "realistic_cost_validated",
        ],
        "validated_performance": {
            "top_sharpe": 2.37,
            "top_cagr": "92.3%",
            "avg_sharpe": 1.96,
            "max_drawdown": "-30%",
            "stocks": [
                "SK하이닉스",
                "삼성전자",
                "포스코홀딩스",
                "한미반도체",
                "삼성SDI",
            ],
        },
        "recommendation": {
            "capital_range": "50만-100만원",
            "rebalance_frequency": "monthly",
            "execution_time": "15:20 KST",
            "notes": [
                "소수점 거래로 소액 투자 가능",
                "월 1회 리밸런싱으로 거래비용 최소화",
                "종가 단일가 매매로 슬리피지 최소화",
            ],
        },
    }
)


def register_strategy(strategy_class: type[BaseStrategy]) -> None:
    """
    Register a strategy class in the registry.

    Args:
        strategy_class: Strategy class to register
    """
    STRATEGY_REGISTRY[strategy_class.strategy_id] = strategy_class


def load_strategy_class(strategy_id: str) -> Optional[Type[BaseStrategy]]:
    """Load strategy class dynamically."""
    if strategy_id in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[strategy_id]

    for entry in AVAILABLE_STRATEGIES:
        if entry.get("strategy_id") == strategy_id:
            module_path = entry.get("module")
            class_name = entry.get("class_name")

            if not module_path or not class_name:
                print(f"[Loader] Warning: Missing module/class_name for {strategy_id}")
                return None

            try:
                module = importlib.import_module(module_path)
                strategy_class = getattr(module, class_name)
                STRATEGY_REGISTRY[strategy_id] = strategy_class
                print(f"[Loader] Dynamically loaded: {strategy_id}")
                return strategy_class
            except Exception as exc:
                print(f"[Loader] Failed to load {strategy_id}: {exc}")
                return None

    return None


def get_strategy(strategy_id: str) -> Optional[BaseStrategy]:
    """
    Get a strategy instance by ID.

    Args:
        strategy_id: Strategy identifier

    Returns:
        Strategy instance or None if not found
    """
    strategy_class = load_strategy_class(strategy_id)
    if strategy_class:
        return strategy_class()
    return None


def load_strategies(strategy_ids: list[str]) -> list[BaseStrategy]:
    """
    Load multiple strategies by their IDs.

    Args:
        strategy_ids: List of strategy identifiers

    Returns:
        List of strategy instances (skips unknown IDs with warning)
    """
    strategies = []

    for strategy_id in strategy_ids:
        strategy = get_strategy(strategy_id)
        if strategy:
            strategies.append(strategy)
            print(f"[Loader] Loaded strategy: {strategy}")
        else:
            print(f"[Loader] Warning: Unknown strategy ID '{strategy_id}', skipping")

    return strategies


def _get_attr(cls, *names, default: str = "unknown") -> str:
    """Get attribute from class, trying multiple names (case-insensitive fallback)."""
    for name in names:
        if hasattr(cls, name):
            return getattr(cls, name)
    return default


def list_available_strategies() -> list[dict]:
    """
    List all available strategies.

    Returns:
        List of strategy metadata dicts
    """
    result: dict[str, dict] = {}
    for strategy_id, strategy_class in STRATEGY_REGISTRY.items():
        result[strategy_id] = {
            "strategy_id": strategy_id,
            "id": strategy_id,
            "name": _get_attr(strategy_class, "name", "NAME"),
            "version": _get_attr(strategy_class, "version", "VERSION"),
            "description": _get_attr(strategy_class, "description", "DESCRIPTION"),
        }
    for entry in AVAILABLE_STRATEGIES:
        strategy_id = entry.get("strategy_id")
        if not strategy_id:
            continue
        if strategy_id not in result:
            merged = dict(entry)
            if "id" not in merged:
                merged["id"] = strategy_id
            result[strategy_id] = merged
        else:
            merged = dict(result[strategy_id])
            merged.update(entry)
            if "id" not in merged:
                merged["id"] = strategy_id
            result[strategy_id] = merged
    return list(result.values())
