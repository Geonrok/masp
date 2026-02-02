"""
Backtesting Framework

Provides comprehensive backtesting capabilities for strategy evaluation.

Bias-Free Backtester (v2.0):
- Signal at Day T close -> Execute at Day T+1 open
- No look-ahead bias
- Includes slippage and commission

Monte Carlo Simulation:
- Return distribution simulation
- Confidence intervals
- Maximum drawdown estimation
- Probability of ruin calculation
"""

from libs.backtest.engine import BacktestEngine, BacktestResult
from libs.backtest.historical_data import HistoricalDataLoader, OHLCVDataset
from libs.backtest.portfolio_simulator import PortfolioSimulator, PortfolioState
from libs.backtest.parameter_optimizer import (
    ParameterOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
)
from libs.backtest.bias_free_backtester import (
    BiasFreeBacktester,
    BacktestConfig,
    ExecutionConfig,
    BacktestMetrics,
)
from libs.backtest.validation import (
    WalkForwardOptimizer,
    WFOConfig,
    WFOResult,
    CPCVValidator,
    CPCVConfig,
    CPCVResult,
    DeflatedSharpeRatio,
    DSRResult,
    validate_strategy,
)
from libs.backtest.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    ReturnDistribution,
    DrawdownDistribution,
    RuinProbability,
    ConfidenceInterval,
    format_monte_carlo_report,
)

__all__ = [
    # Legacy
    "BacktestEngine",
    "BacktestResult",
    "HistoricalDataLoader",
    "OHLCVDataset",
    "PortfolioSimulator",
    "PortfolioState",
    "ParameterOptimizer",
    "GridSearchOptimizer",
    "OptimizationResult",
    # Bias-Free (v2.0)
    "BiasFreeBacktester",
    "BacktestConfig",
    "ExecutionConfig",
    "BacktestMetrics",
    # Validation (v2.0)
    "WalkForwardOptimizer",
    "WFOConfig",
    "WFOResult",
    "CPCVValidator",
    "CPCVConfig",
    "CPCVResult",
    "DeflatedSharpeRatio",
    "DSRResult",
    "validate_strategy",
    # Monte Carlo
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResult",
    "ReturnDistribution",
    "DrawdownDistribution",
    "RuinProbability",
    "ConfidenceInterval",
    "format_monte_carlo_report",
]
