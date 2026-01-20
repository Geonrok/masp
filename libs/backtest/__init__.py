"""
Backtesting Framework

Provides comprehensive backtesting capabilities for strategy evaluation.
"""

from libs.backtest.engine import BacktestEngine, BacktestResult
from libs.backtest.historical_data import HistoricalDataLoader, OHLCVDataset
from libs.backtest.portfolio_simulator import PortfolioSimulator, PortfolioState
from libs.backtest.parameter_optimizer import (
    ParameterOptimizer,
    GridSearchOptimizer,
    OptimizationResult,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "HistoricalDataLoader",
    "OHLCVDataset",
    "PortfolioSimulator",
    "PortfolioState",
    "ParameterOptimizer",
    "GridSearchOptimizer",
    "OptimizationResult",
]
