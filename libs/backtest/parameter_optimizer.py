"""
Parameter Optimizer

Grid search and optimization for strategy parameters.
Includes walk-forward analysis for robust parameter selection.
"""

from __future__ import annotations

import logging
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from libs.backtest.historical_data import OHLCVDataset
from libs.backtest.portfolio_simulator import PortfolioSimulator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a single parameter combination test."""

    params: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    def __lt__(self, other: OptimizationResult) -> bool:
        """Compare by Sharpe ratio for sorting."""
        return self.sharpe_ratio < other.sharpe_ratio


@dataclass
class WalkForwardResult:
    """Result of walk-forward analysis."""

    in_sample_results: List[OptimizationResult]
    out_of_sample_results: List[OptimizationResult]
    best_params_per_fold: List[Dict[str, Any]]
    overall_oos_return: float
    overall_oos_sharpe: float
    robustness_score: float  # Consistency of parameters across folds
    efficiency_ratio: float  # OOS performance / IS performance


class ParameterOptimizer(ABC):
    """
    Abstract base class for parameter optimization.
    """

    def __init__(
        self,
        strategy_class: type,
        param_ranges: Dict[str, List[Any]],
        objective: str = "sharpe",  # "sharpe", "return", "calmar", "sortino"
        n_jobs: int = 1,
    ):
        """
        Initialize optimizer.

        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Dictionary of parameter names to list of values
            objective: Optimization objective
            n_jobs: Number of parallel jobs (1 = sequential)
        """
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.objective = objective
        self.n_jobs = n_jobs
        self.results: List[OptimizationResult] = []

    @abstractmethod
    def optimize(
        self,
        dataset: OHLCVDataset,
        initial_capital: float = 10_000_000,
    ) -> List[OptimizationResult]:
        """Run optimization and return results sorted by objective."""
        pass

    def _get_objective_value(self, result: OptimizationResult) -> float:
        """Get the value of the optimization objective."""
        if self.objective == "sharpe":
            return result.sharpe_ratio
        elif self.objective == "return":
            return result.total_return
        elif self.objective == "calmar":
            return result.calmar_ratio
        elif self.objective == "sortino":
            return result.sortino_ratio
        else:
            return result.sharpe_ratio


class GridSearchOptimizer(ParameterOptimizer):
    """
    Exhaustive grid search over parameter combinations.
    """

    def _generate_param_combinations(self) -> Generator[Dict[str, Any], None, None]:
        """Generate all parameter combinations."""
        keys = list(self.param_ranges.keys())
        values = list(self.param_ranges.values())

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def _run_single_backtest(
        self,
        params: Dict[str, Any],
        dataset: OHLCVDataset,
        initial_capital: float,
    ) -> OptimizationResult:
        """
        Run a single backtest with given parameters.

        Args:
            params: Strategy parameters
            dataset: Historical data
            initial_capital: Starting capital

        Returns:
            OptimizationResult
        """
        try:
            # Create strategy instance
            strategy = self.strategy_class(**params)

            # Create simulator
            simulator = PortfolioSimulator(initial_capital=initial_capital)

            # Define strategy function
            def strategy_fn(bar_data: Dict[str, Any]) -> Optional[Tuple[str, float]]:
                closes = bar_data.get("closes", [])
                if len(closes) < 2:
                    return None

                # Set price data for strategy
                strategy.set_price_data(bar_data["symbol"], closes)

                # Generate signals
                signals = strategy.generate_signals([bar_data["symbol"]])

                if signals:
                    signal = signals[0]
                    if signal.signal.value == "BUY":
                        return ("BUY", bar_data["close"])
                    elif signal.signal.value == "SELL":
                        return ("SELL", bar_data["close"])

                return None

            # Run backtest
            simulator.run_backtest(dataset, strategy_fn)
            summary = simulator.get_summary()

            # Calculate additional metrics
            max_dd = summary.get("max_drawdown_pct", 0.01)
            total_return = summary.get("total_return", 0)

            calmar = total_return / (max_dd / 100) if max_dd > 0 else 0

            return OptimizationResult(
                params=params,
                total_return=total_return,
                sharpe_ratio=summary.get("sharpe_ratio", 0),
                max_drawdown=max_dd,
                win_rate=summary.get("win_rate", 0),
                total_trades=summary.get("total_trades", 0),
                calmar_ratio=calmar,
            )

        except Exception as e:
            logger.warning(f"[GridSearch] Backtest failed for {params}: {e}")
            return OptimizationResult(
                params=params,
                total_return=-999,
                sharpe_ratio=-999,
                max_drawdown=100,
                win_rate=0,
                total_trades=0,
            )

    def optimize(
        self,
        dataset: OHLCVDataset,
        initial_capital: float = 10_000_000,
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization.

        Args:
            dataset: Historical data for backtesting
            initial_capital: Starting capital

        Returns:
            List of OptimizationResult sorted by objective (best first)
        """
        combinations = list(self._generate_param_combinations())
        total = len(combinations)

        logger.info(
            f"[GridSearch] Starting optimization with {total} combinations, "
            f"objective={self.objective}"
        )

        self.results = []

        if self.n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(combinations):
                result = self._run_single_backtest(params, dataset, initial_capital)
                self.results.append(result)

                if (i + 1) % 10 == 0:
                    logger.info(f"[GridSearch] Progress: {i + 1}/{total}")

        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self._run_single_backtest, params, dataset, initial_capital
                    ): params
                    for params in combinations
                }

                for future in as_completed(futures):
                    result = future.result()
                    self.results.append(result)

        # Sort by objective (descending)
        self.results.sort(key=lambda r: self._get_objective_value(r), reverse=True)

        best = self.results[0] if self.results else None
        if best:
            logger.info(
                f"[GridSearch] Best params: {best.params}, "
                f"Sharpe={best.sharpe_ratio:.2f}, Return={best.total_return*100:.2f}%"
            )

        return self.results

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameter combination."""
        if self.results:
            return self.results[0].params
        return None

    def get_top_n(self, n: int = 10) -> List[OptimizationResult]:
        """Get top N results."""
        return self.results[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        for r in self.results:
            row = r.params.copy()
            row["total_return"] = r.total_return
            row["sharpe_ratio"] = r.sharpe_ratio
            row["max_drawdown"] = r.max_drawdown
            row["win_rate"] = r.win_rate
            row["total_trades"] = r.total_trades
            row["calmar_ratio"] = r.calmar_ratio
            data.append(row)
        return pd.DataFrame(data)


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter selection.

    Divides data into multiple in-sample/out-of-sample periods
    and tests parameter stability across periods.
    """

    def __init__(
        self,
        strategy_class: type,
        param_ranges: Dict[str, List[Any]],
        n_folds: int = 5,
        train_pct: float = 0.7,  # 70% in-sample, 30% out-of-sample
        objective: str = "sharpe",
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_class: Strategy class to optimize
            param_ranges: Parameter ranges for grid search
            n_folds: Number of walk-forward folds
            train_pct: Percentage of each fold for training
            objective: Optimization objective
        """
        self.strategy_class = strategy_class
        self.param_ranges = param_ranges
        self.n_folds = n_folds
        self.train_pct = train_pct
        self.objective = objective

    def _split_data(
        self,
        dataset: OHLCVDataset,
    ) -> List[Tuple[OHLCVDataset, OHLCVDataset]]:
        """
        Split dataset into walk-forward folds.

        Returns:
            List of (train_dataset, test_dataset) tuples
        """
        n = dataset.length
        fold_size = n // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start_idx = i * fold_size
            end_idx = min((i + 1) * fold_size + fold_size // 2, n)

            if end_idx - start_idx < 50:
                continue

            train_end = start_idx + int((end_idx - start_idx) * self.train_pct)

            train_data = dataset.slice(start_idx, train_end)
            test_data = dataset.slice(train_end, end_idx)

            if train_data.length > 20 and test_data.length > 10:
                folds.append((train_data, test_data))

        return folds

    def optimize(
        self,
        dataset: OHLCVDataset,
        initial_capital: float = 10_000_000,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            dataset: Full historical dataset
            initial_capital: Starting capital

        Returns:
            WalkForwardResult with detailed analysis
        """
        folds = self._split_data(dataset)
        logger.info(f"[WalkForward] Created {len(folds)} folds")

        in_sample_results = []
        out_of_sample_results = []
        best_params_per_fold = []

        for i, (train, test) in enumerate(folds):
            logger.info(
                f"[WalkForward] Fold {i + 1}/{len(folds)}: "
                f"train={train.length} bars, test={test.length} bars"
            )

            # Optimize on in-sample
            optimizer = GridSearchOptimizer(
                strategy_class=self.strategy_class,
                param_ranges=self.param_ranges,
                objective=self.objective,
            )

            is_results = optimizer.optimize(train, initial_capital)

            if not is_results:
                continue

            best_params = is_results[0].params
            best_params_per_fold.append(best_params)
            in_sample_results.append(is_results[0])

            # Test on out-of-sample with best params
            oos_result = optimizer._run_single_backtest(
                best_params, test, initial_capital
            )
            out_of_sample_results.append(oos_result)

            logger.info(
                f"[WalkForward] Fold {i + 1}: IS Sharpe={is_results[0].sharpe_ratio:.2f}, "
                f"OOS Sharpe={oos_result.sharpe_ratio:.2f}"
            )

        # Calculate overall metrics
        if out_of_sample_results:
            overall_oos_return = np.mean([r.total_return for r in out_of_sample_results])
            overall_oos_sharpe = np.mean([r.sharpe_ratio for r in out_of_sample_results])

            # Robustness score: how consistent are parameters across folds
            param_stability = self._calculate_param_stability(best_params_per_fold)

            # Efficiency ratio: OOS / IS performance
            avg_is_sharpe = np.mean([r.sharpe_ratio for r in in_sample_results])
            efficiency = overall_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0

        else:
            overall_oos_return = 0
            overall_oos_sharpe = 0
            param_stability = 0
            efficiency = 0

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            best_params_per_fold=best_params_per_fold,
            overall_oos_return=overall_oos_return,
            overall_oos_sharpe=overall_oos_sharpe,
            robustness_score=param_stability,
            efficiency_ratio=efficiency,
        )

    def _calculate_param_stability(
        self,
        params_list: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate how stable parameters are across folds.

        Returns:
            Score from 0 (unstable) to 1 (perfectly stable)
        """
        if len(params_list) < 2:
            return 1.0

        stability_scores = []

        # For each parameter
        all_keys = set()
        for p in params_list:
            all_keys.update(p.keys())

        for key in all_keys:
            values = [p.get(key) for p in params_list if key in p]

            if not values:
                continue

            # Calculate consistency
            if isinstance(values[0], (int, float)):
                # Numeric: use coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability_scores.append(max(0, 1 - cv))
                else:
                    stability_scores.append(1.0)
            else:
                # Categorical: use mode frequency
                from collections import Counter
                counts = Counter(values)
                mode_freq = counts.most_common(1)[0][1] / len(values)
                stability_scores.append(mode_freq)

        return np.mean(stability_scores) if stability_scores else 0.0


class MonteCarloAnalyzer:
    """
    Monte Carlo simulation for strategy robustness testing.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize Monte Carlo analyzer.

        Args:
            n_simulations: Number of simulations
            confidence_level: Confidence level for statistics
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def analyze_trade_sequence(
        self,
        trades: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo analysis on trade sequence.

        Randomly shuffles trade order to test if results are dependent
        on specific sequence or are statistically robust.

        Args:
            trades: List of trade dictionaries with 'pnl' key

        Returns:
            Dictionary with Monte Carlo statistics
        """
        if len(trades) < 10:
            return {"error": "Need at least 10 trades for Monte Carlo"}

        pnl_list = [t["pnl"] for t in trades]
        original_total = sum(pnl_list)

        # Run simulations
        simulated_totals = []
        simulated_drawdowns = []

        for _ in range(self.n_simulations):
            # Shuffle trade order
            shuffled = pnl_list.copy()
            np.random.shuffle(shuffled)

            # Calculate equity curve
            equity = [0]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl)

            simulated_totals.append(equity[-1])

            # Calculate max drawdown
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = peak - e
                if dd > max_dd:
                    max_dd = dd
            simulated_drawdowns.append(max_dd)

        # Calculate statistics
        alpha = 1 - self.confidence_level
        lower_idx = int(self.n_simulations * (alpha / 2))
        upper_idx = int(self.n_simulations * (1 - alpha / 2))

        sorted_totals = sorted(simulated_totals)
        sorted_drawdowns = sorted(simulated_drawdowns)

        return {
            "original_total_pnl": original_total,
            "mean_simulated_pnl": np.mean(simulated_totals),
            "std_simulated_pnl": np.std(simulated_totals),
            "confidence_interval_pnl": (
                sorted_totals[lower_idx],
                sorted_totals[upper_idx],
            ),
            "mean_max_drawdown": np.mean(simulated_drawdowns),
            "worst_case_drawdown": sorted_drawdowns[upper_idx],
            "probability_profit": sum(1 for t in simulated_totals if t > 0) / self.n_simulations,
            "n_simulations": self.n_simulations,
            "confidence_level": self.confidence_level,
        }
