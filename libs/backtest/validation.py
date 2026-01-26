"""
Strategy Validation Module

Implements robust validation techniques to prevent overfitting:
1. Walk-Forward Optimization (WFO)
2. Combinatorial Purged Cross-Validation (CPCV)
3. Deflated Sharpe Ratio (DSR)

References:
- Bailey et al. (2014) "The Deflated Sharpe Ratio"
- Lopez de Prado (2018) "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration."""

    train_months: int = 12  # Training window
    test_months: int = 3  # Out-of-sample test window
    min_train_samples: int = 252  # Minimum training days
    step_months: int = 3  # Step size for rolling


@dataclass
class WFOResult:
    """Result from Walk-Forward Optimization."""

    periods: List[Dict[str, Any]] = field(default_factory=list)
    oos_returns: List[float] = field(default_factory=list)  # Out-of-sample returns
    is_returns: List[float] = field(default_factory=list)  # In-sample returns
    oos_sharpe: float = 0.0
    is_sharpe: float = 0.0
    efficiency_ratio: float = 0.0  # OOS/IS performance ratio
    robustness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_periods": len(self.periods),
            "oos_sharpe": self.oos_sharpe,
            "is_sharpe": self.is_sharpe,
            "efficiency_ratio": self.efficiency_ratio,
            "robustness_score": self.robustness_score,
            "oos_returns": self.oos_returns,
        }


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization (WFO)

    Splits data into rolling train/test windows:
    - Train on window 1 → Test on window 2
    - Train on window 2 → Test on window 3
    - ...

    Measures out-of-sample (OOS) performance consistency.
    """

    def __init__(self, config: WFOConfig):
        """
        Initialize WFO.

        Args:
            config: WFO configuration
        """
        self.config = config

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
        optimize_func: Optional[Callable] = None,
    ) -> WFOResult:
        """
        Run walk-forward optimization.

        Args:
            data: Dict of symbol -> DataFrame with OHLCV data
            strategy_func: Function(data, params) -> returns Series
            optimize_func: Optional function to optimize params on train data

        Returns:
            WFOResult
        """
        # Get date range from data
        all_dates = self._get_all_dates(data)
        if len(all_dates) < self.config.min_train_samples + 60:
            logger.warning("[WFO] Insufficient data for walk-forward optimization")
            return WFOResult()

        result = WFOResult()
        train_days = self.config.train_months * 21  # ~21 trading days per month
        test_days = self.config.test_months * 21
        step_days = self.config.step_months * 21

        start_idx = 0
        period_num = 0

        while start_idx + train_days + test_days <= len(all_dates):
            train_start = all_dates[start_idx]
            train_end = all_dates[start_idx + train_days - 1]
            test_start = all_dates[start_idx + train_days]
            test_end_idx = min(start_idx + train_days + test_days - 1, len(all_dates) - 1)
            test_end = all_dates[test_end_idx]

            # Filter data for train period
            train_data = self._filter_data(data, train_start, train_end)

            # Optimize on train if optimizer provided
            if optimize_func:
                best_params = optimize_func(train_data)
            else:
                best_params = {}

            # Calculate in-sample return
            is_returns = strategy_func(train_data, best_params)
            is_return = float(np.sum(is_returns)) if len(is_returns) > 0 else 0

            # Filter data for test period
            test_data = self._filter_data(data, test_start, test_end)

            # Calculate out-of-sample return
            oos_returns = strategy_func(test_data, best_params)
            oos_return = float(np.sum(oos_returns)) if len(oos_returns) > 0 else 0

            result.periods.append({
                "period": period_num,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "is_return": is_return,
                "oos_return": oos_return,
                "params": best_params,
            })

            result.is_returns.append(is_return)
            result.oos_returns.append(oos_return)

            start_idx += step_days
            period_num += 1

        # Calculate summary statistics
        if result.oos_returns:
            oos_arr = np.array(result.oos_returns)
            is_arr = np.array(result.is_returns)

            if np.std(oos_arr) > 0:
                result.oos_sharpe = np.mean(oos_arr) / np.std(oos_arr) * np.sqrt(4)  # Quarterly

            if np.std(is_arr) > 0:
                result.is_sharpe = np.mean(is_arr) / np.std(is_arr) * np.sqrt(4)

            if result.is_sharpe != 0:
                result.efficiency_ratio = result.oos_sharpe / result.is_sharpe

            # Robustness: % of periods with positive OOS return
            result.robustness_score = np.mean(oos_arr > 0)

        logger.info(
            "[WFO] Completed: %d periods, OOS Sharpe=%.2f, Efficiency=%.2f",
            len(result.periods),
            result.oos_sharpe,
            result.efficiency_ratio,
        )

        return result

    def _get_all_dates(self, data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        """Get sorted list of all dates in data."""
        all_dates = set()
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
            elif "date" in df.columns:
                all_dates.update(pd.to_datetime(df["date"]).tolist())
        return sorted(all_dates)

    def _filter_data(
        self,
        data: Dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Filter data to date range."""
        filtered = {}
        for symbol, df in data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                mask = (df.index >= start) & (df.index <= end)
                filtered[symbol] = df[mask].copy()
            elif "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= start) & (df["date"] <= end)
                filtered[symbol] = df[mask].copy()
        return filtered


@dataclass
class CPCVConfig:
    """Combinatorial Purged Cross-Validation configuration."""

    n_splits: int = 5  # Number of CV splits
    embargo_pct: float = 0.05  # 5% embargo between train/test
    purge_pct: float = 0.01  # 1% purge before test


@dataclass
class CPCVResult:
    """Result from CPCV."""

    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    mean_return: float = 0.0
    std_return: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    pbo: float = 0.0  # Probability of Backtest Overfitting

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_folds": len(self.fold_results),
            "mean_return": self.mean_return,
            "std_return": self.std_return,
            "mean_sharpe": self.mean_sharpe,
            "std_sharpe": self.std_sharpe,
            "pbo": self.pbo,
        }


class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation (CPCV)

    Features:
    - Purging: Remove observations between train/test to prevent leakage
    - Embargo: Add gap between train and test periods
    - Combinatorial: Test all combinations of folds

    Reference: Lopez de Prado (2018)
    """

    def __init__(self, config: CPCVConfig):
        """
        Initialize CPCV.

        Args:
            config: CPCV configuration
        """
        self.config = config

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_func: Callable,
    ) -> CPCVResult:
        """
        Run CPCV validation.

        Args:
            data: Dict of symbol -> DataFrame
            strategy_func: Function(data) -> returns Series

        Returns:
            CPCVResult
        """
        all_dates = self._get_all_dates(data)
        n = len(all_dates)

        if n < 100:
            logger.warning("[CPCV] Insufficient data for cross-validation")
            return CPCVResult()

        result = CPCVResult()
        fold_size = n // self.config.n_splits
        embargo_size = int(n * self.config.embargo_pct)
        purge_size = int(n * self.config.purge_pct)

        fold_returns = []
        fold_sharpes = []

        for test_fold in range(self.config.n_splits):
            # Define test range
            test_start_idx = test_fold * fold_size
            test_end_idx = min((test_fold + 1) * fold_size, n)

            # Apply purge and embargo
            train_end_idx = max(0, test_start_idx - purge_size)
            train_start_after_test = min(n, test_end_idx + embargo_size)

            # Collect train dates (before test with purge, after test with embargo)
            train_dates_before = all_dates[:train_end_idx]
            train_dates_after = all_dates[train_start_after_test:]
            train_dates = train_dates_before + train_dates_after

            test_dates = all_dates[test_start_idx:test_end_idx]

            if not train_dates or not test_dates:
                continue

            # Filter data
            train_data = self._filter_data_by_dates(data, train_dates)
            test_data = self._filter_data_by_dates(data, test_dates)

            # Run strategy on test data
            try:
                returns = strategy_func(test_data)
                if len(returns) > 0:
                    fold_return = float(np.sum(returns))
                    fold_sharpe = (
                        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
                        if np.std(returns) > 0
                        else 0
                    )

                    fold_returns.append(fold_return)
                    fold_sharpes.append(fold_sharpe)

                    result.fold_results.append({
                        "fold": test_fold,
                        "test_start": test_dates[0],
                        "test_end": test_dates[-1],
                        "return": fold_return,
                        "sharpe": fold_sharpe,
                    })
            except Exception as e:
                logger.debug("[CPCV] Fold %d failed: %s", test_fold, e)
                continue

        # Calculate summary
        if fold_returns:
            result.mean_return = float(np.mean(fold_returns))
            result.std_return = float(np.std(fold_returns))
            result.mean_sharpe = float(np.mean(fold_sharpes))
            result.std_sharpe = float(np.std(fold_sharpes))

            # Probability of Backtest Overfitting (PBO)
            # Simplified: proportion of folds with negative return
            result.pbo = float(np.mean(np.array(fold_returns) < 0))

        logger.info(
            "[CPCV] Completed: %d folds, Mean Sharpe=%.2f, PBO=%.2f",
            len(result.fold_results),
            result.mean_sharpe,
            result.pbo,
        )

        return result

    def _get_all_dates(self, data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
        """Get sorted list of all dates."""
        all_dates = set()
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_dates.update(df.index.tolist())
        return sorted(all_dates)

    def _filter_data_by_dates(
        self,
        data: Dict[str, pd.DataFrame],
        dates: List[pd.Timestamp],
    ) -> Dict[str, pd.DataFrame]:
        """Filter data to specific dates."""
        date_set = set(dates)
        filtered = {}
        for symbol, df in data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                mask = df.index.isin(date_set)
                filtered[symbol] = df[mask].copy()
        return filtered


@dataclass
class DSRResult:
    """Result from Deflated Sharpe Ratio calculation."""

    raw_sharpe: float = 0.0
    deflated_sharpe: float = 0.0
    p_value: float = 1.0  # Probability under null hypothesis
    n_trials: int = 1
    is_significant: bool = False
    haircut_pct: float = 0.0  # % reduction from multiple testing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_sharpe": self.raw_sharpe,
            "deflated_sharpe": self.deflated_sharpe,
            "p_value": self.p_value,
            "n_trials": self.n_trials,
            "is_significant": self.is_significant,
            "haircut_pct": self.haircut_pct,
        }


class DeflatedSharpeRatio:
    """
    Deflated Sharpe Ratio (DSR)

    Adjusts Sharpe ratio for multiple testing bias.
    When you test N strategies and pick the best, the expected
    Sharpe of the best is inflated even under the null hypothesis.

    Formula (Bailey et al. 2014):
    DSR = SR * sqrt(1 - γ * (N-1) / T)

    Where:
    - SR: Observed Sharpe ratio
    - γ: Euler-Mascheroni constant (~0.5772)
    - N: Number of trials/strategies tested
    - T: Number of observations

    Reference: Bailey, Borwein, Lopez de Prado, Zhu (2014)
    """

    EULER_MASCHERONI = 0.5772156649

    def __init__(self, significance_level: float = 0.05):
        """
        Initialize DSR calculator.

        Args:
            significance_level: P-value threshold for significance
        """
        self.significance_level = significance_level

    def calculate(
        self,
        returns: np.ndarray,
        n_trials: int = 1,
        skewness: Optional[float] = None,
        kurtosis: Optional[float] = None,
    ) -> DSRResult:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            returns: Array of strategy returns
            n_trials: Number of strategies/parameter combos tested
            skewness: Optional skewness (auto-calculated if None)
            kurtosis: Optional excess kurtosis (auto-calculated if None)

        Returns:
            DSRResult
        """
        if len(returns) < 30:
            logger.warning("[DSR] Insufficient data (n=%d, need 30+)", len(returns))
            return DSRResult()

        T = len(returns)

        # Calculate raw Sharpe
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return DSRResult()

        raw_sharpe = mean_ret / std_ret * np.sqrt(252)  # Annualized

        # Calculate skewness and kurtosis if not provided
        if skewness is None:
            skewness = float(stats.skew(returns))
        if kurtosis is None:
            kurtosis = float(stats.kurtosis(returns))  # Excess kurtosis

        # Expected maximum Sharpe under null hypothesis (E[max(SR)])
        # Using approximation from Bailey et al.
        expected_max_sr = self._expected_max_sharpe(n_trials, T)

        # Variance of Sharpe ratio estimator
        # Var(SR) ≈ (1 + 0.5*SR^2 - γ_3*SR + (γ_4-1)/4 * SR^2) / T
        # where γ_3 = skewness, γ_4 = kurtosis
        var_sr = (
            1
            + 0.5 * raw_sharpe**2
            - skewness * raw_sharpe
            + (kurtosis - 1) / 4 * raw_sharpe**2
        ) / T

        std_sr = np.sqrt(var_sr) if var_sr > 0 else 0.001

        # Calculate DSR and p-value
        if std_sr > 0:
            z_score = (raw_sharpe - expected_max_sr) / std_sr
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            p_value = 1.0

        # Deflated Sharpe (adjusted for multiple testing)
        deflated_sharpe = raw_sharpe - expected_max_sr

        # Haircut percentage
        haircut_pct = (
            (raw_sharpe - deflated_sharpe) / raw_sharpe * 100
            if raw_sharpe != 0
            else 0
        )

        result = DSRResult(
            raw_sharpe=raw_sharpe,
            deflated_sharpe=deflated_sharpe,
            p_value=p_value,
            n_trials=n_trials,
            is_significant=p_value < self.significance_level,
            haircut_pct=haircut_pct,
        )

        logger.info(
            "[DSR] Raw=%.2f, Deflated=%.2f, p=%.3f, Trials=%d",
            raw_sharpe,
            deflated_sharpe,
            p_value,
            n_trials,
        )

        return result

    def _expected_max_sharpe(self, n: int, T: int) -> float:
        """
        Calculate expected maximum Sharpe ratio under null hypothesis.

        Using approximation: E[max(SR_1, ..., SR_n)] ≈ sqrt(2 * log(n)) / sqrt(T)

        For small n, uses exact formula.
        """
        if n <= 1:
            return 0.0

        # Approximation for expected max of n standard normal variables
        # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * log(n)) for large n
        if n > 5:
            expected_max_z = np.sqrt(2 * np.log(n))
        else:
            # More accurate for small n
            expected_max_z = self.EULER_MASCHERONI + np.log(n)

        # Scale by 1/sqrt(T) and annualize
        return expected_max_z / np.sqrt(T) * np.sqrt(252)


def validate_strategy(
    data: Dict[str, pd.DataFrame],
    strategy_func: Callable,
    n_trials: int = 1,
    wfo_config: Optional[WFOConfig] = None,
    cpcv_config: Optional[CPCVConfig] = None,
) -> Dict[str, Any]:
    """
    Comprehensive strategy validation.

    Runs WFO, CPCV, and DSR validation.

    Args:
        data: Dict of symbol -> DataFrame
        strategy_func: Function(data) -> returns array
        n_trials: Number of strategies tested for DSR
        wfo_config: Optional WFO configuration
        cpcv_config: Optional CPCV configuration

    Returns:
        Dict with validation results
    """
    results = {}

    # Walk-Forward Optimization
    if wfo_config is None:
        wfo_config = WFOConfig()

    wfo = WalkForwardOptimizer(wfo_config)
    wfo_result = wfo.run(data, lambda d, p: strategy_func(d))
    results["wfo"] = wfo_result.to_dict()

    # CPCV
    if cpcv_config is None:
        cpcv_config = CPCVConfig()

    cpcv = CPCVValidator(cpcv_config)
    cpcv_result = cpcv.run(data, strategy_func)
    results["cpcv"] = cpcv_result.to_dict()

    # Deflated Sharpe Ratio
    # Combine all OOS returns for DSR calculation
    all_returns = np.array(wfo_result.oos_returns) if wfo_result.oos_returns else np.array([])

    if len(all_returns) >= 4:  # Need minimum samples
        dsr = DeflatedSharpeRatio()
        dsr_result = dsr.calculate(all_returns, n_trials=n_trials)
        results["dsr"] = dsr_result.to_dict()
    else:
        results["dsr"] = {"error": "Insufficient OOS periods for DSR"}

    # Overall assessment
    results["verdict"] = _assess_validation(results)

    return results


def _assess_validation(results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall validation results."""
    verdict = {
        "is_robust": False,
        "confidence": "low",
        "warnings": [],
    }

    warnings = []

    # Check WFO efficiency
    wfo = results.get("wfo", {})
    efficiency = wfo.get("efficiency_ratio", 0)
    if efficiency < 0.5:
        warnings.append("WFO efficiency ratio < 0.5 (high overfit risk)")
    elif efficiency < 0.7:
        warnings.append("WFO efficiency ratio < 0.7 (moderate overfit risk)")

    robustness = wfo.get("robustness_score", 0)
    if robustness < 0.5:
        warnings.append("WFO robustness < 50% (inconsistent OOS performance)")

    # Check CPCV
    cpcv = results.get("cpcv", {})
    pbo = cpcv.get("pbo", 1.0)
    if pbo > 0.5:
        warnings.append(f"CPCV PBO = {pbo:.0%} (high probability of overfitting)")

    # Check DSR
    dsr = results.get("dsr", {})
    if "error" not in dsr:
        if not dsr.get("is_significant", False):
            warnings.append("DSR not statistically significant")
        if dsr.get("haircut_pct", 0) > 50:
            warnings.append(f"DSR haircut = {dsr.get('haircut_pct', 0):.0f}%")

    verdict["warnings"] = warnings

    # Determine confidence level
    if len(warnings) == 0:
        verdict["is_robust"] = True
        verdict["confidence"] = "high"
    elif len(warnings) <= 2:
        verdict["confidence"] = "medium"
    else:
        verdict["confidence"] = "low"

    return verdict
