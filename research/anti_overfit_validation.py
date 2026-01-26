"""
Anti-Overfitting Validation Suite
Strategy Quality: B+ → A+ Upgrade

Key techniques:
1. CSCV (Combinatorially Symmetric Cross-Validation)
2. Probability of Backtest Overfitting (PBO)
3. Deflated Sharpe Ratio
4. Monte Carlo Permutation Test
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class AntiOverfitValidator:
    """
    Implements rigorous anti-overfitting validation methods.

    References:
    - Bailey et al. (2014) "Probability of Backtest Overfitting"
    - Harvey et al. (2016) "... and the Cross-Section of Expected Returns"
    """

    def __init__(self, returns: pd.Series, n_trials: int = 1000):
        """
        Args:
            returns: Strategy daily returns
            n_trials: Number of trials for Monte Carlo tests
        """
        self.returns = returns
        self.n_trials = n_trials
        self.results = {}

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_strategies_tested: int,
        lookback_years: float,
        skewness: float = 0,
        kurtosis: float = 3
    ) -> Tuple[float, float]:
        """
        Calculate Deflated Sharpe Ratio (DSR).

        Adjusts for multiple testing bias - the more strategies you test,
        the more likely you find one that looks good by chance.

        Args:
            sharpe: Observed Sharpe ratio
            n_strategies_tested: Total number of strategy variants tested
            lookback_years: Years of data used
            skewness: Return skewness
            kurtosis: Return kurtosis

        Returns:
            (deflated_sharpe, p_value)
        """
        # Expected maximum Sharpe under null hypothesis
        euler_mascheroni = 0.5772156649

        # Approximation for E[max(Z_1, ..., Z_N)] where Z_i ~ N(0,1)
        if n_strategies_tested > 1:
            expected_max = (1 - euler_mascheroni) * stats.norm.ppf(1 - 1/n_strategies_tested) + \
                          euler_mascheroni * stats.norm.ppf(1 - 1/(n_strategies_tested * np.e))
        else:
            expected_max = 0

        # Variance of Sharpe ratio estimator
        # From Lo (2002) "The Statistics of Sharpe Ratios"
        n_obs = lookback_years * 252
        var_sharpe = (1 + 0.5 * sharpe**2 - skewness * sharpe +
                      (kurtosis - 3) / 4 * sharpe**2) / n_obs

        # Deflated Sharpe
        dsr = (sharpe - expected_max * np.sqrt(var_sharpe)) / np.sqrt(var_sharpe)

        # P-value
        p_value = 1 - stats.norm.cdf(dsr)

        return dsr, p_value

    def probability_of_backtest_overfitting(
        self,
        strategy_returns: pd.DataFrame,
        n_partitions: int = 16
    ) -> Tuple[float, dict]:
        """
        Calculate Probability of Backtest Overfitting (PBO).

        Uses CSCV to estimate the probability that the best in-sample
        strategy will underperform out-of-sample.

        Args:
            strategy_returns: DataFrame with columns as different strategy variants
            n_partitions: Number of time partitions (must be even)

        Returns:
            (pbo, details_dict)
        """
        if n_partitions % 2 != 0:
            n_partitions = n_partitions - 1

        n_rows = len(strategy_returns)
        partition_size = n_rows // n_partitions

        # Create partitions
        partitions = []
        for i in range(n_partitions):
            start = i * partition_size
            end = start + partition_size if i < n_partitions - 1 else n_rows
            partitions.append(strategy_returns.iloc[start:end])

        # Generate all combinations of n/2 partitions for training
        train_combos = list(combinations(range(n_partitions), n_partitions // 2))

        # Track results
        is_best_ranks = []
        oos_ranks = []
        logits = []

        for train_idx in train_combos:
            test_idx = tuple(i for i in range(n_partitions) if i not in train_idx)

            # Combine partitions
            train_data = pd.concat([partitions[i] for i in train_idx])
            test_data = pd.concat([partitions[i] for i in test_idx])

            # Calculate Sharpe ratios
            train_sharpes = train_data.mean() / train_data.std() * np.sqrt(252)
            test_sharpes = test_data.mean() / test_data.std() * np.sqrt(252)

            # Find best IS strategy and its OOS rank
            best_is_strategy = train_sharpes.idxmax()

            # Rank (1 = best)
            oos_rank = test_sharpes.rank(ascending=False)[best_is_strategy]
            n_strategies = len(strategy_returns.columns)

            # Relative rank (0 = best, 1 = worst)
            relative_rank = (oos_rank - 1) / (n_strategies - 1)

            is_best_ranks.append(best_is_strategy)
            oos_ranks.append(relative_rank)

            # Logit for PBO calculation
            if relative_rank > 0 and relative_rank < 1:
                logits.append(np.log(relative_rank / (1 - relative_rank)))

        # PBO = probability that OOS rank > 0.5 (worse than median)
        pbo = np.mean([r > 0.5 for r in oos_ranks])

        return pbo, {
            'mean_oos_rank': np.mean(oos_ranks),
            'std_oos_rank': np.std(oos_ranks),
            'n_combinations': len(train_combos),
            'oos_ranks': oos_ranks
        }

    def monte_carlo_permutation_test(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> Tuple[float, float]:
        """
        Monte Carlo permutation test for strategy significance.

        Shuffles returns to destroy any predictability, then compares
        actual strategy performance to random distribution.

        Args:
            strategy_returns: Strategy daily returns
            benchmark_returns: Benchmark returns (default: None = test vs 0)

        Returns:
            (p_value, percentile)
        """
        actual_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

        random_sharpes = []
        returns_array = strategy_returns.values.copy()

        for _ in range(self.n_trials):
            # Shuffle returns (destroys any temporal pattern)
            np.random.shuffle(returns_array)
            random_sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            random_sharpes.append(random_sharpe)

        # P-value: proportion of random Sharpes >= actual
        p_value = np.mean([rs >= actual_sharpe for rs in random_sharpes])

        # Percentile of actual Sharpe in random distribution
        percentile = stats.percentileofscore(random_sharpes, actual_sharpe)

        return p_value, percentile

    def time_series_bootstrap(
        self,
        strategy_returns: pd.Series,
        block_size: int = 20,
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Stationary bootstrap for time series (preserves autocorrelation).

        Args:
            strategy_returns: Strategy daily returns
            block_size: Average block length
            confidence_level: Confidence level for interval

        Returns:
            (sharpe_mean, ci_lower, ci_upper)
        """
        n = len(strategy_returns)
        bootstrap_sharpes = []

        for _ in range(self.n_trials):
            # Stationary bootstrap
            indices = []
            i = 0
            while len(indices) < n:
                # Random starting point
                if i == 0 or np.random.random() < 1/block_size:
                    i = np.random.randint(0, n)
                indices.append(i)
                i = (i + 1) % n

            bootstrap_sample = strategy_returns.iloc[indices[:n]]
            sharpe = bootstrap_sample.mean() / bootstrap_sample.std() * np.sqrt(252)
            bootstrap_sharpes.append(sharpe)

        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_sharpes, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_sharpes, (1 - alpha/2) * 100)

        return np.mean(bootstrap_sharpes), ci_lower, ci_upper

    def minimum_track_record_length(
        self,
        target_sharpe: float,
        observed_sharpe: float,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate minimum track record length needed to confirm skill.

        From Bailey & López de Prado (2012).

        Args:
            target_sharpe: Minimum Sharpe to claim skill (e.g., 0.5)
            observed_sharpe: Observed Sharpe ratio
            confidence_level: Required confidence level

        Returns:
            Minimum years of track record needed
        """
        if observed_sharpe <= target_sharpe:
            return np.inf

        z = stats.norm.ppf(confidence_level)

        # MinTRL formula
        min_trl = 1 + (1 - observed_sharpe * target_sharpe +
                       observed_sharpe**2 / 4 * (observed_sharpe**2 - 4)) * \
                  (z / (observed_sharpe - target_sharpe))**2

        return min_trl / 252  # Convert to years

    def run_full_validation(
        self,
        n_strategies_tested: int = 100,
        lookback_years: float = 5
    ) -> dict:
        """Run all validation tests."""

        results = {}

        # 1. Deflated Sharpe Ratio
        actual_sharpe = self.returns.mean() / self.returns.std() * np.sqrt(252)
        skewness = stats.skew(self.returns)
        kurtosis = stats.kurtosis(self.returns) + 3  # scipy returns excess kurtosis

        dsr, dsr_pvalue = self.deflated_sharpe_ratio(
            actual_sharpe, n_strategies_tested, lookback_years, skewness, kurtosis
        )
        results['deflated_sharpe_ratio'] = {
            'raw_sharpe': actual_sharpe,
            'deflated_sharpe': dsr,
            'p_value': dsr_pvalue,
            'pass': dsr_pvalue < 0.05
        }

        # 2. Monte Carlo Permutation
        mc_pvalue, mc_percentile = self.monte_carlo_permutation_test(self.returns)
        results['monte_carlo_test'] = {
            'p_value': mc_pvalue,
            'percentile': mc_percentile,
            'pass': mc_pvalue < 0.05
        }

        # 3. Bootstrap Confidence Interval
        bs_mean, bs_lower, bs_upper = self.time_series_bootstrap(self.returns)
        results['bootstrap_ci'] = {
            'mean_sharpe': bs_mean,
            'ci_lower': bs_lower,
            'ci_upper': bs_upper,
            'ci_lower_positive': bs_lower > 0,
            'pass': bs_lower > 0
        }

        # 4. Minimum Track Record
        min_trl = self.minimum_track_record_length(0.5, actual_sharpe)
        results['minimum_track_record'] = {
            'years_needed': min_trl,
            'years_available': lookback_years,
            'pass': lookback_years >= min_trl
        }

        # Overall Assessment
        n_passed = sum([
            results['deflated_sharpe_ratio']['pass'],
            results['monte_carlo_test']['pass'],
            results['bootstrap_ci']['pass'],
            results['minimum_track_record']['pass']
        ])

        results['overall'] = {
            'tests_passed': n_passed,
            'total_tests': 4,
            'grade': 'A+' if n_passed == 4 else 'A' if n_passed == 3 else 'B' if n_passed == 2 else 'C'
        }

        return results


def example_usage():
    """Example of how to use the validator."""

    # Generate sample returns (replace with real strategy returns)
    np.random.seed(42)
    n_days = 252 * 5  # 5 years

    # Simulated strategy with slight edge
    daily_returns = pd.Series(
        np.random.normal(0.0005, 0.02, n_days),  # ~12% annual, 32% vol
        index=pd.date_range('2020-01-01', periods=n_days, freq='D')
    )

    # Run validation
    validator = AntiOverfitValidator(daily_returns, n_trials=1000)
    results = validator.run_full_validation(
        n_strategies_tested=50,  # How many variants were tested
        lookback_years=5
    )

    # Print results
    print("=" * 60)
    print("ANTI-OVERFITTING VALIDATION REPORT")
    print("=" * 60)

    print("\n1. Deflated Sharpe Ratio:")
    print(f"   Raw Sharpe: {results['deflated_sharpe_ratio']['raw_sharpe']:.3f}")
    print(f"   Deflated Sharpe: {results['deflated_sharpe_ratio']['deflated_sharpe']:.3f}")
    print(f"   P-value: {results['deflated_sharpe_ratio']['p_value']:.4f}")
    print(f"   Pass: {results['deflated_sharpe_ratio']['pass']}")

    print("\n2. Monte Carlo Permutation Test:")
    print(f"   P-value: {results['monte_carlo_test']['p_value']:.4f}")
    print(f"   Percentile: {results['monte_carlo_test']['percentile']:.1f}%")
    print(f"   Pass: {results['monte_carlo_test']['pass']}")

    print("\n3. Bootstrap 95% Confidence Interval:")
    print(f"   Mean Sharpe: {results['bootstrap_ci']['mean_sharpe']:.3f}")
    print(f"   CI: [{results['bootstrap_ci']['ci_lower']:.3f}, {results['bootstrap_ci']['ci_upper']:.3f}]")
    print(f"   Pass: {results['bootstrap_ci']['pass']}")

    print("\n4. Minimum Track Record Length:")
    print(f"   Years Needed: {results['minimum_track_record']['years_needed']:.2f}")
    print(f"   Years Available: {results['minimum_track_record']['years_available']:.2f}")
    print(f"   Pass: {results['minimum_track_record']['pass']}")

    print("\n" + "=" * 60)
    print(f"OVERALL GRADE: {results['overall']['grade']}")
    print(f"Tests Passed: {results['overall']['tests_passed']}/{results['overall']['total_tests']}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    example_usage()
