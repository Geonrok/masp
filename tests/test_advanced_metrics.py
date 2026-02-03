"""
Tests for advanced performance metrics.
"""

import pytest
import numpy as np
from libs.analytics.advanced_metrics import (
    calculate_omega_ratio,
    calculate_var,
    calculate_cvar,
    calculate_information_ratio,
    calculate_treynor_ratio,
    calculate_tail_ratio,
    calculate_gain_to_pain,
    calculate_skewness,
    calculate_kurtosis,
    analyze_drawdowns,
    calculate_expectancy,
    calculate_payoff_ratio,
    calculate_advanced_metrics,
    generate_performance_report,
)


class TestOmegaRatio:
    """Tests for Omega Ratio calculation."""

    def test_omega_all_positive(self):
        """Test Omega with all positive returns."""
        returns = [0.01, 0.02, 0.015, 0.01, 0.02]
        omega = calculate_omega_ratio(returns)
        assert omega == float("inf")

    def test_omega_mixed_returns(self):
        """Test Omega with mixed returns."""
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        omega = calculate_omega_ratio(returns)
        assert omega > 1  # More gains than losses

    def test_omega_all_negative(self):
        """Test Omega with all negative returns."""
        returns = [-0.01, -0.02, -0.015]
        omega = calculate_omega_ratio(returns)
        assert omega == 0  # No gains

    def test_omega_with_threshold(self):
        """Test Omega with non-zero threshold."""
        returns = [0.01, 0.02, 0.015, 0.005, 0.025]
        omega = calculate_omega_ratio(returns, threshold=0.01)
        assert omega > 0


class TestVaR:
    """Tests for Value at Risk calculation."""

    def test_var_historical(self):
        """Test historical VaR."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))

        var_95 = calculate_var(returns, 0.95, "historical")
        var_99 = calculate_var(returns, 0.99, "historical")

        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher

    def test_var_parametric(self):
        """Test parametric VaR."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))

        var_95 = calculate_var(returns, 0.95, "parametric")
        assert var_95 > 0

    def test_var_insufficient_data(self):
        """Test VaR with insufficient data."""
        returns = [0.01]
        var = calculate_var(returns, 0.95)
        assert var == 0


class TestCVaR:
    """Tests for Conditional VaR calculation."""

    def test_cvar_greater_than_var(self):
        """CVaR should be >= VaR."""
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.02, 100))

        var_95 = calculate_var(returns, 0.95)
        cvar_95 = calculate_cvar(returns, 0.95)

        assert cvar_95 >= var_95

    def test_cvar_99(self):
        """Test 99% CVaR."""
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.02, 100))

        cvar_95 = calculate_cvar(returns, 0.95)
        cvar_99 = calculate_cvar(returns, 0.99)

        assert cvar_99 >= cvar_95


class TestInformationRatio:
    """Tests for Information Ratio calculation."""

    def test_ir_outperformance(self):
        """Test IR when portfolio outperforms."""
        portfolio = [0.02, 0.015, 0.025, 0.01, 0.03]
        benchmark = [0.01, 0.01, 0.01, 0.01, 0.01]

        ir = calculate_information_ratio(portfolio, benchmark)
        assert ir > 0

    def test_ir_underperformance(self):
        """Test IR when portfolio underperforms."""
        portfolio = [0.005, 0.008, 0.004, 0.006, 0.007]
        benchmark = [0.01, 0.01, 0.01, 0.01, 0.01]

        ir = calculate_information_ratio(portfolio, benchmark)
        assert ir < 0

    def test_ir_mismatched_lengths(self):
        """Test IR with mismatched data lengths."""
        portfolio = [0.01, 0.02]
        benchmark = [0.01]

        ir = calculate_information_ratio(portfolio, benchmark)
        assert ir == 0


class TestTreynorRatio:
    """Tests for Treynor Ratio calculation."""

    def test_treynor_positive_beta(self):
        """Test Treynor with positive beta."""
        np.random.seed(42)
        benchmark = list(np.random.normal(0.001, 0.02, 50))
        # Portfolio with positive correlation to benchmark
        portfolio = [b * 1.2 + 0.001 for b in benchmark]

        treynor = calculate_treynor_ratio(portfolio, benchmark)
        assert treynor != 0


class TestTailRatio:
    """Tests for Tail Ratio calculation."""

    def test_tail_ratio_symmetric(self):
        """Test tail ratio with symmetric distribution."""
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.02, 100))

        tail = calculate_tail_ratio(returns)
        # Should be close to 1 for symmetric
        assert 0.5 < tail < 2

    def test_tail_ratio_right_skewed(self):
        """Test tail ratio with right-skewed distribution."""
        # More large positive returns
        returns = list(np.random.exponential(0.01, 50)) + list(
            -np.random.exponential(0.005, 50)
        )

        tail = calculate_tail_ratio(returns)
        assert tail > 0


class TestGainToPain:
    """Tests for Gain-to-Pain Ratio calculation."""

    def test_gain_to_pain_positive(self):
        """Test G2P with net positive returns."""
        returns = [0.02, -0.01, 0.03, -0.005, 0.015]
        g2p = calculate_gain_to_pain(returns)
        assert g2p > 0

    def test_gain_to_pain_no_losses(self):
        """Test G2P with no losses."""
        returns = [0.01, 0.02, 0.015]
        g2p = calculate_gain_to_pain(returns)
        assert g2p == float("inf")


class TestSkewnessKurtosis:
    """Tests for distribution metrics."""

    def test_skewness_positive(self):
        """Test positive skewness."""
        # Right-skewed: more extreme positive values
        returns = [0.01, 0.02, 0.01, 0.015, 0.10]  # One large positive
        skew = calculate_skewness(returns)
        assert skew > 0

    def test_skewness_negative(self):
        """Test negative skewness."""
        # Left-skewed: more extreme negative values
        returns = [0.01, 0.02, 0.01, 0.015, -0.10]  # One large negative
        skew = calculate_skewness(returns)
        assert skew < 0

    def test_kurtosis_normal(self):
        """Test kurtosis near zero for normal distribution."""
        np.random.seed(42)
        returns = list(np.random.normal(0, 0.02, 1000))
        kurt = calculate_kurtosis(returns)
        # Should be close to 0 for normal
        assert -0.5 < kurt < 0.5


class TestDrawdownAnalysis:
    """Tests for drawdown analysis."""

    def test_drawdown_increasing(self):
        """Test with constantly increasing equity."""
        equity = [100, 101, 102, 103, 104, 105]
        dd = analyze_drawdowns(equity)

        assert dd.max_drawdown == 0
        assert dd.num_drawdowns == 0

    def test_drawdown_single(self):
        """Test with single drawdown."""
        equity = [100, 95, 90, 92, 100, 105]
        dd = analyze_drawdowns(equity)

        assert dd.max_drawdown == 0.10  # 10% drawdown
        assert dd.max_drawdown_pct == 10

    def test_drawdown_recovery_time(self):
        """Test recovery time calculation."""
        equity = [100, 90, 85, 90, 95, 100]
        dd = analyze_drawdowns(equity)

        assert dd.recovery_time is not None
        assert dd.recovery_time > 0

    def test_underwater_curve(self):
        """Test underwater curve generation."""
        equity = [100, 95, 90, 95, 100, 95]
        dd = analyze_drawdowns(equity)

        assert len(dd.underwater_curve) == len(equity)
        assert dd.underwater_curve[0] == 0  # No drawdown at start


class TestTradeStatistics:
    """Tests for trade statistics."""

    def test_expectancy_positive(self):
        """Test positive expectancy."""
        wins = [100, 150, 120, 80]  # Avg: 112.5
        losses = [50, 40, 60]  # Avg: 50

        # Win rate = 4/7 = 0.57
        # Expectancy = (0.57 * 112.5) - (0.43 * 50) = 64 - 21.5 = 42.5
        exp = calculate_expectancy(wins, losses)
        assert exp > 0

    def test_payoff_ratio(self):
        """Test payoff ratio calculation."""
        wins = [100, 200, 150]  # Avg: 150
        losses = [-50, -75, -25]  # Avg loss: 50

        payoff = calculate_payoff_ratio(wins, losses)
        assert payoff == 3.0  # 150/50


class TestAdvancedMetrics:
    """Tests for comprehensive metrics calculation."""

    def test_calculate_advanced_metrics(self):
        """Test full metrics calculation."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        equity = [10000]
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        metrics = calculate_advanced_metrics(returns, equity)

        assert metrics.sharpe_ratio != 0
        assert metrics.var_95 > 0
        assert metrics.max_drawdown_pct >= 0
        assert metrics.win_rate >= 0

    def test_empty_data(self):
        """Test with empty data."""
        metrics = calculate_advanced_metrics([], [])

        assert metrics.sharpe_ratio == 0
        assert metrics.omega_ratio == 0


class TestPerformanceReport:
    """Tests for report generation."""

    def test_generate_report(self):
        """Test report generation."""
        np.random.seed(42)
        returns = list(np.random.normal(0.001, 0.02, 100))
        equity = [10000]
        for r in returns:
            equity.append(equity[-1] * (1 + r))

        metrics = calculate_advanced_metrics(returns, equity)
        report = generate_performance_report(metrics, "Test Strategy")

        assert "PERFORMANCE REPORT" in report
        assert "Test Strategy" in report
        assert "Sharpe Ratio" in report
        assert "VaR" in report
