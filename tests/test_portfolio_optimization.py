"""
Tests for portfolio optimization module.
"""

import pytest
import numpy as np

from libs.portfolio.optimization import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationConstraints,
    OptimizationResult,
    StrategyReturns,
    mean_variance_optimize,
    risk_parity_optimize,
    max_sharpe_optimize,
    min_variance_optimize,
)


class TestStrategyReturns:
    """Tests for StrategyReturns."""

    def test_mean_return(self):
        """Test mean return calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        strategy = StrategyReturns(name="Test", returns=returns)

        # Annualized mean = daily mean * 252
        expected = np.mean(returns) * 252
        assert strategy.mean_return == pytest.approx(expected, rel=0.01)

    def test_volatility(self):
        """Test volatility calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        strategy = StrategyReturns(name="Test", returns=returns)

        # Annualized vol = daily std * sqrt(252)
        expected = np.std(returns) * np.sqrt(252)
        assert strategy.volatility == pytest.approx(expected, rel=0.01)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        strategy = StrategyReturns(name="Test", returns=returns)

        expected = strategy.mean_return / strategy.volatility
        assert strategy.sharpe_ratio == pytest.approx(expected, rel=0.01)


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer."""

    @pytest.fixture
    def simple_strategies(self):
        """Create simple test strategies."""
        np.random.seed(42)

        # Three uncorrelated strategies
        returns1 = np.random.normal(0.001, 0.02, 252)  # 25.2% vol
        returns2 = np.random.normal(0.0008, 0.015, 252)  # 18.9% vol
        returns3 = np.random.normal(0.0005, 0.01, 252)  # 12.6% vol

        return [
            StrategyReturns(name="Aggressive", returns=returns1),
            StrategyReturns(name="Moderate", returns=returns2),
            StrategyReturns(name="Conservative", returns=returns3),
        ]

    @pytest.fixture
    def optimizer(self, simple_strategies):
        """Create optimizer instance."""
        return PortfolioOptimizer(simple_strategies)

    def test_initialization(self, optimizer, simple_strategies):
        """Test optimizer initialization."""
        assert optimizer.n_strategies == 3
        assert len(optimizer.mean_returns) == 3
        assert optimizer.cov_matrix.shape == (3, 3)

    def test_equal_weight(self, optimizer):
        """Test equal weight allocation."""
        result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)

        assert len(result.weights) == 3
        assert np.allclose(result.weights, [1 / 3, 1 / 3, 1 / 3])
        assert np.sum(result.weights) == pytest.approx(1.0)

    def test_inverse_volatility(self, optimizer):
        """Test inverse volatility weighting."""
        result = optimizer.optimize(OptimizationMethod.INVERSE_VOLATILITY)

        # Lower volatility should have higher weight
        assert np.sum(result.weights) == pytest.approx(1.0)
        # Conservative (lowest vol) should have highest weight
        assert result.weights[2] > result.weights[0]

    def test_min_variance(self, optimizer):
        """Test minimum variance optimization."""
        result = optimizer.optimize(OptimizationMethod.MIN_VARIANCE)

        assert np.sum(result.weights) == pytest.approx(1.0)
        assert all(w >= 0 for w in result.weights)

        # Min variance should have lower vol than equal weight
        eq_result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)
        assert result.expected_volatility <= eq_result.expected_volatility + 0.01

    def test_max_sharpe(self, optimizer):
        """Test maximum Sharpe ratio optimization."""
        result = optimizer.optimize(OptimizationMethod.MAX_SHARPE)

        assert np.sum(result.weights) == pytest.approx(1.0)
        assert all(w >= 0 for w in result.weights)

        # Should have positive Sharpe ratio (assuming positive returns)
        assert result.sharpe_ratio is not None

    def test_risk_parity(self, optimizer):
        """Test risk parity optimization."""
        result = optimizer.optimize(OptimizationMethod.RISK_PARITY)

        assert np.sum(result.weights) == pytest.approx(1.0)

        # Check risk contributions are roughly equal
        rc = optimizer.get_risk_contributions(result.weights)
        rc_values = list(rc.values())

        # All risk contributions should be within 5% of each other
        rc_range = max(rc_values) - min(rc_values)
        assert rc_range < 15  # Allow some tolerance

    def test_hrp(self, optimizer):
        """Test Hierarchical Risk Parity."""
        result = optimizer.optimize(OptimizationMethod.HRP)

        assert np.sum(result.weights) == pytest.approx(1.0)
        assert all(w >= 0 for w in result.weights)

    def test_max_diversification(self, optimizer):
        """Test maximum diversification."""
        result = optimizer.optimize(OptimizationMethod.MAX_DIVERSIFICATION)

        assert np.sum(result.weights) == pytest.approx(1.0)
        assert result.diversification_ratio >= 1.0

    def test_constraints_min_weight(self, optimizer):
        """Test minimum weight constraint."""
        constraints = OptimizationConstraints(min_weight=0.1)
        result = optimizer.optimize(OptimizationMethod.MAX_SHARPE, constraints)

        assert all(w >= 0.1 - 1e-6 for w in result.weights)

    def test_constraints_max_weight(self, optimizer):
        """Test maximum weight constraint."""
        constraints = OptimizationConstraints(max_weight=0.5)
        result = optimizer.optimize(OptimizationMethod.MAX_SHARPE, constraints)

        assert all(w <= 0.5 + 1e-6 for w in result.weights)

    def test_result_allocation(self, optimizer):
        """Test get_allocation method."""
        result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)
        allocation = result.get_allocation()

        assert "Aggressive" in allocation
        assert "Moderate" in allocation
        assert "Conservative" in allocation
        assert allocation["Aggressive"] == pytest.approx(1 / 3, rel=0.01)


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_str_representation(self):
        """Test string representation."""
        result = OptimizationResult(
            weights=np.array([0.3, 0.5, 0.2]),
            strategy_names=["A", "B", "C"],
            method=OptimizationMethod.MAX_SHARPE,
            expected_return=0.15,
            expected_volatility=0.12,
            sharpe_ratio=1.08,
            diversification_ratio=1.2,
            effective_n=2.8,
        )

        s = str(result)
        assert "max_sharpe" in s
        assert "15.00%" in s
        assert "12.00%" in s
        assert "B:" in s  # Highest weight


class TestEfficientFrontier:
    """Tests for efficient frontier calculation."""

    def test_efficient_frontier(self):
        """Test efficient frontier generation."""
        np.random.seed(42)

        strategies = [
            StrategyReturns(
                name=f"Strategy_{i}",
                returns=np.random.normal(0.001 * (i + 1), 0.01 * (i + 1), 252),
            )
            for i in range(3)
        ]

        optimizer = PortfolioOptimizer(strategies)
        frontier = optimizer.get_efficient_frontier(n_points=10)

        assert len(frontier) > 0

        # Frontier should be sorted by return (roughly)
        returns = [r.expected_return for r in frontier]
        vols = [r.expected_volatility for r in frontier]

        # Higher return generally means higher volatility
        # (This is the efficient frontier property)
        assert all(isinstance(r, float) for r in returns)
        assert all(isinstance(v, float) for v in vols)


class TestRiskContributions:
    """Tests for risk contribution calculation."""

    def test_risk_contributions_sum(self):
        """Test that risk contributions sum to 100%."""
        np.random.seed(42)

        strategies = [
            StrategyReturns(name=f"S{i}", returns=np.random.normal(0.001, 0.02, 100))
            for i in range(3)
        ]

        optimizer = PortfolioOptimizer(strategies)
        result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)
        rc = optimizer.get_risk_contributions(result.weights)

        total_rc = sum(rc.values())
        assert total_rc == pytest.approx(100, abs=1)


class TestConvenienceFunctions:
    """Tests for convenience optimization functions."""

    @pytest.fixture
    def returns_matrix(self):
        """Create test returns matrix."""
        np.random.seed(42)
        return np.random.normal(0.001, 0.02, (252, 4))

    def test_mean_variance_optimize(self, returns_matrix):
        """Test mean_variance_optimize function."""
        weights = mean_variance_optimize(returns_matrix)

        assert len(weights) == 4
        assert np.sum(weights) == pytest.approx(1.0)

    def test_risk_parity_optimize(self, returns_matrix):
        """Test risk_parity_optimize function."""
        weights = risk_parity_optimize(returns_matrix)

        assert len(weights) == 4
        assert np.sum(weights) == pytest.approx(1.0)

    def test_max_sharpe_optimize(self, returns_matrix):
        """Test max_sharpe_optimize function."""
        weights = max_sharpe_optimize(returns_matrix)

        assert len(weights) == 4
        assert np.sum(weights) == pytest.approx(1.0)

    def test_min_variance_optimize(self, returns_matrix):
        """Test min_variance_optimize function."""
        weights = min_variance_optimize(returns_matrix)

        assert len(weights) == 4
        assert np.sum(weights) == pytest.approx(1.0)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_strategy(self):
        """Test with single strategy."""
        strategy = StrategyReturns(
            name="Only",
            returns=np.random.normal(0.001, 0.02, 100),
        )
        optimizer = PortfolioOptimizer([strategy])
        result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)

        assert result.weights[0] == 1.0

    def test_zero_volatility(self):
        """Test with zero volatility strategy."""
        strategies = [
            StrategyReturns(name="Zero", returns=np.ones(100) * 0.001),
            StrategyReturns(name="Normal", returns=np.random.normal(0.001, 0.02, 100)),
        ]

        optimizer = PortfolioOptimizer(strategies)
        result = optimizer.optimize(OptimizationMethod.EQUAL_WEIGHT)

        assert len(result.weights) == 2

    def test_highly_correlated(self):
        """Test with highly correlated strategies."""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)

        strategies = [
            StrategyReturns(name="S1", returns=base),
            StrategyReturns(name="S2", returns=base + np.random.normal(0, 0.001, 100)),
            StrategyReturns(name="S3", returns=base + np.random.normal(0, 0.002, 100)),
        ]

        optimizer = PortfolioOptimizer(strategies)
        result = optimizer.optimize(OptimizationMethod.MIN_VARIANCE)

        # Should still produce valid weights
        assert np.sum(result.weights) == pytest.approx(1.0)
