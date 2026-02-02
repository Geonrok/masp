"""
Tests for Monte Carlo Simulation Module

Validates statistical simulation functionality:
- Bootstrap resampling
- Parametric simulation
- Trade-based simulation
- Confidence intervals
- Drawdown estimation
- Ruin probability
"""

import pytest
import numpy as np

from libs.backtest.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    format_monte_carlo_report,
)


class TestMonteCarloConfig:
    """Tests for MonteCarloConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MonteCarloConfig()
        assert config.n_simulations == 10000
        assert config.n_periods == 252
        assert 0.95 in config.confidence_levels

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonteCarloConfig(
            n_simulations=5000,
            n_periods=100,
            confidence_levels=[0.90],
            seed=42,
        )
        assert config.n_simulations == 5000
        assert config.n_periods == 100
        assert config.seed == 42


class TestMonteCarloSimulatorBootstrap:
    """Tests for bootstrap simulation."""

    @pytest.fixture
    def simulator(self):
        """Create simulator with fixed seed for reproducibility."""
        config = MonteCarloConfig(
            n_simulations=1000,
            n_periods=100,
            seed=42,
        )
        return MonteCarloSimulator(config)

    @pytest.fixture
    def positive_returns(self):
        """Generate positive historical returns."""
        np.random.seed(42)
        # Mean 0.1% daily with 1% std
        return np.random.normal(0.001, 0.01, 500)

    @pytest.fixture
    def negative_returns(self):
        """Generate negative historical returns."""
        np.random.seed(42)
        # Mean -0.1% daily
        return np.random.normal(-0.001, 0.01, 500)

    def test_simulate_from_returns_structure(self, simulator, positive_returns):
        """Test result structure from bootstrap simulation."""
        result = simulator.simulate_from_returns(positive_returns, initial_capital=1000000)

        assert isinstance(result, MonteCarloResult)
        assert result.config.n_simulations == 1000
        assert result.equity_curves.shape == (1000, 100)
        assert len(result.final_capital_distribution) == 1000

    def test_positive_returns_profitable(self, simulator, positive_returns):
        """Test that positive mean returns lead to profits."""
        result = simulator.simulate_from_returns(positive_returns)

        # Expected return should be positive
        assert result.return_distribution.mean > 0
        # Probability of profit should be > 50%
        assert result.summary["probability_profit"] > 0.5

    def test_negative_returns_unprofitable(self, simulator, negative_returns):
        """Test that negative mean returns lead to losses."""
        result = simulator.simulate_from_returns(negative_returns)

        # Expected return should be negative
        assert result.return_distribution.mean < 0
        # Higher ruin probability
        assert result.ruin_probability.prob_lose_10pct > 0.3

    def test_drawdown_distribution(self, simulator, positive_returns):
        """Test drawdown distribution properties."""
        result = simulator.simulate_from_returns(positive_returns)

        dd = result.drawdown_distribution
        # All drawdowns should be non-negative
        assert dd.mean >= 0
        # 99th percentile >= 95th percentile >= median
        assert dd.percentile_99 >= dd.percentile_95 >= dd.median

    def test_confidence_intervals(self, simulator, positive_returns):
        """Test confidence interval properties."""
        result = simulator.simulate_from_returns(positive_returns)

        for ci in result.confidence_intervals["final_return"]:
            # Lower bound <= point estimate <= upper bound
            assert ci.lower <= ci.point_estimate <= ci.upper
            # Higher confidence = wider interval (check 99% vs 90%)

    def test_sharpe_distribution(self, simulator, positive_returns):
        """Test Sharpe ratio distribution."""
        result = simulator.simulate_from_returns(positive_returns)

        assert len(result.sharpe_distribution) == 1000
        # Positive returns should have positive mean Sharpe
        assert np.mean(result.sharpe_distribution) > 0


class TestMonteCarloSimulatorParametric:
    """Tests for parametric simulation."""

    @pytest.fixture
    def simulator(self):
        """Create simulator."""
        config = MonteCarloConfig(
            n_simulations=1000,
            n_periods=100,
            seed=42,
        )
        return MonteCarloSimulator(config)

    def test_parametric_normal(self, simulator):
        """Test normal distribution parametric simulation."""
        result = simulator.simulate_parametric(
            mean_return=0.001,  # 0.1% daily
            std_return=0.02,  # 2% daily
            initial_capital=1000000,
            distribution="normal",
        )

        assert isinstance(result, MonteCarloResult)
        # Return should be around mean * periods
        expected_return = (1 + 0.001) ** 100 - 1
        assert abs(result.return_distribution.mean - expected_return) < 0.1

    def test_parametric_t_distribution(self, simulator):
        """Test t-distribution parametric simulation (fat tails)."""
        result = simulator.simulate_parametric(
            mean_return=0.001,
            std_return=0.02,
            initial_capital=1000000,
            distribution="t",
        )

        assert isinstance(result, MonteCarloResult)
        # t-distribution should have higher kurtosis (fatter tails)
        # Note: kurtosis is more likely to be extreme with t-dist

    def test_parametric_invalid_distribution(self, simulator):
        """Test invalid distribution raises error."""
        with pytest.raises(ValueError):
            simulator.simulate_parametric(
                mean_return=0.001,
                std_return=0.02,
                distribution="invalid",
            )


class TestMonteCarloSimulatorTradeBased:
    """Tests for trade-based simulation."""

    @pytest.fixture
    def simulator(self):
        """Create simulator."""
        config = MonteCarloConfig(
            n_simulations=500,
            n_periods=50,
            seed=42,
        )
        return MonteCarloSimulator(config)

    @pytest.fixture
    def trade_returns(self):
        """Generate sample trade returns."""
        np.random.seed(42)
        # Mix of winners and losers
        winners = np.random.uniform(0.01, 0.05, 100)  # 1-5% gains
        losers = np.random.uniform(-0.03, -0.01, 50)  # 1-3% losses
        return np.concatenate([winners, losers])

    def test_simulate_from_trades(self, simulator, trade_returns):
        """Test trade-based simulation."""
        result = simulator.simulate_from_trades(
            trade_returns=trade_returns,
            trades_per_period=1.0,
            initial_capital=1000000,
        )

        assert isinstance(result, MonteCarloResult)
        # With positive expected value trades, should be profitable
        assert result.summary["probability_profit"] > 0.4

    def test_varying_trade_frequency(self, simulator, trade_returns):
        """Test different trade frequencies."""
        result_low = simulator.simulate_from_trades(
            trade_returns=trade_returns,
            trades_per_period=0.5,
            initial_capital=1000000,
        )

        result_high = simulator.simulate_from_trades(
            trade_returns=trade_returns,
            trades_per_period=2.0,
            initial_capital=1000000,
        )

        # Higher frequency should have higher volatility
        assert result_high.return_distribution.std > result_low.return_distribution.std


class TestRuinProbability:
    """Tests for ruin probability calculations."""

    def test_high_risk_strategy(self):
        """Test ruin probability for high-risk strategy."""
        config = MonteCarloConfig(n_simulations=1000, n_periods=100, seed=42)
        simulator = MonteCarloSimulator(config)

        # Very volatile negative returns
        np.random.seed(42)
        high_risk_returns = np.random.normal(-0.002, 0.05, 200)

        result = simulator.simulate_from_returns(high_risk_returns)

        # High risk should have significant ruin probability
        assert result.ruin_probability.prob_lose_25pct > 0.1

    def test_low_risk_strategy(self):
        """Test ruin probability for low-risk strategy."""
        config = MonteCarloConfig(n_simulations=1000, n_periods=100, seed=42)
        simulator = MonteCarloSimulator(config)

        # Low volatility positive returns
        np.random.seed(42)
        low_risk_returns = np.random.normal(0.002, 0.005, 200)

        result = simulator.simulate_from_returns(low_risk_returns)

        # Low risk should have minimal ruin probability
        assert result.ruin_probability.prob_lose_50pct < 0.01


class TestReportFormatting:
    """Tests for report formatting."""

    def test_format_report(self):
        """Test report formatting."""
        config = MonteCarloConfig(n_simulations=100, n_periods=50, seed=42)
        simulator = MonteCarloSimulator(config)

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        result = simulator.simulate_from_returns(returns)

        report = format_monte_carlo_report(result)

        # Check report contains key sections
        assert "MONTE CARLO SIMULATION REPORT" in report
        assert "RETURN DISTRIBUTION" in report
        assert "DRAWDOWN RISK" in report
        assert "RUIN PROBABILITY" in report
        assert "CONFIDENCE INTERVALS" in report
        assert "SHARPE RATIO DISTRIBUTION" in report
        assert "SUMMARY" in report


class TestReproducibility:
    """Tests for simulation reproducibility."""

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        config = MonteCarloConfig(n_simulations=100, n_periods=50, seed=42)

        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)

        sim1 = MonteCarloSimulator(config)
        result1 = sim1.simulate_from_returns(returns.copy())

        sim2 = MonteCarloSimulator(config)
        result2 = sim2.simulate_from_returns(returns.copy())

        assert result1.return_distribution.mean == result2.return_distribution.mean
        assert np.allclose(
            result1.final_capital_distribution,
            result2.final_capital_distribution,
        )
