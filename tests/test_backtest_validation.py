"""
Tests for Strategy Validation Module (Backtest)

Tests WFO, CPCV, and Deflated Sharpe Ratio implementations.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from libs.backtest.validation import (
    CPCVConfig,
    CPCVResult,
    CPCVValidator,
    DeflatedSharpeRatio,
    DSRResult,
    WalkForwardOptimizer,
    WFOConfig,
    WFOResult,
    validate_strategy,
)


def create_mock_data(n_days: int = 500, n_symbols: int = 3, seed: int = 42) -> dict:
    """Create mock OHLCV data for testing."""
    np.random.seed(seed)
    data = {}

    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    for i in range(n_symbols):
        symbol = f"SYM{i}" if i > 0 else "BTC"
        base_price = 100 * (i + 1)

        # Random walk with slight positive drift
        returns = np.random.randn(n_days) * 0.02 + 0.0001
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n_days) * 0.001),
                "high": prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "low": prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "close": prices,
                "volume": np.random.uniform(1e6, 1e7, n_days),
            },
            index=dates,
        )

        data[symbol] = df

    return data


def mock_strategy(data: dict) -> np.ndarray:
    """Mock strategy that returns random daily returns."""
    # Get number of days from first symbol
    if not data:
        return np.array([])

    first_df = list(data.values())[0]
    n = len(first_df)

    # Return small random returns
    np.random.seed(123)
    return np.random.randn(n) * 0.01


class TestWFOConfig:
    """Test WFOConfig dataclass."""

    def test_default_config(self):
        """Test default values."""
        config = WFOConfig()

        assert config.train_months == 12
        assert config.test_months == 3
        assert config.step_months == 3


class TestWalkForwardOptimizer:
    """Test Walk-Forward Optimization."""

    def test_initialization(self):
        """Test WFO initialization."""
        config = WFOConfig(train_months=6, test_months=2)
        wfo = WalkForwardOptimizer(config)

        assert wfo.config.train_months == 6
        assert wfo.config.test_months == 2

    def test_run_with_mock_data(self):
        """Test WFO run with mock data."""
        config = WFOConfig(train_months=3, test_months=1, step_months=1)
        wfo = WalkForwardOptimizer(config)
        data = create_mock_data(n_days=500)

        result = wfo.run(data, lambda d, p: mock_strategy(d))

        assert isinstance(result, WFOResult)
        assert len(result.periods) > 0
        assert len(result.oos_returns) == len(result.periods)

    def test_efficiency_ratio_calculation(self):
        """Test that efficiency ratio is calculated."""
        config = WFOConfig(train_months=3, test_months=1, step_months=1)
        wfo = WalkForwardOptimizer(config)
        data = create_mock_data(n_days=500)

        result = wfo.run(data, lambda d, p: mock_strategy(d))

        # Efficiency ratio should be defined
        assert isinstance(result.efficiency_ratio, float)

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        config = WFOConfig()
        wfo = WalkForwardOptimizer(config)
        data = create_mock_data(n_days=50)  # Too short

        result = wfo.run(data, lambda d, p: mock_strategy(d))

        assert len(result.periods) == 0

    def test_result_to_dict(self):
        """Test WFOResult.to_dict()."""
        result = WFOResult(
            periods=[{"period": 0}],
            oos_returns=[0.05],
            is_returns=[0.10],
            oos_sharpe=1.5,
            is_sharpe=2.0,
            efficiency_ratio=0.75,
        )

        d = result.to_dict()

        assert d["n_periods"] == 1
        assert d["oos_sharpe"] == 1.5
        assert d["efficiency_ratio"] == 0.75


class TestCPCVConfig:
    """Test CPCVConfig dataclass."""

    def test_default_config(self):
        """Test default values."""
        config = CPCVConfig()

        assert config.n_splits == 5
        assert config.embargo_pct == 0.05
        assert config.purge_pct == 0.01


class TestCPCVValidator:
    """Test Combinatorial Purged Cross-Validation."""

    def test_initialization(self):
        """Test CPCV initialization."""
        config = CPCVConfig(n_splits=10)
        cpcv = CPCVValidator(config)

        assert cpcv.config.n_splits == 10

    def test_run_with_mock_data(self):
        """Test CPCV run with mock data."""
        config = CPCVConfig(n_splits=5)
        cpcv = CPCVValidator(config)
        data = create_mock_data(n_days=500)

        result = cpcv.run(data, mock_strategy)

        assert isinstance(result, CPCVResult)
        assert len(result.fold_results) > 0

    def test_pbo_calculation(self):
        """Test PBO (Probability of Backtest Overfitting) calculation."""
        config = CPCVConfig(n_splits=5)
        cpcv = CPCVValidator(config)
        data = create_mock_data(n_days=500)

        result = cpcv.run(data, mock_strategy)

        # PBO should be between 0 and 1
        assert 0 <= result.pbo <= 1

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        config = CPCVConfig()
        cpcv = CPCVValidator(config)
        data = create_mock_data(n_days=50)  # Too short

        result = cpcv.run(data, mock_strategy)

        # Should handle gracefully
        assert isinstance(result, CPCVResult)

    def test_result_to_dict(self):
        """Test CPCVResult.to_dict()."""
        result = CPCVResult(
            fold_results=[{"fold": 0}],
            mean_return=0.05,
            std_return=0.02,
            mean_sharpe=1.5,
            pbo=0.2,
        )

        d = result.to_dict()

        assert d["n_folds"] == 1
        assert d["mean_sharpe"] == 1.5
        assert d["pbo"] == 0.2


class TestDeflatedSharpeRatio:
    """Test Deflated Sharpe Ratio calculation."""

    def test_initialization(self):
        """Test DSR initialization."""
        dsr = DeflatedSharpeRatio(significance_level=0.01)

        assert dsr.significance_level == 0.01

    def test_basic_calculation(self):
        """Test basic DSR calculation."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005  # Slight positive drift

        dsr = DeflatedSharpeRatio()
        result = dsr.calculate(returns, n_trials=1)

        assert isinstance(result, DSRResult)
        assert result.raw_sharpe != 0
        assert 0 <= result.p_value <= 1

    def test_multiple_trials_haircut(self):
        """Test that multiple trials increase haircut."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        dsr = DeflatedSharpeRatio()

        result_1 = dsr.calculate(returns, n_trials=1)
        result_10 = dsr.calculate(returns, n_trials=10)
        result_100 = dsr.calculate(returns, n_trials=100)

        # More trials should result in lower deflated Sharpe
        assert result_1.deflated_sharpe >= result_10.deflated_sharpe
        assert result_10.deflated_sharpe >= result_100.deflated_sharpe

    def test_haircut_increases_with_trials(self):
        """Test that haircut % increases with more trials."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001

        dsr = DeflatedSharpeRatio()

        result_1 = dsr.calculate(returns, n_trials=1)
        result_50 = dsr.calculate(returns, n_trials=50)

        assert result_50.haircut_pct >= result_1.haircut_pct

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        returns = np.random.randn(10) * 0.01  # Too short

        dsr = DeflatedSharpeRatio()
        result = dsr.calculate(returns)

        # Should return empty result
        assert result.raw_sharpe == 0

    def test_result_to_dict(self):
        """Test DSRResult.to_dict()."""
        result = DSRResult(
            raw_sharpe=2.0,
            deflated_sharpe=1.5,
            p_value=0.03,
            n_trials=10,
            is_significant=True,
            haircut_pct=25.0,
        )

        d = result.to_dict()

        assert d["raw_sharpe"] == 2.0
        assert d["deflated_sharpe"] == 1.5
        assert d["is_significant"] is True


class TestValidateStrategy:
    """Test comprehensive strategy validation."""

    def test_validate_strategy(self):
        """Test full validation pipeline."""
        data = create_mock_data(n_days=500)

        results = validate_strategy(
            data=data,
            strategy_func=mock_strategy,
            n_trials=5,
        )

        assert "wfo" in results
        assert "cpcv" in results
        assert "dsr" in results or "error" in results.get("dsr", {})
        assert "verdict" in results

    def test_verdict_structure(self):
        """Test that verdict has correct structure."""
        data = create_mock_data(n_days=500)

        results = validate_strategy(data, mock_strategy)
        verdict = results["verdict"]

        assert "is_robust" in verdict
        assert "confidence" in verdict
        assert "warnings" in verdict
        assert verdict["confidence"] in ["low", "medium", "high"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        config = WFOConfig()
        wfo = WalkForwardOptimizer(config)

        result = wfo.run({}, lambda d, p: np.array([]))

        assert len(result.periods) == 0

    def test_strategy_returning_zeros(self):
        """Test strategy that returns all zeros."""
        config = CPCVConfig()
        cpcv = CPCVValidator(config)
        data = create_mock_data(n_days=500)

        def zero_strategy(data):
            return np.zeros(100)

        result = cpcv.run(data, zero_strategy)

        # Should handle gracefully
        assert isinstance(result, CPCVResult)

    def test_negative_sharpe(self):
        """Test strategy with negative returns."""
        np.random.seed(42)
        # Negative drift
        returns = np.random.randn(252) * 0.01 - 0.002

        dsr = DeflatedSharpeRatio()
        result = dsr.calculate(returns)

        assert result.raw_sharpe < 0
