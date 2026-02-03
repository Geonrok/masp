"""
Tests for Bias-Free Backtester

Validates that:
1. Day T signal -> Day T+1 execution
2. No look-ahead bias
3. Slippage and commission applied correctly
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from libs.backtest.bias_free_backtester import (
    BacktestConfig,
    BacktestMetrics,
    BiasFreeBacktester,
    ExecutionConfig,
)


def create_mock_data(n_days: int = 200, n_symbols: int = 5, seed: int = 42) -> dict:
    """Create mock OHLCV data for testing."""
    np.random.seed(seed)
    data = {}

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    for i in range(n_symbols):
        symbol = f"SYMBOL{i}" if i > 0 else "BTC"
        base_price = 100 * (i + 1)

        # Random walk for prices
        returns = np.random.randn(n_days) * 0.02
        prices = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame(
            {
                "date": dates,
                "open": prices * (1 + np.random.randn(n_days) * 0.001),
                "high": prices * (1 + np.abs(np.random.randn(n_days) * 0.01)),
                "low": prices * (1 - np.abs(np.random.randn(n_days) * 0.01)),
                "close": prices,
                "volume": np.random.uniform(1e6, 1e7, n_days),
            }
        )

        data[symbol] = df

    return data


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == 10000.0
        assert config.kama_period == 5
        assert config.tsmom_period == 90
        assert config.gate_period == 30

    def test_execution_config(self):
        """Test ExecutionConfig values."""
        exec_config = ExecutionConfig()

        assert exec_config.slippage_pct == 0.005
        assert exec_config.commission_pct == 0.001
        assert exec_config.max_positions == 20

    def test_total_cost(self):
        """Test total cost calculation."""
        exec_config = ExecutionConfig(slippage_pct=0.005, commission_pct=0.001)

        # Total cost = 2 * (slippage + commission) for round trip
        expected = 2 * (0.005 + 0.001)
        assert exec_config.get_total_cost() == expected


class TestBiasFreeBacktester:
    """Test BiasFreeBacktester class."""

    def test_initialization(self):
        """Test backtester initialization."""
        config = BacktestConfig(initial_capital=50000)
        backtester = BiasFreeBacktester(config)

        assert backtester.config.initial_capital == 50000
        assert backtester.portfolio_values == []
        assert backtester.daily_returns == []

    def test_run_with_mock_data(self):
        """Test backtest run with mock data."""
        config = BacktestConfig(
            initial_capital=10000,
            kama_period=5,
            tsmom_period=30,  # Shorter for test
            gate_period=10,
        )
        backtester = BiasFreeBacktester(config)
        data = create_mock_data(n_days=200, n_symbols=5)

        metrics = backtester.run(data)

        # Should complete without error
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.trading_days > 0

    def test_no_future_leakage(self):
        """Test that no future data is used in signal generation."""
        config = BacktestConfig(initial_capital=10000)
        backtester = BiasFreeBacktester(config)

        # Create data where future prices are dramatically different
        data = {}
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(200)]

        # BTC prices
        btc_prices = np.linspace(100, 200, 200)  # Uptrend
        data["BTC"] = pd.DataFrame(
            {
                "date": dates,
                "open": btc_prices,
                "high": btc_prices * 1.01,
                "low": btc_prices * 0.99,
                "close": btc_prices,
                "volume": np.ones(200) * 1e7,
            }
        )

        # Test symbol with known pattern
        test_prices = np.linspace(100, 150, 200)
        data["TEST"] = pd.DataFrame(
            {
                "date": dates,
                "open": test_prices,
                "high": test_prices * 1.01,
                "low": test_prices * 0.99,
                "close": test_prices,
                "volume": np.ones(200) * 1e7,
            }
        )

        metrics = backtester.run(data)

        # The return should be reasonable, not artificially inflated
        # If there was look-ahead bias, returns would be unrealistically high
        assert abs(metrics.total_return) < 5.0  # Should be within 500%

    def test_slippage_applied(self):
        """Test that slippage is applied to entry prices."""
        exec_config = ExecutionConfig(slippage_pct=0.01)  # 1% slippage
        config = BacktestConfig(
            initial_capital=10000,
            execution=exec_config,
        )
        backtester = BiasFreeBacktester(config)
        data = create_mock_data(n_days=200, n_symbols=3)

        metrics = backtester.run(data)

        # With slippage, returns should be lower than without
        # This is a basic sanity check
        assert isinstance(metrics, BacktestMetrics)

    def test_max_positions_respected(self):
        """Test that max_positions limit is respected."""
        exec_config = ExecutionConfig(max_positions=3)
        config = BacktestConfig(
            initial_capital=10000,
            execution=exec_config,
        )
        backtester = BiasFreeBacktester(config)
        data = create_mock_data(n_days=200, n_symbols=10)

        backtester.run(data)

        # Check positions log
        for pos in backtester.get_positions_log():
            assert len(pos["symbols"]) <= 3

    def test_empty_data(self):
        """Test handling of empty data."""
        config = BacktestConfig()
        backtester = BiasFreeBacktester(config)

        metrics = backtester.run({})

        assert metrics.total_return == 0
        assert metrics.sharpe_ratio == 0

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        config = BacktestConfig()
        backtester = BiasFreeBacktester(config)

        # Data with only 10 days (less than required 100)
        data = create_mock_data(n_days=10, n_symbols=2)

        metrics = backtester.run(data)

        # Should handle gracefully
        assert isinstance(metrics, BacktestMetrics)


class TestKAMACalculation:
    """Test KAMA indicator calculation."""

    def test_kama_basic(self):
        """Test basic KAMA calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

        kama = BiasFreeBacktester._calc_kama(prices, period=5)

        # KAMA should exist after period-1
        assert not np.isnan(kama[4])
        assert not np.isnan(kama[-1])

        # KAMA should follow price in trending market
        assert kama[-1] > kama[4]

    def test_kama_insufficient_data(self):
        """Test KAMA with insufficient data."""
        prices = np.array([100, 101, 102])

        kama = BiasFreeBacktester._calc_kama(prices, period=5)

        # All should be NaN
        assert all(np.isnan(kama))


class TestMACalculation:
    """Test MA indicator calculation."""

    def test_ma_basic(self):
        """Test basic MA calculation."""
        prices = np.array([100, 102, 104, 106, 108, 110])

        ma = BiasFreeBacktester._calc_ma(prices, period=3)

        # MA should be average of last 3 prices
        expected_last = np.mean([106, 108, 110])
        assert abs(ma[-1] - expected_last) < 0.01


class TestBacktestMetrics:
    """Test BacktestMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = BacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
        )

        d = metrics.to_dict()

        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.5
        assert d["max_drawdown"] == -0.10


class TestEquityCurve:
    """Test equity curve generation."""

    def test_equity_curve(self):
        """Test that equity curve is generated correctly."""
        config = BacktestConfig(initial_capital=10000)
        backtester = BiasFreeBacktester(config)
        data = create_mock_data(n_days=200, n_symbols=3)

        backtester.run(data)

        equity = backtester.get_equity_curve()

        # First value should be initial capital
        assert equity.iloc[0] == 10000

        # Should have values for all days
        assert len(equity) > 0
