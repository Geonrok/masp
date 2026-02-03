"""
Tests for Integrated Strategy

Tests the combination of:
- Bias-Free Backtester
- Validation System
- Veto Manager
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from libs.strategy.integrated_strategy import (
    IntegratedStrategy,
    IntegratedConfig,
    IntegratedResult,
)
from libs.risk.veto_manager import VetoResult


def create_mock_data(n_days: int = 300, n_symbols: int = 5, seed: int = 42) -> dict:
    """Create mock OHLCV data for testing."""
    np.random.seed(seed)
    data = {}

    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    for i in range(n_symbols):
        symbol = f"SYM{i}" if i > 0 else "BTC"
        base_price = 100 * (i + 1)

        # Random walk with slight positive drift
        returns = np.random.randn(n_days) * 0.02 + 0.0002
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


class TestIntegratedConfig:
    """Test IntegratedConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntegratedConfig()

        assert config.initial_capital == 10000.0
        assert config.slippage_pct == 0.005
        assert config.commission_pct == 0.001
        assert config.kama_period == 5
        assert config.tsmom_period == 90
        assert config.enable_veto is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = IntegratedConfig(
            initial_capital=50000.0,
            slippage_pct=0.01,
            kama_period=10,
        )

        assert config.initial_capital == 50000.0
        assert config.slippage_pct == 0.01
        assert config.kama_period == 10

    def test_to_dict(self):
        """Test configuration to_dict."""
        config = IntegratedConfig()
        d = config.to_dict()

        assert "initial_capital" in d
        assert "slippage_pct" in d
        assert "kama_period" in d


class TestIntegratedResult:
    """Test IntegratedResult dataclass."""

    def test_empty_result(self):
        """Test empty result."""
        result = IntegratedResult()

        assert result.metrics is None
        assert result.validation is None

    def test_to_dict(self):
        """Test result to_dict."""
        result = IntegratedResult()
        d = result.to_dict()

        assert "metrics" in d
        assert "validation" in d
        assert "timestamp" in d

    def test_summary(self):
        """Test result summary generation."""
        result = IntegratedResult()
        summary = result.summary()

        assert "INTEGRATED STRATEGY RESULTS" in summary


class TestIntegratedStrategy:
    """Test IntegratedStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = IntegratedConfig()
        strategy = IntegratedStrategy(config)

        assert strategy.config == config
        assert strategy.backtester is not None
        assert strategy.veto_manager is not None

    def test_run_backtest(self):
        """Test backtest execution."""
        config = IntegratedConfig(
            run_validation=False,  # Skip for speed
            enable_veto=False,
        )
        strategy = IntegratedStrategy(config)
        data = create_mock_data(n_days=200, n_symbols=3)

        result = strategy.run_backtest(data, validate=False)

        assert isinstance(result, IntegratedResult)
        assert result.metrics is not None
        assert result.run_time > 0

    def test_run_backtest_with_validation(self):
        """Test backtest with validation."""
        config = IntegratedConfig(
            run_validation=True,
            n_trials=1,
        )
        strategy = IntegratedStrategy(config)
        data = create_mock_data(n_days=300, n_symbols=3)

        result = strategy.run_backtest(data, validate=True)

        assert result.validation is not None

    def test_paper_trade_check(self):
        """Test paper trade veto check."""
        config = IntegratedConfig(enable_veto=True)
        strategy = IntegratedStrategy(config)

        ohlcv = create_mock_data(n_days=100, n_symbols=1)["BTC"]

        result = strategy.paper_trade_check(
            symbol="BTC",
            side="long",
            ohlcv=ohlcv,
        )

        assert isinstance(result, VetoResult)

    def test_kill_switch(self):
        """Test kill switch functionality."""
        strategy = IntegratedStrategy()

        # Initially disabled
        assert strategy.veto_manager.is_kill_switch_active() is False

        # Enable
        strategy.enable_kill_switch()
        assert strategy.veto_manager.is_kill_switch_active() is True

        # Disable
        strategy.disable_kill_switch()
        assert strategy.veto_manager.is_kill_switch_active() is False


class TestIntegratedResultSummary:
    """Test IntegratedResult summary generation."""

    def test_summary_with_metrics(self):
        """Test summary with metrics."""
        from libs.backtest.bias_free_backtester import BacktestMetrics

        metrics = BacktestMetrics(
            total_return=0.15,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.55,
            trading_days=252,
            invested_days=200,
        )

        result = IntegratedResult(metrics=metrics)
        summary = result.summary()

        assert "15.0%" in summary or "+15.0%" in summary
        assert "1.50" in summary or "1.5" in summary

    def test_summary_with_validation(self):
        """Test summary with validation results."""
        validation = {
            "verdict": {
                "is_robust": True,
                "confidence": "high",
                "warnings": [],
            },
            "wfo": {
                "oos_sharpe": 1.2,
                "efficiency_ratio": 0.8,
                "robustness_score": 0.7,
            },
        }

        result = IntegratedResult(validation=validation)
        summary = result.summary()

        assert "VALIDATION" in summary
        assert "high" in summary

    def test_summary_with_veto_stats(self):
        """Test summary with veto statistics."""
        veto_stats = {
            "total_checks": 100,
            "vetoed_count": 25,
            "veto_rate": 0.25,
        }

        result = IntegratedResult(veto_stats=veto_stats)
        summary = result.summary()

        assert "VETO" in summary
        assert "100" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test with empty data."""
        strategy = IntegratedStrategy()
        result = strategy.run_backtest({}, validate=False)

        # Should handle gracefully
        assert isinstance(result, IntegratedResult)

    def test_single_symbol(self):
        """Test with single symbol."""
        strategy = IntegratedStrategy()
        data = create_mock_data(n_days=200, n_symbols=1)

        result = strategy.run_backtest(data, validate=False)

        assert isinstance(result, IntegratedResult)

    def test_veto_with_funding_rate(self):
        """Test veto check with funding rate."""
        config = IntegratedConfig(
            enable_veto=True,
            funding_rate_threshold=0.001,
        )
        strategy = IntegratedStrategy(config)

        ohlcv = create_mock_data(n_days=100, n_symbols=1)["BTC"]

        # High funding rate should trigger veto for longs
        result = strategy.paper_trade_check(
            symbol="BTC",
            side="long",
            ohlcv=ohlcv,
            funding_rate=0.005,  # 0.5%
        )

        assert result.can_trade is False


class TestVetoStats:
    """Test veto statistics calculation."""

    def test_veto_stats_enabled(self):
        """Test veto stats when enabled."""
        config = IntegratedConfig(enable_veto=True)
        strategy = IntegratedStrategy(config)
        data = create_mock_data(n_days=200, n_symbols=3)

        result = strategy.run_backtest(data, validate=False)

        assert result.veto_stats["enabled"] is True

    def test_veto_stats_disabled(self):
        """Test veto stats when disabled."""
        config = IntegratedConfig(enable_veto=False)
        strategy = IntegratedStrategy(config)
        data = create_mock_data(n_days=200, n_symbols=3)

        result = strategy.run_backtest(data, validate=False)

        assert result.veto_stats["enabled"] is False
