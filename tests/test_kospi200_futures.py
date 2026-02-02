"""
Tests for KOSPI200 Futures Strategy.

Tests cover:
- Strategy initialization
- Signal generation
- Composite calculation
- Individual sub-strategies
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from libs.strategies.kospi200_futures import (
    KOSPI200FuturesStrategy,
    KOSPI200FuturesConfig,
    VIXBelowSMA20Strategy,
    VIXDecliningStrategy,
    SemiconForeignStrategy,
    KOSPI200HourlyStrategy,
    KOSPI200StablePortfolioStrategy,
    KOSPI200AggressivePortfolioStrategy,
    KOSPI200SubStrategy,
)
from libs.strategies.base import Signal


class TestKOSPI200FuturesConfig:
    """Tests for strategy configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KOSPI200FuturesConfig()

        assert "vix_below_sma20" in config.enabled_strategies
        assert "vix_declining" in config.enabled_strategies
        assert "semicon_foreign" in config.enabled_strategies
        assert config.vix_sma_period == 20
        assert config.round_trip_cost == 0.0009

    def test_custom_config(self):
        """Test custom configuration."""
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20"],
            vix_sma_period=30,
        )

        assert config.enabled_strategies == ["vix_below_sma20"]
        assert config.vix_sma_period == 30

    def test_strategy_weights_sum(self):
        """Test that default weights sum to 1.0."""
        config = KOSPI200FuturesConfig()
        total = sum(config.strategy_weights.values())
        assert abs(total - 1.0) < 0.01


class TestKOSPI200FuturesStrategy:
    """Tests for main strategy class."""

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = KOSPI200FuturesStrategy()

        assert strategy.strategy_id == "kospi200_futures_v1"
        assert strategy.name == "KOSPI200 Futures Strategy"
        assert strategy.version == "1.0.0"

    def test_strategy_with_custom_config(self):
        """Test strategy with custom configuration."""
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20"],
            strategy_weights={"vix_below_sma20": 1.0},
        )
        strategy = KOSPI200FuturesStrategy(config=config)

        assert strategy.config.enabled_strategies == ["vix_below_sma20"]

    def test_vix_below_sma20_signal_long(self):
        """Test VIX Below SMA20 returns LONG when VIX(T-1) < SMA."""
        strategy = KOSPI200FuturesStrategy()

        # Mock VIX data where T-1 VIX is below SMA
        # Uses T-1 shift: vix_values[-2] is T-1 VIX, vix_values[-1] is "today" (ignored)
        vix_values = [20.0] * 19 + [15.0] + [99.0]  # T-1=15, today=99 (ignored)
        strategy._vix_data = pd.Series(vix_values)

        signal = strategy._calculate_vix_below_sma20_signal()
        assert signal == 1  # LONG (T-1 VIX 15 < SMA ~19.75)

    def test_vix_below_sma20_signal_cash(self):
        """Test VIX Below SMA20 returns CASH when VIX(T-1) > SMA."""
        strategy = KOSPI200FuturesStrategy()

        # Mock VIX data where T-1 VIX is above SMA
        vix_values = [15.0] * 19 + [25.0] + [1.0]  # T-1=25, today=1 (ignored)
        strategy._vix_data = pd.Series(vix_values)

        signal = strategy._calculate_vix_below_sma20_signal()
        assert signal == 0  # CASH (T-1 VIX 25 > SMA ~15.5)

    def test_vix_declining_signal_long(self):
        """Test VIX Declining returns LONG when VIX(T-1) < VIX(T-2)."""
        strategy = KOSPI200FuturesStrategy()

        # Mock VIX data where VIX is declining (T-2=20, T-1=18)
        # Uses T-1 shift: vix_values[-3]=T-2, vix_values[-2]=T-1, vix_values[-1]=today
        strategy._vix_data = pd.Series([20.0, 18.0, 99.0])  # T-2=20, T-1=18, today=99

        signal = strategy._calculate_vix_declining_signal()
        assert signal == 1  # LONG (T-1=18 < T-2=20)

    def test_vix_declining_signal_cash(self):
        """Test VIX Declining returns CASH when VIX(T-1) > VIX(T-2)."""
        strategy = KOSPI200FuturesStrategy()

        # Mock VIX data where VIX is rising (T-2=18, T-1=20)
        strategy._vix_data = pd.Series([18.0, 20.0, 1.0])  # T-2=18, T-1=20, today=1

        signal = strategy._calculate_vix_declining_signal()
        assert signal == 0  # CASH (T-1=20 > T-2=18)

    def test_semicon_foreign_signal_long(self):
        """Test Semicon+Foreign returns LONG when both conditions met."""
        strategy = KOSPI200FuturesStrategy()

        # Mock semicon data above SMA
        semicon_values = [100.0] * 19 + [120.0]  # SMA = 101, current = 120
        strategy._semicon_data = pd.Series(semicon_values)

        # Mock foreign data with positive flow
        foreign_values = [1e10] * 20  # Positive 20-day sum
        strategy._foreign_data = pd.Series(foreign_values)

        signal = strategy._calculate_semicon_foreign_signal()
        assert signal == 1  # LONG

    def test_semicon_foreign_signal_cash_semicon_weak(self):
        """Test Semicon+Foreign returns CASH when semicon below SMA."""
        strategy = KOSPI200FuturesStrategy()

        # Mock semicon data below SMA
        semicon_values = [100.0] * 19 + [80.0]  # SMA = 99, current = 80
        strategy._semicon_data = pd.Series(semicon_values)

        # Mock foreign data with positive flow
        foreign_values = [1e10] * 20
        strategy._foreign_data = pd.Series(foreign_values)

        signal = strategy._calculate_semicon_foreign_signal()
        assert signal == 0  # CASH

    def test_semicon_foreign_signal_cash_foreign_negative(self):
        """Test Semicon+Foreign returns CASH when foreign flow negative."""
        strategy = KOSPI200FuturesStrategy()

        # Mock semicon data above SMA
        semicon_values = [100.0] * 19 + [120.0]
        strategy._semicon_data = pd.Series(semicon_values)

        # Mock foreign data with negative flow
        foreign_values = [-1e10] * 20  # Negative 20-day sum
        strategy._foreign_data = pd.Series(foreign_values)

        signal = strategy._calculate_semicon_foreign_signal()
        assert signal == 0  # CASH

    def test_hourly_sma_15_30_signal_long(self):
        """Test SMA 15/30 returns LONG in uptrend."""
        strategy = KOSPI200FuturesStrategy()

        # Create uptrending price data
        prices = list(range(100, 150))  # Strong uptrend
        hourly_df = pd.DataFrame({'close': prices})
        strategy._hourly_data = hourly_df

        signal = strategy._calculate_sma_15_30_hourly_signal()
        assert signal == 1  # LONG (SMA15 > SMA30 in uptrend)

    def test_hourly_sma_15_30_signal_cash(self):
        """Test SMA 15/30 returns CASH in downtrend."""
        strategy = KOSPI200FuturesStrategy()

        # Create downtrending price data
        prices = list(range(150, 100, -1))  # Strong downtrend
        hourly_df = pd.DataFrame({'close': prices})
        strategy._hourly_data = hourly_df

        signal = strategy._calculate_sma_15_30_hourly_signal()
        assert signal == 0  # CASH (SMA15 < SMA30 in downtrend)

    def test_composite_signal_calculation(self):
        """Test composite signal calculation with weights."""
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20", "vix_declining"],
            strategy_weights={
                "vix_below_sma20": 0.6,
                "vix_declining": 0.4,
            },
        )
        strategy = KOSPI200FuturesStrategy(config=config)

        # Set up for both LONG signals (T-1 shift applied)
        # T-2=20, T-1=15 (declining), today=99 (ignored)
        # SMA uses T-1 basis, T-1 VIX=15 < SMA(20*19+15)/20=19.75
        vix_values = [20.0] * 19 + [15.0] + [99.0]  # T-1=15 below SMA, declining
        strategy._vix_data = pd.Series(vix_values)

        composite, signals = strategy.calculate_composite_signal()

        # Both signals should be 1
        assert signals.get("vix_below_sma20") == 1
        assert signals.get("vix_declining") == 1
        assert composite == 1.0  # 0.6 * 1 + 0.4 * 1 = 1.0

    def test_composite_signal_partial(self):
        """Test composite signal with partial LONG."""
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20", "vix_declining"],
            strategy_weights={
                "vix_below_sma20": 0.6,
                "vix_declining": 0.4,
            },
        )
        strategy = KOSPI200FuturesStrategy(config=config)

        # T-2=15, T-1=16 (rising), today=99 (ignored)
        # vix_below_sma20: 1 (T-1=16 < SMA~19.8)
        # vix_declining: 0 (T-1=16 > T-2=15)
        strategy._vix_data = pd.Series([20.0] * 18 + [15.0, 16.0, 99.0])

        composite, signals = strategy.calculate_composite_signal()

        assert signals.get("vix_below_sma20") == 1
        assert signals.get("vix_declining") == 0
        assert composite == 0.6  # 0.6 * 1 + 0.4 * 0 = 0.6

    def test_get_current_status(self):
        """Test status retrieval."""
        strategy = KOSPI200FuturesStrategy()

        # Set up minimal data
        strategy._vix_data = pd.Series([20.0] * 20)
        strategy._last_signals = {"vix_below_sma20": 0}
        strategy._last_update = datetime.now()

        status = strategy.get_current_status()

        assert status["strategy_id"] == "kospi200_futures_v1"
        assert "indicators" in status
        assert "vix" in status["indicators"]


class TestIndividualStrategies:
    """Tests for individual strategy convenience classes."""

    def test_vix_below_sma20_strategy(self):
        """Test VIXBelowSMA20Strategy initialization."""
        strategy = VIXBelowSMA20Strategy()

        assert strategy.strategy_id == "kospi200_vix_below_sma20"
        assert "vix_below_sma20" in strategy.config.enabled_strategies
        assert strategy.config.strategy_weights["vix_below_sma20"] == 1.0

    def test_vix_declining_strategy(self):
        """Test VIXDecliningStrategy initialization."""
        strategy = VIXDecliningStrategy()

        assert strategy.strategy_id == "kospi200_vix_declining"
        assert "vix_declining" in strategy.config.enabled_strategies

    def test_semicon_foreign_strategy(self):
        """Test SemiconForeignStrategy initialization."""
        strategy = SemiconForeignStrategy()

        assert strategy.strategy_id == "kospi200_semicon_foreign"
        assert "semicon_foreign" in strategy.config.enabled_strategies

    def test_hourly_strategy(self):
        """Test KOSPI200HourlyStrategy initialization."""
        strategy = KOSPI200HourlyStrategy()

        assert strategy.strategy_id == "kospi200_hourly_ma"
        assert "sma_15_30_hourly" in strategy.config.enabled_strategies
        assert "ema_15_20_hourly" in strategy.config.enabled_strategies
        assert "sma_20_30_hourly" in strategy.config.enabled_strategies

    def test_stable_portfolio_strategy(self):
        """Test KOSPI200StablePortfolioStrategy initialization."""
        strategy = KOSPI200StablePortfolioStrategy()

        assert strategy.strategy_id == "kospi200_stable_portfolio"
        assert strategy.config.strategy_weights["vix_below_sma20"] == 0.5
        assert strategy.config.strategy_weights["vix_declining"] == 0.3
        assert strategy.config.strategy_weights["semicon_foreign"] == 0.2

    def test_aggressive_portfolio_strategy(self):
        """Test KOSPI200AggressivePortfolioStrategy initialization."""
        strategy = KOSPI200AggressivePortfolioStrategy()

        assert strategy.strategy_id == "kospi200_aggressive_portfolio"
        assert "ema_15_20_hourly" in strategy.config.enabled_strategies
        assert "sma_20_30_hourly" in strategy.config.enabled_strategies
        assert "vix_below_sma20" in strategy.config.enabled_strategies


class TestStrategyLoader:
    """Tests for strategy loader integration."""

    def test_strategy_in_registry(self):
        """Test that KOSPI200 strategies are in registry."""
        from libs.strategies.loader import STRATEGY_REGISTRY

        assert "kospi200_futures_v1" in STRATEGY_REGISTRY
        assert "kospi200_vix_below_sma20" in STRATEGY_REGISTRY
        assert "kospi200_stable_portfolio" in STRATEGY_REGISTRY

    def test_load_strategy(self):
        """Test loading KOSPI200 strategy via loader."""
        from libs.strategies.loader import get_strategy

        strategy = get_strategy("kospi200_stable_portfolio")

        assert strategy is not None
        assert strategy.strategy_id == "kospi200_stable_portfolio"

    def test_list_available_strategies(self):
        """Test KOSPI200 strategies in available list."""
        from libs.strategies.loader import list_available_strategies

        strategies = list_available_strategies()
        strategy_ids = [s["strategy_id"] for s in strategies]

        assert "kospi200_stable_portfolio" in strategy_ids
        assert "kospi200_vix_below_sma20" in strategy_ids


class TestSignalGeneration:
    """Tests for generate_signals method."""

    def test_generate_signals_without_data(self):
        """Test signal generation when data load fails."""
        strategy = KOSPI200FuturesStrategy()

        # Don't load data, should return HOLD
        with patch.object(strategy, 'load_data', return_value=False):
            signals = strategy.generate_signals(["KOSPI200"])

        assert len(signals) == 1
        assert signals[0].signal == Signal.HOLD
        assert "Data load failed" in signals[0].reason

    def test_generate_signals_buy(self):
        """Test BUY signal generation."""
        strategy = KOSPI200FuturesStrategy()

        # Mock data for all LONG signals (T-1 shift applied)
        # VIX: T-1=10 (below SMA ~19.5), declining from T-2=20
        strategy._daily_data = pd.DataFrame({'close': [100.0] * 30})
        strategy._vix_data = pd.Series([20.0] * 19 + [10.0] + [99.0])  # T-1=10, today=99
        strategy._semicon_data = pd.Series([100.0] * 19 + [120.0])
        strategy._foreign_data = pd.Series([1e10] * 20)

        signals = strategy.generate_signals(["KOSPI200"])

        assert len(signals) == 1
        assert signals[0].signal == Signal.BUY
        assert signals[0].strength >= 0.5

    def test_generate_signals_hold(self):
        """Test HOLD signal generation."""
        config = KOSPI200FuturesConfig(
            enabled_strategies=["vix_below_sma20"],
            strategy_weights={"vix_below_sma20": 1.0},
        )
        strategy = KOSPI200FuturesStrategy(config=config)

        # Mock data for CASH signal (T-1 VIX above SMA)
        strategy._daily_data = pd.DataFrame({'close': [100.0] * 30})
        strategy._vix_data = pd.Series([15.0] * 19 + [25.0] + [1.0])  # T-1=25 above SMA

        signals = strategy.generate_signals(["KOSPI200"])

        assert len(signals) == 1
        # Low composite weight without position = HOLD
        assert signals[0].signal == Signal.HOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
