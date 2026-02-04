"""
Unit tests for MR_ADAPTIVE_AGGRESSIVE Strategy.
"""

import numpy as np
import pytest

from libs.strategies.mr_adaptive_aggressive import (
    MRAdaptiveAggressiveStrategy,
    MRAdaptiveConfig,
)
from libs.strategies.base import Signal
from libs.strategies.indicators import BollingerBands, RSI, MA


class TestMRAdaptiveAggressiveStrategy:
    """Test cases for MRAdaptiveAggressiveStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return MRAdaptiveAggressiveStrategy()

    @pytest.fixture
    def sample_uptrend_data(self):
        """Generate sample data in uptrend (price > MA50)."""
        np.random.seed(42)
        n = 100
        # Generate uptrending prices
        base = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        return {
            "close": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "volume": np.random.rand(n) * 1000000,
        }

    @pytest.fixture
    def sample_downtrend_data(self):
        """Generate sample data in downtrend (price < MA50)."""
        np.random.seed(43)
        n = 100
        # Generate downtrending prices
        base = 100 - np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        base = np.maximum(base, 10)  # Ensure positive
        return {
            "close": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "volume": np.random.rand(n) * 1000000,
        }

    @pytest.fixture
    def oversold_data(self):
        """Generate data with oversold conditions (RSI < 35, below BB lower)."""
        np.random.seed(44)
        n = 100
        # Start high, then sharp drop to create oversold
        base = np.concatenate([
            np.linspace(100, 120, 70),  # Rise
            np.linspace(120, 80, 30),   # Sharp drop
        ])
        return {
            "close": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "volume": np.random.rand(n) * 1000000,
        }

    def test_strategy_initialization(self, strategy):
        """Test strategy initializes with correct defaults."""
        assert strategy.strategy_id == "mr_adaptive_aggressive"
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0
        assert strategy.rsi_period == 14
        assert strategy.rsi_low == 35
        assert strategy.rsi_exit == 55
        assert strategy.trend_ma == 50
        assert strategy.trend_scale == 0.3
        assert strategy.max_positions == 30

    def test_strategy_custom_params(self):
        """Test strategy with custom parameters."""
        strategy = MRAdaptiveAggressiveStrategy(
            bb_period=25,
            rsi_low=30,
            rsi_exit=60,
            trend_scale=0.5,
        )
        assert strategy.bb_period == 25
        assert strategy.rsi_low == 30
        assert strategy.rsi_exit == 60
        assert strategy.trend_scale == 0.5

    def test_update_ohlcv(self, strategy):
        """Test OHLCV cache update."""
        close = [100, 101, 102]
        high = [101, 102, 103]
        low = [99, 100, 101]
        volume = [1000, 1100, 1200]

        strategy.update_ohlcv("TEST/USDT:PERP", close, high, low, volume)

        assert "TEST/USDT:PERP" in strategy._ohlcv_cache
        assert len(strategy._ohlcv_cache["TEST/USDT:PERP"]["close"]) == 3

    def test_calculate_indicators(self, strategy, sample_uptrend_data):
        """Test indicator calculation."""
        strategy.update_ohlcv(
            "BTC/USDT:PERP",
            sample_uptrend_data["close"].tolist(),
            sample_uptrend_data["high"].tolist(),
            sample_uptrend_data["low"].tolist(),
            sample_uptrend_data["volume"].tolist(),
        )

        indicators = strategy._calculate_indicators(sample_uptrend_data)

        assert indicators is not None
        assert "close" in indicators
        assert "bb_upper" in indicators
        assert "bb_mid" in indicators
        assert "bb_lower" in indicators
        assert "rsi" in indicators
        assert "trend_ma" in indicators
        assert "in_uptrend" in indicators
        assert "position_scale" in indicators

        # BB order should be upper > mid > lower
        assert indicators["bb_upper"] >= indicators["bb_mid"]
        assert indicators["bb_mid"] >= indicators["bb_lower"]

        # RSI should be 0-100
        assert 0 <= indicators["rsi"] <= 100

    def test_entry_conditions(self, strategy):
        """Test entry condition check."""
        # Entry: close < bb_lower AND rsi < 35
        entry_indicators = {
            "close": 95,
            "bb_lower": 100,
            "rsi": 30,
        }
        assert strategy._check_entry(entry_indicators) is True

        # No entry: RSI too high
        no_entry_rsi = {
            "close": 95,
            "bb_lower": 100,
            "rsi": 40,
        }
        assert strategy._check_entry(no_entry_rsi) is False

        # No entry: Price above BB lower
        no_entry_price = {
            "close": 105,
            "bb_lower": 100,
            "rsi": 30,
        }
        assert strategy._check_entry(no_entry_price) is False

    def test_exit_conditions(self, strategy):
        """Test exit condition check."""
        # Exit: RSI > 55
        exit_rsi = {
            "close": 100,
            "bb_mid": 105,
            "rsi": 60,
        }
        assert strategy._check_exit(exit_rsi) is True

        # Exit: Close > BB mid
        exit_price = {
            "close": 110,
            "bb_mid": 105,
            "rsi": 50,
        }
        assert strategy._check_exit(exit_price) is True

        # No exit: Neither condition met
        no_exit = {
            "close": 100,
            "bb_mid": 105,
            "rsi": 50,
        }
        assert strategy._check_exit(no_exit) is False

    def test_position_scale_uptrend(self, strategy, sample_uptrend_data):
        """Test position scale is 1.0 in uptrend."""
        strategy.update_ohlcv(
            "BTC/USDT:PERP",
            sample_uptrend_data["close"].tolist(),
            sample_uptrend_data["high"].tolist(),
            sample_uptrend_data["low"].tolist(),
            sample_uptrend_data["volume"].tolist(),
        )

        indicators = strategy._calculate_indicators(sample_uptrend_data)

        if indicators["in_uptrend"]:
            assert indicators["position_scale"] == 1.0

    def test_position_scale_downtrend(self, strategy, sample_downtrend_data):
        """Test position scale is 0.3 in downtrend."""
        strategy.update_ohlcv(
            "BTC/USDT:PERP",
            sample_downtrend_data["close"].tolist(),
            sample_downtrend_data["high"].tolist(),
            sample_downtrend_data["low"].tolist(),
            sample_downtrend_data["volume"].tolist(),
        )

        indicators = strategy._calculate_indicators(sample_downtrend_data)

        if not indicators["in_uptrend"]:
            assert indicators["position_scale"] == 0.3

    def test_generate_signal_no_data(self, strategy):
        """Test signal generation with no data."""
        signal = strategy.generate_signal("UNKNOWN/USDT:PERP")

        assert signal.signal == Signal.HOLD
        assert "unavailable" in signal.reason.lower()

    def test_generate_signal_insufficient_data(self, strategy):
        """Test signal generation with insufficient data."""
        # Only 10 bars (need at least 70)
        strategy.update_ohlcv(
            "TEST/USDT:PERP",
            [100] * 10,
            [101] * 10,
            [99] * 10,
            [1000] * 10,
        )

        signal = strategy.generate_signal("TEST/USDT:PERP")

        assert signal.signal == Signal.HOLD
        assert "insufficient" in signal.reason.lower()

    def test_generate_signals_multiple(self, strategy, sample_uptrend_data):
        """Test generating signals for multiple symbols."""
        for symbol in ["BTC/USDT:PERP", "ETH/USDT:PERP"]:
            strategy.update_ohlcv(
                symbol,
                sample_uptrend_data["close"].tolist(),
                sample_uptrend_data["high"].tolist(),
                sample_uptrend_data["low"].tolist(),
                sample_uptrend_data["volume"].tolist(),
            )

        signals = strategy.generate_signals(["BTC/USDT:PERP", "ETH/USDT:PERP"])

        assert len(signals) == 2
        for signal in signals:
            assert signal.signal in [Signal.BUY, Signal.SELL, Signal.HOLD]

    def test_position_tracking(self, strategy):
        """Test position entry and exit tracking."""
        # Simulate entry
        strategy._entry_prices["TEST/USDT:PERP"] = 100.0
        strategy._position_scales["TEST/USDT:PERP"] = 0.3
        strategy.update_position("TEST/USDT:PERP", 1.0)

        assert strategy.has_position("TEST/USDT:PERP")
        assert strategy.get_position("TEST/USDT:PERP") == 1.0
        assert strategy.get_position_scale("TEST/USDT:PERP") == 0.3

        # Simulate exit
        strategy.update_position("TEST/USDT:PERP", 0)
        assert not strategy.has_position("TEST/USDT:PERP")

    def test_get_parameters(self, strategy):
        """Test get_parameters returns correct values."""
        params = strategy.get_parameters()

        assert params["bb_period"] == 20
        assert params["bb_std"] == 2.0
        assert params["rsi_period"] == 14
        assert params["rsi_low"] == 35
        assert params["rsi_exit"] == 55
        assert params["trend_ma"] == 50
        assert params["trend_scale"] == 0.3
        assert params["max_positions"] == 30


class TestBollingerBandsIndicator:
    """Test Bollinger Bands indicator function."""

    def test_bollinger_bands_basic(self):
        """Test basic BB calculation."""
        close = [100, 101, 102, 103, 104] * 10  # 50 values
        upper, mid, lower = BollingerBands(close, period=20, std_dev=2.0)

        assert upper > mid
        assert mid > lower

    def test_bollinger_bands_insufficient_data(self):
        """Test BB with insufficient data."""
        close = [100, 101, 102]  # Only 3 values
        upper, mid, lower = BollingerBands(close, period=20)

        # Should return last value for all
        assert upper == close[-1]
        assert mid == close[-1]
        assert lower == close[-1]


class TestRSIIndicator:
    """Test RSI indicator function."""

    def test_rsi_range(self):
        """Test RSI is always 0-100."""
        np.random.seed(45)
        close = 100 + np.cumsum(np.random.randn(100))
        rsi = RSI(close.tolist())

        assert 0 <= rsi <= 100

    def test_rsi_oversold(self):
        """Test RSI detects oversold condition."""
        # Consistent downward prices should give low RSI
        close = [100 - i * 0.5 for i in range(50)]
        rsi = RSI(close)

        assert rsi < 50  # Should be in lower half

    def test_rsi_overbought(self):
        """Test RSI detects overbought condition."""
        # Consistent upward prices should give high RSI
        close = [100 + i * 0.5 for i in range(50)]
        rsi = RSI(close)

        assert rsi > 50  # Should be in upper half
