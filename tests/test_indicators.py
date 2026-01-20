"""
Tests for technical indicators module.
"""

import numpy as np
import pytest

from libs.strategies.indicators import (
    MA,
    KAMA,
    KAMA_series,
    TSMOM,
    TSMOM_signal,
    RSI,
    RSI_series,
    EMA,
    EMA_series,
    MACD,
    MACD_series,
    ATR,
)


class TestMA:
    """Tests for Simple Moving Average."""

    def test_basic_calculation(self):
        """Test basic MA calculation."""
        data = [10, 20, 30, 40, 50]
        result = MA(data, period=3)

        # MA of last 3: (30+40+50)/3 = 40
        assert result == pytest.approx(40.0)

    def test_insufficient_data(self):
        """Test MA with insufficient data returns last value."""
        data = [10, 20]
        result = MA(data, period=5)

        assert result == pytest.approx(20.0)

    def test_empty_data(self):
        """Test MA with empty data returns 0."""
        result = MA([], period=3)
        assert result == 0.0

    def test_numpy_array_input(self):
        """Test MA accepts numpy array."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = MA(data, period=5)

        assert result == pytest.approx(30.0)


class TestKAMA:
    """Tests for Kaufman's Adaptive Moving Average."""

    def test_basic_calculation(self):
        """Test basic KAMA calculation."""
        # Create trending data
        data = list(range(1, 21))  # 1 to 20
        result = KAMA(data, period=5)

        assert result > 0
        assert result <= 20  # Should be less than or equal to last price

    def test_insufficient_data(self):
        """Test KAMA with insufficient data."""
        data = [10, 20, 30]
        result = KAMA(data, period=5)

        assert result == pytest.approx(30.0)

    def test_kama_series(self):
        """Test full KAMA series."""
        data = list(range(1, 21))
        series = KAMA_series(data, period=5)

        assert len(series) == len(data)
        assert series[-1] > 0

    def test_volatility_adaptation(self):
        """Test KAMA adapts to volatility."""
        # Low volatility (trending)
        trending = [i * 1.01 for i in range(100, 150)]
        kama_trending = KAMA(trending, period=5)

        # High volatility (choppy)
        choppy = [100 + ((-1) ** i) * 5 for i in range(50)]
        kama_choppy = KAMA(choppy, period=5)

        # Both should compute without errors
        assert kama_trending > 0
        assert kama_choppy > 0


class TestTSMOM:
    """Tests for Time-Series Momentum."""

    def test_positive_momentum(self):
        """Test positive momentum."""
        data = list(range(100, 200))  # Rising prices
        result = TSMOM(data, lookback=50)

        assert result > 0

    def test_negative_momentum(self):
        """Test negative momentum."""
        data = list(range(200, 100, -1))  # Falling prices
        result = TSMOM(data, lookback=50)

        assert result < 0

    def test_signal_positive(self):
        """Test TSMOM signal for positive momentum."""
        data = list(range(100, 200))
        assert TSMOM_signal(data, lookback=50) is True

    def test_signal_negative(self):
        """Test TSMOM signal for negative momentum."""
        data = list(range(200, 100, -1))
        assert TSMOM_signal(data, lookback=50) is False

    def test_insufficient_data(self):
        """Test TSMOM with insufficient data."""
        data = [100, 110]
        result = TSMOM(data, lookback=90)

        # Should calculate return from first to last
        assert result == pytest.approx(0.1)

    def test_zero_price(self):
        """Test TSMOM handles zero price."""
        data = [0] + list(range(1, 100))
        result = TSMOM(data, lookback=90)

        assert result == 0.0  # Can't calculate return from zero


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_overbought(self):
        """Test RSI for consistently rising prices."""
        data = list(range(50, 100))  # Consistently rising
        result = RSI(data, period=14)

        assert result > 70  # Should be overbought

    def test_oversold(self):
        """Test RSI for consistently falling prices."""
        data = list(range(100, 50, -1))  # Consistently falling
        result = RSI(data, period=14)

        assert result < 30  # Should be oversold

    def test_neutral(self):
        """Test RSI with insufficient data returns neutral."""
        data = [100, 101, 102]
        result = RSI(data, period=14)

        assert result == pytest.approx(50.0)

    def test_rsi_bounds(self):
        """Test RSI is bounded 0-100."""
        # Extreme rising
        data = [i * 1.1 for i in range(1, 100)]
        result = RSI(data, period=14)
        assert 0 <= result <= 100

        # Extreme falling
        data = [100 - i * 0.5 for i in range(100)]
        result = RSI(data, period=14)
        assert 0 <= result <= 100

    def test_rsi_series(self):
        """Test full RSI series."""
        data = [100 + np.sin(i / 5) * 10 for i in range(100)]
        series = RSI_series(data, period=14)

        assert len(series) == len(data)
        assert all(0 <= v <= 100 for v in series)


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_basic_calculation(self):
        """Test basic EMA calculation."""
        data = [10, 20, 30, 40, 50]
        result = EMA(data, period=3)

        # EMA should be weighted toward recent values
        assert result > MA(data, period=3)  # EMA > SMA for uptrend

    def test_insufficient_data(self):
        """Test EMA with insufficient data."""
        data = [10, 20]
        result = EMA(data, period=5)

        assert result == pytest.approx(20.0)

    def test_ema_series(self):
        """Test full EMA series."""
        data = list(range(1, 21))
        series = EMA_series(data, period=5)

        assert len(series) == len(data)
        assert series[0] == data[0]  # First value should be same

    def test_empty_data(self):
        """Test EMA with empty data."""
        result = EMA([], period=5)
        assert result == 0.0

        series = EMA_series([], period=5)
        assert len(series) == 0


class TestMACD:
    """Tests for MACD."""

    def test_basic_calculation(self):
        """Test basic MACD calculation."""
        data = list(range(50, 100))
        macd_line, signal_line, histogram = MACD(data)

        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert isinstance(histogram, float)

    def test_uptrend(self):
        """Test MACD in uptrend."""
        data = [i * 1.02 for i in range(50, 150)]
        macd_line, signal_line, histogram = MACD(data)

        assert macd_line > 0  # Fast EMA > Slow EMA in uptrend

    def test_downtrend(self):
        """Test MACD in downtrend."""
        data = [150 - i * 0.5 for i in range(100)]
        macd_line, signal_line, histogram = MACD(data)

        assert macd_line < 0  # Fast EMA < Slow EMA in downtrend

    def test_insufficient_data(self):
        """Test MACD with insufficient data."""
        data = [100, 101, 102]
        macd_line, signal_line, histogram = MACD(data)

        assert macd_line == 0.0
        assert signal_line == 0.0
        assert histogram == 0.0

    def test_macd_series(self):
        """Test full MACD series."""
        data = list(range(50, 150))
        macd_line, signal_line, histogram = MACD_series(data)

        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)


class TestATR:
    """Tests for Average True Range."""

    def test_basic_calculation(self):
        """Test basic ATR calculation."""
        high = [110, 115, 120, 118, 122]
        low = [100, 105, 110, 108, 112]
        close = [105, 112, 115, 110, 118]

        result = ATR(high, low, close, period=3)

        assert result > 0
        assert result < 20  # Should be reasonable

    def test_constant_range(self):
        """Test ATR with constant range."""
        n = 20
        high = [110] * n
        low = [100] * n
        close = [105] * n

        result = ATR(high, low, close, period=14)

        # ATR should be around 10 (high - low)
        assert result == pytest.approx(10.0, rel=0.1)

    def test_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = [110]
        low = [100]
        close = [105]

        result = ATR(high, low, close, period=14)

        assert result == 0.0

    def test_gaps(self):
        """Test ATR handles gaps correctly."""
        # Gap up scenario
        high = [100, 120, 125]
        low = [90, 115, 118]
        close = [95, 118, 120]

        result = ATR(high, low, close, period=3)

        # Should capture the gap
        assert result > 10


class TestIndicatorsEdgeCases:
    """Tests for edge cases across indicators."""

    def test_single_value_arrays(self):
        """Test indicators with single value."""
        data = [100.0]

        assert MA(data, 5) == 100.0
        assert KAMA(data, 5) == 100.0
        assert EMA(data, 5) == 100.0

    def test_identical_values(self):
        """Test indicators with identical values."""
        data = [100.0] * 50

        assert MA(data, 10) == 100.0
        assert EMA(data, 10) == pytest.approx(100.0, rel=0.01)
        assert RSI(data, 14) == pytest.approx(50.0)  # No movement = neutral

    def test_large_numbers(self):
        """Test indicators with large numbers."""
        data = [1e10 + i * 1e8 for i in range(50)]

        ma = MA(data, 10)
        ema = EMA(data, 10)
        rsi = RSI(data, 14)

        assert ma > 0
        assert ema > 0
        assert 0 <= rsi <= 100

    def test_small_numbers(self):
        """Test indicators with small numbers."""
        data = [1e-6 + i * 1e-8 for i in range(50)]

        ma = MA(data, 10)
        ema = EMA(data, 10)
        rsi = RSI(data, 14)

        assert ma > 0
        assert ema > 0
        assert 0 <= rsi <= 100

    def test_negative_numbers(self):
        """Test indicators with negative numbers (e.g., returns)."""
        data = [-0.05, -0.03, 0.02, 0.01, -0.02, 0.04]

        ma = MA(data, 3)
        ema = EMA(data, 3)

        assert isinstance(ma, float)
        assert isinstance(ema, float)

    def test_pandas_series_input(self):
        """Test indicators accept pandas Series."""
        import pandas as pd

        data = pd.Series([10, 20, 30, 40, 50])

        assert MA(data, 3) == pytest.approx(40.0)
        assert EMA(data, 3) > 0
        assert RSI(list(range(50, 100)), 14) > 50


class TestIndicatorsNumericalStability:
    """Tests for numerical stability."""

    def test_rsi_all_gains(self):
        """Test RSI when all changes are gains."""
        data = list(range(100, 200))  # All positive changes
        result = RSI(data, period=14)

        assert result == pytest.approx(100.0)

    def test_rsi_all_losses(self):
        """Test RSI when all changes are losses."""
        data = list(range(200, 100, -1))  # All negative changes
        result = RSI(data, period=14)

        assert result == pytest.approx(0.0, abs=1.0)

    def test_macd_convergence(self):
        """Test MACD converges for flat price."""
        data = [100.0] * 100
        macd_line, signal_line, histogram = MACD(data)

        # Should converge to zero
        assert abs(macd_line) < 0.1
        assert abs(histogram) < 0.1

    def test_kama_zero_volatility(self):
        """Test KAMA handles zero volatility."""
        data = [100.0] * 20
        result = KAMA(data, period=5)

        assert result == pytest.approx(100.0)
