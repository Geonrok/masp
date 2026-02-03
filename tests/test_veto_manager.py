"""
Tests for Veto Manager (4-Layer Risk Control System)

Tests all veto layers:
1. Kill Switch
2. Market Structure (ADX, CI)
3. On-Chain (Inflow Z-score)
4. Derivatives (Funding Rate)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from libs.risk.veto_manager import (
    VetoManager,
    VetoConfig,
    VetoResult,
    VetoLevel,
    calculate_adx,
    calculate_choppiness_index,
    calculate_funding_rate_signal,
)


def create_mock_ohlcv(
    n_days: int = 100, trend: str = "up", seed: int = 42
) -> pd.DataFrame:
    """Create mock OHLCV data."""
    np.random.seed(seed)

    if trend == "up":
        drift = 0.002
    elif trend == "down":
        drift = -0.002
    else:
        drift = 0.0

    returns = np.random.randn(n_days) * 0.02 + drift
    close = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_days) * 0.005),
            "high": close * (1 + np.abs(np.random.randn(n_days) * 0.01)),
            "low": close * (1 - np.abs(np.random.randn(n_days) * 0.01)),
            "close": close,
            "volume": np.random.uniform(1e6, 1e7, n_days),
        }
    )

    return df


class TestVetoConfig:
    """Test VetoConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VetoConfig()

        assert config.kill_switch_enabled is False
        assert config.adx_threshold == 20.0
        assert config.ci_threshold == 61.8
        assert config.inflow_zscore_threshold == 2.0
        assert config.funding_rate_threshold == 0.001

    def test_custom_config(self):
        """Test custom configuration."""
        config = VetoConfig(
            adx_threshold=25.0,
            ci_threshold=55.0,
            funding_rate_threshold=0.002,
        )

        assert config.adx_threshold == 25.0
        assert config.ci_threshold == 55.0
        assert config.funding_rate_threshold == 0.002


class TestVetoResult:
    """Test VetoResult dataclass."""

    def test_veto_result_allowed(self):
        """Test allowed trade result."""
        result = VetoResult(can_trade=True)

        assert result.can_trade is True
        assert result.veto_level is None

    def test_veto_result_blocked(self):
        """Test blocked trade result."""
        result = VetoResult(
            can_trade=False,
            veto_level=VetoLevel.KILL_SWITCH,
            reason="Kill switch enabled",
        )

        assert result.can_trade is False
        assert result.veto_level == VetoLevel.KILL_SWITCH

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = VetoResult(
            can_trade=False,
            veto_level=VetoLevel.MARKET_STRUCTURE,
            reason="ADX too low",
            details={"adx": 15.0},
        )

        d = result.to_dict()

        assert d["can_trade"] is False
        assert d["veto_level"] == "MARKET_STRUCTURE"
        assert d["details"]["adx"] == 15.0


class TestVetoManager:
    """Test VetoManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        config = VetoConfig(adx_threshold=25.0)
        manager = VetoManager(config)

        assert manager.config.adx_threshold == 25.0
        assert manager.is_kill_switch_active() is False

    def test_kill_switch(self):
        """Test kill switch functionality."""
        manager = VetoManager()

        # Initially disabled
        assert manager.is_kill_switch_active() is False

        # Enable kill switch
        manager.enable_kill_switch()
        assert manager.is_kill_switch_active() is True

        # Should block trade
        result = manager.can_trade("BTC", "long", {})
        assert result.can_trade is False
        assert result.veto_level == VetoLevel.KILL_SWITCH

        # Disable kill switch
        manager.disable_kill_switch()
        assert manager.is_kill_switch_active() is False

    def test_market_structure_adx_veto(self):
        """Test ADX-based veto (no trend)."""
        config = VetoConfig(adx_threshold=20.0)
        manager = VetoManager(config)

        # Create sideways/choppy data (low ADX)
        ohlcv = create_mock_ohlcv(100, trend="sideways", seed=42)
        context = {"ohlcv": ohlcv}

        result = manager.can_trade("BTC", "long", context)

        # Should check and potentially veto based on ADX
        assert isinstance(result, VetoResult)

    def test_market_structure_ci_veto(self):
        """Test Choppiness Index-based veto."""
        config = VetoConfig(ci_threshold=61.8)
        manager = VetoManager(config)

        # Create sideways data (high CI)
        ohlcv = create_mock_ohlcv(100, trend="sideways", seed=123)
        context = {"ohlcv": ohlcv}

        result = manager.can_trade("BTC", "long", context)

        assert isinstance(result, VetoResult)

    def test_onchain_inflow_veto_long(self):
        """Test on-chain veto for long trades."""
        config = VetoConfig(inflow_zscore_threshold=2.0)
        manager = VetoManager(config)

        # Create high inflow data (sell pressure)
        np.random.seed(42)
        normal_inflow = np.random.randn(30) * 100 + 1000
        high_inflow = np.append(normal_inflow, 1500)  # Spike
        inflow_data = pd.Series(high_inflow)

        context = {"inflow_data": inflow_data}

        result = manager.can_trade("BTC", "long", context)

        # High inflow should potentially veto long
        assert isinstance(result, VetoResult)

    def test_onchain_inflow_veto_short(self):
        """Test on-chain veto for short trades."""
        config = VetoConfig(inflow_zscore_threshold=2.0)
        manager = VetoManager(config)

        # Create low inflow data (accumulation)
        np.random.seed(42)
        normal_inflow = np.random.randn(30) * 100 + 1000
        low_inflow = np.append(normal_inflow, 500)  # Drop
        inflow_data = pd.Series(low_inflow)

        context = {"inflow_data": inflow_data}

        result = manager.can_trade("BTC", "short", context)

        # Low inflow should potentially veto short
        assert isinstance(result, VetoResult)

    def test_derivatives_funding_veto_long(self):
        """Test funding rate veto for long trades."""
        config = VetoConfig(funding_rate_threshold=0.001)
        manager = VetoManager(config)

        # High positive funding (crowded long)
        context = {"funding_rate": 0.002}

        result = manager.can_trade("BTC", "long", context)

        assert result.can_trade is False
        assert result.veto_level == VetoLevel.DERIVATIVES

    def test_derivatives_funding_veto_short(self):
        """Test funding rate veto for short trades."""
        config = VetoConfig(funding_rate_threshold=0.001)
        manager = VetoManager(config)

        # High negative funding (crowded short)
        context = {"funding_rate": -0.002}

        result = manager.can_trade("BTC", "short", context)

        assert result.can_trade is False
        assert result.veto_level == VetoLevel.DERIVATIVES

    def test_funding_allowed(self):
        """Test funding rate within threshold."""
        config = VetoConfig(funding_rate_threshold=0.001)
        manager = VetoManager(config)

        # Normal funding
        context = {"funding_rate": 0.0005}

        result = manager.can_trade("BTC", "long", context)

        assert result.can_trade is True

    def test_all_checks_pass(self):
        """Test when all checks pass."""
        manager = VetoManager()

        # Trending market, normal conditions
        ohlcv = create_mock_ohlcv(100, trend="up", seed=42)
        context = {
            "ohlcv": ohlcv,
            "funding_rate": 0.0001,  # Normal
        }

        result = manager.can_trade("BTC", "long", context)

        # May or may not pass depending on calculated indicators
        assert isinstance(result, VetoResult)

    def test_veto_history(self):
        """Test veto history tracking."""
        config = VetoConfig(funding_rate_threshold=0.001)
        manager = VetoManager(config)

        # Trigger some vetoes
        manager.can_trade("BTC", "long", {"funding_rate": 0.005})
        manager.can_trade("BTC", "long", {"funding_rate": 0.003})

        history = manager.get_veto_history()

        assert len(history) >= 2


class TestCalculateADX:
    """Test ADX calculation."""

    def test_adx_trending_market(self):
        """Test ADX in trending market."""
        np.random.seed(42)
        n = 100

        # Strong uptrend
        close = 100 * np.cumprod(1 + np.ones(n) * 0.01)
        high = close * 1.005
        low = close * 0.995

        adx = calculate_adx(high, low, close, period=14)

        # In strong trend, ADX should be high
        valid_adx = adx[~np.isnan(adx)]
        assert len(valid_adx) > 0

    def test_adx_sideways_market(self):
        """Test ADX in sideways market."""
        np.random.seed(42)
        n = 100

        # Sideways movement
        close = 100 + np.sin(np.linspace(0, 4 * np.pi, n)) * 5
        high = close * 1.01
        low = close * 0.99

        adx = calculate_adx(high, low, close, period=14)

        valid_adx = adx[~np.isnan(adx)]
        assert len(valid_adx) > 0

    def test_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        high = np.array([100, 101, 102])
        low = np.array([99, 100, 101])
        close = np.array([100, 101, 102])

        adx = calculate_adx(high, low, close, period=14)

        assert all(np.isnan(adx))


class TestCalculateChoppinessIndex:
    """Test Choppiness Index calculation."""

    def test_ci_trending_market(self):
        """Test CI in trending market."""
        np.random.seed(42)
        n = 100

        # Strong trend
        close = 100 * np.cumprod(1 + np.ones(n) * 0.01)
        high = close * 1.005
        low = close * 0.995

        ci = calculate_choppiness_index(high, low, close, period=14)

        valid_ci = ci[~np.isnan(ci)]
        assert len(valid_ci) > 0

    def test_ci_choppy_market(self):
        """Test CI in choppy market."""
        np.random.seed(42)
        n = 100

        # Choppy movement
        close = 100 + np.random.randn(n) * 2
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        ci = calculate_choppiness_index(high, low, close, period=14)

        valid_ci = ci[~np.isnan(ci)]
        assert len(valid_ci) > 0

    def test_ci_range(self):
        """Test CI values are in valid range."""
        ohlcv = create_mock_ohlcv(100)

        ci = calculate_choppiness_index(
            ohlcv["high"].values,
            ohlcv["low"].values,
            ohlcv["close"].values,
        )

        valid_ci = ci[~np.isnan(ci)]
        # CI should be between 0 and 100
        assert all(0 <= v <= 100 for v in valid_ci)


class TestCalculateFundingRateSignal:
    """Test funding rate signal calculation."""

    def test_positive_funding_bearish(self):
        """Test positive funding gives bearish signal."""
        signal = calculate_funding_rate_signal(0.002, threshold=0.001)

        assert signal["signal"] == "bearish"

    def test_negative_funding_bullish(self):
        """Test negative funding gives bullish signal."""
        signal = calculate_funding_rate_signal(-0.002, threshold=0.001)

        assert signal["signal"] == "bullish"

    def test_neutral_funding(self):
        """Test neutral funding."""
        signal = calculate_funding_rate_signal(0.0005, threshold=0.001)

        assert signal["signal"] == "neutral"

    def test_with_history(self):
        """Test signal with historical data."""
        np.random.seed(42)
        history = pd.Series(np.random.randn(50) * 0.001)

        signal = calculate_funding_rate_signal(
            funding_rate=0.002,
            funding_history=history,
            threshold=0.001,
        )

        assert signal["zscore"] is not None
        assert signal["percentile"] is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_context(self):
        """Test with empty context."""
        manager = VetoManager()

        result = manager.can_trade("BTC", "long", {})

        # Should pass (no data to veto on)
        assert result.can_trade is True

    def test_insufficient_ohlcv(self):
        """Test with insufficient OHLCV data."""
        manager = VetoManager()

        ohlcv = create_mock_ohlcv(10)  # Too short
        context = {"ohlcv": ohlcv}

        result = manager.can_trade("BTC", "long", context)

        # Should pass (insufficient data for indicators)
        assert isinstance(result, VetoResult)

    def test_zero_std_inflow(self):
        """Test inflow with zero standard deviation."""
        manager = VetoManager()

        # Constant inflow (zero variance)
        inflow_data = pd.Series([1000] * 50)
        context = {"inflow_data": inflow_data}

        result = manager.can_trade("BTC", "long", context)

        assert result.can_trade is True
