"""
Tests for Order Validator with enhanced position limits.
"""

import pytest
from unittest.mock import MagicMock, patch
import os

from libs.core.order_validator import (
    OrderValidator,
    ValidationResult,
    PositionLimits,
)


@pytest.fixture
def mock_config():
    """Create mock config."""
    config = MagicMock()
    config.is_kill_switch_active.return_value = False
    return config


@pytest.fixture
def default_limits():
    """Create default position limits."""
    return PositionLimits()


@pytest.fixture
def validator(mock_config, default_limits):
    """Create validator with default limits."""
    return OrderValidator(mock_config, limits=default_limits)


class TestPositionLimits:
    """Tests for PositionLimits dataclass."""

    def test_default_values(self):
        """Test default limit values."""
        limits = PositionLimits()
        assert limits.max_position_pct == 0.10
        assert limits.max_position_value_krw == 10_000_000
        assert limits.max_total_positions == 20
        assert limits.max_daily_orders == 100

    def test_custom_values(self):
        """Test custom limit values."""
        limits = PositionLimits(
            max_position_pct=0.20,
            max_total_positions=10,
        )
        assert limits.max_position_pct == 0.20
        assert limits.max_total_positions == 10

    @patch.dict(os.environ, {
        "MASP_MAX_POSITION_PCT": "0.15",
        "MASP_MAX_TOTAL_POSITIONS": "30",
    })
    def test_from_env(self):
        """Test loading limits from environment."""
        limits = PositionLimits.from_env()
        assert limits.max_position_pct == 0.15
        assert limits.max_total_positions == 30


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.reason is None
        assert result.warnings == []

    def test_invalid_result(self):
        """Test invalid result with reason."""
        result = ValidationResult(valid=False, reason="Test reason")
        assert result.valid is False
        assert result.reason == "Test reason"

    def test_warnings(self):
        """Test result with warnings."""
        result = ValidationResult(valid=True, warnings=["Warning 1"])
        assert result.valid is True
        assert len(result.warnings) == 1


class TestOrderValidator:
    """Tests for OrderValidator class."""

    def test_init_default(self, mock_config):
        """Test initialization with defaults."""
        validator = OrderValidator(mock_config)
        assert validator.quote_currency == "KRW"
        assert validator._daily_order_count == 0

    def test_init_usdt(self, mock_config, default_limits):
        """Test initialization with USDT."""
        validator = OrderValidator(mock_config, limits=default_limits, quote_currency="USDT")
        assert validator.quote_currency == "USDT"

    def test_kill_switch_blocks(self, mock_config, default_limits):
        """Test kill switch blocks all orders."""
        mock_config.is_kill_switch_active.return_value = True
        validator = OrderValidator(mock_config, limits=default_limits)

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.001,
            price=50_000_000,
            balance=1_000_000,
            total_equity=10_000_000,
        )

        assert result.valid is False
        assert "Kill-Switch" in result.reason


class TestOrderValueLimits:
    """Tests for order value limits."""

    def test_order_too_small(self, validator):
        """Test order below minimum value."""
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.0001,
            price=50_000,  # 5 KRW order
            balance=1_000_000,
            total_equity=10_000_000,
        )
        assert result.valid is False
        assert "too small" in result.reason

    def test_order_too_large(self, validator):
        """Test order exceeds maximum value."""
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=1,
            price=100_000_000,  # 100M KRW order
            balance=200_000_000,
            total_equity=500_000_000,
        )
        assert result.valid is False
        assert "exceeds max value" in result.reason

    def test_order_valid_value(self, validator):
        """Test valid order value."""
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
            price=50_000_000,  # 500K KRW order
            balance=1_000_000,
            total_equity=10_000_000,
        )
        assert result.valid is True


class TestPositionSizeLimits:
    """Tests for position size limits."""

    def test_exceeds_position_pct(self, mock_config):
        """Test order exceeds position percentage limit."""
        limits = PositionLimits(max_position_pct=0.05)  # 5%
        validator = OrderValidator(mock_config, limits=limits)

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.02,
            price=50_000_000,  # 1M KRW = 10% of 10M equity
            balance=5_000_000,
            total_equity=10_000_000,
        )
        assert result.valid is False
        assert "position limit" in result.reason


class TestConcentrationLimits:
    """Tests for portfolio concentration limits."""

    def test_exceeds_concentration(self, mock_config):
        """Test exceeding concentration limit."""
        # Set position_pct high so concentration limit triggers first
        limits = PositionLimits(max_concentration_pct=0.30, max_position_pct=0.50)
        validator = OrderValidator(mock_config, limits=limits)

        # Already have 25% in BTC
        current_positions = {"BTC/KRW": 2_500_000}

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.02,
            price=50_000_000,  # Another 1M KRW = 35% total > 30% concentration
            balance=5_000_000,
            total_equity=10_000_000,
            current_positions=current_positions,
        )
        assert result.valid is False
        assert "concentration limit" in result.reason

    def test_within_concentration(self, mock_config):
        """Test within concentration limit."""
        # Set both limits high enough for 3M position to pass
        limits = PositionLimits(max_concentration_pct=0.50, max_position_pct=0.40)
        validator = OrderValidator(mock_config, limits=limits)

        current_positions = {"BTC/KRW": 2_000_000}

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.02,
            price=50_000_000,  # 1M + 2M = 3M = 30% < 40% position limit, < 50% concentration
            balance=5_000_000,
            total_equity=10_000_000,
            current_positions=current_positions,
        )
        assert result.valid is True


class TestTotalPositionsLimit:
    """Tests for maximum number of positions."""

    def test_max_positions_reached(self, mock_config):
        """Test max positions limit."""
        limits = PositionLimits(max_total_positions=2)
        validator = OrderValidator(mock_config, limits=limits)

        # Already have 2 positions
        current_positions = {
            "BTC/KRW": 1_000_000,
            "ETH/KRW": 1_000_000,
        }

        result = validator.validate(
            symbol="XRP/KRW",  # New symbol
            side="BUY",
            quantity=10,
            price=1_000,
            balance=100_000,
            total_equity=10_000_000,
            current_positions=current_positions,
        )
        assert result.valid is False
        assert "Maximum positions" in result.reason

    def test_adding_to_existing_position(self, mock_config):
        """Test adding to existing position doesn't count as new."""
        # Set position_pct high enough to allow cumulative position
        limits = PositionLimits(max_total_positions=2, max_position_pct=0.20)
        validator = OrderValidator(mock_config, limits=limits)

        current_positions = {
            "BTC/KRW": 500_000,  # 5% of equity
            "ETH/KRW": 500_000,  # 5% of equity
        }

        result = validator.validate(
            symbol="BTC/KRW",  # Existing symbol
            side="BUY",
            quantity=0.01,
            price=50_000_000,  # 500K + 500K = 1M = 10% < 20% position limit
            balance=600_000,
            total_equity=10_000_000,
            current_positions=current_positions,
        )
        # Should pass because we're adding to existing position (within limits)
        assert result.valid is True


class TestDailyLimits:
    """Tests for daily order limits."""

    def test_daily_order_limit(self, mock_config):
        """Test daily order count limit."""
        limits = PositionLimits(max_daily_orders=5)
        validator = OrderValidator(mock_config, limits=limits)
        validator._daily_order_count = 5  # Already at limit

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.001,
            price=50_000_000,
            balance=100_000,
            total_equity=10_000_000,
        )
        assert result.valid is False
        assert "Daily order limit" in result.reason

    def test_daily_volume_limit(self, mock_config):
        """Test daily volume limit."""
        limits = PositionLimits(max_daily_volume_krw=1_000_000)
        validator = OrderValidator(mock_config, limits=limits)
        validator._daily_volume = 900_000  # Almost at limit

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.004,
            price=50_000_000,  # 200K order
            balance=500_000,
            total_equity=10_000_000,
        )
        assert result.valid is False
        assert "Daily volume limit" in result.reason


class TestBalanceCheck:
    """Tests for balance checks."""

    def test_insufficient_balance(self, validator):
        """Test insufficient balance for BUY."""
        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
            price=50_000_000,  # 500K KRW
            balance=100_000,  # Only 100K
            total_equity=10_000_000,
        )
        assert result.valid is False
        assert "Insufficient balance" in result.reason

    def test_sell_no_balance_check(self, validator):
        """Test SELL doesn't require balance."""
        result = validator.validate(
            symbol="BTC/KRW",
            side="SELL",
            quantity=0.01,
            price=50_000_000,
            balance=0,  # No balance
            total_equity=10_000_000,
        )
        assert result.valid is True


class TestWarnings:
    """Tests for validation warnings."""

    def test_near_position_limit_warning(self, mock_config):
        """Test warning when near position limit."""
        limits = PositionLimits(max_position_pct=0.10)
        validator = OrderValidator(mock_config, limits=limits)

        result = validator.validate(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.018,
            price=50_000_000,  # 900K = 9% of equity
            balance=1_000_000,
            total_equity=10_000_000,
        )
        assert result.valid is True
        assert len(result.warnings) > 0
        assert "close to position limit" in result.warnings[0]


class TestRecordingAndStats:
    """Tests for order recording and statistics."""

    def test_record_order(self, validator):
        """Test recording an order."""
        validator.record_order(100_000)
        assert validator._daily_order_count == 1
        assert validator._daily_volume == 100_000

    def test_reset_daily_counters(self, validator):
        """Test resetting daily counters."""
        validator._daily_order_count = 50
        validator._daily_volume = 5_000_000

        validator.reset_daily_counters()

        assert validator._daily_order_count == 0
        assert validator._daily_volume == 0

    def test_get_stats(self, validator):
        """Test getting validator stats."""
        validator._daily_order_count = 10
        validator._daily_volume = 1_000_000
        validator._current_positions = {"BTC/KRW": 500_000}

        stats = validator.get_stats()

        assert stats["daily_order_count"] == 10
        assert stats["daily_volume"] == 1_000_000
        assert stats["position_count"] == 1
        assert "limits" in stats


class TestUSDTValidator:
    """Tests for USDT-based validation."""

    def test_usdt_min_order(self, mock_config):
        """Test USDT minimum order."""
        limits = PositionLimits(min_order_value_usdt=5)
        validator = OrderValidator(mock_config, limits=limits, quote_currency="USDT")

        result = validator.validate(
            symbol="BTC/USDT",
            side="BUY",
            quantity=0.00001,
            price=50000,  # 0.5 USDT
            balance=1000,
            total_equity=10000,
        )
        assert result.valid is False
        assert "too small" in result.reason

    def test_usdt_valid_order(self, mock_config):
        """Test valid USDT order."""
        limits = PositionLimits()
        validator = OrderValidator(mock_config, limits=limits, quote_currency="USDT")

        result = validator.validate(
            symbol="BTC/USDT",
            side="BUY",
            quantity=0.001,
            price=50000,  # 50 USDT
            balance=1000,
            total_equity=10000,
        )
        assert result.valid is True
