"""
Tests for Execution Helpers.
"""

import time
from unittest.mock import MagicMock

import pytest

from libs.adapters.execution_helpers import (
    BalanceInfo,
    KillSwitchMixin,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationMixin,
    StandardOrderResult,
    calculate_slippage,
    format_price,
    format_quantity,
    rate_limit,
    retry_with_backoff,
    validate_order_params,
)


class TestOrderSide:
    """Tests for OrderSide enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


class TestOrderType:
    """Tests for OrderType enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_values(self):
        """Test enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"


class TestStandardOrderResult:
    """Tests for StandardOrderResult dataclass."""

    def test_creation(self):
        """Test creating an order result."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=0.01,
            filled_quantity=0.01,
            average_fill_price=50000000,
        )
        assert result.order_id == "123"
        assert result.exchange == "upbit"
        assert result.is_filled

    def test_is_filled(self):
        """Test is_filled property."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=0.01,
        )
        assert result.is_filled
        assert not result.is_partial

    def test_is_partial(self):
        """Test is_partial property."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PARTIAL,
            requested_quantity=0.01,
            filled_quantity=0.005,
        )
        assert result.is_partial
        assert not result.is_filled

    def test_fill_rate(self):
        """Test fill_rate property."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.PARTIAL,
            requested_quantity=0.01,
            filled_quantity=0.005,
        )
        assert result.fill_rate == 0.5

    def test_total_value(self):
        """Test total_value property."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=0.01,
            filled_quantity=0.01,
            average_fill_price=50000000,
        )
        assert result.total_value == 500000

    def test_to_dict(self):
        """Test to_dict method."""
        result = StandardOrderResult(
            order_id="123",
            exchange="upbit",
            symbol="BTC/KRW",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=0.01,
        )
        d = result.to_dict()
        assert d["order_id"] == "123"
        assert d["side"] == "BUY"
        assert d["order_type"] == "MARKET"


class TestBalanceInfo:
    """Tests for BalanceInfo dataclass."""

    def test_creation(self):
        """Test creating balance info."""
        balance = BalanceInfo(
            currency="KRW",
            available=1000000,
            locked=100000,
        )
        assert balance.currency == "KRW"
        assert balance.available == 1000000
        assert balance.locked == 100000
        assert balance.total == 1100000

    def test_total_calculation(self):
        """Test automatic total calculation."""
        balance = BalanceInfo(
            currency="BTC",
            available=0.5,
            locked=0.1,
        )
        assert balance.total == 0.6


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_call(self):
        """Test successful call without retry."""
        call_count = [0]

        @retry_with_backoff(max_retries=3)
        def successful_func():
            call_count[0] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = failing_func()
        assert result == "success"
        assert call_count[0] == 2

    def test_max_retries_exceeded(self):
        """Test max retries exceeded."""
        call_count = [0]

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert call_count[0] == 3  # Initial + 2 retries


class TestRateLimit:
    """Tests for rate_limit decorator."""

    def test_rate_limiting(self):
        """Test rate limiting."""
        call_times = []

        @rate_limit(calls_per_second=10)  # 100ms between calls
        def rate_limited_func():
            call_times.append(time.time())
            return "result"

        # Make 3 calls
        for _ in range(3):
            rate_limited_func()

        # Check intervals
        assert len(call_times) == 3
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i - 1]
            # Should be at least 100ms (0.1s) apart
            assert interval >= 0.09  # Allow small margin


class TestValidateOrderParams:
    """Tests for validate_order_params function."""

    def test_valid_market_order(self):
        """Test validating market order."""
        params = validate_order_params(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
        )
        assert params["side"] == OrderSide.BUY
        assert params["order_type"] == OrderType.MARKET
        assert params["quantity"] == 0.01

    def test_valid_limit_order(self):
        """Test validating limit order."""
        params = validate_order_params(
            symbol="BTC/KRW",
            side="SELL",
            quantity=0.01,
            price=50000000,
            order_type="LIMIT",
        )
        assert params["side"] == OrderSide.SELL
        assert params["order_type"] == OrderType.LIMIT
        assert params["price"] == 50000000

    def test_invalid_side(self):
        """Test invalid side validation."""
        with pytest.raises(ValueError, match="Invalid side"):
            validate_order_params(
                symbol="BTC/KRW",
                side="INVALID",
                quantity=0.01,
            )

    def test_negative_quantity(self):
        """Test negative quantity validation."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            validate_order_params(
                symbol="BTC/KRW",
                side="BUY",
                quantity=-0.01,
            )

    def test_limit_order_without_price(self):
        """Test limit order without price."""
        with pytest.raises(ValueError, match="Price is required"):
            validate_order_params(
                symbol="BTC/KRW",
                side="BUY",
                quantity=0.01,
                order_type="LIMIT",
            )


class TestFormatQuantity:
    """Tests for format_quantity function."""

    def test_basic_formatting(self):
        """Test basic quantity formatting."""
        result = format_quantity(0.123456789, precision=4)
        assert result == 0.1235

    def test_min_quantity(self):
        """Test minimum quantity enforcement."""
        result = format_quantity(0.00001, precision=4, min_quantity=0.001)
        assert result == 0.001


class TestFormatPrice:
    """Tests for format_price function."""

    def test_tick_size_formatting(self):
        """Test price formatting with tick size."""
        result = format_price(50000123, tick_size=100)
        assert result == 50000100

    def test_precision_formatting(self):
        """Test price formatting with precision."""
        result = format_price(0.123456789, precision=4)
        assert result == 0.1235


class TestCalculateSlippage:
    """Tests for calculate_slippage function."""

    def test_buy_positive_slippage(self):
        """Test buy with unfavorable slippage."""
        slippage = calculate_slippage(
            requested_price=100,
            executed_price=101,
            side=OrderSide.BUY,
        )
        assert slippage == 1.0  # 1%

    def test_buy_negative_slippage(self):
        """Test buy with favorable slippage."""
        slippage = calculate_slippage(
            requested_price=100,
            executed_price=99,
            side=OrderSide.BUY,
        )
        assert slippage == -1.0  # -1%

    def test_sell_positive_slippage(self):
        """Test sell with unfavorable slippage."""
        slippage = calculate_slippage(
            requested_price=100,
            executed_price=99,
            side=OrderSide.SELL,
        )
        assert slippage == 1.0  # 1%

    def test_sell_negative_slippage(self):
        """Test sell with favorable slippage."""
        slippage = calculate_slippage(
            requested_price=100,
            executed_price=101,
            side=OrderSide.SELL,
        )
        assert slippage == -1.0  # -1%


class TestKillSwitchMixin:
    """Tests for KillSwitchMixin."""

    def test_kill_switch_inactive(self):
        """Test kill switch inactive."""

        class TestAdapter(KillSwitchMixin):
            def __init__(self):
                self._config = MagicMock()
                self._config.is_kill_switch_active.return_value = False

        adapter = TestAdapter()
        assert not adapter._is_kill_switch_active()

    def test_kill_switch_active(self):
        """Test kill switch active."""

        class TestAdapter(KillSwitchMixin):
            def __init__(self):
                self._config = MagicMock()
                self._config.is_kill_switch_active.return_value = True

        adapter = TestAdapter()
        assert adapter._is_kill_switch_active()

    def test_check_kill_switch_raises(self):
        """Test check_kill_switch raises when active."""

        class TestAdapter(KillSwitchMixin):
            def __init__(self):
                self._config = MagicMock()
                self._config.is_kill_switch_active.return_value = True

        adapter = TestAdapter()
        with pytest.raises(RuntimeError, match="Kill switch is active"):
            adapter._check_kill_switch()


class TestOrderValidationMixin:
    """Tests for OrderValidationMixin."""

    def test_validation_passes(self):
        """Test validation passes when no validator."""

        class TestAdapter(OrderValidationMixin):
            pass

        adapter = TestAdapter()
        result = adapter._validate_order(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
        )
        assert result["valid"]

    def test_validation_with_validator(self):
        """Test validation with validator."""

        class TestAdapter(OrderValidationMixin):
            def __init__(self):
                self._order_validator = MagicMock()
                validation_result = MagicMock()
                validation_result.valid = True
                validation_result.warnings = []
                self._order_validator.validate.return_value = validation_result

        adapter = TestAdapter()
        result = adapter._validate_order(
            symbol="BTC/KRW",
            side="BUY",
            quantity=0.01,
            price=50000000,
            balance=1000000,
        )
        assert result["valid"]
