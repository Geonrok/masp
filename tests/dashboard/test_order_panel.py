"""Tests for order panel component."""
from __future__ import annotations

from datetime import datetime

import pytest


def test_import_order_panel():
    """Test module imports correctly."""
    from services.dashboard.components import order_panel

    assert hasattr(order_panel, "render_order_panel")
    assert hasattr(order_panel, "OrderRequest")
    assert hasattr(order_panel, "OrderResult")
    assert hasattr(order_panel, "OrderSide")
    assert hasattr(order_panel, "OrderType")


def test_order_side_enum():
    """Test OrderSide enum values."""
    from services.dashboard.components.order_panel import OrderSide

    assert OrderSide.BUY.value == "BUY"
    assert OrderSide.SELL.value == "SELL"


def test_order_type_enum():
    """Test OrderType enum values."""
    from services.dashboard.components.order_panel import OrderType

    assert OrderType.MARKET.value == "MARKET"
    assert OrderType.LIMIT.value == "LIMIT"


def test_order_request_dataclass():
    """Test OrderRequest dataclass."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    assert request.symbol == "BTC"
    assert request.side == OrderSide.BUY
    assert request.order_type == OrderType.MARKET
    assert request.quantity == 0.1
    assert request.price is None
    assert request.amount_krw is None


def test_order_request_with_price():
    """Test OrderRequest with limit price."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
    )

    request = OrderRequest(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=3_000_000.0,
    )

    assert request.price == 3_000_000.0


def test_order_result_dataclass():
    """Test OrderResult dataclass."""
    from services.dashboard.components.order_panel import OrderResult

    result = OrderResult(
        success=True,
        order_id="TEST-001",
        symbol="BTC",
        side="BUY",
        executed_qty=0.1,
        executed_price=54_000_000.0,
        total_value=5_400_000.0,
        fee=2_700.0,
        message="Order executed",
        timestamp=datetime.now(),
    )

    assert result.success is True
    assert result.order_id == "TEST-001"
    assert result.executed_qty == 0.1


def test_key_function():
    """Test _key generates namespaced keys."""
    from services.dashboard.components.order_panel import _key, _KEY_PREFIX

    result = _key("symbol")
    assert result == f"{_KEY_PREFIX}symbol"
    assert result == "order_panel.symbol"


def test_safe_float_valid():
    """Test _safe_float with valid values."""
    from services.dashboard.components.order_panel import _safe_float

    assert _safe_float(1.5) == 1.5
    assert _safe_float(0) == 0.0
    assert _safe_float(-2.5) == -2.5
    assert _safe_float("3.14") == 3.14


def test_safe_float_invalid():
    """Test _safe_float with invalid values."""
    from services.dashboard.components.order_panel import _safe_float

    assert _safe_float(None) == 0.0
    assert _safe_float("invalid") == 0.0
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0


def test_safe_float_custom_default():
    """Test _safe_float with custom default."""
    from services.dashboard.components.order_panel import _safe_float

    assert _safe_float(None, default=1.0) == 1.0
    assert _safe_float("invalid", default=-1.0) == -1.0


def test_is_finite():
    """Test _is_finite function."""
    from services.dashboard.components.order_panel import _is_finite

    assert _is_finite(1.0) is True
    assert _is_finite(0.0) is True
    assert _is_finite(-100.5) is True
    assert _is_finite(float("inf")) is False
    assert _is_finite(float("-inf")) is False
    assert _is_finite(float("nan")) is False


def test_validate_order_request_valid():
    """Test _validate_order_request with valid request."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is True
    assert error == ""


def test_validate_order_request_empty_symbol():
    """Test _validate_order_request with empty symbol."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Symbol" in error


def test_validate_order_request_negative_quantity():
    """Test _validate_order_request with negative quantity."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=-0.1,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Quantity" in error


def test_validate_order_request_no_quantity_no_amount():
    """Test _validate_order_request with neither quantity nor amount."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=None,
        amount_krw=None,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "quantity" in error.lower() or "amount" in error.lower()


def test_validate_order_request_limit_no_price():
    """Test _validate_order_request for LIMIT order without price."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=None,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Price" in error or "LIMIT" in error


def test_validate_order_request_with_amount_krw():
    """Test _validate_order_request with amount_krw instead of quantity."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=None,
        amount_krw=1_000_000.0,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is True


def test_get_demo_symbols():
    """Test _get_demo_symbols returns expected symbols."""
    from services.dashboard.components.order_panel import _get_demo_symbols

    symbols = _get_demo_symbols()

    assert "BTC" in symbols
    assert "ETH" in symbols
    assert "XRP" in symbols
    assert len(symbols) >= 5


def test_get_demo_prices():
    """Test _get_demo_prices returns deterministic prices."""
    from services.dashboard.components.order_panel import _get_demo_prices

    prices1 = _get_demo_prices()
    prices2 = _get_demo_prices()

    assert prices1 == prices2
    assert prices1["BTC"] == 54_000_000.0
    assert prices1["ETH"] == 2_900_000.0


def test_get_demo_balances():
    """Test _get_demo_balances returns expected structure."""
    from services.dashboard.components.order_panel import _get_demo_balances

    balances = _get_demo_balances()

    assert "KRW" in balances
    assert "BTC" in balances
    assert "available" in balances["KRW"]
    assert "locked" in balances["KRW"]
    assert balances["KRW"]["available"] == 10_000_000.0


def test_format_krw():
    """Test _format_krw formatting."""
    from services.dashboard.components.order_panel import _format_krw

    assert _format_krw(1_000_000) == "1,000,000 KRW"
    assert _format_krw(54_000_000) == "54,000,000 KRW"
    assert _format_krw(500) == "500.00 KRW"


def test_format_quantity_btc():
    """Test _format_quantity for BTC."""
    from services.dashboard.components.order_panel import _format_quantity

    result = _format_quantity(0.12345678, "BTC")
    assert "0.12345678" in result


def test_format_quantity_xrp():
    """Test _format_quantity for XRP."""
    from services.dashboard.components.order_panel import _format_quantity

    result = _format_quantity(1000.5, "XRP")
    assert "1000.50" in result


def test_calculate_order_estimate_with_quantity():
    """Test _calculate_order_estimate with quantity."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=0.1,
        amount_krw=None,
        price=54_000_000.0,
    )

    assert estimate["quantity"] == 0.1
    assert estimate["total_value"] == 5_400_000.0
    assert estimate["fee"] == 5_400_000.0 * 0.0005  # 0.05%
    assert estimate["net_value"] > estimate["total_value"]  # BUY: cost includes fee


def test_calculate_order_estimate_with_amount():
    """Test _calculate_order_estimate with amount_krw."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=None,
        amount_krw=1_000_000.0,
        price=50_000_000.0,
    )

    assert estimate["quantity"] == 0.02  # 1M / 50M
    assert estimate["total_value"] == 1_000_000.0


def test_calculate_order_estimate_sell():
    """Test _calculate_order_estimate for SELL order."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="ETH",
        side=OrderSide.SELL,
        quantity=1.0,
        amount_krw=None,
        price=3_000_000.0,
    )

    assert estimate["total_value"] == 3_000_000.0
    assert estimate["net_value"] < estimate["total_value"]  # SELL: proceeds minus fee


def test_calculate_order_estimate_zero_price():
    """Test _calculate_order_estimate with zero price."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=0.1,
        amount_krw=None,
        price=0.0,
    )

    assert estimate["quantity"] == 0.0
    assert estimate["total_value"] == 0.0


def test_calculate_order_estimate_no_input():
    """Test _calculate_order_estimate with no quantity or amount."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=None,
        amount_krw=None,
        price=54_000_000.0,
    )

    assert estimate["quantity"] == 0.0
    assert estimate["total_value"] == 0.0


def test_execute_demo_order():
    """Test _execute_demo_order creates valid result."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _execute_demo_order,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    result = _execute_demo_order(request, price=54_000_000.0)

    assert result.success is True
    assert result.order_id.startswith("PAPER-")
    assert result.symbol == "BTC"
    assert result.side == "BUY"
    assert result.executed_qty == 0.1
    assert result.executed_price == 54_000_000.0
    assert result.total_value == 5_400_000.0
    assert result.fee > 0
    assert isinstance(result.timestamp, datetime)


def test_execute_demo_order_with_amount():
    """Test _execute_demo_order with amount_krw (BUY only)."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _execute_demo_order,
    )

    # BUY can use amount_krw
    request = OrderRequest(
        symbol="ETH",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=None,
        amount_krw=1_000_000.0,
    )

    result = _execute_demo_order(request, price=2_500_000.0)

    assert result.success is True
    assert result.executed_qty == 0.4  # 1M / 2.5M


def test_validate_order_request_negative_amount():
    """Test _validate_order_request with negative amount_krw."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=None,
        amount_krw=-1000.0,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Amount" in error


def test_validate_order_request_whitespace_symbol():
    """Test _validate_order_request with whitespace-only symbol."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="   ",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Symbol" in error


def test_validate_order_request_limit_zero_price():
    """Test _validate_order_request for LIMIT order with zero price."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.1,
        price=0.0,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "Price" in error


def test_calculate_order_estimate_custom_fee_rate():
    """Test _calculate_order_estimate with custom fee rate."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=1.0,
        amount_krw=None,
        price=1_000_000.0,
        fee_rate=0.001,  # 0.1%
    )

    assert estimate["fee"] == 1_000.0  # 1M * 0.1%


def test_format_quantity_default():
    """Test _format_quantity for default symbols."""
    from services.dashboard.components.order_panel import _format_quantity

    result = _format_quantity(1.23456789, "SOL")
    assert "1.2346" in result  # 4 decimal places


def test_get_demo_balances_deterministic():
    """Test _get_demo_balances returns consistent values."""
    from services.dashboard.components.order_panel import _get_demo_balances

    balances1 = _get_demo_balances()
    balances2 = _get_demo_balances()

    assert balances1 == balances2


def test_order_result_with_failure():
    """Test OrderResult for failed order."""
    from services.dashboard.components.order_panel import OrderResult

    result = OrderResult(
        success=False,
        order_id="",
        symbol="BTC",
        side="BUY",
        executed_qty=0.0,
        executed_price=0.0,
        total_value=0.0,
        fee=0.0,
        message="Insufficient balance",
        timestamp=datetime.now(),
    )

    assert result.success is False
    assert result.order_id == ""
    assert "Insufficient" in result.message


def test_validate_order_request_sell_requires_quantity():
    """Test SELL orders require quantity (not just amount_krw)."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    # SELL with only amount_krw should fail
    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=None,
        amount_krw=1_000_000.0,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "SELL" in error and "quantity" in error.lower()


def test_validate_order_request_sell_with_quantity_valid():
    """Test SELL orders with quantity are valid."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    request = OrderRequest(
        symbol="ETH",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.5,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is True
    assert error == ""


def test_fee_rate_constant():
    """Test _FEE_RATE constant is defined and used correctly."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _FEE_RATE,
        _calculate_order_estimate,
    )

    # Verify constant value
    assert _FEE_RATE == 0.0005  # 0.05%

    # Verify it's used as default in estimate calculation
    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=1.0,
        amount_krw=None,
        price=1_000_000.0,
    )
    assert estimate["fee"] == 1_000_000.0 * _FEE_RATE


def test_order_request_quantity_optional():
    """Test OrderRequest.quantity is Optional[float]."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
    )

    # quantity can be None (for BUY with amount_krw)
    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        amount_krw=1_000_000.0,
    )

    assert request.quantity is None
    assert request.amount_krw == 1_000_000.0


def test_safe_float_normalizes_price():
    """Test _safe_float handles price provider edge cases."""
    from services.dashboard.components.order_panel import _safe_float

    # Handles None
    assert _safe_float(None) == 0.0

    # Handles inf/nan
    assert _safe_float(float("inf")) == 0.0
    assert _safe_float(float("-inf")) == 0.0
    assert _safe_float(float("nan")) == 0.0

    # Normal values pass through
    assert _safe_float(54_000_000.0) == 54_000_000.0
    assert _safe_float(-100.0) == -100.0  # Negative but finite


def test_calculate_order_estimate_nan_price():
    """Test _calculate_order_estimate handles NaN price."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=0.1,
        amount_krw=None,
        price=float("nan"),
    )

    # NaN price should result in zero values
    assert estimate["quantity"] == 0.0
    assert estimate["total_value"] == 0.0
    assert estimate["fee"] == 0.0
    assert estimate["net_value"] == 0.0


def test_calculate_order_estimate_inf_price():
    """Test _calculate_order_estimate handles Inf price."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="BTC",
        side=OrderSide.BUY,
        quantity=0.1,
        amount_krw=None,
        price=float("inf"),
    )

    # Inf price should result in zero values
    assert estimate["quantity"] == 0.0
    assert estimate["total_value"] == 0.0


def test_calculate_order_estimate_negative_inf_price():
    """Test _calculate_order_estimate handles -Inf price."""
    from services.dashboard.components.order_panel import (
        OrderSide,
        _calculate_order_estimate,
    )

    estimate = _calculate_order_estimate(
        symbol="ETH",
        side=OrderSide.SELL,
        quantity=1.0,
        amount_krw=None,
        price=float("-inf"),
    )

    # -Inf price should result in zero values
    assert estimate["quantity"] == 0.0
    assert estimate["total_value"] == 0.0


def test_validate_order_request_sell_with_amount_krw_blocked():
    """Test SELL orders with amount_krw are explicitly blocked."""
    from services.dashboard.components.order_panel import (
        OrderRequest,
        OrderSide,
        OrderType,
        _validate_order_request,
    )

    # SELL with quantity AND amount_krw should fail
    request = OrderRequest(
        symbol="BTC",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=0.1,
        amount_krw=1_000_000.0,
    )

    is_valid, error = _validate_order_request(request)
    assert is_valid is False
    assert "SELL" in error and "amount" in error.lower()
