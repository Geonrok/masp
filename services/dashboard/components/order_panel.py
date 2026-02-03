"""Manual order panel component for Paper trading mode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

# Session state key prefix
_KEY_PREFIX = "order_panel."

# Fee rate constant (0.05%)
_FEE_RATE = 0.0005


class OrderSide(str, Enum):
    """Order side enumeration."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class OrderRequest:
    """Order request data structure."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Optional[float] = (
        None  # Required for SELL, optional for BUY with amount_krw
    )
    price: Optional[float] = None  # Required for LIMIT orders
    amount_krw: Optional[float] = None  # Alternative to quantity (BUY only)


@dataclass
class OrderResult:
    """Order execution result."""

    success: bool
    order_id: str
    symbol: str
    side: str
    executed_qty: float
    executed_price: float
    total_value: float
    fee: float
    message: str
    timestamp: datetime


def _key(name: str) -> str:
    """Generate namespaced session state key."""
    return f"{_KEY_PREFIX}{name}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default for invalid values."""
    if value is None:
        return default
    try:
        result = float(value)
        if not _is_finite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def _is_finite(value: float) -> bool:
    """Check if value is finite (not inf or nan)."""
    return math.isfinite(value)


def _validate_order_request(request: OrderRequest) -> Tuple[bool, str]:
    """Validate order request parameters.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not request.symbol or not request.symbol.strip():
        return False, "Symbol is required"

    if request.quantity is not None and request.quantity <= 0:
        return False, "Quantity must be positive"

    if request.amount_krw is not None and request.amount_krw <= 0:
        return False, "Amount (KRW) must be positive"

    # SELL requires quantity (most exchanges don't support amount-based sell)
    if request.side == OrderSide.SELL:
        if request.quantity is None or request.quantity <= 0:
            return False, "SELL orders require quantity"
        if request.amount_krw is not None:
            return False, "SELL orders do not support amount (KRW)"

    # BUY can use either quantity or amount_krw
    if request.side == OrderSide.BUY:
        if request.quantity is None and request.amount_krw is None:
            return False, "Either quantity or amount (KRW) is required"

    if request.order_type == OrderType.LIMIT:
        if request.price is None or request.price <= 0:
            return False, "Price is required for LIMIT orders"

    return True, ""


def _get_demo_symbols() -> List[str]:
    """Get list of available symbols for demo mode."""
    return ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "AVAX", "DOT"]


def _get_demo_prices() -> Dict[str, float]:
    """Get demo prices for symbols (deterministic)."""
    return {
        "BTC": 54_000_000.0,
        "ETH": 2_900_000.0,
        "XRP": 820.0,
        "SOL": 160_000.0,
        "DOGE": 180.0,
        "ADA": 650.0,
        "AVAX": 45_000.0,
        "DOT": 9_500.0,
    }


def _get_demo_balances() -> Dict[str, Dict[str, float]]:
    """Get demo balances for paper trading."""
    return {
        "KRW": {"available": 10_000_000.0, "locked": 0.0},
        "BTC": {"available": 0.5, "locked": 0.0},
        "ETH": {"available": 2.0, "locked": 0.0},
        "XRP": {"available": 5000.0, "locked": 0.0},
        "SOL": {"available": 10.0, "locked": 0.0},
    }


def _format_krw(value: float) -> str:
    """Format value as KRW currency."""
    if value >= 1_000_000:
        return f"{value:,.0f} KRW"
    elif value >= 1000:
        return f"{value:,.0f} KRW"
    else:
        return f"{value:,.2f} KRW"


def _format_quantity(value: float, symbol: str) -> str:
    """Format quantity with appropriate decimal places."""
    if symbol in ("BTC", "ETH"):
        return f"{value:.8f}"
    elif symbol in ("XRP", "DOGE", "ADA"):
        return f"{value:.2f}"
    else:
        return f"{value:.4f}"


def _calculate_order_estimate(
    symbol: str,
    side: OrderSide,
    quantity: Optional[float],
    amount_krw: Optional[float],
    price: float,
    fee_rate: float = _FEE_RATE,
) -> Dict[str, float]:
    """Calculate order estimate including fees.

    Args:
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        quantity: Order quantity (optional)
        amount_krw: Order amount in KRW (optional)
        price: Current or limit price
        fee_rate: Fee rate (default 0.05%)

    Returns:
        Dict with quantity, total_value, fee, net_value
    """
    # Normalize price to handle NaN/Inf
    safe_price = _safe_float(price, default=0.0)
    if safe_price <= 0:
        return {"quantity": 0.0, "total_value": 0.0, "fee": 0.0, "net_value": 0.0}

    if quantity is not None and quantity > 0:
        calc_qty = quantity
        total_value = calc_qty * safe_price
    elif amount_krw is not None and amount_krw > 0:
        total_value = amount_krw
        calc_qty = amount_krw / safe_price
    else:
        return {"quantity": 0.0, "total_value": 0.0, "fee": 0.0, "net_value": 0.0}

    fee = total_value * fee_rate

    if side == OrderSide.BUY:
        net_value = total_value + fee  # Total cost
    else:
        net_value = total_value - fee  # Total proceeds

    return {
        "quantity": calc_qty,
        "total_value": total_value,
        "fee": fee,
        "net_value": net_value,
    }


def _execute_demo_order(request: OrderRequest, price: float) -> OrderResult:
    """Execute order in demo mode (no actual execution)."""
    estimate = _calculate_order_estimate(
        symbol=request.symbol,
        side=request.side,
        quantity=request.quantity,
        amount_krw=request.amount_krw,
        price=price,
    )

    order_id = f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return OrderResult(
        success=True,
        order_id=order_id,
        symbol=request.symbol,
        side=request.side.value,
        executed_qty=estimate["quantity"],
        executed_price=price,
        total_value=estimate["total_value"],
        fee=estimate["fee"],
        message=f"Demo order executed: {request.side.value} {estimate['quantity']:.8f} {request.symbol}",
        timestamp=datetime.now(),
    )


def _get_order_history() -> List[Dict[str, Any]]:
    """Get order history from session state."""
    return st.session_state.get(_key("order_history"), [])


def _add_order_to_history(result: OrderResult) -> None:
    """Add order result to history in session state."""
    history = _get_order_history()
    history.insert(
        0,
        {
            "order_id": result.order_id,
            "timestamp": result.timestamp,
            "symbol": result.symbol,
            "side": result.side,
            "quantity": result.executed_qty,
            "price": result.executed_price,
            "total": result.total_value,
            "fee": result.fee,
            "status": "FILLED" if result.success else "FAILED",
        },
    )
    # Keep last 50 orders
    st.session_state[_key("order_history")] = history[:50]


def _clear_order_form() -> None:
    """Clear order form inputs."""
    st.session_state[_key("quantity_input")] = 0.0
    st.session_state[_key("amount_input")] = 0.0
    st.session_state[_key("price_input")] = 0.0


def render_order_panel(
    execution_adapter: Optional[Any] = None,
    price_provider: Optional[Callable[[str], float]] = None,
    balance_provider: Optional[Callable[[], Dict[str, Dict[str, float]]]] = None,
) -> None:
    """Render manual order panel for Paper trading mode.

    Args:
        execution_adapter: Paper execution adapter for order submission
        price_provider: Function to get current price for a symbol
        balance_provider: Function to get current balances
    """
    st.subheader("수동 주문 (모의 거래)")

    # Determine if we're in demo mode
    is_demo = execution_adapter is None

    if is_demo:
        st.caption("데모 모드 - 주문이 시뮬레이션됩니다")
        symbols = _get_demo_symbols()
        prices = _get_demo_prices()
        balances = _get_demo_balances()
    else:
        symbols = _get_demo_symbols()  # Use same symbols
        prices = _get_demo_prices() if price_provider is None else {}
        balances = (
            balance_provider() if balance_provider is not None else _get_demo_balances()
        )

    # Balance display
    st.markdown("**사용 가능 잔고**")
    bal_cols = st.columns(4)
    display_assets = ["KRW", "BTC", "ETH", "XRP"]
    for i, asset in enumerate(display_assets):
        with bal_cols[i]:
            bal = balances.get(asset, {}).get("available", 0.0)
            if asset == "KRW":
                st.metric(asset, _format_krw(bal))
            else:
                st.metric(asset, _format_quantity(bal, asset))

    st.divider()

    # Order form
    col_left, col_right = st.columns(2)

    with col_left:
        # Symbol selection
        selected_symbol = st.selectbox(
            "종목",
            options=symbols,
            key=_key("symbol"),
        )

        # Get current price (with defensive normalization for all paths)
        if price_provider is not None:
            raw_price = price_provider(selected_symbol)
            current_price = _safe_float(raw_price, default=0.0)
        else:
            current_price = _safe_float(prices.get(selected_symbol, 0.0), default=0.0)

        # Validate price
        price_invalid = current_price <= 0
        if price_invalid:
            st.warning("가격 정보 없음 - 주문 비활성화")
        else:
            st.caption(f"현재가: {_format_krw(current_price)}")

        # Order side
        side = st.radio(
            "매수/매도",
            options=[OrderSide.BUY.value, OrderSide.SELL.value],
            horizontal=True,
            key=_key("side"),
        )

        # Order type (LIMIT only available in demo mode)
        if is_demo:
            order_type = st.radio(
                "주문 유형",
                options=[OrderType.MARKET.value, OrderType.LIMIT.value],
                horizontal=True,
                key=_key("order_type"),
            )
        else:
            # Non-demo: LIMIT not supported by most adapters
            order_type = OrderType.MARKET.value
            st.caption("주문 유형: 시장가 (지정가 미지원)")

    with col_right:
        # Quantity input
        quantity = st.number_input(
            f"수량 ({selected_symbol})",
            min_value=0.0,
            step=0.00000001 if selected_symbol in ("BTC", "ETH") else 1.0,
            format="%.8f" if selected_symbol in ("BTC", "ETH") else "%.4f",
            key=_key("quantity_input"),
        )

        # Amount input (alternative, BUY only)
        is_sell = side == OrderSide.SELL.value
        amount_krw = st.number_input(
            "금액 (KRW)" + (" (매수만)" if is_sell else ""),
            min_value=0.0,
            step=10000.0,
            format="%.0f",
            key=_key("amount_input"),
            disabled=is_sell,
        )

        # Price input for LIMIT orders
        if order_type == OrderType.LIMIT.value:
            # Safe default value and step for LIMIT price input
            default_limit_price = current_price if not price_invalid else 0.0
            price_step = (
                1000.0
                if current_price > 100000
                else 10.0 if current_price > 0 else 1000.0
            )
            limit_price = st.number_input(
                "지정가 (KRW)",
                min_value=0.0,
                value=default_limit_price,
                step=price_step,
                format="%.0f",
                key=_key("price_input"),
            )
        else:
            limit_price = current_price

    # Order estimate
    st.divider()
    st.markdown("**주문 예상**")

    # Normalize order_price
    raw_order_price = (
        limit_price if order_type == OrderType.LIMIT.value else current_price
    )
    order_price = _safe_float(raw_order_price, default=0.0)

    qty_for_estimate = quantity if quantity > 0 else None
    # SELL: force amount_krw to None (not supported by adapters)
    # BUY: if quantity is set, ignore amount_krw (single parameter only)
    if is_sell:
        amt_for_estimate = None
    elif qty_for_estimate is not None:
        amt_for_estimate = None  # quantity takes priority
    else:
        amt_for_estimate = amount_krw if amount_krw > 0 else None

    estimate = _calculate_order_estimate(
        symbol=selected_symbol,
        side=OrderSide(side),
        quantity=qty_for_estimate,
        amount_krw=amt_for_estimate,
        price=order_price,
    )

    est_cols = st.columns(4)
    with est_cols[0]:
        st.metric("수량", _format_quantity(estimate["quantity"], selected_symbol))
    with est_cols[1]:
        st.metric("총 금액", _format_krw(estimate["total_value"]))
    with est_cols[2]:
        fee_pct = _FEE_RATE * 100
        st.metric(f"수수료 ({fee_pct:.2f}%)", _format_krw(estimate["fee"]))
    with est_cols[3]:
        label = "총 비용" if side == OrderSide.BUY.value else "순 수익"
        st.metric(label, _format_krw(estimate["net_value"]))

    # Submit button
    st.divider()
    btn_cols = st.columns([3, 1, 1])

    with btn_cols[1]:
        if st.button("초기화", key=_key("clear_btn")):
            _clear_order_form()
            st.rerun()

    with btn_cols[2]:
        # Disable submit if no quantity or price is invalid
        # For MARKET orders, current_price must be valid
        # For LIMIT orders, limit_price must be valid (user can still input)
        order_price_invalid = (
            order_type == OrderType.MARKET.value and price_invalid
        ) or (order_type == OrderType.LIMIT.value and limit_price <= 0)
        submit_disabled = estimate["quantity"] <= 0 or order_price_invalid

        if st.button(
            f"{side} {selected_symbol}",
            type="primary",
            disabled=submit_disabled,
            key=_key("submit_btn"),
        ):
            request = OrderRequest(
                symbol=selected_symbol,
                side=OrderSide(side),
                order_type=OrderType(order_type),
                quantity=qty_for_estimate,
                price=limit_price if order_type == OrderType.LIMIT.value else None,
                amount_krw=amt_for_estimate,
            )

            is_valid, error_msg = _validate_order_request(request)

            if not is_valid:
                st.error(error_msg)
            else:
                if is_demo or execution_adapter is None:
                    result = _execute_demo_order(request, order_price)
                else:
                    # Use actual execution adapter
                    try:
                        exec_result = execution_adapter.place_order(
                            symbol=request.symbol,
                            side=request.side.value.lower(),
                            units=request.quantity,
                            amount_krw=request.amount_krw,
                        )
                        result = OrderResult(
                            success=exec_result.get("success", False),
                            order_id=exec_result.get("order_id", ""),
                            symbol=request.symbol,
                            side=request.side.value,
                            executed_qty=exec_result.get("executed_qty", 0.0),
                            executed_price=exec_result.get("executed_price", 0.0),
                            total_value=exec_result.get("total_value", 0.0),
                            fee=exec_result.get("fee", 0.0),
                            message=exec_result.get("message", ""),
                            timestamp=datetime.now(),
                        )
                    except Exception as e:
                        result = OrderResult(
                            success=False,
                            order_id="",
                            symbol=request.symbol,
                            side=request.side.value,
                            executed_qty=0.0,
                            executed_price=0.0,
                            total_value=0.0,
                            fee=0.0,
                            message=f"Order failed: {str(e)}",
                            timestamp=datetime.now(),
                        )

                _add_order_to_history(result)

                if result.success:
                    st.success(result.message)
                else:
                    st.error(result.message)

    # Order history
    st.divider()
    st.markdown("**최근 주문 (현재 세션)**")

    history = _get_order_history()

    if not history:
        st.info("이 세션에서 주문한 내역이 없습니다.")
    else:
        history_data = [
            {
                "시간": h["timestamp"].strftime("%H:%M:%S"),
                "종목": h["symbol"],
                "구분": h["side"],
                "수량": _format_quantity(h["quantity"], h["symbol"]),
                "가격": _format_krw(h["price"]),
                "총액": _format_krw(h["total"]),
                "상태": h["status"],
            }
            for h in history[:10]  # Show last 10
        ]
        st.dataframe(history_data, use_container_width=True, hide_index=True)
