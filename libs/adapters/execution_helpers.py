"""
Execution Adapter Helpers.

Common utilities and mixins for execution adapters to reduce code duplication.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OrderSide(Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class StandardOrderResult:
    """
    Standardized order result across all exchanges.

    This provides a common structure for order results
    that can be used across different exchange adapters.
    """

    order_id: str
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus

    # Quantities
    requested_quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0

    # Prices
    requested_price: Optional[float] = None
    average_fill_price: Optional[float] = None

    # Fees
    fee: float = 0.0
    fee_currency: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    # Additional info
    message: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_partial(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIAL

    @property
    def fill_rate(self) -> float:
        """Get fill rate (0.0 to 1.0)."""
        if self.requested_quantity == 0:
            return 0.0
        return self.filled_quantity / self.requested_quantity

    @property
    def total_value(self) -> float:
        """Get total filled value."""
        if self.average_fill_price is None:
            return 0.0
        return self.filled_quantity * self.average_fill_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "status": self.status.value,
            "requested_quantity": self.requested_quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "requested_price": self.requested_price,
            "average_fill_price": self.average_fill_price,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "message": self.message,
            "fill_rate": self.fill_rate,
            "total_value": self.total_value,
        }


@dataclass
class BalanceInfo:
    """
    Standardized balance information.
    """

    currency: str
    available: float
    locked: float = 0.0
    total: float = 0.0
    exchange: str = ""

    def __post_init__(self):
        if self.total == 0:
            self.total = self.available + self.locked


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retrying API calls with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential: Use exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Tuple of exceptions to retry on

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def place_order(self, symbol, side, quantity):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            "[Retry] %s failed after %d attempts: %s",
                            func.__name__,
                            max_retries + 1,
                            e,
                        )
                        raise

                    # Calculate delay
                    if exponential:
                        delay = base_delay * (2**attempt)
                    else:
                        delay = base_delay

                    delay = min(delay, max_delay)

                    if jitter:
                        delay *= 0.5 + random.random()

                    logger.warning(
                        "[Retry] %s attempt %d failed: %s. Retrying in %.1fs",
                        func.__name__,
                        attempt + 1,
                        e,
                        delay,
                    )
                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


def rate_limit(
    calls_per_second: float = 10.0,
    calls_per_minute: Optional[float] = None,
) -> Callable:
    """
    Decorator for rate limiting API calls.

    Args:
        calls_per_second: Maximum calls per second
        calls_per_minute: Maximum calls per minute (optional)

    Example:
        @rate_limit(calls_per_second=5)
        def get_ticker(self, symbol):
            ...
    """
    min_interval = 1.0 / calls_per_second

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        last_call_time = [0.0]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_time = time.time()
            elapsed = current_time - last_call_time[0]

            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)

            last_call_time[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_order_params(
    symbol: str,
    side: Union[str, OrderSide],
    quantity: float,
    price: Optional[float] = None,
    order_type: Union[str, OrderType] = OrderType.MARKET,
) -> Dict[str, Any]:
    """
    Validate and normalize order parameters.

    Args:
        symbol: Trading pair symbol
        side: Order side (BUY/SELL)
        quantity: Order quantity
        price: Order price (required for LIMIT orders)
        order_type: Order type (MARKET/LIMIT)

    Returns:
        Validated parameters dictionary

    Raises:
        ValueError: If parameters are invalid
    """
    # Normalize side
    if isinstance(side, str):
        try:
            side = OrderSide(side.upper())
        except ValueError:
            raise ValueError(f"Invalid side: {side}. Must be BUY or SELL.")

    # Normalize order type
    if isinstance(order_type, str):
        try:
            order_type = OrderType(order_type.upper())
        except ValueError:
            raise ValueError(f"Invalid order type: {order_type}")

    # Validate quantity
    if quantity <= 0:
        raise ValueError(f"Quantity must be positive: {quantity}")

    # Validate price for limit orders
    if order_type == OrderType.LIMIT and (price is None or price <= 0):
        raise ValueError("Price is required for LIMIT orders and must be positive")

    # Validate symbol format
    if "/" not in symbol and "_" not in symbol:
        logger.warning(
            "Symbol '%s' doesn't contain separator. May need normalization.", symbol
        )

    return {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "price": price,
        "order_type": order_type,
    }


def format_quantity(
    quantity: float,
    precision: int = 8,
    min_quantity: float = 0.0,
) -> float:
    """
    Format quantity to exchange precision.

    Args:
        quantity: Raw quantity
        precision: Decimal precision
        min_quantity: Minimum allowed quantity

    Returns:
        Formatted quantity
    """
    formatted = round(quantity, precision)
    return max(formatted, min_quantity)


def format_price(
    price: float,
    tick_size: Optional[float] = None,
    precision: int = 8,
) -> float:
    """
    Format price to exchange tick size.

    Args:
        price: Raw price
        tick_size: Price tick size (e.g., 0.01 for KRW)
        precision: Decimal precision if no tick size

    Returns:
        Formatted price
    """
    if tick_size and tick_size > 0:
        return round(price / tick_size) * tick_size
    return round(price, precision)


def calculate_slippage(
    requested_price: float,
    executed_price: float,
    side: OrderSide,
) -> float:
    """
    Calculate slippage percentage.

    Args:
        requested_price: Original requested price
        executed_price: Actual execution price
        side: Order side

    Returns:
        Slippage percentage (positive = unfavorable)
    """
    if requested_price == 0:
        return 0.0

    if side == OrderSide.BUY:
        # For buys, higher price is worse
        slippage = (executed_price - requested_price) / requested_price
    else:
        # For sells, lower price is worse
        slippage = (requested_price - executed_price) / requested_price

    return slippage * 100  # Return as percentage


class KillSwitchMixin:
    """
    Mixin for kill switch functionality.

    Provides methods for checking and enforcing kill switch status
    before executing orders.
    """

    def _is_kill_switch_active(self) -> bool:
        """
        Check if kill switch is active.

        Subclasses should override this to implement
        their kill switch checking logic.
        """
        if hasattr(self, "_config") and hasattr(self._config, "is_kill_switch_active"):
            return self._config.is_kill_switch_active()
        return False

    def _check_kill_switch(self, operation: str = "order") -> None:
        """
        Check kill switch and raise if active.

        Args:
            operation: Operation being attempted

        Raises:
            RuntimeError: If kill switch is active
        """
        if self._is_kill_switch_active():
            raise RuntimeError(f"Kill switch is active. Cannot execute {operation}.")


class OrderValidationMixin:
    """
    Mixin for order validation functionality.

    Provides methods for validating orders against
    position limits and other constraints.
    """

    def _validate_order(
        self,
        symbol: str,
        side: Union[str, OrderSide],
        quantity: float,
        price: Optional[float] = None,
        balance: Optional[float] = None,
        total_equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate order against constraints.

        Args:
            symbol: Trading pair
            side: Order side
            quantity: Order quantity
            price: Order price
            balance: Available balance
            total_equity: Total portfolio equity

        Returns:
            Dict with 'valid' bool and optional 'reason' string
        """
        result = {"valid": True, "warnings": []}

        # Check kill switch
        if hasattr(self, "_is_kill_switch_active") and self._is_kill_switch_active():
            return {"valid": False, "reason": "Kill switch active"}

        # Check position limits if validator is available
        if hasattr(self, "_order_validator") and self._order_validator:
            validation = self._order_validator.validate(
                symbol=symbol,
                side=side if isinstance(side, str) else side.value,
                quantity=quantity,
                price=price or 0,
                balance=balance or 0,
                total_equity=total_equity or balance or 0,
            )
            if not validation.valid:
                return {"valid": False, "reason": validation.reason}
            if validation.warnings:
                result["warnings"] = validation.warnings

        return result


def log_order_execution(
    exchange: str,
    symbol: str,
    side: Union[str, OrderSide],
    quantity: float,
    price: Optional[float] = None,
    order_type: str = "MARKET",
) -> Callable:
    """
    Decorator for logging order execution.

    Example:
        @log_order_execution("upbit", "BTC/KRW", "BUY", 0.001)
        def place_order(...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            side_str = side if isinstance(side, str) else side.value

            logger.info(
                "[%s] Executing %s order: %s %s %.8f @ %s",
                exchange,
                order_type,
                side_str,
                symbol,
                quantity,
                price or "MARKET",
            )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.time() - start_time) * 1000

                logger.info(
                    "[%s] Order completed in %.0fms",
                    exchange,
                    elapsed,
                )
                return result
            except Exception as e:
                elapsed = (time.time() - start_time) * 1000
                logger.error(
                    "[%s] Order failed after %.0fms: %s",
                    exchange,
                    elapsed,
                    e,
                )
                raise

        return wrapper

    return decorator
