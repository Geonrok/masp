"""
Custom Exception Hierarchy for MASP

Provides structured exceptions for better error handling,
debugging, and error recovery throughout the platform.
"""

from __future__ import annotations

from typing import Optional, Dict, Any


class MASPError(Exception):
    """
    Base exception for all MASP errors.

    All custom exceptions should inherit from this class.
    """

    error_code: str = "MASP_ERROR"

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize MASP error.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional context/details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.error_code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.details:
            parts.append(f" Details: {self.details}")
        if self.cause:
            parts.append(f" Caused by: {self.cause}")
        return "".join(parts)


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(MASPError):
    """Base class for configuration-related errors."""

    error_code = "CONFIG_ERROR"


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    error_code = "CONFIG_MISSING"


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    error_code = "CONFIG_INVALID"


class EnvironmentError(ConfigurationError):
    """Environment variable is missing or invalid."""

    error_code = "CONFIG_ENV_ERROR"


# =============================================================================
# Adapter Errors
# =============================================================================


class AdapterError(MASPError):
    """Base class for adapter-related errors."""

    error_code = "ADAPTER_ERROR"


class ConnectionError(AdapterError):
    """Failed to connect to external service."""

    error_code = "ADAPTER_CONNECTION"


class AuthenticationError(AdapterError):
    """Authentication with external service failed."""

    error_code = "ADAPTER_AUTH"


class RateLimitError(AdapterError):
    """Rate limit exceeded for external service."""

    error_code = "ADAPTER_RATE_LIMIT"

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        if retry_after:
            self.details["retry_after_seconds"] = retry_after


class APIError(AdapterError):
    """External API returned an error."""

    error_code = "ADAPTER_API_ERROR"

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body
        if status_code:
            self.details["status_code"] = status_code


class DataUnavailableError(AdapterError):
    """Requested data is not available."""

    error_code = "ADAPTER_DATA_UNAVAILABLE"


# =============================================================================
# Trading/Execution Errors
# =============================================================================


class TradingError(MASPError):
    """Base class for trading-related errors."""

    error_code = "TRADING_ERROR"


class InsufficientFundsError(TradingError):
    """Insufficient funds for the requested operation."""

    error_code = "TRADING_INSUFFICIENT_FUNDS"

    def __init__(
        self,
        message: str,
        *,
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if required is not None:
            self.details["required"] = required
        if available is not None:
            self.details["available"] = available


class OrderRejectedError(TradingError):
    """Order was rejected by the exchange/broker."""

    error_code = "TRADING_ORDER_REJECTED"

    def __init__(
        self,
        message: str,
        *,
        order_id: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if order_id:
            self.details["order_id"] = order_id
        if rejection_reason:
            self.details["rejection_reason"] = rejection_reason


class OrderNotFoundError(TradingError):
    """Order was not found."""

    error_code = "TRADING_ORDER_NOT_FOUND"


class PositionError(TradingError):
    """Error related to position management."""

    error_code = "TRADING_POSITION_ERROR"


class MarketClosedError(TradingError):
    """Market is closed for trading."""

    error_code = "TRADING_MARKET_CLOSED"


# =============================================================================
# Strategy Errors
# =============================================================================


class StrategyError(MASPError):
    """Base class for strategy-related errors."""

    error_code = "STRATEGY_ERROR"


class StrategyNotFoundError(StrategyError):
    """Strategy was not found."""

    error_code = "STRATEGY_NOT_FOUND"


class StrategyExecutionError(StrategyError):
    """Error during strategy execution."""

    error_code = "STRATEGY_EXECUTION"


class InvalidSignalError(StrategyError):
    """Strategy generated an invalid signal."""

    error_code = "STRATEGY_INVALID_SIGNAL"


class InsufficientDataError(StrategyError):
    """Insufficient data for strategy calculation."""

    error_code = "STRATEGY_INSUFFICIENT_DATA"


# =============================================================================
# Risk Management Errors
# =============================================================================


class RiskError(MASPError):
    """Base class for risk management errors."""

    error_code = "RISK_ERROR"


class RiskLimitExceededError(RiskError):
    """Risk limit has been exceeded."""

    error_code = "RISK_LIMIT_EXCEEDED"

    def __init__(
        self,
        message: str,
        *,
        limit_type: Optional[str] = None,
        limit_value: Optional[float] = None,
        current_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        if limit_type:
            self.details["limit_type"] = limit_type
        if limit_value is not None:
            self.details["limit_value"] = limit_value
        if current_value is not None:
            self.details["current_value"] = current_value


class DrawdownExceededError(RiskError):
    """Maximum drawdown limit exceeded."""

    error_code = "RISK_DRAWDOWN_EXCEEDED"


class PositionLimitError(RiskError):
    """Position size limit exceeded."""

    error_code = "RISK_POSITION_LIMIT"


class TradingHaltedError(RiskError):
    """Trading has been halted due to risk limits."""

    error_code = "RISK_TRADING_HALTED"


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(MASPError):
    """Base class for validation errors."""

    error_code = "VALIDATION_ERROR"


class InvalidSymbolError(ValidationError):
    """Invalid trading symbol."""

    error_code = "VALIDATION_INVALID_SYMBOL"


class InvalidQuantityError(ValidationError):
    """Invalid order quantity."""

    error_code = "VALIDATION_INVALID_QUANTITY"


class InvalidPriceError(ValidationError):
    """Invalid price value."""

    error_code = "VALIDATION_INVALID_PRICE"


class InvalidOrderTypeError(ValidationError):
    """Invalid order type."""

    error_code = "VALIDATION_INVALID_ORDER_TYPE"


# =============================================================================
# Database/Storage Errors
# =============================================================================


class StorageError(MASPError):
    """Base class for storage-related errors."""

    error_code = "STORAGE_ERROR"


class DatabaseConnectionError(StorageError):
    """Failed to connect to database."""

    error_code = "STORAGE_DB_CONNECTION"


class DatabaseQueryError(StorageError):
    """Database query failed."""

    error_code = "STORAGE_DB_QUERY"


class CacheError(StorageError):
    """Cache operation failed."""

    error_code = "STORAGE_CACHE"


# =============================================================================
# Network/Communication Errors
# =============================================================================


class NetworkError(MASPError):
    """Base class for network-related errors."""

    error_code = "NETWORK_ERROR"


class TimeoutError(NetworkError):
    """Operation timed out."""

    error_code = "NETWORK_TIMEOUT"


# =============================================================================
# Exchange Errors
# =============================================================================


class ExchangeError(MASPError):
    """Base class for exchange-related errors."""

    error_code = "EXCHANGE_ERROR"


# =============================================================================
# Data Errors
# =============================================================================


class DataError(MASPError):
    """Base class for data-related errors."""

    error_code = "DATA_ERROR"


# =============================================================================
# Execution Errors
# =============================================================================


class ExecutionError(MASPError):
    """Base class for execution-related errors."""

    error_code = "EXECUTION_ERROR"


# =============================================================================
# Notification Errors
# =============================================================================


class NotificationError(MASPError):
    """Base class for notification errors."""

    error_code = "NOTIFICATION_ERROR"


class NotificationDeliveryError(NotificationError):
    """Failed to deliver notification."""

    error_code = "NOTIFICATION_DELIVERY"


class NotificationConfigError(NotificationError):
    """Notification configuration error."""

    error_code = "NOTIFICATION_CONFIG"


# =============================================================================
# Backtest Errors
# =============================================================================


class BacktestError(MASPError):
    """Base class for backtest errors."""

    error_code = "BACKTEST_ERROR"


class InsufficientHistoricalDataError(BacktestError):
    """Not enough historical data for backtest."""

    error_code = "BACKTEST_INSUFFICIENT_DATA"


class BacktestConfigError(BacktestError):
    """Invalid backtest configuration."""

    error_code = "BACKTEST_CONFIG"


# =============================================================================
# Helper Functions
# =============================================================================


def wrap_exception(
    exception: Exception,
    wrapper_class: type = MASPError,
    message: Optional[str] = None,
) -> MASPError:
    """
    Wrap a standard exception in a MASP exception.

    Args:
        exception: The original exception
        wrapper_class: MASP exception class to use
        message: Optional custom message

    Returns:
        Wrapped MASPError
    """
    if isinstance(exception, MASPError):
        return exception

    return wrapper_class(
        message=message or str(exception),
        cause=exception,
    )
