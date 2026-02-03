"""
Exception Handling Utilities

Centralized exception handling patterns for consistent
error handling across the MASP platform.
"""

from __future__ import annotations

import functools
import inspect
import logging
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Optional, Type, TypeVar, Union, Tuple

from libs.core.exceptions import (
    MASPError,
    AdapterError,
    ConfigurationError,
    DataError,
    NetworkError,
    ExchangeError,
    ValidationError,
    ExecutionError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# Exception mapping for common error types
# Note: More specific types must come before their parent types
# FileNotFoundError/PermissionError must be checked before OSError
EXCEPTION_MAP: list[tuple[Type[Exception], Type[MASPError]]] = [
    (FileNotFoundError, ConfigurationError),
    (PermissionError, ConfigurationError),
    (ConnectionError, NetworkError),
    (TimeoutError, NetworkError),
    (OSError, NetworkError),
    (ValueError, ValidationError),
    (KeyError, DataError),
    (TypeError, DataError),
]


def classify_exception(exc: Exception) -> Type[MASPError]:
    """
    Classify a generic exception into a MASP exception type.

    Args:
        exc: The exception to classify

    Returns:
        Appropriate MASPError subclass
    """
    for exc_type, masp_type in EXCEPTION_MAP:
        if isinstance(exc, exc_type):
            return masp_type

    # Check for common patterns in exception messages
    exc_str = str(exc).lower()
    if any(x in exc_str for x in ["timeout", "timed out"]):
        return NetworkError
    if any(x in exc_str for x in ["connect", "connection", "network"]):
        return NetworkError
    if any(x in exc_str for x in ["api", "exchange", "rate limit"]):
        return ExchangeError
    # Check validation patterns BEFORE config patterns
    # ("invalid parameter" should be ValidationError, not ConfigurationError)
    if any(x in exc_str for x in ["invalid", "validation", "required"]):
        return ValidationError
    if any(x in exc_str for x in ["config", "setting"]):
        return ConfigurationError

    return AdapterError  # Default to adapter error


def wrap_exception(
    exc: Exception,
    context: str = "",
    error_code: Optional[str] = None,
) -> MASPError:
    """
    Wrap a generic exception in an appropriate MASPError.

    Args:
        exc: Original exception
        context: Additional context about where the error occurred
        error_code: Optional error code

    Returns:
        MASPError instance
    """
    if isinstance(exc, MASPError):
        return exc

    masp_class = classify_exception(exc)
    message = f"{context}: {exc}" if context else str(exc)

    return masp_class(
        message=message,
        error_code=error_code or masp_class.__name__.upper(),
        cause=exc,
    )


def safe_execute(
    func: Callable[..., T],
    *args,
    default: T = None,
    context: str = "",
    log_level: int = logging.WARNING,
    reraise: bool = False,
    **kwargs,
) -> T:
    """
    Safely execute a function with proper exception handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on failure
        context: Context for error message
        log_level: Logging level for errors
        reraise: Whether to re-raise as MASPError
        **kwargs: Keyword arguments

    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except MASPError:
        raise
    except Exception as exc:
        ctx = context or f"{func.__module__}.{func.__name__}"
        logger.log(
            log_level,
            f"[{ctx}] Error: {exc}",
            exc_info=log_level <= logging.DEBUG,
        )

        if reraise:
            raise wrap_exception(exc, ctx)

        return default


def handle_exceptions(
    default: Any = None,
    context: str = "",
    log_level: int = logging.WARNING,
    reraise: bool = False,
    specific_handlers: Optional[dict[Type[Exception], Callable]] = None,
) -> Callable[[F], F]:
    """
    Decorator for consistent exception handling.

    Args:
        default: Default return value on exception
        context: Context string for error messages
        log_level: Logging level for caught exceptions
        reraise: Re-raise exceptions as MASPError
        specific_handlers: Dict mapping exception types to handlers

    Returns:
        Decorated function

    Example:
        @handle_exceptions(default=[], context="fetch_orders")
        def fetch_orders(self):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MASPError:
                raise
            except Exception as exc:
                ctx = context or f"{func.__module__}.{func.__name__}"

                # Check for specific handlers
                if specific_handlers:
                    for exc_type, handler in specific_handlers.items():
                        if isinstance(exc, exc_type):
                            return handler(exc, *args, **kwargs)

                logger.log(
                    log_level,
                    f"[{ctx}] {type(exc).__name__}: {exc}",
                    exc_info=log_level <= logging.DEBUG,
                )

                if reraise:
                    raise wrap_exception(exc, ctx)

                return default

        return wrapper  # type: ignore

    return decorator


def handle_adapter_exceptions(
    operation: str,
    exchange: str = "unknown",
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Specialized decorator for exchange adapter methods.

    Args:
        operation: Operation name (e.g., "get_quote", "place_order")
        exchange: Exchange name
        reraise: Whether to re-raise as ExchangeError

    Example:
        @handle_adapter_exceptions("get_quote", "binance")
        async def get_quote(self, symbol: str):
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except MASPError:
                raise
            except Exception as exc:
                logger.error(
                    f"[{exchange}:{operation}] Error: {exc}",
                    extra={
                        "exchange": exchange,
                        "operation": operation,
                        "error_type": type(exc).__name__,
                    },
                )
                if reraise:
                    raise ExchangeError(
                        message=f"{exchange} {operation} failed: {exc}",
                        error_code=f"{exchange.upper()}_{operation.upper()}_ERROR",
                        cause=exc,
                        details={"exchange": exchange, "operation": operation},
                    )
                return None

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MASPError:
                raise
            except Exception as exc:
                logger.error(
                    f"[{exchange}:{operation}] Error: {exc}",
                    extra={
                        "exchange": exchange,
                        "operation": operation,
                        "error_type": type(exc).__name__,
                    },
                )
                if reraise:
                    raise ExchangeError(
                        message=f"{exchange} {operation} failed: {exc}",
                        error_code=f"{exchange.upper()}_{operation.upper()}_ERROR",
                        cause=exc,
                        details={"exchange": exchange, "operation": operation},
                    )
                return None

        import asyncio

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


@contextmanager
def exception_context(
    context: str,
    log_level: int = logging.ERROR,
    reraise: bool = True,
    default: Any = None,
):
    """
    Context manager for exception handling.

    Args:
        context: Context description
        log_level: Logging level for errors
        reraise: Whether to re-raise as MASPError
        default: Default value (yields this on error if not reraising)

    Example:
        with exception_context("loading config", reraise=False):
            config = load_config()
    """
    try:
        yield
    except MASPError:
        raise
    except Exception as exc:
        logger.log(
            log_level,
            f"[{context}] {type(exc).__name__}: {exc}",
            exc_info=log_level <= logging.DEBUG,
        )
        if reraise:
            raise wrap_exception(exc, context)


def log_and_suppress(
    exc: Exception,
    context: str,
    log_level: int = logging.WARNING,
) -> None:
    """
    Log an exception and suppress it.

    Use this instead of bare `except: pass` patterns.

    Args:
        exc: The exception
        context: Context description
        log_level: Logging level
    """
    logger.log(
        log_level,
        f"[{context}] Suppressed {type(exc).__name__}: {exc}",
    )


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    context: str = "",
) -> Callable[[F], F]:
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential: Use exponential backoff
        retryable_exceptions: Tuple of exception types to retry
        context: Context for logging

    Example:
        @retry_with_backoff(max_retries=3, context="api_call")
        def call_api():
            ...
    """
    import asyncio
    import random
    import time

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break

                    delay = base_delay * (2**attempt if exponential else 1)
                    delay = min(delay, max_delay)
                    delay *= 0.5 + random.random()  # Jitter

                    ctx = context or func.__name__
                    logger.warning(
                        f"[{ctx}] Retry {attempt + 1}/{max_retries} "
                        f"after {delay:.1f}s: {exc}"
                    )
                    await asyncio.sleep(delay)
                except Exception as exc:
                    # Non-retryable exception, wrap and re-raise immediately
                    raise wrap_exception(exc, context or func.__name__)

            raise wrap_exception(last_exc, context or func.__name__)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        break

                    delay = base_delay * (2**attempt if exponential else 1)
                    delay = min(delay, max_delay)
                    delay *= 0.5 + random.random()  # Jitter

                    ctx = context or func.__name__
                    logger.warning(
                        f"[{ctx}] Retry {attempt + 1}/{max_retries} "
                        f"after {delay:.1f}s: {exc}"
                    )
                    time.sleep(delay)
                except Exception as exc:
                    # Non-retryable exception, wrap and re-raise immediately
                    raise wrap_exception(exc, context or func.__name__)

            raise wrap_exception(last_exc, context or func.__name__)

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class ExceptionAggregator:
    """
    Aggregates multiple exceptions for batch operations.

    Example:
        agg = ExceptionAggregator("batch_process")
        for item in items:
            with agg.catch():
                process(item)
        agg.raise_if_any()
    """

    def __init__(self, context: str):
        self.context = context
        self.exceptions: list[tuple[int, Exception]] = []
        self._index = 0

    @contextmanager
    def catch(self, index: Optional[int] = None):
        """Catch and store exception."""
        idx = index if index is not None else self._index
        self._index += 1
        try:
            yield
        except Exception as exc:
            self.exceptions.append((idx, exc))
            logger.warning(f"[{self.context}] Item {idx} failed: {exc}")

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.exceptions) > 0

    @property
    def error_count(self) -> int:
        """Number of errors."""
        return len(self.exceptions)

    def raise_if_any(self, threshold: float = 1.0):
        """
        Raise aggregated error if failure rate exceeds threshold.

        Args:
            threshold: Failure rate threshold (0.0-1.0)
        """
        if not self.exceptions:
            return

        failure_rate = len(self.exceptions) / max(self._index, 1)
        if failure_rate >= threshold:
            errors_str = "; ".join(
                f"[{idx}] {type(e).__name__}: {e}" for idx, e in self.exceptions[:5]
            )
            raise AdapterError(
                message=f"{self.context}: {len(self.exceptions)} errors: {errors_str}",
                error_code="BATCH_ERROR",
                details={
                    "total_items": self._index,
                    "failed_items": len(self.exceptions),
                    "failure_rate": failure_rate,
                },
            )

    def get_summary(self) -> dict:
        """Get error summary."""
        return {
            "context": self.context,
            "total": self._index,
            "errors": len(self.exceptions),
            "success_rate": 1 - (len(self.exceptions) / max(self._index, 1)),
        }
