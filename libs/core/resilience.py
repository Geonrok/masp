"""
Resilience Patterns

Circuit breaker, retry policies, and fallback mechanisms
for building resilient adapters and services.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Optional, Tuple, Type, TypeVar

from libs.core.exceptions import MASPError, NetworkError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential: Use exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Exception types that should trigger retry
        non_retryable_exceptions: Exception types that should not retry
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential: bool = True
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.exponential:
            delay = self.base_delay * (2**attempt)
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            delay *= 0.5 + random.random()

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Check if exception should trigger retry."""
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        return isinstance(exception, self.retryable_exceptions)


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Successes needed to close from half-open
        timeout: Seconds before attempting recovery (half-open)
        half_open_max_calls: Max calls to allow in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Example:
        cb = CircuitBreaker("payment_service")

        @cb.protect
        async def call_payment_api():
            return await http_client.post("/charge")
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.timeout:
                    self._transition_to_half_open()
        return self._state

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.warning(
            f"[CircuitBreaker:{self.name}] OPEN - failures: {self._failure_count}"
        )
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN - testing recovery")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info(f"[CircuitBreaker:{self.name}] CLOSED - recovered")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state  # This may update state based on timeout

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # Half-open: allow limited calls
        if self._half_open_calls < self.config.half_open_max_calls:
            self._half_open_calls += 1
            return True

        return False

    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with circuit breaker."""

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not self.allow_request():
                raise NetworkError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    error_code="CIRCUIT_OPEN",
                )
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not self.allow_request():
                raise NetworkError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    error_code="CIRCUIT_OPEN",
                )
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    _instance: Optional["CircuitBreakerRegistry"] = None
    _breakers: Dict[str, CircuitBreaker] = {}

    @classmethod
    def get_instance(cls) -> "CircuitBreakerRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        for cb in self._breakers.values():
            cb._transition_to_closed()


def with_retry(
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for adding retry behavior to functions.

    Args:
        policy: Retry policy (uses defaults if None)
        on_retry: Callback called before each retry

    Example:
        @with_retry(RetryPolicy(max_retries=3))
        async def fetch_data():
            return await api.get("/data")
    """
    policy = policy or RetryPolicy()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e):
                        raise

                    if attempt == policy.max_retries:
                        break

                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    await asyncio.sleep(delay)

            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e):
                        raise

                    if attempt == policy.max_retries:
                        break

                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"[Retry] {func.__name__} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(delay)

            raise last_exception

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def with_fallback(
    fallback_value: T = None, fallback_fn: Optional[Callable[..., T]] = None
):
    """
    Decorator for adding fallback behavior.

    Args:
        fallback_value: Value to return on failure
        fallback_fn: Function to call on failure (receives original args)

    Example:
        @with_fallback(fallback_value=[])
        def get_items():
            return api.fetch_items()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[Fallback] {func.__name__} failed: {e}")
                if fallback_fn:
                    return fallback_fn(*args, **kwargs)
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[Fallback] {func.__name__} failed: {e}")
                if fallback_fn:
                    return fallback_fn(*args, **kwargs)
                return fallback_value

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def with_timeout(seconds: float):
    """
    Decorator for adding timeout to async functions.

    Args:
        seconds: Timeout in seconds

    Example:
        @with_timeout(5.0)
        async def slow_operation():
            await do_something_slow()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise NetworkError(
                    f"{func.__name__} timed out after {seconds}s",
                    error_code="TIMEOUT",
                )

        return wrapper

    return decorator


class Bulkhead:
    """
    Bulkhead pattern for limiting concurrent executions.

    Prevents resource exhaustion by limiting parallel calls.

    Example:
        bulkhead = Bulkhead("database", max_concurrent=10)

        async with bulkhead.acquire():
            await db.query(...)
    """

    def __init__(self, name: str, max_concurrent: int = 10):
        self.name = name
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active = 0

    @property
    def available(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active

    async def acquire(self):
        """Async context manager for acquiring a slot."""
        return _BulkheadContext(self)

    def protect(self, func: Callable) -> Callable:
        """Decorator to protect a function with bulkhead."""

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async with self.acquire():
                return await func(*args, **kwargs)

        return wrapper


class _BulkheadContext:
    """Context manager for bulkhead acquisition."""

    def __init__(self, bulkhead: Bulkhead):
        self._bulkhead = bulkhead

    async def __aenter__(self):
        await self._bulkhead._semaphore.acquire()
        self._bulkhead._active += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._bulkhead._active -= 1
        self._bulkhead._semaphore.release()


class ResilienceBuilder:
    """
    Builder for composing resilience patterns.

    Example:
        resilience = (
            ResilienceBuilder("api_call")
            .with_retry(RetryPolicy(max_retries=3))
            .with_circuit_breaker(CircuitBreakerConfig(failure_threshold=5))
            .with_timeout(10.0)
            .build()
        )

        @resilience
        async def call_api():
            return await http.get("/endpoint")
    """

    def __init__(self, name: str):
        self.name = name
        self._retry_policy: Optional[RetryPolicy] = None
        self._circuit_config: Optional[CircuitBreakerConfig] = None
        self._timeout: Optional[float] = None
        self._fallback_value: Any = None
        self._fallback_fn: Optional[Callable] = None
        self._bulkhead_limit: Optional[int] = None

    def with_retry(self, policy: RetryPolicy) -> "ResilienceBuilder":
        """Add retry behavior."""
        self._retry_policy = policy
        return self

    def with_circuit_breaker(self, config: CircuitBreakerConfig) -> "ResilienceBuilder":
        """Add circuit breaker."""
        self._circuit_config = config
        return self

    def with_timeout(self, seconds: float) -> "ResilienceBuilder":
        """Add timeout."""
        self._timeout = seconds
        return self

    def with_fallback(
        self,
        value: Any = None,
        fn: Optional[Callable] = None,
    ) -> "ResilienceBuilder":
        """Add fallback."""
        self._fallback_value = value
        self._fallback_fn = fn
        return self

    def with_bulkhead(self, max_concurrent: int) -> "ResilienceBuilder":
        """Add bulkhead."""
        self._bulkhead_limit = max_concurrent
        return self

    def build(self) -> Callable:
        """Build the composed decorator."""
        decorators = []

        if self._fallback_value is not None or self._fallback_fn:
            decorators.append(with_fallback(self._fallback_value, self._fallback_fn))

        if self._retry_policy:
            decorators.append(with_retry(self._retry_policy))

        if self._circuit_config:
            cb = CircuitBreakerRegistry.get_instance().get_or_create(
                self.name, self._circuit_config
            )
            decorators.append(cb.protect)

        if self._timeout:
            decorators.append(with_timeout(self._timeout))

        def composed_decorator(func: Callable) -> Callable:
            result = func
            for decorator in reversed(decorators):
                result = decorator(result)
            return result

        return composed_decorator


# Pre-configured policies for common use cases
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential=True,
    jitter=True,
)

AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    base_delay=0.5,
    max_delay=60.0,
    exponential=True,
    jitter=True,
)

CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    base_delay=2.0,
    max_delay=10.0,
    exponential=False,
    jitter=True,
)
