"""
Tests for resilience patterns.
"""

import asyncio
import time

import pytest

from libs.core.exceptions import NetworkError
from libs.core.resilience import (
    DEFAULT_RETRY_POLICY,
    Bulkhead,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    ResilienceBuilder,
    RetryPolicy,
    with_fallback,
    with_retry,
    with_timeout,
)


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_default_values(self):
        """Test default policy values."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.base_delay == 1.0
        assert policy.exponential is True

    def test_get_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        policy = RetryPolicy(base_delay=1.0, exponential=True, jitter=False)

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0

    def test_get_delay_linear(self):
        """Test linear delay calculation."""
        policy = RetryPolicy(base_delay=1.0, exponential=False, jitter=False)

        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 1.0

    def test_max_delay(self):
        """Test max delay cap."""
        policy = RetryPolicy(
            base_delay=1.0, max_delay=5.0, exponential=True, jitter=False
        )

        assert policy.get_delay(10) == 5.0  # Capped at max

    def test_should_retry(self):
        """Test should_retry logic."""
        policy = RetryPolicy(
            retryable_exceptions=(ConnectionError, TimeoutError),
            non_retryable_exceptions=(ValueError,),
        )

        assert policy.should_retry(ConnectionError()) is True
        assert policy.should_retry(TimeoutError()) is True
        assert policy.should_retry(ValueError()) is False
        assert policy.should_retry(RuntimeError()) is False


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test initial state is closed."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_rejects_when_open(self):
        """Test circuit rejects requests when open."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_half_open_after_timeout(self):
        """Test circuit becomes half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=0.1,
        )
        cb = CircuitBreaker("test", config)

        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_protect_decorator_sync(self):
        """Test protect decorator with sync function."""
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        call_count = 0

        @cb.protect
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("error")

        # First two calls should fail
        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        # Circuit should be open now
        assert cb.state == CircuitState.OPEN

        # Third call should be rejected
        with pytest.raises(NetworkError) as exc_info:
            failing_func()
        assert "OPEN" in str(exc_info.value)
        assert call_count == 2  # Function not called

    @pytest.mark.asyncio
    async def test_protect_decorator_async(self):
        """Test protect decorator with async function."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)

        @cb.protect
        async def async_failing():
            raise ConnectionError("network error")

        with pytest.raises(ConnectionError):
            await async_failing()

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test get_or_create returns same instance."""
        registry = CircuitBreakerRegistry.get_instance()

        cb1 = registry.get_or_create("test_service")
        cb2 = registry.get_or_create("test_service")

        assert cb1 is cb2

    def test_get_all_stats(self):
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry.get_instance()
        registry.get_or_create("service_a")
        registry.get_or_create("service_b")

        stats = registry.get_all_stats()
        assert "service_a" in stats
        assert "service_b" in stats


class TestWithRetry:
    """Tests for with_retry decorator."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=3))
        def successful():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful()
        assert result == "success"
        assert call_count == 1

    def test_retry_then_success(self):
        """Test retry until success."""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=3, base_delay=0.01))
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry")
            return "success"

        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test failure after max retries."""

        @with_retry(RetryPolicy(max_retries=2, base_delay=0.01))
        def always_fails():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            always_fails()

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry."""
        call_count = 0

        @with_retry(RetryPolicy(max_retries=3, base_delay=0.01))
        async def async_eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry")
            return "async success"

        result = await async_eventually_succeeds()
        assert result == "async success"
        assert call_count == 2


class TestWithFallback:
    """Tests for with_fallback decorator."""

    def test_fallback_value(self):
        """Test fallback with static value."""

        @with_fallback(fallback_value="default")
        def failing():
            raise ValueError("error")

        result = failing()
        assert result == "default"

    def test_fallback_function(self):
        """Test fallback with function."""

        def fallback_fn(*args):
            return f"fallback: {args}"

        @with_fallback(fallback_fn=fallback_fn)
        def failing(x):
            raise ValueError("error")

        result = failing(42)
        assert result == "fallback: (42,)"

    def test_no_fallback_on_success(self):
        """Test fallback not used on success."""

        @with_fallback(fallback_value="default")
        def successful():
            return "success"

        result = successful()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_fallback(self):
        """Test async fallback."""

        @with_fallback(fallback_value=[])
        async def async_failing():
            raise ConnectionError("error")

        result = await async_failing()
        assert result == []


class TestWithTimeout:
    """Tests for with_timeout decorator."""

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Test function completes within timeout."""

        @with_timeout(1.0)
        async def fast_func():
            await asyncio.sleep(0.01)
            return "done"

        result = await fast_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """Test timeout exceeded."""

        @with_timeout(0.1)
        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"

        with pytest.raises(NetworkError) as exc_info:
            await slow_func()
        assert "timeout" in str(exc_info.value).lower()


class TestBulkhead:
    """Tests for Bulkhead."""

    @pytest.mark.asyncio
    async def test_limits_concurrent_calls(self):
        """Test bulkhead limits concurrent executions."""
        bulkhead = Bulkhead("test", max_concurrent=2)
        concurrent_count = 0
        max_concurrent = 0

        async def task():
            nonlocal concurrent_count, max_concurrent
            async with await bulkhead.acquire():
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
                await asyncio.sleep(0.1)
                concurrent_count -= 1

        # Run 5 tasks with bulkhead limit of 2
        await asyncio.gather(*[task() for _ in range(5)])

        assert max_concurrent == 2

    def test_available_slots(self):
        """Test available slots tracking."""
        bulkhead = Bulkhead("test", max_concurrent=3)
        assert bulkhead.available == 3


class TestResilienceBuilder:
    """Tests for ResilienceBuilder."""

    def test_with_retry(self):
        """Test builder with retry."""
        call_count = 0

        resilience = (
            ResilienceBuilder("test")
            .with_retry(RetryPolicy(max_retries=2, base_delay=0.01))
            .build()
        )

        @resilience
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry")
            return "success"

        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 2

    def test_with_fallback(self):
        """Test builder with fallback."""
        resilience = ResilienceBuilder("test").with_fallback(value="fallback").build()

        @resilience
        def failing():
            raise ValueError("error")

        result = failing()
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_combined_patterns(self):
        """Test combining multiple patterns."""
        call_count = 0

        resilience = (
            ResilienceBuilder("test")
            .with_retry(RetryPolicy(max_retries=2, base_delay=0.01))
            .with_fallback(value="fallback")
            .build()
        )

        @resilience
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("error")

        result = await always_fails()
        assert result == "fallback"
        assert call_count == 3  # Initial + 2 retries


class TestPreConfiguredPolicies:
    """Tests for pre-configured policies."""

    def test_default_policy(self):
        """Test default retry policy."""
        assert DEFAULT_RETRY_POLICY.max_retries == 3
        assert DEFAULT_RETRY_POLICY.exponential is True
        assert DEFAULT_RETRY_POLICY.jitter is True
