"""
Tests for exception handling utilities.
"""

import logging

import pytest

from libs.core.exception_handlers import (
    ExceptionAggregator,
    classify_exception,
    exception_context,
    handle_adapter_exceptions,
    handle_exceptions,
    log_and_suppress,
    retry_with_backoff,
    safe_execute,
    wrap_exception,
)
from libs.core.exceptions import (
    AdapterError,
    ConfigurationError,
    DataError,
    ExchangeError,
    MASPError,
    NetworkError,
    ValidationError,
)


class TestClassifyException:
    """Tests for exception classification."""

    def test_classify_connection_error(self):
        """Test ConnectionError classification."""
        assert classify_exception(ConnectionError("test")) == NetworkError

    def test_classify_timeout_error(self):
        """Test TimeoutError classification."""
        assert classify_exception(TimeoutError("test")) == NetworkError

    def test_classify_value_error(self):
        """Test ValueError classification."""
        assert classify_exception(ValueError("test")) == ValidationError

    def test_classify_key_error(self):
        """Test KeyError classification."""
        assert classify_exception(KeyError("test")) == DataError

    def test_classify_file_not_found(self):
        """Test FileNotFoundError classification."""
        assert classify_exception(FileNotFoundError("test")) == ConfigurationError

    def test_classify_by_message(self):
        """Test classification by exception message."""
        assert classify_exception(Exception("connection refused")) == NetworkError
        assert classify_exception(Exception("timeout occurred")) == NetworkError
        assert classify_exception(Exception("rate limit exceeded")) == ExchangeError
        assert classify_exception(Exception("invalid parameter")) == ValidationError

    def test_classify_unknown(self):
        """Test unknown exception defaults to AdapterError."""
        assert classify_exception(Exception("random error")) == AdapterError


class TestWrapException:
    """Tests for wrap_exception function."""

    def test_wrap_generic_exception(self):
        """Test wrapping a generic exception."""
        original = ValueError("invalid value")
        wrapped = wrap_exception(original, "test context")

        assert isinstance(wrapped, ValidationError)
        assert "test context" in str(wrapped)
        assert wrapped.cause is original

    def test_wrap_masp_error_unchanged(self):
        """Test that MASPError is returned unchanged."""
        original = AdapterError("adapter failed")
        wrapped = wrap_exception(original, "context")

        assert wrapped is original

    def test_wrap_with_error_code(self):
        """Test wrapping with custom error code."""
        original = Exception("error")
        wrapped = wrap_exception(original, "context", error_code="CUSTOM_ERROR")

        assert wrapped.error_code == "CUSTOM_ERROR"


class TestSafeExecute:
    """Tests for safe_execute function."""

    def test_successful_execution(self):
        """Test successful function execution."""
        result = safe_execute(lambda: 42)
        assert result == 42

    def test_exception_returns_default(self):
        """Test that exception returns default value."""

        def failing():
            raise ValueError("error")

        result = safe_execute(failing, default="default")
        assert result == "default"

    def test_reraise_option(self):
        """Test reraise option."""

        def failing():
            raise ValueError("error")

        with pytest.raises(ValidationError):
            safe_execute(failing, reraise=True)

    def test_with_arguments(self):
        """Test with function arguments."""

        def add(a, b):
            return a + b

        result = safe_execute(add, 1, 2)
        assert result == 3


class TestHandleExceptionsDecorator:
    """Tests for handle_exceptions decorator."""

    def test_successful_function(self):
        """Test decorator with successful function."""

        @handle_exceptions(default=None)
        def successful():
            return 42

        assert successful() == 42

    def test_failing_function_returns_default(self):
        """Test decorator returns default on failure."""

        @handle_exceptions(default=-1, context="test")
        def failing():
            raise ValueError("error")

        assert failing() == -1

    def test_reraise_option(self):
        """Test decorator reraise option."""

        @handle_exceptions(reraise=True)
        def failing():
            raise ValueError("error")

        with pytest.raises(ValidationError):
            failing()

    def test_masp_error_propagates(self):
        """Test that MASPError propagates unchanged."""

        @handle_exceptions(default=None)
        def raising_masp():
            raise AdapterError("adapter error")

        with pytest.raises(AdapterError):
            raising_masp()

    def test_specific_handlers(self):
        """Test specific exception handlers."""
        handled = []

        def value_handler(exc, *args, **kwargs):
            handled.append(exc)
            return "handled"

        @handle_exceptions(
            default=None,
            specific_handlers={ValueError: value_handler},
        )
        def failing():
            raise ValueError("specific error")

        result = failing()
        assert result == "handled"
        assert len(handled) == 1


class TestHandleAdapterExceptionsDecorator:
    """Tests for handle_adapter_exceptions decorator."""

    def test_successful_sync_function(self):
        """Test decorator with successful sync function."""

        @handle_adapter_exceptions("get_quote", "test_exchange")
        def get_quote():
            return {"price": 100}

        assert get_quote() == {"price": 100}

    def test_failing_sync_function(self):
        """Test decorator wraps exception as ExchangeError."""

        @handle_adapter_exceptions("get_quote", "test_exchange")
        def get_quote():
            raise ValueError("bad symbol")

        with pytest.raises(ExchangeError) as exc_info:
            get_quote()

        assert "test_exchange" in str(exc_info.value)
        assert "get_quote" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_async_function(self):
        """Test decorator with successful async function."""

        @handle_adapter_exceptions("get_quote", "test_exchange")
        async def get_quote():
            return {"price": 100}

        assert await get_quote() == {"price": 100}

    @pytest.mark.asyncio
    async def test_failing_async_function(self):
        """Test decorator wraps async exception."""

        @handle_adapter_exceptions("place_order", "test_exchange")
        async def place_order():
            raise ConnectionError("network error")

        with pytest.raises(ExchangeError):
            await place_order()


class TestExceptionContext:
    """Tests for exception_context context manager."""

    def test_successful_block(self):
        """Test successful code block."""
        result = None
        with exception_context("test", reraise=False):
            result = 42

        assert result == 42

    def test_exception_reraise(self):
        """Test exception is re-raised when configured."""
        with pytest.raises(MASPError):
            with exception_context("test", reraise=True):
                raise ValueError("error")

    def test_exception_suppressed(self):
        """Test exception is suppressed when configured."""
        result = "not changed"
        with exception_context("test", reraise=False):
            raise ValueError("error")
            result = "changed"

        assert result == "not changed"

    def test_masp_error_always_propagates(self):
        """Test MASPError always propagates."""
        with pytest.raises(AdapterError):
            with exception_context("test", reraise=False):
                raise AdapterError("error")


class TestLogAndSuppress:
    """Tests for log_and_suppress function."""

    def test_logs_exception(self, caplog):
        """Test that exception is logged."""
        exc = ValueError("test error")

        with caplog.at_level(logging.WARNING):
            log_and_suppress(exc, "test_context")

        assert "Suppressed" in caplog.text
        assert "ValueError" in caplog.text
        assert "test error" in caplog.text


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_first_try(self):
        """Test no retry needed on success."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def successful():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful()
        assert result == "success"
        assert call_count == 1

    def test_retry_then_success(self):
        """Test retry then eventual success."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("retry me")
            return "success"

        result = eventually_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exceeded(self):
        """Test failure after max retries."""

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            raise ConnectionError("always fails")

        with pytest.raises(MASPError):
            always_fails()

    def test_specific_retryable_exceptions(self):
        """Test only specific exceptions are retried."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(MASPError):
            raises_value_error()

        assert call_count == 1  # No retries for ValueError

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async function retry."""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def async_eventually_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry me")
            return "async success"

        result = await async_eventually_succeeds()
        assert result == "async success"
        assert call_count == 2


class TestExceptionAggregator:
    """Tests for ExceptionAggregator."""

    def test_no_errors(self):
        """Test aggregator with no errors."""
        agg = ExceptionAggregator("test")

        for i in range(5):
            with agg.catch():
                pass  # All successful

        assert not agg.has_errors
        assert agg.error_count == 0

    def test_some_errors(self):
        """Test aggregator with some errors."""
        agg = ExceptionAggregator("test")

        for i in range(5):
            with agg.catch():
                if i % 2 == 0:
                    raise ValueError(f"error {i}")

        assert agg.has_errors
        assert agg.error_count == 3

    def test_raise_if_any_below_threshold(self):
        """Test raise_if_any respects threshold."""
        agg = ExceptionAggregator("test")

        for i in range(10):
            with agg.catch():
                if i < 3:
                    raise ValueError(f"error {i}")

        # 30% failure rate, threshold 50%
        agg.raise_if_any(threshold=0.5)  # Should not raise

    def test_raise_if_any_above_threshold(self):
        """Test raise_if_any raises above threshold."""
        agg = ExceptionAggregator("test")

        for i in range(10):
            with agg.catch():
                if i < 6:
                    raise ValueError(f"error {i}")

        # 60% failure rate, threshold 50%
        with pytest.raises(AdapterError):
            agg.raise_if_any(threshold=0.5)

    def test_get_summary(self):
        """Test summary generation."""
        agg = ExceptionAggregator("batch_test")

        for i in range(10):
            with agg.catch(index=i):
                if i % 2 == 0:
                    raise ValueError(f"error {i}")

        summary = agg.get_summary()
        assert summary["context"] == "batch_test"
        assert summary["total"] == 10
        assert summary["errors"] == 5
        assert summary["success_rate"] == 0.5
