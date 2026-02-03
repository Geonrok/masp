"""
Tests for exception hierarchy and structured logging.
"""

import json
import logging
from io import StringIO

import pytest

from libs.core.exceptions import (
    AdapterError,
    APIError,
    ConfigurationError,
    ConnectionError,
    InsufficientFundsError,
    MASPError,
    MissingConfigError,
    OrderRejectedError,
    RateLimitError,
    RiskLimitExceededError,
    TradingError,
    ValidationError,
    wrap_exception,
)
from libs.core.structured_logging import (
    AuditLogger,
    JSONFormatter,
    LogContext,
    configure_logging,
    generate_trace_id,
    get_logger,
    get_trace_id,
)


class TestMASPError:
    """Tests for base MASP exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = MASPError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.error_code == "MASP_ERROR"
        assert error.details == {}
        assert error.cause is None

    def test_error_with_details(self):
        """Test error with details."""
        error = MASPError(
            "Failed operation",
            error_code="OP_FAILED",
            details={"key": "value"},
        )
        assert error.error_code == "OP_FAILED"
        assert error.details["key"] == "value"

    def test_error_with_cause(self):
        """Test error with cause."""
        original = ValueError("original error")
        error = MASPError("Wrapped error", cause=original)
        assert error.cause is original

    def test_to_dict(self):
        """Test error serialization."""
        error = MASPError(
            "Test error",
            error_code="TEST",
            details={"foo": "bar"},
        )
        d = error.to_dict()
        assert d["error_type"] == "MASPError"
        assert d["error_code"] == "TEST"
        assert d["message"] == "Test error"
        assert d["details"]["foo"] == "bar"


class TestConfigurationErrors:
    """Tests for configuration exceptions."""

    def test_missing_config_error(self):
        """Test missing configuration error."""
        error = MissingConfigError(
            "Required config 'API_KEY' is missing",
            details={"config_key": "API_KEY"},
        )
        assert error.error_code == "CONFIG_MISSING"

    def test_inheritance(self):
        """Test exception hierarchy."""
        error = MissingConfigError("Missing")
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, MASPError)
        assert isinstance(error, Exception)


class TestAdapterErrors:
    """Tests for adapter exceptions."""

    def test_rate_limit_error(self):
        """Test rate limit error with retry info."""
        error = RateLimitError(
            "Rate limit exceeded",
            retry_after=60,
        )
        assert error.error_code == "ADAPTER_RATE_LIMIT"
        assert error.retry_after == 60
        assert error.details["retry_after_seconds"] == 60

    def test_api_error(self):
        """Test API error with status code."""
        error = APIError(
            "Server error",
            status_code=500,
            response_body='{"error": "Internal"}',
        )
        assert error.error_code == "ADAPTER_API_ERROR"
        assert error.status_code == 500
        assert error.details["status_code"] == 500


class TestTradingErrors:
    """Tests for trading exceptions."""

    def test_insufficient_funds(self):
        """Test insufficient funds error."""
        error = InsufficientFundsError(
            "Not enough KRW",
            required=100000,
            available=50000,
        )
        assert error.details["required"] == 100000
        assert error.details["available"] == 50000

    def test_order_rejected(self):
        """Test order rejected error."""
        error = OrderRejectedError(
            "Order rejected by exchange",
            order_id="ORD-123",
            rejection_reason="Invalid quantity",
        )
        assert error.details["order_id"] == "ORD-123"
        assert error.details["rejection_reason"] == "Invalid quantity"


class TestRiskErrors:
    """Tests for risk management exceptions."""

    def test_risk_limit_exceeded(self):
        """Test risk limit exceeded error."""
        error = RiskLimitExceededError(
            "Daily loss limit exceeded",
            limit_type="daily_loss",
            limit_value=0.02,
            current_value=0.025,
        )
        assert error.error_code == "RISK_LIMIT_EXCEEDED"
        assert error.details["limit_type"] == "daily_loss"
        assert error.details["limit_value"] == 0.02


class TestWrapException:
    """Tests for exception wrapping utility."""

    def test_wrap_standard_exception(self):
        """Test wrapping standard exception."""
        original = ValueError("bad value")
        wrapped = wrap_exception(original, ValidationError)

        assert isinstance(wrapped, ValidationError)
        assert wrapped.cause is original
        assert "bad value" in wrapped.message

    def test_wrap_already_masp_error(self):
        """Test wrapping already MASP error returns same."""
        original = MASPError("already wrapped")
        wrapped = wrap_exception(original)

        assert wrapped is original


class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_basic_formatting(self):
        """Test basic JSON log output."""
        formatter = JSONFormatter(include_timestamp=False)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"

    def test_extra_fields(self):
        """Test formatter with extra fields."""
        formatter = JSONFormatter(
            include_timestamp=False,
            extra_fields={"app": "masp", "env": "test"},
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["app"] == "masp"
        assert data["env"] == "test"

    def test_exception_formatting(self):
        """Test exception formatting."""
        formatter = JSONFormatter(include_timestamp=False)

        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "test error" in data["exception"]["message"]


class TestAuditLogger:
    """Tests for audit logger."""

    def test_log_trade(self, caplog):
        """Test trade logging."""
        audit = AuditLogger()

        with caplog.at_level(logging.INFO):
            audit.log_trade(
                action="ORDER_PLACED",
                symbol="BTC",
                side="BUY",
                quantity=0.1,
                price=50000,
                order_id="ORD-001",
            )

        assert "TRADE" in caplog.text
        assert "ORDER_PLACED" in caplog.text

    def test_log_signal(self, caplog):
        """Test signal logging."""
        audit = AuditLogger()

        with caplog.at_level(logging.INFO):
            audit.log_signal(
                strategy="KAMA-TSMOM",
                symbol="ETH",
                signal="BUY",
                strength=0.8,
            )

        assert "SIGNAL" in caplog.text
        assert "KAMA-TSMOM" in caplog.text

    def test_log_risk_event(self, caplog):
        """Test risk event logging."""
        audit = AuditLogger()

        with caplog.at_level(logging.WARNING):
            audit.log_risk_event(
                event="Drawdown approaching limit",
                level="WARNING",
                metric="drawdown",
                value=0.08,
                threshold=0.10,
            )

        assert "RISK" in caplog.text


class TestTraceContext:
    """Tests for trace context management."""

    def test_generate_trace_id(self):
        """Test trace ID generation."""
        trace_id = generate_trace_id()
        assert trace_id is not None
        assert len(trace_id) == 8

    def test_get_trace_id(self):
        """Test getting current trace ID."""
        generate_trace_id()
        trace_id = get_trace_id()
        assert trace_id is not None

    def test_log_context(self):
        """Test log context manager."""
        with LogContext(trace_id="test123"):
            assert get_trace_id() == "test123"


class TestConfigureLogging:
    """Tests for logging configuration."""

    def test_configure_json_logging(self):
        """Test JSON logging configuration."""
        configure_logging(
            level=logging.DEBUG,
            json_format=True,
        )

        logger = get_logger("test.json")
        assert logger is not None

    def test_configure_text_logging(self):
        """Test text logging configuration."""
        configure_logging(
            level=logging.INFO,
            json_format=False,
        )

        logger = get_logger("test.text")
        assert logger is not None
