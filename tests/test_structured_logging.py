"""
Tests for structured logging module.
"""

import json
import logging

from libs.core.structured_logging import (
    AuditLogger,
    JSONFormatter,
    LogContext,
    configure_logging,
    generate_trace_id,
    get_logger,
    get_trace_id,
    set_trace_id,
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        formatter = JSONFormatter()

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
        log_data = json.loads(output)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test.logger"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data
        assert "source" in log_data
        assert log_data["source"]["line"] == 10

    def test_timestamp_optional(self):
        """Test timestamp can be disabled."""
        formatter = JSONFormatter(include_timestamp=False)

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
        log_data = json.loads(output)

        assert "timestamp" not in log_data

    def test_exception_formatting(self):
        """Test exception info is included."""
        formatter = JSONFormatter(include_traceback=True)

        try:
            raise ValueError("Test error")
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
        log_data = json.loads(output)

        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
        assert "traceback" in log_data["exception"]

    def test_exception_without_traceback(self):
        """Test exception without traceback."""
        formatter = JSONFormatter(include_traceback=False)

        try:
            raise RuntimeError("Runtime error")
        except RuntimeError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        log_data = json.loads(output)

        assert "exception" in log_data
        assert "traceback" not in log_data["exception"]

    def test_extra_fields(self):
        """Test global extra fields are included."""
        formatter = JSONFormatter(extra_fields={"service": "masp", "version": "1.0"})

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
        log_data = json.loads(output)

        assert log_data["service"] == "masp"
        assert log_data["version"] == "1.0"

    def test_record_extra_fields(self):
        """Test extra fields from log record."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        record.order_id = "ORD123"

        output = formatter.format(record)
        log_data = json.loads(output)

        assert log_data["custom_field"] == "custom_value"
        assert log_data["order_id"] == "ORD123"

    def test_trace_id_included(self):
        """Test trace ID is included when set."""
        formatter = JSONFormatter()

        set_trace_id("abc123")

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
        log_data = json.loads(output)

        assert log_data["trace_id"] == "abc123"

    def test_indent_option(self):
        """Test JSON indent option."""
        formatter = JSONFormatter(indent=2)

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

        # Indented JSON has newlines
        assert "\n" in output

    def test_serialization_of_complex_types(self):
        """Test serialization of complex types."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.data = {"nested": {"list": [1, 2, 3]}}
        record.obj = object()  # Should be converted to string

        output = formatter.format(record)
        log_data = json.loads(output)

        assert log_data["data"] == {"nested": {"list": [1, 2, 3]}}
        assert isinstance(log_data["obj"], str)


class TestTraceId:
    """Tests for trace ID functions."""

    def test_generate_trace_id(self):
        """Test trace ID generation."""
        trace_id = generate_trace_id()

        assert trace_id is not None
        assert len(trace_id) == 8

    def test_get_trace_id(self):
        """Test getting trace ID."""
        set_trace_id("test123")
        assert get_trace_id() == "test123"

    def test_trace_id_uniqueness(self):
        """Test trace IDs are unique."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestAuditLogger:
    """Tests for AuditLogger."""

    def test_initialization(self):
        """Test audit logger initialization."""
        audit = AuditLogger(name="test.audit")

        assert audit.logger is not None
        assert audit.logger.name == "test.audit"

    def test_log_trade(self, caplog):
        """Test trade logging."""
        audit = AuditLogger(name="test.audit.trade")

        with caplog.at_level(logging.INFO, logger="test.audit.trade"):
            audit.log_trade(
                action="ORDER_PLACED",
                symbol="BTC/USDT",
                side="BUY",
                quantity=1.0,
                price=50000.0,
                order_id="ORD001",
                strategy="KAMA",
                exchange="binance",
            )

        assert "TRADE: ORDER_PLACED BUY 1.0 BTC/USDT @ 50000.0" in caplog.text

    def test_log_signal(self, caplog):
        """Test signal logging."""
        audit = AuditLogger(name="test.audit.signal")

        with caplog.at_level(logging.INFO, logger="test.audit.signal"):
            audit.log_signal(
                strategy="KAMA-TSMOM",
                symbol="ETH/USDT",
                signal="BUY",
                strength=0.85,
                reason="Positive momentum",
            )

        assert "SIGNAL: KAMA-TSMOM -> BUY ETH/USDT" in caplog.text

    def test_log_risk_event(self, caplog):
        """Test risk event logging."""
        audit = AuditLogger(name="test.audit.risk")

        with caplog.at_level(logging.WARNING, logger="test.audit.risk"):
            audit.log_risk_event(
                event="Drawdown limit exceeded",
                level="WARNING",
                metric="max_drawdown",
                value=0.15,
                threshold=0.10,
            )

        assert "RISK: Drawdown limit exceeded" in caplog.text

    def test_log_system_event(self, caplog):
        """Test system event logging."""
        audit = AuditLogger(name="test.audit.system")

        with caplog.at_level(logging.INFO, logger="test.audit.system"):
            audit.log_system_event(
                event="Scheduler started",
                component="scheduler",
                status="STARTED",
            )

        assert "SYSTEM: [scheduler] Scheduler started" in caplog.text


class TestLogContext:
    """Tests for LogContext."""

    def test_context_manager(self):
        """Test context manager sets trace ID."""
        with LogContext(trace_id="ctx123"):
            assert get_trace_id() == "ctx123"

    def test_multiple_contexts(self):
        """Test nested contexts."""
        with LogContext(trace_id="outer"):
            assert get_trace_id() == "outer"

            with LogContext(trace_id="inner"):
                assert get_trace_id() == "inner"


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_json_format(self):
        """Test JSON format configuration."""
        configure_logging(level=logging.DEBUG, json_format=True)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        assert len(root_logger.handlers) > 0

        # Check formatter is JSON
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_text_format(self):
        """Test text format configuration."""
        configure_logging(level=logging.INFO, json_format=False)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert not isinstance(handler.formatter, JSONFormatter)

    def test_extra_fields_propagation(self):
        """Test extra fields are propagated."""
        configure_logging(
            json_format=True,
            extra_fields={"environment": "test"},
        )

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert handler.formatter.extra_fields["environment"] == "test"


class TestGetLogger:
    """Tests for get_logger."""

    def test_get_logger(self):
        """Test getting logger by name."""
        logger = get_logger("test.module")

        assert logger is not None
        assert logger.name == "test.module"

    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent = get_logger("parent")
        child = get_logger("parent.child")

        assert child.parent is parent or child.parent.name == "parent"
