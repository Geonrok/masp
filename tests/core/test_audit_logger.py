"""
Tests for Audit Logger.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from libs.core.audit_logger import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditLogWriter,
    AuditSeverity,
    audit_config_change,
    audit_order_placed,
    get_audit_logger,
)


@pytest.fixture
def temp_log_dir():
    """Create temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audit_logger(temp_log_dir):
    """Create test audit logger."""
    AuditLogger.reset()
    return AuditLogger(
        log_dir=temp_log_dir,
        enabled=True,
        console_output=False,
    )


@pytest.fixture
def log_writer(temp_log_dir):
    """Create test log writer."""
    return AuditLogWriter(
        log_dir=temp_log_dir,
        max_file_size_mb=1.0,
        max_files=5,
    )


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_trading_events(self):
        """Test trading event types."""
        assert AuditEventType.ORDER_PLACED.value == "order.placed"
        assert AuditEventType.ORDER_FILLED.value == "order.filled"
        assert AuditEventType.ORDER_CANCELLED.value == "order.cancelled"
        assert AuditEventType.ORDER_REJECTED.value == "order.rejected"

    def test_security_events(self):
        """Test security event types."""
        assert AuditEventType.AUTH_SUCCESS.value == "auth.success"
        assert AuditEventType.AUTH_FAILURE.value == "auth.failure"
        assert AuditEventType.PERMISSION_DENIED.value == "permission.denied"

    def test_system_events(self):
        """Test system event types."""
        assert AuditEventType.SYSTEM_STARTUP.value == "system.startup"
        assert AuditEventType.SYSTEM_SHUTDOWN.value == "system.shutdown"
        assert AuditEventType.KILL_SWITCH_ACTIVATED.value == "killswitch.activated"


class TestAuditSeverity:
    """Tests for AuditSeverity enum."""

    def test_values(self):
        """Test severity values."""
        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
        )
        assert event.event_type == AuditEventType.ORDER_PLACED
        assert event.severity == AuditSeverity.INFO
        assert event.exchange == "upbit"
        assert event.order_id == "123"

    def test_event_id_generated(self):
        """Test event ID is generated."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test",
        )
        assert hasattr(event, "event_id")
        assert len(event.event_id) == 16

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
            exchange="upbit",
            user_id="user1",
        )
        d = event.to_dict()
        assert d["event_type"] == "order.placed"
        assert d["severity"] == "info"
        assert d["message"] == "Test order"
        assert d["exchange"] == "upbit"
        assert d["user_id"] == "user1"

    def test_to_json(self):
        """Test conversion to JSON."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "order.placed"

    def test_with_details(self):
        """Test event with additional details."""
        event = AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGED,
            severity=AuditSeverity.INFO,
            message="Config changed",
            details={"setting": "max_position_pct", "value": 0.15},
            before_state={"value": 0.10},
            after_state={"value": 0.15},
        )
        assert event.details["setting"] == "max_position_pct"
        assert event.before_state["value"] == 0.10
        assert event.after_state["value"] == 0.15


class TestAuditLogWriter:
    """Tests for AuditLogWriter class."""

    def test_init(self, log_writer, temp_log_dir):
        """Test initialization."""
        assert log_writer._log_dir == Path(temp_log_dir)
        assert log_writer._log_dir.exists()

    def test_write_event(self, log_writer):
        """Test writing an event."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
        )
        success = log_writer.write(event)
        assert success

    def test_file_created(self, log_writer, temp_log_dir):
        """Test log file is created."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
        )
        log_writer.write(event)

        files = list(Path(temp_log_dir).glob("audit_*.jsonl"))
        assert len(files) == 1

    def test_multiple_events(self, log_writer, temp_log_dir):
        """Test writing multiple events."""
        for i in range(5):
            event = AuditEvent(
                event_type=AuditEventType.ORDER_PLACED,
                severity=AuditSeverity.INFO,
                message=f"Test order {i}",
            )
            log_writer.write(event)

        files = list(Path(temp_log_dir).glob("audit_*.jsonl"))
        assert len(files) == 1

        # Check content
        with open(files[0], "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_get_log_files(self, log_writer, temp_log_dir):
        """Test getting log files."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test",
        )
        log_writer.write(event)

        files = log_writer.get_log_files()
        assert len(files) == 1


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_init(self, audit_logger):
        """Test initialization."""
        assert audit_logger._enabled

    def test_singleton(self, temp_log_dir):
        """Test singleton pattern."""
        AuditLogger.reset()
        with patch.object(AuditLogger, "__init__", lambda self, **kwargs: None):
            AuditLogger._instance = AuditLogger.__new__(AuditLogger)
            AuditLogger._instance._enabled = True
            AuditLogger._instance._console_output = False
            AuditLogger._instance._writer = AuditLogWriter(log_dir=temp_log_dir)
            AuditLogger._instance._callbacks = []

        a1 = AuditLogger.get_instance()
        a2 = AuditLogger.get_instance()
        assert a1 is a2
        AuditLogger.reset()

    def test_log_event(self, audit_logger):
        """Test logging an event."""
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test order",
        )
        success = audit_logger.log(event)
        assert success

    def test_disabled_logger(self, temp_log_dir):
        """Test disabled logger."""
        AuditLogger.reset()
        logger = AuditLogger(
            log_dir=temp_log_dir,
            enabled=False,
        )
        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test",
        )
        success = logger.log(event)
        assert not success
        AuditLogger.reset()

    def test_callback(self, audit_logger):
        """Test event callback."""
        received = []

        def callback(event):
            received.append(event)

        audit_logger.register_callback(callback)

        event = AuditEvent(
            event_type=AuditEventType.ORDER_PLACED,
            severity=AuditSeverity.INFO,
            message="Test",
        )
        audit_logger.log(event)

        assert len(received) == 1
        assert received[0].message == "Test"


class TestAuditLoggerTradingOperations:
    """Tests for trading operation logging."""

    def test_log_order_placed(self, audit_logger):
        """Test logging order placement."""
        success = audit_logger.log_order_placed(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            side="BUY",
            quantity=0.01,
            price=50000000,
        )
        assert success

    def test_log_order_filled(self, audit_logger):
        """Test logging order fill."""
        success = audit_logger.log_order_filled(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            filled_quantity=0.01,
            filled_price=50000000,
        )
        assert success

    def test_log_order_cancelled(self, audit_logger):
        """Test logging order cancellation."""
        success = audit_logger.log_order_cancelled(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            reason="User requested",
        )
        assert success

    def test_log_order_rejected(self, audit_logger):
        """Test logging order rejection."""
        success = audit_logger.log_order_rejected(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            reason="Insufficient balance",
        )
        assert success


class TestAuditLoggerConfiguration:
    """Tests for configuration logging."""

    def test_log_config_change(self, audit_logger):
        """Test logging config change."""
        success = audit_logger.log_config_change(
            setting="max_position_pct",
            old_value=0.10,
            new_value=0.15,
            user_id="admin",
        )
        assert success

    def test_log_settings_updated(self, audit_logger):
        """Test logging settings update."""
        success = audit_logger.log_settings_updated(
            changes={"max_position_pct": 0.15, "max_daily_orders": 50},
            user_id="admin",
        )
        assert success

    def test_log_strategy_enabled(self, audit_logger):
        """Test logging strategy enabled."""
        success = audit_logger.log_strategy_enabled(
            strategy_id="KAMA-TSMOM",
            exchange="upbit",
            user_id="admin",
        )
        assert success

    def test_log_strategy_disabled(self, audit_logger):
        """Test logging strategy disabled."""
        success = audit_logger.log_strategy_disabled(
            strategy_id="KAMA-TSMOM",
            exchange="upbit",
            reason="Manual stop",
            user_id="admin",
        )
        assert success


class TestAuditLoggerSecurity:
    """Tests for security logging."""

    def test_log_auth_success(self, audit_logger):
        """Test logging auth success."""
        success = audit_logger.log_auth_success(
            user_id="user1",
            method="api_key",
            ip_address="192.168.1.1",
        )
        assert success

    def test_log_auth_failure(self, audit_logger):
        """Test logging auth failure."""
        success = audit_logger.log_auth_failure(
            user_id="user1",
            reason="Invalid API key",
            ip_address="192.168.1.1",
        )
        assert success

    def test_log_permission_denied(self, audit_logger):
        """Test logging permission denied."""
        success = audit_logger.log_permission_denied(
            user_id="user1",
            resource="admin/settings",
            action="write",
        )
        assert success


class TestAuditLoggerSystem:
    """Tests for system logging."""

    def test_log_system_startup(self, audit_logger):
        """Test logging system startup."""
        success = audit_logger.log_system_startup()
        assert success

    def test_log_system_shutdown(self, audit_logger):
        """Test logging system shutdown."""
        success = audit_logger.log_system_shutdown(reason="normal")
        assert success

    def test_log_system_error(self, audit_logger):
        """Test logging system error."""
        success = audit_logger.log_system_error(
            error="Connection timeout",
            component="upbit_adapter",
        )
        assert success

    def test_log_kill_switch_activated(self, audit_logger):
        """Test logging kill switch activation."""
        success = audit_logger.log_kill_switch_activated(
            reason="Manual emergency stop",
            user_id="admin",
        )
        assert success

    def test_log_circuit_breaker_triggered(self, audit_logger):
        """Test logging circuit breaker trigger."""
        success = audit_logger.log_circuit_breaker_triggered(
            trigger_type="drawdown",
            value=0.12,
            threshold=0.10,
        )
        assert success


class TestAuditLoggerDataOperations:
    """Tests for data operation logging."""

    def test_log_backup_created(self, audit_logger):
        """Test logging backup creation."""
        success = audit_logger.log_backup_created(
            backup_path="/backups/masp_20240101.tar.gz",
            size_bytes=1024000,
            user_id="system",
        )
        assert success

    def test_log_backup_restored(self, audit_logger):
        """Test logging backup restoration."""
        success = audit_logger.log_backup_restored(
            backup_path="/backups/masp_20240101.tar.gz",
            user_id="admin",
        )
        assert success


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_audit_logger(self, temp_log_dir):
        """Test get_audit_logger function."""
        AuditLogger.reset()
        with patch.object(AuditLogger, "__init__", lambda self, **kwargs: None):
            AuditLogger._instance = AuditLogger.__new__(AuditLogger)
            AuditLogger._instance._enabled = True
            AuditLogger._instance._console_output = False
            AuditLogger._instance._writer = AuditLogWriter(log_dir=temp_log_dir)
            AuditLogger._instance._callbacks = []

        logger = get_audit_logger()
        assert isinstance(logger, AuditLogger)
        AuditLogger.reset()

    def test_audit_order_placed(self, audit_logger):
        """Test audit_order_placed convenience function."""
        # Need to set up singleton properly
        AuditLogger.reset()
        AuditLogger._instance = audit_logger

        success = audit_order_placed(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            side="BUY",
            quantity=0.01,
        )
        assert success
        AuditLogger.reset()

    def test_audit_config_change(self, audit_logger):
        """Test audit_config_change convenience function."""
        AuditLogger.reset()
        AuditLogger._instance = audit_logger

        success = audit_config_change(
            setting="max_position_pct",
            old_value=0.10,
            new_value=0.15,
        )
        assert success
        AuditLogger.reset()
