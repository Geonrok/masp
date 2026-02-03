"""
Audit Logger for MASP.

Provides comprehensive audit logging for:
- Trading operations (orders, fills, cancellations)
- Configuration changes
- Security events (authentication, authorization)
- System operations (startup, shutdown, errors)

Audit logs are stored in JSON format for easy parsing and analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""

    # Trading operations
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_MODIFIED = "order.modified"

    # Position operations
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_MODIFIED = "position.modified"

    # Configuration
    CONFIG_CHANGED = "config.changed"
    SETTINGS_UPDATED = "settings.updated"
    STRATEGY_ENABLED = "strategy.enabled"
    STRATEGY_DISABLED = "strategy.disabled"

    # Security
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    API_KEY_CREATED = "apikey.created"
    API_KEY_DELETED = "apikey.deleted"
    PERMISSION_DENIED = "permission.denied"

    # System
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_ERROR = "system.error"
    KILL_SWITCH_ACTIVATED = "killswitch.activated"
    CIRCUIT_BREAKER_TRIGGERED = "circuitbreaker.triggered"

    # Data operations
    BACKUP_CREATED = "backup.created"
    BACKUP_RESTORED = "backup.restored"
    DATA_EXPORTED = "data.exported"


class AuditSeverity(Enum):
    """Audit event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Individual audit event record."""

    event_type: AuditEventType
    severity: AuditSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Actor information
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Context
    exchange: Optional[str] = None
    symbol: Optional[str] = None
    order_id: Optional[str] = None
    strategy_id: Optional[str] = None

    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None

    # Metadata
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def __post_init__(self):
        """Generate unique event ID."""
        self.event_id = self._generate_event_id()

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        content = f"{self.timestamp.isoformat()}{self.event_type.value}{id(self)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "details": self.details,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "correlation_id": self.correlation_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogWriter:
    """
    Writes audit logs to files.

    Files are organized by date and rotated automatically.
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        max_file_size_mb: float = 100.0,
        max_files: int = 30,
    ):
        """
        Initialize audit log writer.

        Args:
            log_dir: Directory for audit logs
            max_file_size_mb: Maximum size per log file
            max_files: Maximum number of files to keep
        """
        self._log_dir = Path(log_dir)
        self._max_file_size = max_file_size_mb * 1024 * 1024
        self._max_files = max_files
        self._lock = threading.Lock()

        self._current_file: Optional[Path] = None
        self._current_date: Optional[str] = None

        # Ensure log directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event: AuditEvent) -> bool:
        """
        Write audit event to log file.

        Args:
            event: Audit event to write

        Returns:
            True if write was successful
        """
        with self._lock:
            try:
                self._ensure_current_file()
                line = event.to_json() + "\n"
                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(line)
                return True
            except Exception as e:
                logger.error("[AuditLog] Write failed: %s", e)
                return False

    def _ensure_current_file(self) -> None:
        """Ensure current file is valid and not too large."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Check if we need a new file
        if (
            self._current_file is None
            or self._current_date != today
            or (
                self._current_file.exists()
                and self._current_file.stat().st_size >= self._max_file_size
            )
        ):
            self._rotate_if_needed()
            self._current_date = today
            self._current_file = self._get_new_filename(today)

    def _get_new_filename(self, date: str) -> Path:
        """Generate new log filename."""
        base = self._log_dir / f"audit_{date}"
        suffix = ""
        counter = 0

        while True:
            filename = Path(f"{base}{suffix}.jsonl")
            if not filename.exists() or filename.stat().st_size < self._max_file_size:
                return filename
            counter += 1
            suffix = f"_{counter}"

    def _rotate_if_needed(self) -> None:
        """Remove old log files if needed."""
        try:
            files = sorted(self._log_dir.glob("audit_*.jsonl"))
            while len(files) > self._max_files:
                oldest = files.pop(0)
                oldest.unlink()
                logger.info("[AuditLog] Rotated old file: %s", oldest.name)
        except Exception as e:
            logger.error("[AuditLog] Rotation failed: %s", e)

    def get_log_files(self) -> List[Path]:
        """Get list of log files."""
        return sorted(self._log_dir.glob("audit_*.jsonl"))


class AuditLogger:
    """
    Main audit logger class.

    Provides methods for logging various types of audit events.

    Example:
        audit = AuditLogger()

        # Log an order
        audit.log_order_placed(
            exchange="upbit",
            symbol="BTC/KRW",
            order_id="123",
            side="BUY",
            quantity=0.01,
            price=50000000,
            user_id="user1",
        )

        # Log a config change
        audit.log_config_change(
            setting="max_position_pct",
            old_value=0.1,
            new_value=0.15,
            user_id="admin",
        )
    """

    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        log_dir: str = "logs/audit",
        enabled: bool = True,
        console_output: bool = False,
    ):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit logs
            enabled: Whether audit logging is enabled
            console_output: Also output to console logger
        """
        self._enabled = enabled
        self._console_output = console_output
        self._writer = AuditLogWriter(log_dir=log_dir)
        self._callbacks: List[Callable[[AuditEvent], None]] = []

        # Load settings from environment
        if os.getenv("MASP_AUDIT_ENABLED", "1") == "0":
            self._enabled = False

        if os.getenv("MASP_AUDIT_CONSOLE", "0") == "1":
            self._console_output = True

        logger.info(
            "[AuditLogger] Initialized: enabled=%s, console=%s",
            self._enabled,
            self._console_output,
        )

    @classmethod
    def get_instance(cls) -> "AuditLogger":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register_callback(
        self,
        callback: Callable[[AuditEvent], None],
    ) -> None:
        """Register callback for audit events."""
        self._callbacks.append(callback)

    def log(self, event: AuditEvent) -> bool:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            True if logging was successful
        """
        if not self._enabled:
            return False

        # Write to file
        success = self._writer.write(event)

        # Console output
        if self._console_output:
            log_level = getattr(logging, event.severity.value.upper(), logging.INFO)
            logger.log(
                log_level, "[Audit] %s: %s", event.event_type.value, event.message
            )

        # Callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("[AuditLogger] Callback error: %s", e)

        return success

    # ==================== Trading Operations ====================

    def log_order_placed(
        self,
        exchange: str,
        symbol: str,
        order_id: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "LIMIT",
        **kwargs,
    ) -> bool:
        """Log order placement."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.ORDER_PLACED,
                severity=AuditSeverity.INFO,
                message=f"Order placed: {side} {quantity} {symbol} @ {price}",
                exchange=exchange,
                symbol=symbol,
                order_id=order_id,
                details={
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                },
                **kwargs,
            )
        )

    def log_order_filled(
        self,
        exchange: str,
        symbol: str,
        order_id: str,
        filled_quantity: float,
        filled_price: float,
        **kwargs,
    ) -> bool:
        """Log order fill."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.ORDER_FILLED,
                severity=AuditSeverity.INFO,
                message=f"Order filled: {filled_quantity} {symbol} @ {filled_price}",
                exchange=exchange,
                symbol=symbol,
                order_id=order_id,
                details={
                    "filled_quantity": filled_quantity,
                    "filled_price": filled_price,
                },
                **kwargs,
            )
        )

    def log_order_cancelled(
        self,
        exchange: str,
        symbol: str,
        order_id: str,
        reason: str = "",
        **kwargs,
    ) -> bool:
        """Log order cancellation."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.ORDER_CANCELLED,
                severity=AuditSeverity.INFO,
                message=f"Order cancelled: {order_id} ({reason})",
                exchange=exchange,
                symbol=symbol,
                order_id=order_id,
                details={"reason": reason},
                **kwargs,
            )
        )

    def log_order_rejected(
        self,
        exchange: str,
        symbol: str,
        order_id: Optional[str],
        reason: str,
        **kwargs,
    ) -> bool:
        """Log order rejection."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.ORDER_REJECTED,
                severity=AuditSeverity.WARNING,
                message=f"Order rejected: {reason}",
                exchange=exchange,
                symbol=symbol,
                order_id=order_id,
                details={"reason": reason},
                **kwargs,
            )
        )

    # ==================== Configuration ====================

    def log_config_change(
        self,
        setting: str,
        old_value: Any,
        new_value: Any,
        **kwargs,
    ) -> bool:
        """Log configuration change."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.CONFIG_CHANGED,
                severity=AuditSeverity.INFO,
                message=f"Config changed: {setting}",
                details={"setting": setting},
                before_state={"value": old_value},
                after_state={"value": new_value},
                **kwargs,
            )
        )

    def log_settings_updated(
        self,
        changes: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """Log settings update."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.SETTINGS_UPDATED,
                severity=AuditSeverity.INFO,
                message=f"Settings updated: {len(changes)} changes",
                details={"changes": changes},
                **kwargs,
            )
        )

    def log_strategy_enabled(
        self,
        strategy_id: str,
        exchange: str,
        **kwargs,
    ) -> bool:
        """Log strategy enabled."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.STRATEGY_ENABLED,
                severity=AuditSeverity.INFO,
                message=f"Strategy enabled: {strategy_id} on {exchange}",
                strategy_id=strategy_id,
                exchange=exchange,
                **kwargs,
            )
        )

    def log_strategy_disabled(
        self,
        strategy_id: str,
        exchange: str,
        reason: str = "",
        **kwargs,
    ) -> bool:
        """Log strategy disabled."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.STRATEGY_DISABLED,
                severity=AuditSeverity.INFO,
                message=f"Strategy disabled: {strategy_id} ({reason})",
                strategy_id=strategy_id,
                exchange=exchange,
                details={"reason": reason},
                **kwargs,
            )
        )

    # ==================== Security ====================

    def log_auth_success(
        self,
        user_id: str,
        method: str = "api_key",
        **kwargs,
    ) -> bool:
        """Log successful authentication."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_SUCCESS,
                severity=AuditSeverity.INFO,
                message=f"Auth success: {user_id} via {method}",
                user_id=user_id,
                details={"method": method},
                **kwargs,
            )
        )

    def log_auth_failure(
        self,
        user_id: Optional[str],
        reason: str,
        **kwargs,
    ) -> bool:
        """Log authentication failure."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.AUTH_FAILURE,
                severity=AuditSeverity.WARNING,
                message=f"Auth failure: {reason}",
                user_id=user_id,
                details={"reason": reason},
                **kwargs,
            )
        )

    def log_permission_denied(
        self,
        user_id: str,
        resource: str,
        action: str,
        **kwargs,
    ) -> bool:
        """Log permission denied."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.PERMISSION_DENIED,
                severity=AuditSeverity.WARNING,
                message=f"Permission denied: {user_id} attempted {action} on {resource}",
                user_id=user_id,
                details={"resource": resource, "action": action},
                **kwargs,
            )
        )

    # ==================== System ====================

    def log_system_startup(self, **kwargs) -> bool:
        """Log system startup."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_STARTUP,
                severity=AuditSeverity.INFO,
                message="System startup",
                **kwargs,
            )
        )

    def log_system_shutdown(self, reason: str = "normal", **kwargs) -> bool:
        """Log system shutdown."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_SHUTDOWN,
                severity=AuditSeverity.INFO,
                message=f"System shutdown: {reason}",
                details={"reason": reason},
                **kwargs,
            )
        )

    def log_system_error(
        self,
        error: str,
        component: str = "",
        **kwargs,
    ) -> bool:
        """Log system error."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.SYSTEM_ERROR,
                severity=AuditSeverity.ERROR,
                message=f"System error: {error}",
                details={"error": error, "component": component},
                **kwargs,
            )
        )

    def log_kill_switch_activated(
        self,
        reason: str,
        **kwargs,
    ) -> bool:
        """Log kill switch activation."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.KILL_SWITCH_ACTIVATED,
                severity=AuditSeverity.CRITICAL,
                message=f"Kill switch activated: {reason}",
                details={"reason": reason},
                **kwargs,
            )
        )

    def log_circuit_breaker_triggered(
        self,
        trigger_type: str,
        value: float,
        threshold: float,
        **kwargs,
    ) -> bool:
        """Log circuit breaker trigger."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.CIRCUIT_BREAKER_TRIGGERED,
                severity=AuditSeverity.CRITICAL,
                message=f"Circuit breaker triggered: {trigger_type}",
                details={
                    "trigger_type": trigger_type,
                    "value": value,
                    "threshold": threshold,
                },
                **kwargs,
            )
        )

    # ==================== Data Operations ====================

    def log_backup_created(
        self,
        backup_path: str,
        size_bytes: int,
        **kwargs,
    ) -> bool:
        """Log backup creation."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.BACKUP_CREATED,
                severity=AuditSeverity.INFO,
                message=f"Backup created: {backup_path}",
                details={"path": backup_path, "size_bytes": size_bytes},
                **kwargs,
            )
        )

    def log_backup_restored(
        self,
        backup_path: str,
        **kwargs,
    ) -> bool:
        """Log backup restoration."""
        return self.log(
            AuditEvent(
                event_type=AuditEventType.BACKUP_RESTORED,
                severity=AuditSeverity.WARNING,
                message=f"Backup restored: {backup_path}",
                details={"path": backup_path},
                **kwargs,
            )
        )


# ============================================================================
# Convenience functions
# ============================================================================


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    return AuditLogger.get_instance()


def audit_order_placed(
    exchange: str,
    symbol: str,
    order_id: str,
    side: str,
    quantity: float,
    **kwargs,
) -> bool:
    """Convenience function to log order placement."""
    return get_audit_logger().log_order_placed(
        exchange=exchange,
        symbol=symbol,
        order_id=order_id,
        side=side,
        quantity=quantity,
        **kwargs,
    )


def audit_config_change(
    setting: str,
    old_value: Any,
    new_value: Any,
    **kwargs,
) -> bool:
    """Convenience function to log config change."""
    return get_audit_logger().log_config_change(
        setting=setting,
        old_value=old_value,
        new_value=new_value,
        **kwargs,
    )
