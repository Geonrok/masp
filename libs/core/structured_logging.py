"""
Structured Logging for MASP

Provides JSON-formatted logging with contextual information
for production monitoring, debugging, and audit trails.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar

# Context variables for request/trace tracking
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_trace_id() -> Optional[str]:
    """Get current trace ID."""
    return _trace_id.get()


def set_trace_id(trace_id: str) -> None:
    """Set trace ID for current context."""
    _trace_id.set(trace_id)


def generate_trace_id() -> str:
    """Generate a new trace ID."""
    trace_id = str(uuid.uuid4())[:8]
    set_trace_id(trace_id)
    return trace_id


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs log records as JSON objects with standardized fields.
    """

    STANDARD_FIELDS = {
        "timestamp",
        "level",
        "logger",
        "message",
        "trace_id",
        "span_id",
        "correlation_id",
    }

    def __init__(
        self,
        *,
        include_timestamp: bool = True,
        include_traceback: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
        indent: Optional[int] = None,
    ):
        """
        Initialize JSON formatter.

        Args:
            include_timestamp: Include ISO timestamp
            include_traceback: Include traceback for exceptions
            extra_fields: Additional fields to include in every log
            indent: JSON indent (None for compact)
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_traceback = include_traceback
        self.extra_fields = extra_fields or {}
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {}

        # Timestamp
        if self.include_timestamp:
            log_data["timestamp"] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

        # Standard fields
        log_data["level"] = record.levelname
        log_data["logger"] = record.name
        log_data["message"] = record.getMessage()

        # Source location
        log_data["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Trace context
        if trace_id := get_trace_id():
            log_data["trace_id"] = trace_id
        if span_id := _span_id.get():
            log_data["span_id"] = span_id
        if correlation_id := _correlation_id.get():
            log_data["correlation_id"] = correlation_id

        # Exception information
        if record.exc_info:
            if self.include_traceback:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "traceback": self._format_traceback(record.exc_info),
                }
            else:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                }

        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                if key not in self.STANDARD_FIELDS:
                    log_data[key] = self._serialize_value(value)

        # Global extra fields
        log_data.update(self.extra_fields)

        return json.dumps(log_data, default=str, indent=self.indent)

    def _format_traceback(self, exc_info) -> list:
        """Format exception traceback as list of frames."""
        if not exc_info[2]:
            return []

        frames = []
        for frame in traceback.extract_tb(exc_info[2]):
            frames.append(
                {
                    "file": frame.filename,
                    "line": frame.lineno,
                    "function": frame.name,
                    "code": frame.line,
                }
            )
        return frames

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON output."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return str(value)


class AuditLogger:
    """
    Specialized logger for audit trails.

    Records important events with full context for compliance
    and debugging purposes.
    """

    def __init__(
        self,
        name: str = "masp.audit",
        log_file: Optional[str] = None,
    ):
        """
        Initialize audit logger.

        Args:
            name: Logger name
            log_file: Path to audit log file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Ensure we have at least one handler
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter(indent=2))
            self.logger.addHandler(file_handler)

    def log_trade(
        self,
        *,
        action: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        strategy: Optional[str] = None,
        exchange: Optional[str] = None,
        **extra,
    ) -> None:
        """
        Log a trade event.

        Args:
            action: Trade action (e.g., "ORDER_PLACED", "ORDER_FILLED")
            symbol: Trading symbol
            side: Buy/Sell
            quantity: Order quantity
            price: Order price
            order_id: Order identifier
            strategy: Strategy name
            exchange: Exchange name
            **extra: Additional fields
        """
        self.logger.info(
            f"TRADE: {action} {side} {quantity} {symbol} @ {price}",
            extra={
                "event_type": "TRADE",
                "action": action,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_id": order_id,
                "strategy": strategy,
                "exchange": exchange,
                "value": quantity * price,
                **extra,
            },
        )

    def log_signal(
        self,
        *,
        strategy: str,
        symbol: str,
        signal: str,
        strength: float = 0.0,
        reason: Optional[str] = None,
        **extra,
    ) -> None:
        """
        Log a trading signal.

        Args:
            strategy: Strategy that generated the signal
            symbol: Trading symbol
            signal: Signal type (BUY, SELL, HOLD)
            strength: Signal strength (0-1)
            reason: Reason for signal
            **extra: Additional fields
        """
        self.logger.info(
            f"SIGNAL: {strategy} -> {signal} {symbol}",
            extra={
                "event_type": "SIGNAL",
                "strategy": strategy,
                "symbol": symbol,
                "signal": signal,
                "strength": strength,
                "reason": reason,
                **extra,
            },
        )

    def log_risk_event(
        self,
        *,
        event: str,
        level: str = "WARNING",
        metric: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        **extra,
    ) -> None:
        """
        Log a risk management event.

        Args:
            event: Risk event description
            level: Event severity (INFO, WARNING, CRITICAL)
            metric: Risk metric name
            value: Current value
            threshold: Threshold that was breached
            **extra: Additional fields
        """
        log_level = getattr(logging, level.upper(), logging.WARNING)
        self.logger.log(
            log_level,
            f"RISK: {event}",
            extra={
                "event_type": "RISK",
                "event": event,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                **extra,
            },
        )

    def log_system_event(
        self,
        *,
        event: str,
        component: str,
        status: str = "INFO",
        **extra,
    ) -> None:
        """
        Log a system event.

        Args:
            event: Event description
            component: System component
            status: Event status
            **extra: Additional fields
        """
        self.logger.info(
            f"SYSTEM: [{component}] {event}",
            extra={
                "event_type": "SYSTEM",
                "event": event,
                "component": component,
                "status": status,
                **extra,
            },
        )


def configure_logging(
    *,
    level: Union[str, int] = logging.INFO,
    json_format: bool = True,
    log_file: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level
        json_format: Use JSON formatting
        log_file: Optional log file path
        extra_fields: Extra fields to include in all logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if json_format:
        console_handler.setFormatter(JSONFormatter(extra_fields=extra_fields))
    else:
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)

        if json_format:
            file_handler.setFormatter(JSONFormatter(extra_fields=extra_fields))
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )

        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger
    """
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding temporary context to logs.

    Example:
        with LogContext(request_id="abc123", user_id="user1"):
            logger.info("Processing request")  # includes request_id and user_id
    """

    def __init__(self, **context):
        """
        Initialize log context.

        Args:
            **context: Context fields to add to logs
        """
        self.context = context
        self._tokens = {}

    def __enter__(self):
        """Enter context and set trace variables."""
        if "trace_id" in self.context:
            set_trace_id(self.context["trace_id"])
        if "correlation_id" in self.context:
            _correlation_id.set(self.context["correlation_id"])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass
