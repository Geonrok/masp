"""
Event Logger - SSOT for all strategy events.
Provides standardized event types and helper methods.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_serializer
import pytz


class EventType(str, Enum):
    """Standard event types for the platform."""
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    SIGNAL_DECISION = "SIGNAL_DECISION"
    ORDER_ATTEMPT = "ORDER_ATTEMPT"
    FILL_UPDATE = "FILL_UPDATE"
    ERROR = "ERROR"
    HEARTBEAT = "HEARTBEAT"


class Severity(str, Enum):
    """Event severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Event(BaseModel):
    """
    Standard event model - SSOT for all platform events.
    All events share these common fields.
    """
    model_config = ConfigDict()

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ts_kst: datetime = Field(default_factory=lambda: datetime.now(pytz.timezone("Asia/Seoul")))
    
    asset_class: str
    strategy_id: Optional[str] = None
    run_id: str
    symbol: Optional[str] = None
    
    severity: Severity = Severity.INFO
    event_type: EventType
    
    payload: dict[str, Any] = Field(default_factory=dict)
    
    @field_serializer("ts_utc", "ts_kst")
    def _serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()
    
    def to_dict(self) -> dict:
        """Convert event to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "ts_utc": self.ts_utc.isoformat(),
            "ts_kst": self.ts_kst.isoformat(),
            "asset_class": self.asset_class,
            "strategy_id": self.strategy_id,
            "run_id": self.run_id,
            "symbol": self.symbol,
            "severity": self.severity.value,
            "event_type": self.event_type.value,
            "payload": self.payload,
        }


class EventLogger:
    """
    Event logger that emits standardized events.
    Provides helper methods for each event type.
    """
    
    def __init__(
        self,
        asset_class: str,
        run_id: str,
        event_store: Any,  # EventStore instance
        default_strategy_id: Optional[str] = None,
    ):
        self.asset_class = asset_class
        self.run_id = run_id
        self.event_store = event_store
        self.default_strategy_id = default_strategy_id
        self._kst = pytz.timezone("Asia/Seoul")
    
    def _now_utc(self) -> datetime:
        """Get current UTC time."""
        return datetime.now(timezone.utc)
    
    def _now_kst(self) -> datetime:
        """Get current KST time."""
        return datetime.now(self._kst)
    
    def _emit(self, event: Event) -> Event:
        """Emit an event to the store."""
        self.event_store.write_event(event)
        return event
    
    def emit_run_started(
        self,
        config_version: str,
        enabled_strategies: list[str],
        symbols: list[str],
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit RUN_STARTED event."""
        payload = {
            "config_version": config_version,
            "enabled_strategies": enabled_strategies,
            "symbols": symbols,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            run_id=self.run_id,
            event_type=EventType.RUN_STARTED,
            severity=Severity.INFO,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_run_finished(
        self,
        status: str,  # "success" or "fail"
        decision_count: int = 0,
        error_count: int = 0,
        duration_seconds: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit RUN_FINISHED event."""
        payload = {
            "status": status,
            "decision_count": decision_count,
            "error_count": error_count,
            "duration_seconds": duration_seconds,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            run_id=self.run_id,
            event_type=EventType.RUN_FINISHED,
            severity=Severity.INFO if status == "success" else Severity.ERROR,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_signal_decision(
        self,
        strategy_id: str,
        symbol: str,
        action: str,  # BUY/SELL/HOLD/SKIP
        notes: Optional[str] = None,
        metrics: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit SIGNAL_DECISION event."""
        payload = {
            "action": action,
            "notes": notes,
            "metrics": metrics or {},
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            strategy_id=strategy_id,
            run_id=self.run_id,
            symbol=symbol,
            event_type=EventType.SIGNAL_DECISION,
            severity=Severity.INFO,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_order_attempt(
        self,
        strategy_id: str,
        symbol: str,
        side: str,  # BUY/SELL
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET",
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit ORDER_ATTEMPT event (mock only in Phase 0)."""
        payload = {
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_type": order_type,
            "mock": True,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            strategy_id=strategy_id,
            run_id=self.run_id,
            symbol=symbol,
            event_type=EventType.ORDER_ATTEMPT,
            severity=Severity.INFO,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_fill_update(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        filled_quantity: float,
        fill_price: float,
        order_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit FILL_UPDATE event (mock only in Phase 0)."""
        payload = {
            "side": side,
            "filled_quantity": filled_quantity,
            "fill_price": fill_price,
            "order_id": order_id,
            "mock": True,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            strategy_id=strategy_id,
            run_id=self.run_id,
            symbol=symbol,
            event_type=EventType.FILL_UPDATE,
            severity=Severity.INFO,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_error(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        strategy_id: Optional[str] = None,
        symbol: Optional[str] = None,
        traceback: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit ERROR event."""
        payload = {
            "error_message": error_message,
            "error_type": error_type,
            "traceback": traceback,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            strategy_id=strategy_id or self.default_strategy_id,
            run_id=self.run_id,
            symbol=symbol,
            event_type=EventType.ERROR,
            severity=Severity.ERROR,
            payload=payload,
        )
        return self._emit(event)
    
    def emit_heartbeat(
        self,
        build_version: str,
        config_version: str,
        uptime_seconds: Optional[float] = None,
        extra: Optional[dict] = None,
    ) -> Event:
        """Emit HEARTBEAT event."""
        payload = {
            "build_version": build_version,
            "config_version": config_version,
            "uptime_seconds": uptime_seconds,
            **(extra or {}),
        }
        event = Event(
            ts_utc=self._now_utc(),
            ts_kst=self._now_kst(),
            asset_class=self.asset_class,
            run_id=self.run_id,
            event_type=EventType.HEARTBEAT,
            severity=Severity.DEBUG,
            payload=payload,
        )
        return self._emit(event)

