"""
AlertManager - 중앙화된 알림 관리 시스템
- 우선순위 기반 알림 처리
- 알림 규칙 (필터링, 집계)
- 시스템 이상 감지
- 알림 히스토리 저장
"""
from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """알림 우선순위."""

    CRITICAL = 4  # 즉시 처리 필요 (시스템 장애, 큰 손실)
    HIGH = 3  # 중요 (주문 실패, API 오류)
    NORMAL = 2  # 일반 (거래 체결, 시그널)
    LOW = 1  # 낮음 (정보성 메시지)


class AlertType(Enum):
    """알림 유형."""

    TRADE = "TRADE"  # 거래 관련
    SIGNAL = "SIGNAL"  # 시그널 발생
    ERROR = "ERROR"  # 오류 발생
    SYSTEM = "SYSTEM"  # 시스템 상태
    DAILY = "DAILY"  # 일일 리포트
    ANOMALY = "ANOMALY"  # 이상 감지


@dataclass
class Alert:
    """알림 데이터 구조."""

    id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    symbol: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    sent_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "priority": self.priority.name,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "exchange": self.exchange,
            "symbol": self.symbol,
            "metadata": self.metadata,
            "sent": self.sent,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            alert_type=AlertType(data.get("alert_type", "SYSTEM")),
            priority=AlertPriority[data.get("priority", "NORMAL")],
            title=data.get("title", ""),
            message=data.get("message", ""),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if data.get("timestamp")
            else datetime.now(),
            exchange=data.get("exchange", ""),
            symbol=data.get("symbol", ""),
            metadata=data.get("metadata", {}),
            sent=data.get("sent", False),
            sent_at=datetime.fromisoformat(data["sent_at"])
            if data.get("sent_at")
            else None,
        )


@dataclass
class AlertRule:
    """알림 규칙 (필터링/라우팅)."""

    name: str
    enabled: bool = True
    alert_types: List[AlertType] = field(default_factory=list)  # 빈 리스트 = 모든 타입
    min_priority: AlertPriority = AlertPriority.LOW
    exchanges: List[str] = field(default_factory=list)  # 빈 리스트 = 모든 거래소
    symbols: List[str] = field(default_factory=list)  # 빈 리스트 = 모든 심볼
    cooldown_seconds: int = 0  # 동일 타입 알림 간격 (0=무제한)
    aggregate_count: int = 0  # N개 이상 누적 시 알림 (0=즉시)

    def matches(self, alert: Alert) -> bool:
        """Check if alert matches this rule."""
        if not self.enabled:
            return False

        # Check alert type
        if self.alert_types and alert.alert_type not in self.alert_types:
            return False

        # Check priority
        if alert.priority.value < self.min_priority.value:
            return False

        # Check exchange
        if self.exchanges and alert.exchange.lower() not in [
            e.lower() for e in self.exchanges
        ]:
            return False

        # Check symbol
        if self.symbols and alert.symbol.upper() not in [s.upper() for s in self.symbols]:
            return False

        return True


class AlertManager:
    """중앙화된 알림 관리자."""

    DEFAULT_HISTORY_DIR = "data/alerts"

    def __init__(
        self,
        notifier: Optional[Any] = None,
        history_dir: Optional[str] = None,
        max_history: int = 1000,
    ):
        """초기화.

        Args:
            notifier: TelegramNotifier 인스턴스 (None이면 비활성)
            history_dir: 알림 히스토리 저장 경로
            max_history: 메모리에 유지할 최대 알림 수
        """
        self.notifier = notifier
        self.history_dir = Path(history_dir or self.DEFAULT_HISTORY_DIR)
        self.max_history = max_history

        self._alerts: List[Alert] = []
        self._rules: List[AlertRule] = []
        self._last_sent: Dict[str, datetime] = {}  # type -> last_sent_time
        self._pending: Dict[str, List[Alert]] = {}  # type -> pending alerts (for aggregation)
        self._counter = 0

        self._ensure_directory()
        self._load_rules()

    def _ensure_directory(self) -> None:
        """Create history directory."""
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _load_rules(self) -> None:
        """Load rules from config file."""
        rules_file = self.history_dir / "rules.json"
        if rules_file.exists():
            try:
                with open(rules_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for rule_data in data.get("rules", []):
                    rule = AlertRule(
                        name=rule_data.get("name", ""),
                        enabled=rule_data.get("enabled", True),
                        alert_types=[
                            AlertType(t) for t in rule_data.get("alert_types", [])
                        ],
                        min_priority=AlertPriority[
                            rule_data.get("min_priority", "LOW")
                        ],
                        exchanges=rule_data.get("exchanges", []),
                        symbols=rule_data.get("symbols", []),
                        cooldown_seconds=rule_data.get("cooldown_seconds", 0),
                        aggregate_count=rule_data.get("aggregate_count", 0),
                    )
                    self._rules.append(rule)
                logger.info("[AlertManager] Loaded %d rules", len(self._rules))
            except Exception as e:
                logger.warning("[AlertManager] Failed to load rules: %s", e)

        # Default rule if none exist
        if not self._rules:
            self._rules.append(
                AlertRule(
                    name="default",
                    enabled=True,
                    min_priority=AlertPriority.NORMAL,
                )
            )

    def save_rules(self) -> None:
        """Save rules to config file."""
        rules_file = self.history_dir / "rules.json"
        try:
            data = {
                "rules": [
                    {
                        "name": r.name,
                        "enabled": r.enabled,
                        "alert_types": [t.value for t in r.alert_types],
                        "min_priority": r.min_priority.name,
                        "exchanges": r.exchanges,
                        "symbols": r.symbols,
                        "cooldown_seconds": r.cooldown_seconds,
                        "aggregate_count": r.aggregate_count,
                    }
                    for r in self._rules
                ]
            }
            with open(rules_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("[AlertManager] Failed to save rules: %s", e)

    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        self._counter += 1
        return f"ALT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._counter:04d}"

    def create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        exchange: str = "",
        symbol: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and process a new alert.

        Args:
            alert_type: 알림 유형
            priority: 우선순위
            title: 제목
            message: 메시지
            exchange: 거래소 (선택)
            symbol: 심볼 (선택)
            metadata: 추가 데이터 (선택)

        Returns:
            생성된 Alert 객체
        """
        alert = Alert(
            id=self._generate_id(),
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            exchange=exchange,
            symbol=symbol,
            metadata=metadata or {},
        )

        # Add to history
        self._alerts.append(alert)
        if len(self._alerts) > self.max_history:
            self._alerts = self._alerts[-self.max_history :]

        # Process through rules
        self._process_alert(alert)

        return alert

    def _process_alert(self, alert: Alert) -> None:
        """Process alert through rules and send if needed."""
        for rule in self._rules:
            if not rule.matches(alert):
                continue

            # Check cooldown
            type_key = f"{rule.name}:{alert.alert_type.value}"
            if rule.cooldown_seconds > 0:
                last_sent = self._last_sent.get(type_key)
                if last_sent:
                    elapsed = (datetime.now() - last_sent).total_seconds()
                    if elapsed < rule.cooldown_seconds:
                        logger.debug(
                            "[AlertManager] Cooldown active for %s (%.1fs remaining)",
                            type_key,
                            rule.cooldown_seconds - elapsed,
                        )
                        continue

            # Check aggregation
            if rule.aggregate_count > 0:
                if type_key not in self._pending:
                    self._pending[type_key] = []
                self._pending[type_key].append(alert)

                if len(self._pending[type_key]) >= rule.aggregate_count:
                    self._send_aggregated(type_key, self._pending[type_key])
                    self._pending[type_key] = []
                continue

            # Send immediately
            self._send_alert(alert)
            self._last_sent[type_key] = datetime.now()
            break  # Stop after first matching rule

    def _send_alert(self, alert: Alert) -> bool:
        """Send alert via notifier."""
        if not self.notifier:
            logger.debug("[AlertManager] No notifier configured")
            return False

        try:
            formatted = self._format_alert(alert)
            success = self.notifier.send_message_sync(formatted)

            alert.sent = success
            alert.sent_at = datetime.now() if success else None

            if success:
                logger.info("[AlertManager] Sent alert: %s", alert.id)
            else:
                logger.warning("[AlertManager] Failed to send alert: %s", alert.id)

            return success
        except Exception as e:
            logger.error("[AlertManager] Send error: %s", e)
            return False

    def _send_aggregated(self, type_key: str, alerts: List[Alert]) -> None:
        """Send aggregated alert summary."""
        if not alerts:
            return

        count = len(alerts)
        first = alerts[0]

        summary = self.create_alert(
            alert_type=first.alert_type,
            priority=first.priority,
            title=f"[Aggregated] {count} alerts",
            message=f"{count} {first.alert_type.value} alerts since {first.timestamp.strftime('%H:%M:%S')}",
            exchange=first.exchange,
        )

        # Mark all as sent via aggregation
        for alert in alerts:
            alert.sent = True
            alert.sent_at = datetime.now()

    def _format_alert(self, alert: Alert) -> str:
        """Format alert for Telegram (HTML)."""
        # Priority emoji
        priority_emoji = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.NORMAL: "📢",
            AlertPriority.LOW: "ℹ️",
        }

        # Type emoji
        type_emoji = {
            AlertType.TRADE: "💰",
            AlertType.SIGNAL: "📊",
            AlertType.ERROR: "❌",
            AlertType.SYSTEM: "⚙️",
            AlertType.DAILY: "📅",
            AlertType.ANOMALY: "🔍",
        }

        emoji = priority_emoji.get(alert.priority, "📢")
        type_icon = type_emoji.get(alert.alert_type, "📢")

        lines = [
            f"{emoji}{type_icon} <b>{html.escape(alert.title)}</b>",
        ]

        if alert.exchange:
            lines.append(f"Exchange: {html.escape(alert.exchange.upper())}")

        if alert.symbol:
            lines.append(f"Symbol: {html.escape(alert.symbol)}")

        lines.append(f"\n{html.escape(alert.message)}")
        lines.append(f"\n<code>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</code>")

        return "\n".join(lines)

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        priority: Optional[AlertPriority] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get filtered alerts from history.

        Args:
            alert_type: Filter by type
            priority: Filter by minimum priority
            since: Filter by timestamp (after)
            limit: Maximum number of alerts to return

        Returns:
            List of matching alerts (newest first)
        """
        result = self._alerts.copy()

        if alert_type:
            result = [a for a in result if a.alert_type == alert_type]

        if priority:
            result = [a for a in result if a.priority.value >= priority.value]

        if since:
            result = [a for a in result if a.timestamp >= since]

        # Sort by timestamp descending
        result.sort(key=lambda a: a.timestamp, reverse=True)

        return result[:limit]

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return self._rules.copy()

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self._rules.append(rule)
        self.save_rules()

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name."""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                self.save_rules()
                return True
        return False

    def update_rule(self, name: str, **kwargs) -> bool:
        """Update a rule by name."""
        for rule in self._rules:
            if rule.name == name:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self.save_rules()
                return True
        return False

    # ==========================================================================
    # Convenience methods for common alert types
    # ==========================================================================

    def trade_alert(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        status: str = "FILLED",
        pnl: Optional[float] = None,
    ) -> Alert:
        """Create trade execution alert."""
        side_upper = side.upper()
        title = f"{side_upper} {symbol}"

        lines = [
            f"Quantity: {quantity:.8g}",
            f"Price: {price:,.0f} KRW",
            f"Status: {status}",
        ]
        if pnl is not None:
            lines.append(f"PnL: {pnl:+,.0f} KRW")

        return self.create_alert(
            alert_type=AlertType.TRADE,
            priority=AlertPriority.NORMAL,
            title=title,
            message="\n".join(lines),
            exchange=exchange,
            symbol=symbol,
            metadata={
                "side": side_upper,
                "quantity": quantity,
                "price": price,
                "status": status,
                "pnl": pnl,
            },
        )

    def signal_alert(
        self,
        strategy: str,
        symbol: str,
        signal: str,
        confidence: Optional[float] = None,
    ) -> Alert:
        """Create trading signal alert."""
        title = f"Signal: {signal} {symbol}"

        lines = [f"Strategy: {strategy}"]
        if confidence is not None:
            lines.append(f"Confidence: {confidence:.1%}")

        return self.create_alert(
            alert_type=AlertType.SIGNAL,
            priority=AlertPriority.NORMAL,
            title=title,
            message="\n".join(lines),
            symbol=symbol,
            metadata={
                "strategy": strategy,
                "signal": signal,
                "confidence": confidence,
            },
        )

    def error_alert(
        self,
        title: str,
        error_message: str,
        exchange: str = "",
        critical: bool = False,
    ) -> Alert:
        """Create error alert."""
        return self.create_alert(
            alert_type=AlertType.ERROR,
            priority=AlertPriority.CRITICAL if critical else AlertPriority.HIGH,
            title=title,
            message=error_message,
            exchange=exchange,
        )

    def system_alert(
        self,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
    ) -> Alert:
        """Create system status alert."""
        return self.create_alert(
            alert_type=AlertType.SYSTEM,
            priority=priority,
            title=title,
            message=message,
        )

    def anomaly_alert(
        self,
        metric: str,
        current_value: float,
        threshold: float,
        message: str = "",
    ) -> Alert:
        """Create anomaly detection alert."""
        title = f"Anomaly: {metric}"

        lines = [
            f"Current: {current_value:.2f}",
            f"Threshold: {threshold:.2f}",
        ]
        if message:
            lines.append(message)

        return self.create_alert(
            alert_type=AlertType.ANOMALY,
            priority=AlertPriority.HIGH,
            title=title,
            message="\n".join(lines),
            metadata={
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold,
            },
        )


# =============================================================================
# System Anomaly Detector
# =============================================================================


class AnomalyDetector:
    """시스템 이상 감지기."""

    DEFAULT_THRESHOLDS = {
        "cpu_percent": 90.0,
        "memory_percent": 90.0,
        "disk_percent": 95.0,
        "api_error_rate": 0.1,  # 10%
        "response_time_ms": 5000,
    }

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """초기화.

        Args:
            alert_manager: AlertManager 인스턴스
            thresholds: 임계값 설정 (기본값 사용 시 None)
        """
        self.alert_manager = alert_manager
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._last_check: Dict[str, datetime] = {}
        self._cooldown_seconds = 300  # 동일 메트릭 알림 간격

    def check_system_resources(self) -> List[Dict[str, Any]]:
        """Check system resources and return anomalies.

        Returns:
            List of detected anomalies
        """
        anomalies = []

        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.thresholds["cpu_percent"]:
                anomalies.append(
                    self._create_anomaly(
                        "cpu_percent",
                        cpu_percent,
                        f"High CPU usage: {cpu_percent:.1f}%",
                    )
                )

            # Memory
            memory = psutil.virtual_memory()
            if memory.percent > self.thresholds["memory_percent"]:
                anomalies.append(
                    self._create_anomaly(
                        "memory_percent",
                        memory.percent,
                        f"High memory usage: {memory.percent:.1f}%",
                    )
                )

            # Disk
            disk = psutil.disk_usage("/")
            if disk.percent > self.thresholds["disk_percent"]:
                anomalies.append(
                    self._create_anomaly(
                        "disk_percent",
                        disk.percent,
                        f"Low disk space: {disk.percent:.1f}% used",
                    )
                )

        except ImportError:
            logger.debug("[AnomalyDetector] psutil not available")
        except Exception as e:
            logger.warning("[AnomalyDetector] Error checking resources: %s", e)

        return anomalies

    def check_api_health(
        self,
        total_requests: int,
        error_count: int,
        avg_response_ms: float,
    ) -> List[Dict[str, Any]]:
        """Check API health metrics.

        Args:
            total_requests: Total API requests
            error_count: Number of errors
            avg_response_ms: Average response time in ms

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if total_requests > 0:
            error_rate = error_count / total_requests
            if error_rate > self.thresholds["api_error_rate"]:
                anomalies.append(
                    self._create_anomaly(
                        "api_error_rate",
                        error_rate,
                        f"High API error rate: {error_rate:.1%}",
                    )
                )

        if avg_response_ms > self.thresholds["response_time_ms"]:
            anomalies.append(
                self._create_anomaly(
                    "response_time_ms",
                    avg_response_ms,
                    f"Slow API response: {avg_response_ms:.0f}ms",
                )
            )

        return anomalies

    def _create_anomaly(
        self, metric: str, value: float, message: str
    ) -> Dict[str, Any]:
        """Create anomaly record and optionally send alert."""
        anomaly = {
            "metric": metric,
            "value": value,
            "threshold": self.thresholds.get(metric, 0),
            "message": message,
            "timestamp": datetime.now(),
        }

        # Check cooldown
        last_check = self._last_check.get(metric)
        if last_check:
            elapsed = (datetime.now() - last_check).total_seconds()
            if elapsed < self._cooldown_seconds:
                return anomaly

        # Send alert if manager available
        if self.alert_manager:
            self.alert_manager.anomaly_alert(
                metric=metric,
                current_value=value,
                threshold=self.thresholds.get(metric, 0),
                message=message,
            )
            self._last_check[metric] = datetime.now()

        return anomaly


# =============================================================================
# Factory function
# =============================================================================


def get_alert_manager(
    notifier: Optional[Any] = None,
) -> AlertManager:
    """Get AlertManager instance.

    Args:
        notifier: TelegramNotifier instance (optional)

    Returns:
        AlertManager instance
    """
    if notifier is None:
        try:
            from libs.notifications.telegram import TelegramNotifier

            notifier = TelegramNotifier()
        except Exception:
            pass

    return AlertManager(notifier=notifier)
