"""
Alert Manager for MASP.

Centralized alert management with:
- Multi-level severity
- Deduplication and rate limiting
- Alert history tracking
- Notification routing
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Alert categories for routing."""

    SYSTEM = "system"  # System resources (CPU, Memory)
    TRADING = "trading"  # Trading issues (orders, fills)
    RISK = "risk"  # Risk management (drawdown, limits)
    CONNECTIVITY = "connectivity"  # Exchange connectivity
    SECURITY = "security"  # Security events


@dataclass
class Alert:
    """Individual alert record."""

    id: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    message: str
    source: str = ""  # Component that raised the alert
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    exchange: Optional[str] = None
    symbol: Optional[str] = None

    # State
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    # Deduplication
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            # Generate fingerprint for deduplication
            content = f"{self.category.value}:{self.severity.value}:{self.title}:{self.source}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class AlertRule:
    """Rule for alert routing and processing."""

    name: str
    category: Optional[AlertCategory] = None
    min_severity: AlertSeverity = AlertSeverity.INFO
    callback: Optional[Callable[[Alert], None]] = None
    rate_limit_seconds: float = 60.0  # Min time between alerts for same fingerprint
    enabled: bool = True


class AlertManager:
    """
    Centralized alert management system.

    Features:
    - Alert creation with severity levels
    - Deduplication and rate limiting
    - Alert routing based on rules
    - Alert acknowledgment and resolution
    - History tracking

    Example:
        manager = AlertManager()

        # Register notification callback
        manager.register_rule(AlertRule(
            name="telegram_critical",
            min_severity=AlertSeverity.CRITICAL,
            callback=send_telegram_alert,
        ))

        # Create alert
        manager.alert(
            category=AlertCategory.RISK,
            severity=AlertSeverity.CRITICAL,
            title="Max Drawdown Exceeded",
            message="Portfolio drawdown reached 15%",
            details={"drawdown_pct": 0.15},
        )

        # Get active alerts
        alerts = manager.get_active_alerts()
    """

    _instance: Optional["AlertManager"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        history_hours: int = 72,
        max_alerts: int = 10000,
        default_rate_limit_seconds: float = 60.0,
    ):
        """
        Initialize alert manager.

        Args:
            history_hours: Hours of history to keep
            max_alerts: Maximum alerts to store
            default_rate_limit_seconds: Default rate limit for dedup
        """
        self._history_hours = history_hours
        self._max_alerts = max_alerts
        self._default_rate_limit = default_rate_limit_seconds

        self._alerts: List[Alert] = []
        self._rules: List[AlertRule] = []
        self._lock = threading.Lock()

        # Rate limiting: fingerprint -> last alert time
        self._last_alert_times: Dict[str, float] = {}

        # Statistics
        self._stats = {
            "total_created": 0,
            "total_suppressed": 0,
            "by_severity": {s.value: 0 for s in AlertSeverity},
            "by_category": {c.value: 0 for c in AlertCategory},
        }

    @classmethod
    def get_instance(cls) -> "AlertManager":
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

    def register_rule(self, rule: AlertRule) -> None:
        """
        Register an alert routing rule.

        Args:
            rule: Rule to register
        """
        with self._lock:
            self._rules.append(rule)
        logger.info("[AlertManager] Rule registered: %s", rule.name)

    def unregister_rule(self, name: str) -> bool:
        """
        Unregister a rule by name.

        Args:
            name: Rule name to unregister

        Returns:
            True if rule was found and removed
        """
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.name != name]
            return len(self._rules) < before

    def alert(
        self,
        category: AlertCategory,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "",
        details: Optional[Dict[str, Any]] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None,
        skip_rate_limit: bool = False,
    ) -> Optional[Alert]:
        """
        Create and process a new alert.

        Args:
            category: Alert category
            severity: Alert severity
            title: Short title
            message: Detailed message
            source: Source component
            details: Additional context
            exchange: Related exchange
            symbol: Related symbol
            skip_rate_limit: Bypass rate limiting

        Returns:
            Alert if created, None if rate-limited
        """
        alert = Alert(
            id=self._generate_id(),
            category=category,
            severity=severity,
            title=title,
            message=message,
            source=source,
            details=details or {},
            exchange=exchange,
            symbol=symbol,
        )

        # Check rate limit
        if not skip_rate_limit and self._is_rate_limited(alert):
            self._stats["total_suppressed"] += 1
            logger.debug("[AlertManager] Alert rate-limited: %s", alert.fingerprint)
            return None

        with self._lock:
            self._alerts.append(alert)
            self._last_alert_times[alert.fingerprint] = time.time()

            # Update stats
            self._stats["total_created"] += 1
            self._stats["by_severity"][severity.value] += 1
            self._stats["by_category"][category.value] += 1

            # Trim old alerts
            if len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[-self._max_alerts :]

        # Log based on severity
        log_msg = f"[Alert:{severity.value.upper()}] {title}: {message}"
        if severity == AlertSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity == AlertSeverity.ERROR:
            logger.error(log_msg)
        elif severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Process rules
        self._process_rules(alert)

        return alert

    def _generate_id(self) -> str:
        """Generate unique alert ID."""
        return f"alert_{int(time.time() * 1000)}_{id(self) % 10000}"

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate-limited."""
        last_time = self._last_alert_times.get(alert.fingerprint)
        if last_time is None:
            return False

        elapsed = time.time() - last_time
        return elapsed < self._default_rate_limit

    def _process_rules(self, alert: Alert) -> None:
        """Process alert against registered rules."""
        with self._lock:
            rules = list(self._rules)

        for rule in rules:
            if not rule.enabled:
                continue

            # Check category match
            if rule.category and rule.category != alert.category:
                continue

            # Check severity
            severity_order = list(AlertSeverity)
            if severity_order.index(alert.severity) < severity_order.index(
                rule.min_severity
            ):
                continue

            # Execute callback
            if rule.callback:
                try:
                    rule.callback(alert)
                except Exception as e:
                    logger.error(
                        "[AlertManager] Rule '%s' callback failed: %s",
                        rule.name,
                        e,
                    )

    def acknowledge(
        self,
        alert_id: str,
        by: str = "system",
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            by: Who acknowledged

        Returns:
            True if alert was found and acknowledged
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    alert.acknowledged_at = datetime.now()
                    alert.acknowledged_by = by
                    return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert was found and resolved
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    return True
        return False

    def get_active_alerts(
        self,
        category: Optional[AlertCategory] = None,
        min_severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get active (unresolved) alerts.

        Args:
            category: Filter by category
            min_severity: Filter by minimum severity

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = [a for a in self._alerts if not a.resolved]

        if category:
            alerts = [a for a in alerts if a.category == category]

        if min_severity:
            severity_order = list(AlertSeverity)
            min_idx = severity_order.index(min_severity)
            alerts = [a for a in alerts if severity_order.index(a.severity) >= min_idx]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(
        self,
        hours: int = 24,
        category: Optional[AlertCategory] = None,
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            hours: Hours of history
            category: Filter by category

        Returns:
            List of alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            alerts = [a for a in self._alerts if a.timestamp >= cutoff]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_count(
        self,
        hours: int = 24,
        by: str = "severity",
    ) -> Dict[str, int]:
        """
        Get alert counts grouped by severity or category.

        Args:
            hours: Hours of history
            by: Group by "severity" or "category"

        Returns:
            Dictionary of counts
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            alerts = [a for a in self._alerts if a.timestamp >= cutoff]

        counts: Dict[str, int] = {}

        for alert in alerts:
            if by == "severity":
                key = alert.severity.value
            elif by == "category":
                key = alert.category.value
            else:
                continue

            counts[key] = counts.get(key, 0) + 1

        return counts

    def get_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            active_count = sum(1 for a in self._alerts if not a.resolved)
            unacked_count = sum(
                1 for a in self._alerts if not a.acknowledged and not a.resolved
            )

        return {
            **self._stats,
            "active_alerts": active_count,
            "unacknowledged": unacked_count,
            "rules_count": len(self._rules),
        }

    def cleanup_old_alerts(self) -> int:
        """
        Remove alerts older than history_hours.

        Returns:
            Number of alerts removed
        """
        cutoff = datetime.now() - timedelta(hours=self._history_hours)

        with self._lock:
            before = len(self._alerts)
            self._alerts = [a for a in self._alerts if a.timestamp >= cutoff]
            removed = before - len(self._alerts)

        if removed > 0:
            logger.info("[AlertManager] Cleaned up %d old alerts", removed)

        return removed

    def clear(self) -> None:
        """Clear all alerts (for testing)."""
        with self._lock:
            self._alerts.clear()
            self._last_alert_times.clear()

    # ==================== Convenience Methods ====================

    def info(
        self,
        title: str,
        message: str,
        category: AlertCategory = AlertCategory.SYSTEM,
        **kwargs,
    ) -> Optional[Alert]:
        """Create INFO alert."""
        return self.alert(category, AlertSeverity.INFO, title, message, **kwargs)

    def warning(
        self,
        title: str,
        message: str,
        category: AlertCategory = AlertCategory.SYSTEM,
        **kwargs,
    ) -> Optional[Alert]:
        """Create WARNING alert."""
        return self.alert(category, AlertSeverity.WARNING, title, message, **kwargs)

    def error(
        self,
        title: str,
        message: str,
        category: AlertCategory = AlertCategory.SYSTEM,
        **kwargs,
    ) -> Optional[Alert]:
        """Create ERROR alert."""
        return self.alert(category, AlertSeverity.ERROR, title, message, **kwargs)

    def critical(
        self,
        title: str,
        message: str,
        category: AlertCategory = AlertCategory.SYSTEM,
        **kwargs,
    ) -> Optional[Alert]:
        """Create CRITICAL alert."""
        return self.alert(category, AlertSeverity.CRITICAL, title, message, **kwargs)


# ============================================================================
# Convenience functions
# ============================================================================


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    return AlertManager.get_instance()


def create_alert(
    category: AlertCategory,
    severity: AlertSeverity,
    title: str,
    message: str,
    **kwargs,
) -> Optional[Alert]:
    """Convenience function to create an alert."""
    return get_alert_manager().alert(category, severity, title, message, **kwargs)
