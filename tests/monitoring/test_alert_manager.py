"""
Tests for Alert Manager.
"""

import pytest
import time
from datetime import datetime, timedelta

from libs.monitoring.alert_manager import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertCategory,
    AlertRule,
    get_alert_manager,
    create_alert,
)


@pytest.fixture
def manager():
    """Create test alert manager."""
    AlertManager.reset()
    return AlertManager(
        history_hours=24,
        max_alerts=100,
        default_rate_limit_seconds=1.0,  # Short for testing
    )


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_values(self):
        """Test enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertCategory:
    """Tests for AlertCategory enum."""

    def test_values(self):
        """Test enum values."""
        assert AlertCategory.SYSTEM.value == "system"
        assert AlertCategory.TRADING.value == "trading"
        assert AlertCategory.RISK.value == "risk"
        assert AlertCategory.CONNECTIVITY.value == "connectivity"
        assert AlertCategory.SECURITY.value == "security"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="test-1",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
            source="test",
        )
        assert alert.id == "test-1"
        assert alert.category == AlertCategory.SYSTEM
        assert alert.severity == AlertSeverity.WARNING
        assert alert.acknowledged is False
        assert alert.resolved is False

    def test_fingerprint_generation(self):
        """Test fingerprint is generated."""
        alert = Alert(
            id="test-1",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
        )
        assert alert.fingerprint != ""
        assert len(alert.fingerprint) == 12


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_init(self, manager):
        """Test initialization."""
        assert manager._history_hours == 24
        assert manager._max_alerts == 100

    def test_singleton(self):
        """Test singleton pattern."""
        AlertManager.reset()
        m1 = AlertManager.get_instance()
        m2 = AlertManager.get_instance()
        assert m1 is m2
        AlertManager.reset()

    def test_create_alert(self, manager):
        """Test creating an alert."""
        alert = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="Test message",
        )
        assert alert is not None
        assert alert.category == AlertCategory.SYSTEM
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_with_details(self, manager):
        """Test alert with additional details."""
        alert = manager.alert(
            category=AlertCategory.TRADING,
            severity=AlertSeverity.ERROR,
            title="Order Failed",
            message="Order execution failed",
            details={"order_id": "123", "error": "timeout"},
            exchange="upbit",
            symbol="BTC/KRW",
        )
        assert alert.details["order_id"] == "123"
        assert alert.exchange == "upbit"
        assert alert.symbol == "BTC/KRW"

    def test_alert_rate_limiting(self, manager):
        """Test alert rate limiting."""
        # First alert should succeed
        alert1 = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
        )
        assert alert1 is not None

        # Second alert with same fingerprint should be rate-limited
        alert2 = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
        )
        assert alert2 is None

        # Wait for rate limit
        time.sleep(1.1)

        # Now it should succeed
        alert3 = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
        )
        assert alert3 is not None

    def test_skip_rate_limit(self, manager):
        """Test skipping rate limit."""
        alert1 = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
        )
        assert alert1 is not None

        # Force through rate limit
        alert2 = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Same Alert",
            message="Same message",
            skip_rate_limit=True,
        )
        assert alert2 is not None

    def test_acknowledge_alert(self, manager):
        """Test acknowledging an alert."""
        alert = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
        )

        success = manager.acknowledge(alert.id, by="tester")
        assert success

        # Verify the alert is acknowledged
        active = manager.get_active_alerts()
        for a in active:
            if a.id == alert.id:
                assert a.acknowledged
                assert a.acknowledged_by == "tester"

    def test_acknowledge_nonexistent(self, manager):
        """Test acknowledging nonexistent alert."""
        success = manager.acknowledge("nonexistent-id")
        assert not success

    def test_resolve_alert(self, manager):
        """Test resolving an alert."""
        alert = manager.alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
        )

        success = manager.resolve(alert.id)
        assert success

        # Resolved alerts should not appear in active
        active = manager.get_active_alerts()
        assert all(a.id != alert.id for a in active)

    def test_get_active_alerts(self, manager):
        """Test getting active alerts."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Info", "Message")
        manager.alert(
            AlertCategory.TRADING,
            AlertSeverity.WARNING,
            "Warning",
            "Message",
            skip_rate_limit=True,
        )

        active = manager.get_active_alerts()
        assert len(active) == 2

    def test_get_active_alerts_by_category(self, manager):
        """Test filtering active alerts by category."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "System", "Message")
        manager.alert(
            AlertCategory.TRADING,
            AlertSeverity.INFO,
            "Trading",
            "Message",
            skip_rate_limit=True,
        )

        system_alerts = manager.get_active_alerts(category=AlertCategory.SYSTEM)
        assert len(system_alerts) == 1
        assert system_alerts[0].category == AlertCategory.SYSTEM

    def test_get_active_alerts_by_severity(self, manager):
        """Test filtering active alerts by severity."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Info", "Message")
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.ERROR,
            "Error",
            "Message",
            skip_rate_limit=True,
        )

        error_alerts = manager.get_active_alerts(min_severity=AlertSeverity.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].severity == AlertSeverity.ERROR

    def test_get_alert_history(self, manager):
        """Test getting alert history."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Test1", "Message")
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "Test2",
            "Message",
            skip_rate_limit=True,
        )

        history = manager.get_alert_history(hours=1)
        assert len(history) == 2

    def test_get_alert_count(self, manager):
        """Test getting alert counts."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Info", "Message")
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.WARNING,
            "Warning",
            "Message",
            skip_rate_limit=True,
        )
        manager.alert(
            AlertCategory.TRADING,
            AlertSeverity.ERROR,
            "Error",
            "Message",
            skip_rate_limit=True,
        )

        by_severity = manager.get_alert_count(hours=1, by="severity")
        assert by_severity["info"] == 1
        assert by_severity["warning"] == 1
        assert by_severity["error"] == 1

        by_category = manager.get_alert_count(hours=1, by="category")
        assert by_category["system"] == 2
        assert by_category["trading"] == 1

    def test_get_stats(self, manager):
        """Test getting statistics."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Test1", "Message")
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.WARNING,
            "Test2",
            "Message",
            skip_rate_limit=True,
        )

        stats = manager.get_stats()
        assert stats["total_created"] == 2
        assert stats["active_alerts"] == 2

    def test_alert_rule(self, manager):
        """Test alert rule callback."""
        received = []

        def callback(alert):
            received.append(alert)

        rule = AlertRule(
            name="test_rule",
            min_severity=AlertSeverity.WARNING,
            callback=callback,
        )
        manager.register_rule(rule)

        # INFO should not trigger callback
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Info", "Message")
        assert len(received) == 0

        # WARNING should trigger
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.WARNING,
            "Warning",
            "Message",
            skip_rate_limit=True,
        )
        assert len(received) == 1

    def test_alert_rule_by_category(self, manager):
        """Test alert rule with category filter."""
        received = []

        def callback(alert):
            received.append(alert)

        rule = AlertRule(
            name="trading_only",
            category=AlertCategory.TRADING,
            callback=callback,
        )
        manager.register_rule(rule)

        # System should not trigger
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.ERROR, "System", "Message")
        assert len(received) == 0

        # Trading should trigger
        manager.alert(
            AlertCategory.TRADING,
            AlertSeverity.INFO,
            "Trading",
            "Message",
            skip_rate_limit=True,
        )
        assert len(received) == 1

    def test_unregister_rule(self, manager):
        """Test unregistering a rule."""
        rule = AlertRule(name="test_rule", callback=lambda a: None)
        manager.register_rule(rule)

        success = manager.unregister_rule("test_rule")
        assert success

        success = manager.unregister_rule("nonexistent")
        assert not success

    def test_convenience_methods(self, manager):
        """Test convenience alert methods."""
        info = manager.info("Info", "Message")
        assert info.severity == AlertSeverity.INFO

        warning = manager.warning("Warning", "Message", skip_rate_limit=True)
        assert warning.severity == AlertSeverity.WARNING

        error = manager.error("Error", "Message", skip_rate_limit=True)
        assert error.severity == AlertSeverity.ERROR

        critical = manager.critical("Critical", "Message", skip_rate_limit=True)
        assert critical.severity == AlertSeverity.CRITICAL

    def test_cleanup(self, manager):
        """Test cleanup of old alerts."""
        # Add old alert
        old_alert = manager.alert(
            AlertCategory.SYSTEM, AlertSeverity.INFO, "Old", "Message"
        )
        if old_alert:
            old_alert.timestamp = datetime.now() - timedelta(hours=48)

        # Add new alert
        manager.alert(
            AlertCategory.SYSTEM,
            AlertSeverity.INFO,
            "New",
            "Message",
            skip_rate_limit=True,
        )

        removed = manager.cleanup_old_alerts()
        assert removed == 1

    def test_clear(self, manager):
        """Test clearing all alerts."""
        manager.alert(AlertCategory.SYSTEM, AlertSeverity.INFO, "Test", "Message")
        manager.clear()

        active = manager.get_active_alerts()
        assert len(active) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_alert_manager(self):
        """Test get_alert_manager function."""
        AlertManager.reset()
        manager = get_alert_manager()
        assert isinstance(manager, AlertManager)
        AlertManager.reset()

    def test_create_alert_function(self):
        """Test create_alert convenience function."""
        AlertManager.reset()
        alert = create_alert(
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test message",
        )
        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        AlertManager.reset()
