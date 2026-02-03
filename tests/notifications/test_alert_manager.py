"""Tests for AlertManager and related classes."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


def test_alert_priority_values():
    """Test AlertPriority enum values."""
    from libs.notifications.alert_manager import AlertPriority

    assert AlertPriority.CRITICAL.value == 4
    assert AlertPriority.HIGH.value == 3
    assert AlertPriority.NORMAL.value == 2
    assert AlertPriority.LOW.value == 1


def test_alert_type_values():
    """Test AlertType enum values."""
    from libs.notifications.alert_manager import AlertType

    assert AlertType.TRADE.value == "TRADE"
    assert AlertType.SIGNAL.value == "SIGNAL"
    assert AlertType.ERROR.value == "ERROR"
    assert AlertType.SYSTEM.value == "SYSTEM"
    assert AlertType.DAILY.value == "DAILY"
    assert AlertType.ANOMALY.value == "ANOMALY"


def test_alert_dataclass():
    """Test Alert dataclass."""
    from libs.notifications.alert_manager import Alert, AlertPriority, AlertType

    alert = Alert(
        id="ALT_001",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.NORMAL,
        title="Test Alert",
        message="This is a test",
        exchange="upbit",
        symbol="BTC",
    )

    assert alert.id == "ALT_001"
    assert alert.alert_type == AlertType.TRADE
    assert alert.priority == AlertPriority.NORMAL
    assert alert.exchange == "upbit"
    assert alert.sent is False


def test_alert_to_dict():
    """Test Alert to_dict method."""
    from libs.notifications.alert_manager import Alert, AlertPriority, AlertType

    alert = Alert(
        id="ALT_002",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.HIGH,
        title="Signal Alert",
        message="KAMA crossover",
    )

    data = alert.to_dict()

    assert data["id"] == "ALT_002"
    assert data["alert_type"] == "SIGNAL"
    assert data["priority"] == "HIGH"


def test_alert_from_dict():
    """Test Alert from_dict method."""
    from libs.notifications.alert_manager import Alert, AlertPriority, AlertType

    data = {
        "id": "ALT_003",
        "alert_type": "ERROR",
        "priority": "CRITICAL",
        "title": "Error Alert",
        "message": "API failure",
        "timestamp": "2025-01-15T10:00:00",
    }

    alert = Alert.from_dict(data)

    assert alert.id == "ALT_003"
    assert alert.alert_type == AlertType.ERROR
    assert alert.priority == AlertPriority.CRITICAL


def test_alert_rule_matches():
    """Test AlertRule matching logic."""
    from libs.notifications.alert_manager import (
        Alert,
        AlertPriority,
        AlertRule,
        AlertType,
    )

    rule = AlertRule(
        name="test_rule",
        enabled=True,
        alert_types=[AlertType.TRADE],
        min_priority=AlertPriority.NORMAL,
        exchanges=["upbit"],
    )

    # Matching alert
    alert1 = Alert(
        id="A1",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.HIGH,
        title="Test",
        message="Test",
        exchange="upbit",
    )
    assert rule.matches(alert1) is True

    # Wrong type
    alert2 = Alert(
        id="A2",
        alert_type=AlertType.SIGNAL,
        priority=AlertPriority.HIGH,
        title="Test",
        message="Test",
        exchange="upbit",
    )
    assert rule.matches(alert2) is False

    # Low priority
    alert3 = Alert(
        id="A3",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.LOW,
        title="Test",
        message="Test",
        exchange="upbit",
    )
    assert rule.matches(alert3) is False

    # Wrong exchange
    alert4 = Alert(
        id="A4",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.HIGH,
        title="Test",
        message="Test",
        exchange="bithumb",
    )
    assert rule.matches(alert4) is False


def test_alert_rule_disabled():
    """Test that disabled rules don't match."""
    from libs.notifications.alert_manager import (
        Alert,
        AlertPriority,
        AlertRule,
        AlertType,
    )

    rule = AlertRule(
        name="disabled_rule",
        enabled=False,
        min_priority=AlertPriority.LOW,
    )

    alert = Alert(
        id="A1",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.HIGH,
        title="Test",
        message="Test",
    )

    assert rule.matches(alert) is False


def test_alert_manager_initialization(tmp_path):
    """Test AlertManager initialization."""
    from libs.notifications.alert_manager import AlertManager

    manager = AlertManager(history_dir=str(tmp_path))

    assert manager.history_dir.exists()
    assert len(manager._rules) >= 1  # Default rule


def test_alert_manager_create_alert(tmp_path):
    """Test AlertManager alert creation."""
    from libs.notifications.alert_manager import (
        AlertManager,
        AlertPriority,
        AlertType,
    )

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    alert = manager.create_alert(
        alert_type=AlertType.TRADE,
        priority=AlertPriority.NORMAL,
        title="Test Trade",
        message="BUY BTC",
        exchange="upbit",
    )

    assert alert.id.startswith("ALT_")
    assert alert.alert_type == AlertType.TRADE
    assert len(manager._alerts) == 1


def test_alert_manager_get_alerts(tmp_path):
    """Test AlertManager get_alerts filtering."""
    from libs.notifications.alert_manager import (
        AlertManager,
        AlertPriority,
        AlertType,
    )

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    # Create multiple alerts
    manager.create_alert(AlertType.TRADE, AlertPriority.NORMAL, "Trade 1", "BUY")
    manager.create_alert(AlertType.SIGNAL, AlertPriority.HIGH, "Signal 1", "KAMA")
    manager.create_alert(AlertType.ERROR, AlertPriority.CRITICAL, "Error 1", "API fail")

    # Get all
    all_alerts = manager.get_alerts()
    assert len(all_alerts) == 3

    # Filter by type
    trade_alerts = manager.get_alerts(alert_type=AlertType.TRADE)
    assert len(trade_alerts) == 1

    # Filter by priority
    high_alerts = manager.get_alerts(priority=AlertPriority.HIGH)
    assert len(high_alerts) == 2  # HIGH and CRITICAL


def test_alert_manager_trade_alert(tmp_path):
    """Test trade_alert convenience method."""
    from libs.notifications.alert_manager import AlertManager, AlertType

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    alert = manager.trade_alert(
        exchange="upbit",
        symbol="BTC",
        side="BUY",
        quantity=0.1,
        price=50000000,
        pnl=10000,
    )

    assert alert.alert_type == AlertType.TRADE
    assert alert.exchange == "upbit"
    assert alert.symbol == "BTC"
    assert "BUY BTC" in alert.title


def test_alert_manager_error_alert(tmp_path):
    """Test error_alert convenience method."""
    from libs.notifications.alert_manager import AlertManager, AlertPriority, AlertType

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    # Non-critical error
    alert1 = manager.error_alert("API Error", "Connection timeout", exchange="upbit")
    assert alert1.priority == AlertPriority.HIGH

    # Critical error
    alert2 = manager.error_alert("System Crash", "Fatal error", critical=True)
    assert alert2.priority == AlertPriority.CRITICAL


def test_alert_manager_format_alert(tmp_path):
    """Test alert formatting for Telegram."""
    from libs.notifications.alert_manager import (
        Alert,
        AlertManager,
        AlertPriority,
        AlertType,
    )

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    alert = Alert(
        id="A1",
        alert_type=AlertType.TRADE,
        priority=AlertPriority.NORMAL,
        title="BUY BTC",
        message="Quantity: 0.1",
        exchange="upbit",
        symbol="BTC",
    )

    formatted = manager._format_alert(alert)

    assert "<b>BUY BTC</b>" in formatted
    assert "UPBIT" in formatted
    assert "BTC" in formatted


def test_alert_manager_rules(tmp_path):
    """Test AlertManager rule management."""
    from libs.notifications.alert_manager import (
        AlertManager,
        AlertPriority,
        AlertRule,
        AlertType,
    )

    manager = AlertManager(notifier=None, history_dir=str(tmp_path))

    # Add rule
    rule = AlertRule(
        name="test_rule",
        enabled=True,
        alert_types=[AlertType.TRADE],
        min_priority=AlertPriority.HIGH,
    )
    manager.add_rule(rule)

    rules = manager.get_rules()
    assert len(rules) >= 2  # Default + new rule

    # Update rule
    success = manager.update_rule("test_rule", enabled=False)
    assert success is True

    # Remove rule
    success = manager.remove_rule("test_rule")
    assert success is True


# =============================================================================
# Anomaly Detector Tests
# =============================================================================


def test_anomaly_detector_thresholds():
    """Test AnomalyDetector default thresholds."""
    from libs.notifications.alert_manager import AnomalyDetector

    detector = AnomalyDetector()

    assert "cpu_percent" in detector.thresholds
    assert "memory_percent" in detector.thresholds
    assert detector.thresholds["cpu_percent"] == 90.0


def test_anomaly_detector_custom_thresholds():
    """Test AnomalyDetector with custom thresholds."""
    from libs.notifications.alert_manager import AnomalyDetector

    custom = {"cpu_percent": 80.0, "memory_percent": 85.0}
    detector = AnomalyDetector(thresholds=custom)

    assert detector.thresholds["cpu_percent"] == 80.0
    assert detector.thresholds["memory_percent"] == 85.0


def test_anomaly_detector_check_api_health():
    """Test API health check."""
    from libs.notifications.alert_manager import AnomalyDetector

    detector = AnomalyDetector()

    # Normal case
    anomalies = detector.check_api_health(
        total_requests=100,
        error_count=5,
        avg_response_ms=1000,
    )
    assert len(anomalies) == 0

    # High error rate
    anomalies = detector.check_api_health(
        total_requests=100,
        error_count=20,
        avg_response_ms=1000,
    )
    assert len(anomalies) == 1
    assert anomalies[0]["metric"] == "api_error_rate"

    # Slow response
    anomalies = detector.check_api_health(
        total_requests=100,
        error_count=0,
        avg_response_ms=10000,
    )
    assert len(anomalies) == 1
    assert anomalies[0]["metric"] == "response_time_ms"


def test_anomaly_detector_check_system_resources():
    """Test system resource check (psutil dependent)."""
    from libs.notifications.alert_manager import AnomalyDetector

    detector = AnomalyDetector()

    # Should not raise even if psutil unavailable
    anomalies = detector.check_system_resources()
    assert isinstance(anomalies, list)


def test_get_alert_manager():
    """Test get_alert_manager factory function."""
    from libs.notifications.alert_manager import AlertManager, get_alert_manager

    manager = get_alert_manager()

    assert isinstance(manager, AlertManager)


# =============================================================================
# Provider Tests
# =============================================================================


def test_alert_manager_provider_import():
    """Test alert manager provider imports correctly."""
    from services.dashboard.providers.alert_manager_provider import (
        get_alert_rules,
        get_alert_settings_callbacks,
        get_anomaly_thresholds,
    )

    assert callable(get_alert_rules)
    assert callable(get_anomaly_thresholds)
    assert callable(get_alert_settings_callbacks)


def test_get_alert_settings_callbacks():
    """Test get_alert_settings_callbacks returns all callbacks."""
    from services.dashboard.providers.alert_manager_provider import (
        get_alert_settings_callbacks,
    )

    callbacks = get_alert_settings_callbacks()

    assert "on_rule_toggle" in callbacks
    assert "on_rule_delete" in callbacks
    assert "on_rule_create" in callbacks
    assert "on_thresholds_save" in callbacks
    assert callable(callbacks["on_rule_toggle"])


def test_check_system_anomalies():
    """Test check_system_anomalies function."""
    from services.dashboard.providers.alert_manager_provider import (
        check_system_anomalies,
    )

    # Should return a list (may be empty)
    result = check_system_anomalies()
    assert isinstance(result, list)


# =============================================================================
# Component Tests
# =============================================================================


def test_alert_settings_component_import():
    """Test alert settings component imports correctly."""
    from services.dashboard.components.alert_settings import (
        render_alert_rule_card,
        render_alert_settings,
        render_anomaly_thresholds,
        render_new_rule_form,
    )

    assert callable(render_alert_settings)
    assert callable(render_alert_rule_card)


def test_get_priority_options():
    """Test priority options list."""
    from services.dashboard.components.alert_settings import _get_priority_options

    options = _get_priority_options()

    assert "LOW" in options
    assert "NORMAL" in options
    assert "HIGH" in options
    assert "CRITICAL" in options


def test_get_alert_type_options():
    """Test alert type options list."""
    from services.dashboard.components.alert_settings import _get_alert_type_options

    options = _get_alert_type_options()

    assert "TRADE" in options
    assert "SIGNAL" in options
    assert "ERROR" in options


def test_get_demo_rules():
    """Test demo rules generation."""
    from services.dashboard.components.alert_settings import _get_demo_rules

    rules = _get_demo_rules()

    assert len(rules) >= 1
    assert any(r["name"] == "default" for r in rules)
