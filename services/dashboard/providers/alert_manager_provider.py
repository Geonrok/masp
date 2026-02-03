"""Alert manager provider - connects AlertManager to dashboard components."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance
_alert_manager = None


def _get_alert_manager():
    """Get or create AlertManager instance.

    Returns:
        AlertManager instance or None if unavailable
    """
    global _alert_manager

    if _alert_manager is not None:
        return _alert_manager

    try:
        from libs.notifications.alert_manager import get_alert_manager

        _alert_manager = get_alert_manager()
        return _alert_manager
    except ImportError as e:
        logger.debug("AlertManager import failed: %s", e)
        return None
    except Exception as e:
        logger.debug("AlertManager initialization failed: %s", e)
        return None


def get_alert_rules() -> Optional[List[Dict[str, Any]]]:
    """Get alert rules from AlertManager.

    Returns:
        List of rule dictionaries or None if unavailable
    """
    manager = _get_alert_manager()

    if manager is None:
        return None

    try:
        rules = manager.get_rules()
        return [
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
            for r in rules
        ]
    except Exception as e:
        logger.error("Failed to get alert rules: %s", e)
        return None


def get_anomaly_thresholds() -> Optional[Dict[str, float]]:
    """Get anomaly detection thresholds.

    Returns:
        Threshold dictionary or None if unavailable
    """
    try:
        from libs.notifications.alert_manager import AnomalyDetector

        detector = AnomalyDetector()
        return detector.thresholds.copy()
    except Exception as e:
        logger.debug("Failed to get thresholds: %s", e)
        return None


def toggle_rule(name: str, enabled: bool) -> bool:
    """Toggle alert rule enabled state.

    Args:
        name: Rule name
        enabled: New enabled state

    Returns:
        True if successful
    """
    manager = _get_alert_manager()

    if manager is None:
        return False

    try:
        return manager.update_rule(name, enabled=enabled)
    except Exception as e:
        logger.error("Failed to toggle rule: %s", e)
        return False


def delete_rule(name: str) -> bool:
    """Delete alert rule.

    Args:
        name: Rule name

    Returns:
        True if successful
    """
    manager = _get_alert_manager()

    if manager is None:
        return False

    try:
        return manager.remove_rule(name)
    except Exception as e:
        logger.error("Failed to delete rule: %s", e)
        return False


def create_rule(rule_data: Dict[str, Any]) -> bool:
    """Create new alert rule.

    Args:
        rule_data: Rule configuration dictionary

    Returns:
        True if successful
    """
    manager = _get_alert_manager()

    if manager is None:
        return False

    try:
        from libs.notifications.alert_manager import AlertPriority, AlertRule, AlertType

        rule = AlertRule(
            name=rule_data.get("name", ""),
            enabled=rule_data.get("enabled", True),
            alert_types=[AlertType(t) for t in rule_data.get("alert_types", [])],
            min_priority=AlertPriority[rule_data.get("min_priority", "NORMAL")],
            exchanges=rule_data.get("exchanges", []),
            symbols=rule_data.get("symbols", []),
            cooldown_seconds=rule_data.get("cooldown_seconds", 0),
            aggregate_count=rule_data.get("aggregate_count", 0),
        )
        manager.add_rule(rule)
        return True
    except Exception as e:
        logger.error("Failed to create rule: %s", e)
        return False


def save_thresholds(thresholds: Dict[str, float]) -> bool:
    """Save anomaly detection thresholds.

    Args:
        thresholds: New threshold values

    Returns:
        True if successful
    """
    try:
        # Thresholds are stored in AnomalyDetector instance
        # For persistence, we could save to config file
        import json
        from pathlib import Path

        config_dir = Path("data/alerts")
        config_dir.mkdir(parents=True, exist_ok=True)

        config_file = config_dir / "thresholds.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)

        return True
    except Exception as e:
        logger.error("Failed to save thresholds: %s", e)
        return False


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent alerts from AlertManager.

    Args:
        limit: Maximum number of alerts

    Returns:
        List of alert dictionaries
    """
    manager = _get_alert_manager()

    if manager is None:
        return []

    try:
        alerts = manager.get_alerts(limit=limit)
        return [a.to_dict() for a in alerts]
    except Exception as e:
        logger.error("Failed to get alerts: %s", e)
        return []


def check_system_anomalies() -> List[Dict[str, Any]]:
    """Check for system anomalies.

    Returns:
        List of detected anomalies
    """
    manager = _get_alert_manager()

    try:
        from libs.notifications.alert_manager import AnomalyDetector

        detector = AnomalyDetector(alert_manager=manager)
        return detector.check_system_resources()
    except Exception as e:
        logger.debug("Failed to check anomalies: %s", e)
        return []


def get_alert_settings_callbacks() -> Dict[str, Callable]:
    """Get callbacks for alert settings component.

    Returns:
        Dictionary of callback functions
    """
    return {
        "on_rule_toggle": toggle_rule,
        "on_rule_delete": delete_rule,
        "on_rule_create": create_rule,
        "on_thresholds_save": save_thresholds,
    }
