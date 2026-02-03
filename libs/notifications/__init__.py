"""Notification services for MASP."""

from libs.notifications.alert_manager import (
    Alert,
    AlertManager,
    AlertPriority,
    AlertRule,
    AlertType,
    AnomalyDetector,
    get_alert_manager,
)
from libs.notifications.deduplication import (
    AlertAggregator,
    AlertDeduplicator,
    DeduplicationConfig,
    DeduplicationStrategy,
    RateLimitConfig,
)
from libs.notifications.email import (
    EmailConfig,
    EmailNotifier,
)
from libs.notifications.slack import (
    SlackAttachment,
    SlackNotifier,
)
from libs.notifications.telegram import (
    TelegramNotifier,
    format_daily_summary,
    format_trade_message,
)

__all__ = [
    # Telegram
    "TelegramNotifier",
    "format_trade_message",
    "format_daily_summary",
    # Alert Manager
    "AlertManager",
    "AlertPriority",
    "AlertType",
    "Alert",
    "AlertRule",
    "AnomalyDetector",
    "get_alert_manager",
    # Slack
    "SlackNotifier",
    "SlackAttachment",
    # Email
    "EmailNotifier",
    "EmailConfig",
    # Deduplication
    "AlertDeduplicator",
    "AlertAggregator",
    "DeduplicationConfig",
    "DeduplicationStrategy",
    "RateLimitConfig",
]
