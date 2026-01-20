"""Notification services for MASP."""
from libs.notifications.telegram import (
    TelegramNotifier,
    format_trade_message,
    format_daily_summary,
)
from libs.notifications.alert_manager import (
    AlertManager,
    AlertPriority,
    AlertType,
    Alert,
    AlertRule,
    AnomalyDetector,
    get_alert_manager,
)
from libs.notifications.slack import (
    SlackNotifier,
    SlackAttachment,
)
from libs.notifications.email import (
    EmailNotifier,
    EmailConfig,
)
from libs.notifications.deduplication import (
    AlertDeduplicator,
    AlertAggregator,
    DeduplicationConfig,
    DeduplicationStrategy,
    RateLimitConfig,
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
