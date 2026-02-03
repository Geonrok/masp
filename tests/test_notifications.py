"""
Tests for notification services.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from libs.notifications.deduplication import (
    AlertAggregator,
    AlertDeduplicator,
    DeduplicationConfig,
    DeduplicationStrategy,
    RateLimitConfig,
)
from libs.notifications.email import EmailConfig, EmailNotifier
from libs.notifications.slack import SlackAttachment, SlackNotifier


class TestSlackNotifier:
    """Tests for Slack notifier."""

    @patch.dict("os.environ", {}, clear=True)
    def test_disabled_without_webhook(self):
        """Test notifier is disabled without webhook URL."""
        # Clear env vars to ensure no fallback
        import os

        os.environ.pop("SLACK_WEBHOOK_URL", None)
        notifier = SlackNotifier(webhook_url=None)
        assert not notifier.enabled

    def test_enabled_with_webhook(self):
        """Test notifier is enabled with webhook URL."""
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.enabled

    @patch.dict("os.environ", {}, clear=True)
    def test_send_message_when_disabled(self):
        """Test send returns False when disabled."""
        import os

        os.environ.pop("SLACK_WEBHOOK_URL", None)
        notifier = SlackNotifier(webhook_url=None)
        result = notifier.send_message("Test message")
        assert result is False

    @patch("httpx.Client")
    def test_send_message_success(self, mock_client):
        """Test successful message send."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send_message("Test message")
        assert result is True

    @patch("httpx.Client")
    def test_send_trade_notification(self, mock_client):
        """Test trade notification."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.return_value.__enter__.return_value.post.return_value = (
            mock_response
        )

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        result = notifier.send_trade_notification(
            exchange="upbit",
            symbol="BTC",
            side="BUY",
            quantity=0.1,
            price=50000000,
        )
        assert result is True


class TestSlackAttachment:
    """Tests for Slack attachments."""

    def test_attachment_creation(self):
        """Test attachment dataclass."""
        attachment = SlackAttachment(
            fallback="Test fallback",
            color="#FF0000",
            title="Test Title",
            text="Test text",
        )
        assert attachment.fallback == "Test fallback"
        assert attachment.color == "#FF0000"


class TestEmailNotifier:
    """Tests for email notifier."""

    def test_disabled_without_config(self):
        """Test notifier is disabled without proper config."""
        notifier = EmailNotifier()
        assert not notifier.enabled

    def test_enabled_with_config(self):
        """Test notifier is enabled with proper config."""
        config = EmailConfig(
            username="test@example.com",
            password="password",
            from_address="test@example.com",
        )
        notifier = EmailNotifier(config=config, recipients=["user@example.com"])
        assert notifier.enabled

    def test_send_email_when_disabled(self):
        """Test send returns False when disabled."""
        notifier = EmailNotifier()
        result = notifier.send_email("Subject", "<p>Body</p>")
        assert result is False


class TestAlertDeduplicator:
    """Tests for alert deduplication."""

    def test_first_alert_allowed(self):
        """Test first alert is always allowed."""
        dedup = AlertDeduplicator()
        should_send, reason = dedup.should_send(
            message="Test alert",
            alert_type="SYSTEM",
        )
        assert should_send is True
        assert reason is None

    def test_duplicate_blocked(self):
        """Test duplicate alert is blocked."""
        dedup = AlertDeduplicator()

        # First alert
        dedup.should_send(message="Test alert", alert_type="SYSTEM")

        # Duplicate
        should_send, reason = dedup.should_send(
            message="Test alert",
            alert_type="SYSTEM",
        )
        assert should_send is False
        assert "Duplicate" in reason

    def test_different_alerts_allowed(self):
        """Test different alerts are allowed."""
        # Use EXACT strategy to distinguish different messages
        config = DeduplicationConfig(strategy=DeduplicationStrategy.EXACT)
        dedup = AlertDeduplicator(dedup_config=config)

        dedup.should_send(message="First alert message", alert_type="SYSTEM")

        should_send, reason = dedup.should_send(
            message="Different alert message",
            alert_type="SYSTEM",
        )
        assert should_send is True

    def test_similar_strategy(self):
        """Test similar strategy ignores numbers."""
        config = DeduplicationConfig(strategy=DeduplicationStrategy.SIMILAR)
        dedup = AlertDeduplicator(dedup_config=config)

        dedup.should_send(message="Price is 50000", alert_type="TRADE")

        # Similar message with different number
        should_send, reason = dedup.should_send(
            message="Price is 60000",
            alert_type="TRADE",
        )
        assert should_send is False

    def test_force_send_bypasses_dedup(self):
        """Test force flag bypasses deduplication."""
        dedup = AlertDeduplicator()

        dedup.should_send(message="Test alert", alert_type="SYSTEM")

        should_send, reason = dedup.should_send(
            message="Test alert",
            alert_type="SYSTEM",
            force=True,
        )
        assert should_send is True

    def test_rate_limiting(self):
        """Test rate limiting."""
        rate_limits = {
            "HIGH": RateLimitConfig(max_alerts=2, window_seconds=60),
        }
        # Use EXACT strategy so each message is distinct and gets counted
        config = DeduplicationConfig(strategy=DeduplicationStrategy.EXACT)
        dedup = AlertDeduplicator(dedup_config=config, rate_limits=rate_limits)

        # Send 2 alerts (should be allowed)
        for i in range(2):
            dedup.should_send(
                message=f"Unique alert message number {i}",
                alert_type="SYSTEM",
                priority="HIGH",
            )

        # Third alert should be rate limited
        should_send, reason = dedup.should_send(
            message="Third unique alert message",
            alert_type="SYSTEM",
            priority="HIGH",
        )
        assert should_send is False
        assert "Rate limited" in reason

    def test_get_stats(self):
        """Test statistics retrieval."""
        dedup = AlertDeduplicator()

        dedup.should_send(message="Alert 1", alert_type="SYSTEM")
        dedup.should_send(message="Alert 1", alert_type="SYSTEM")  # Duplicate

        stats = dedup.get_stats()
        assert stats["active_records"] == 1
        assert stats["total_suppressed"] == 1

    def test_reset(self):
        """Test reset clears all state."""
        dedup = AlertDeduplicator()

        dedup.should_send(message="Alert", alert_type="SYSTEM")
        dedup.reset()

        stats = dedup.get_stats()
        assert stats["active_records"] == 0


class TestAlertAggregator:
    """Tests for alert aggregation."""

    def test_add_alert(self):
        """Test adding alerts to aggregator."""
        aggregator = AlertAggregator()

        aggregator.add_alert("key1", {"message": "Alert 1"})
        aggregator.add_alert("key1", {"message": "Alert 2"})

        count = aggregator.get_pending_count("key1")
        assert count == 2

    def test_flush(self):
        """Test flushing aggregated alerts."""
        aggregator = AlertAggregator()

        aggregator.add_alert("key1", {"message": "Alert 1"})
        aggregator.add_alert("key1", {"message": "Alert 2"})

        alerts = aggregator.flush("key1")
        assert len(alerts) == 2

        # Should be empty after flush
        count = aggregator.get_pending_count("key1")
        assert count == 0

    def test_get_all_pending(self):
        """Test getting all pending counts."""
        aggregator = AlertAggregator()

        aggregator.add_alert("key1", {"message": "Alert"})
        aggregator.add_alert("key2", {"message": "Alert"})
        aggregator.add_alert("key2", {"message": "Alert"})

        pending = aggregator.get_all_pending()
        assert pending["key1"] == 1
        assert pending["key2"] == 2


class TestDeduplicationConfig:
    """Tests for deduplication configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeduplicationConfig()
        assert config.strategy == DeduplicationStrategy.SIMILAR
        assert config.window_seconds == 300
        assert config.aggregate_count == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = DeduplicationConfig(
            strategy=DeduplicationStrategy.EXACT,
            window_seconds=60,
            aggregate_count=5,
        )
        assert config.strategy == DeduplicationStrategy.EXACT
        assert config.window_seconds == 60


class TestRateLimitConfig:
    """Tests for rate limit configuration."""

    def test_default_config(self):
        """Test default rate limit values."""
        config = RateLimitConfig()
        assert config.max_alerts == 10
        assert config.window_seconds == 60
        assert config.cooldown_seconds == 300

    def test_custom_config(self):
        """Test custom rate limit."""
        config = RateLimitConfig(
            max_alerts=5,
            window_seconds=30,
            cooldown_seconds=120,
        )
        assert config.max_alerts == 5
