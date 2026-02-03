"""
Alert Deduplication and Rate Limiting

Prevents alert spam by:
- Deduplicating similar alerts within a time window
- Rate limiting alerts per type/priority
- Aggregating multiple similar alerts
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class DeduplicationStrategy(Enum):
    """Deduplication strategies."""

    EXACT = "exact"  # Exact message match
    SIMILAR = "similar"  # Similar content (ignores numbers)
    KEY_BASED = "key_based"  # Based on specific key fields


@dataclass
class RateLimitConfig:
    """Rate limit configuration for alert types."""

    max_alerts: int = 10  # Max alerts per window
    window_seconds: int = 60  # Time window in seconds
    cooldown_seconds: int = 300  # Cooldown after hitting limit


@dataclass
class DeduplicationConfig:
    """Deduplication configuration."""

    strategy: DeduplicationStrategy = DeduplicationStrategy.SIMILAR
    window_seconds: int = 300  # 5 minutes default
    aggregate_count: int = 3  # Aggregate after this many similar alerts
    key_fields: List[str] = field(default_factory=lambda: ["alert_type", "symbol"])


@dataclass
class AlertRecord:
    """Record of a sent alert."""

    alert_hash: str
    first_seen: datetime
    last_seen: datetime
    count: int = 1
    aggregated: bool = False
    suppressed_count: int = 0


class AlertDeduplicator:
    """
    Deduplicates and rate limits alerts.

    Features:
    - Hash-based deduplication
    - Configurable time windows
    - Alert aggregation
    - Rate limiting by type/priority
    - Thread-safe operations
    """

    def __init__(
        self,
        dedup_config: Optional[DeduplicationConfig] = None,
        rate_limits: Optional[Dict[str, RateLimitConfig]] = None,
    ):
        """
        Initialize deduplicator.

        Args:
            dedup_config: Deduplication configuration
            rate_limits: Rate limit configs by alert type
        """
        self.dedup_config = dedup_config or DeduplicationConfig()
        self.rate_limits = rate_limits or self._default_rate_limits()

        # Alert tracking
        self._alert_records: Dict[str, AlertRecord] = {}
        self._rate_counters: Dict[str, List[datetime]] = defaultdict(list)
        self._cooldown_until: Dict[str, datetime] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Callbacks
        self._on_aggregate: Optional[Callable[[str, int], None]] = None
        self._on_rate_limit: Optional[Callable[[str, int], None]] = None

        logger.info(
            f"[Deduplicator] Initialized: "
            f"strategy={self.dedup_config.strategy.value}, "
            f"window={self.dedup_config.window_seconds}s"
        )

    def _default_rate_limits(self) -> Dict[str, RateLimitConfig]:
        """Default rate limits by priority."""
        return {
            "CRITICAL": RateLimitConfig(max_alerts=100, window_seconds=60),
            "HIGH": RateLimitConfig(max_alerts=20, window_seconds=60),
            "NORMAL": RateLimitConfig(max_alerts=10, window_seconds=60),
            "LOW": RateLimitConfig(max_alerts=5, window_seconds=60),
        }

    def _compute_hash(
        self,
        message: str,
        alert_type: str = "",
        symbol: str = "",
        **kwargs,
    ) -> str:
        """
        Compute hash for deduplication.

        Args:
            message: Alert message
            alert_type: Alert type
            symbol: Trading symbol
            **kwargs: Additional fields

        Returns:
            Hash string
        """
        if self.dedup_config.strategy == DeduplicationStrategy.EXACT:
            content = message

        elif self.dedup_config.strategy == DeduplicationStrategy.SIMILAR:
            # Remove numbers to match similar messages
            import re

            content = re.sub(r"\d+\.?\d*", "N", message)

        elif self.dedup_config.strategy == DeduplicationStrategy.KEY_BASED:
            # Use only key fields
            parts = []
            for key in self.dedup_config.key_fields:
                if key == "alert_type":
                    parts.append(alert_type)
                elif key == "symbol":
                    parts.append(symbol)
                elif key in kwargs:
                    parts.append(str(kwargs[key]))
            content = "|".join(parts)

        else:
            content = message

        # Add type and symbol to hash
        full_content = f"{alert_type}:{symbol}:{content}"
        return hashlib.md5(full_content.encode()).hexdigest()[:16]

    def _is_within_window(self, timestamp: datetime) -> bool:
        """Check if timestamp is within deduplication window."""
        cutoff = datetime.now() - timedelta(seconds=self.dedup_config.window_seconds)
        return timestamp > cutoff

    def _cleanup_old_records(self) -> None:
        """Remove expired records."""
        cutoff = datetime.now() - timedelta(
            seconds=self.dedup_config.window_seconds * 2
        )

        expired = [h for h, r in self._alert_records.items() if r.last_seen < cutoff]

        for h in expired:
            del self._alert_records[h]

    def _check_rate_limit(self, priority: str) -> tuple[bool, int]:
        """
        Check if rate limit is exceeded.

        Args:
            priority: Alert priority

        Returns:
            (is_limited, seconds_until_reset)
        """
        config = self.rate_limits.get(priority, self.rate_limits.get("NORMAL"))
        if not config:
            return False, 0

        now = datetime.now()

        # Check cooldown
        if priority in self._cooldown_until:
            if now < self._cooldown_until[priority]:
                remaining = int((self._cooldown_until[priority] - now).total_seconds())
                return True, remaining
            else:
                del self._cooldown_until[priority]

        # Clean old entries
        window_start = now - timedelta(seconds=config.window_seconds)
        self._rate_counters[priority] = [
            t for t in self._rate_counters[priority] if t > window_start
        ]

        # Check limit
        if len(self._rate_counters[priority]) >= config.max_alerts:
            self._cooldown_until[priority] = now + timedelta(
                seconds=config.cooldown_seconds
            )
            logger.warning(
                f"[Deduplicator] Rate limit hit for {priority}, "
                f"cooldown {config.cooldown_seconds}s"
            )
            if self._on_rate_limit:
                self._on_rate_limit(priority, config.cooldown_seconds)
            return True, config.cooldown_seconds

        return False, 0

    def should_send(
        self,
        message: str,
        alert_type: str = "SYSTEM",
        priority: str = "NORMAL",
        symbol: str = "",
        force: bool = False,
        **kwargs,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if an alert should be sent.

        Args:
            message: Alert message
            alert_type: Type of alert
            priority: Alert priority
            symbol: Trading symbol
            force: Force send (bypass deduplication)
            **kwargs: Additional fields

        Returns:
            (should_send, reason_if_blocked)
        """
        if force:
            return True, None

        with self._lock:
            # Cleanup periodically
            self._cleanup_old_records()

            # Check rate limit first
            is_limited, wait_time = self._check_rate_limit(priority)
            if is_limited:
                return False, f"Rate limited ({wait_time}s remaining)"

            # Compute hash
            alert_hash = self._compute_hash(
                message=message,
                alert_type=alert_type,
                symbol=symbol,
                **kwargs,
            )

            now = datetime.now()

            # Check for duplicate
            if alert_hash in self._alert_records:
                record = self._alert_records[alert_hash]

                if self._is_within_window(record.last_seen):
                    record.count += 1
                    record.last_seen = now

                    # Check if we should aggregate
                    if (
                        record.count >= self.dedup_config.aggregate_count
                        and not record.aggregated
                    ):
                        record.aggregated = True
                        if self._on_aggregate:
                            self._on_aggregate(alert_hash, record.count)
                        logger.info(
                            f"[Deduplicator] Aggregating {record.count} similar alerts"
                        )
                        return True, None

                    record.suppressed_count += 1
                    return False, f"Duplicate ({record.count} in window)"

            # New alert - record and allow
            self._alert_records[alert_hash] = AlertRecord(
                alert_hash=alert_hash,
                first_seen=now,
                last_seen=now,
            )

            # Record for rate limiting
            self._rate_counters[priority].append(now)

            return True, None

    def record_sent(
        self,
        message: str,
        alert_type: str = "SYSTEM",
        symbol: str = "",
        **kwargs,
    ) -> None:
        """
        Record that an alert was sent (for tracking).

        Args:
            message: Alert message
            alert_type: Type of alert
            symbol: Trading symbol
            **kwargs: Additional fields
        """
        with self._lock:
            alert_hash = self._compute_hash(
                message=message,
                alert_type=alert_type,
                symbol=symbol,
                **kwargs,
            )

            if alert_hash in self._alert_records:
                self._alert_records[alert_hash].last_seen = datetime.now()

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        with self._lock:
            total_records = len(self._alert_records)
            total_suppressed = sum(
                r.suppressed_count for r in self._alert_records.values()
            )
            aggregated_count = sum(
                1 for r in self._alert_records.values() if r.aggregated
            )

            rate_limit_active = {
                p: (
                    (datetime.now() < until).item()
                    if hasattr((datetime.now() < until), "item")
                    else (datetime.now() < until)
                )
                for p, until in self._cooldown_until.items()
            }

            return {
                "active_records": total_records,
                "total_suppressed": total_suppressed,
                "aggregated_alerts": aggregated_count,
                "rate_limits_active": rate_limit_active,
                "dedup_strategy": self.dedup_config.strategy.value,
                "window_seconds": self.dedup_config.window_seconds,
            }

    def reset(self) -> None:
        """Reset all state."""
        with self._lock:
            self._alert_records.clear()
            self._rate_counters.clear()
            self._cooldown_until.clear()
            logger.info("[Deduplicator] Reset all state")

    def set_aggregate_callback(
        self,
        callback: Callable[[str, int], None],
    ) -> None:
        """Set callback for when alerts are aggregated."""
        self._on_aggregate = callback

    def set_rate_limit_callback(
        self,
        callback: Callable[[str, int], None],
    ) -> None:
        """Set callback for when rate limit is hit."""
        self._on_rate_limit = callback


class AlertAggregator:
    """
    Aggregates multiple similar alerts into summary notifications.
    """

    def __init__(self, aggregation_window: int = 60):
        """
        Initialize aggregator.

        Args:
            aggregation_window: Time window for aggregation in seconds
        """
        self.aggregation_window = aggregation_window
        self._pending: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_alert(
        self,
        alert_key: str,
        alert_data: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Add alert to aggregation queue.

        Args:
            alert_key: Key for grouping similar alerts
            alert_data: Alert data

        Returns:
            List of aggregated alerts if threshold reached, None otherwise
        """
        with self._lock:
            alert_data["timestamp"] = datetime.now()
            self._pending[alert_key].append(alert_data)

            # Clean old entries
            cutoff = datetime.now() - timedelta(seconds=self.aggregation_window)
            self._pending[alert_key] = [
                a for a in self._pending[alert_key] if a["timestamp"] > cutoff
            ]

            return None

    def flush(self, alert_key: str) -> List[Dict[str, Any]]:
        """
        Flush and return all pending alerts for a key.

        Args:
            alert_key: Alert grouping key

        Returns:
            List of pending alerts
        """
        with self._lock:
            alerts = self._pending.pop(alert_key, [])
            return alerts

    def get_pending_count(self, alert_key: str) -> int:
        """Get count of pending alerts for a key."""
        with self._lock:
            return len(self._pending.get(alert_key, []))

    def get_all_pending(self) -> Dict[str, int]:
        """Get counts of all pending alert groups."""
        with self._lock:
            return {k: len(v) for k, v in self._pending.items()}
