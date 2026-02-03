"""
Metrics Collection and Observability

System-wide metrics collection for monitoring adapter performance,
system health, and operational metrics.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"  # Monotonically increasing
    GAUGE = "gauge"  # Current value
    HISTOGRAM = "histogram"  # Distribution
    TIMING = "timing"  # Duration measurements


@dataclass
class MetricValue:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HistogramBuckets:
    """Histogram bucket configuration."""

    boundaries: List[float]
    counts: Dict[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0

    def __post_init__(self):
        for boundary in self.boundaries:
            self.counts[boundary] = 0
        self.counts[float("inf")] = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1

        for boundary in sorted(self.boundaries):
            if value <= boundary:
                self.counts[boundary] += 1
        self.counts[float("inf")] += 1


class MetricsRegistry:
    """
    Central registry for collecting and managing metrics.

    Thread-safe metrics collection with support for counters,
    gauges, histograms, and timing metrics.

    Example:
        registry = MetricsRegistry.get_instance()

        # Counter
        registry.increment("api_calls_total", labels={"exchange": "binance"})

        # Gauge
        registry.set("active_connections", 5)

        # Timing
        with registry.timer("api_latency", labels={"endpoint": "/ticker"}):
            result = await fetch_ticker()

        # Histogram
        registry.observe("response_size", 1024, labels={"exchange": "binance"})
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._counters: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._histograms: Dict[str, Dict[str, HistogramBuckets]] = defaultdict(dict)
        self._timings: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._lock = threading.Lock()

        # Default histogram buckets for latency (milliseconds)
        self.default_latency_buckets = [
            5,
            10,
            25,
            50,
            100,
            250,
            500,
            1000,
            2500,
            5000,
            10000,
        ]
        # Default buckets for general measurements
        self.default_buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]

    @classmethod
    def get_instance(cls) -> "MetricsRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (mainly for testing)."""
        cls._instance = None

    # ==================== Counter Operations ====================

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Increment a counter.

        Args:
            name: Metric name
            value: Amount to increment (default 1)
            labels: Optional labels

        Returns:
            New counter value
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._counters[name][label_key] += value
            return self._counters[name][label_key]

    def get_counter(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get counter value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            return self._counters[name][label_key]

    # ==================== Gauge Operations ====================

    def set(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge value.

        Args:
            name: Metric name
            value: Value to set
            labels: Optional labels
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][label_key] = value

    def get_gauge(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get gauge value."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            return self._gauges[name][label_key]

    def inc_gauge(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """Increment a gauge (allows decrement with negative)."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._gauges[name][label_key] += value
            return self._gauges[name][label_key]

    # ==================== Histogram Operations ====================

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> None:
        """
        Record a histogram observation.

        Args:
            name: Metric name
            value: Observed value
            labels: Optional labels
            buckets: Custom buckets (uses defaults if not provided)
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            if label_key not in self._histograms[name]:
                bucket_boundaries = buckets or self.default_buckets
                self._histograms[name][label_key] = HistogramBuckets(bucket_boundaries)
            self._histograms[name][label_key].observe(value)

    def get_histogram(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get histogram statistics."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            if name not in self._histograms or label_key not in self._histograms[name]:
                return None
            hist = self._histograms[name][label_key]
            return {
                "count": hist.count,
                "sum": hist.sum,
                "avg": hist.sum / hist.count if hist.count > 0 else 0,
                "buckets": dict(hist.counts),
            }

    # ==================== Timing Operations ====================

    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> "TimerContext":
        """
        Context manager for timing operations.

        Args:
            name: Metric name
            labels: Optional labels

        Returns:
            Timer context manager

        Example:
            with registry.timer("api_call", labels={"exchange": "binance"}):
                await api.fetch_data()
        """
        return TimerContext(self, name, labels)

    def record_timing(
        self,
        name: str,
        duration_ms: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a timing value directly.

        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
            labels: Optional labels
        """
        label_key = self._labels_to_key(labels)
        with self._lock:
            self._timings[name][label_key].append(duration_ms)
            # Keep only last 1000 timings per metric
            if len(self._timings[name][label_key]) > 1000:
                self._timings[name][label_key] = self._timings[name][label_key][-1000:]

        # Also record in histogram
        self.observe(name, duration_ms, labels, self.default_latency_buckets)

    def get_timing_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, float]]:
        """Get timing statistics (avg, min, max, p50, p95, p99)."""
        label_key = self._labels_to_key(labels)
        with self._lock:
            if name not in self._timings or label_key not in self._timings[name]:
                return None

            values = self._timings[name][label_key]
            if not values:
                return None

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "count": n,
                "avg": sum(values) / n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "p50": sorted_values[n // 2],
                "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
                "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
            }

    # ==================== Export & Utility ====================

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary with all metrics organized by type
        """
        with self._lock:
            return {
                "counters": {
                    name: dict(values) for name, values in self._counters.items()
                },
                "gauges": {name: dict(values) for name, values in self._gauges.items()},
                "histograms": {
                    name: {
                        label_key: {
                            "count": hist.count,
                            "sum": hist.sum,
                            "avg": hist.sum / hist.count if hist.count > 0 else 0,
                        }
                        for label_key, hist in buckets.items()
                    }
                    for name, buckets in self._histograms.items()
                },
                "timings": {
                    name: {
                        label_key: self._compute_timing_stats(values)
                        for label_key, values in label_values.items()
                    }
                    for name, label_values in self._timings.items()
                },
                "timestamp": datetime.now().isoformat(),
            }

    def _compute_timing_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute timing statistics from values list."""
        if not values:
            return {}
        sorted_values = sorted(values)
        n = len(sorted_values)
        return {
            "count": n,
            "avg": sum(values) / n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[n // 2],
        }

    def _labels_to_key(self, labels: Optional[Dict[str, str]]) -> str:
        """Convert labels dict to a hashable key."""
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def clear(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timings.clear()


class TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        registry: MetricsRegistry,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.registry = registry
        self.name = name
        self.labels = labels
        self.start_time: Optional[float] = None

    def __enter__(self) -> "TimerContext":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            self.registry.record_timing(self.name, duration_ms, self.labels)


# ==================== Decorators ====================


def timed(
    metric_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
):
    """
    Decorator for timing function execution.

    Args:
        metric_name: Metric name (defaults to function name)
        labels: Optional labels

    Example:
        @timed("api_fetch", labels={"exchange": "binance"})
        async def fetch_ticker():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = metric_name or f"function.{func.__name__}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            registry = MetricsRegistry.get_instance()
            with registry.timer(name, labels):
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            registry = MetricsRegistry.get_instance()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                registry.record_timing(name, duration_ms, labels)

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def counted(
    metric_name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
):
    """
    Decorator for counting function calls.

    Args:
        metric_name: Metric name (defaults to function name)
        labels: Optional labels

    Example:
        @counted("api_calls", labels={"exchange": "binance"})
        def make_api_call():
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = metric_name or f"calls.{func.__name__}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            registry = MetricsRegistry.get_instance()
            registry.increment(name, labels=labels)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            registry = MetricsRegistry.get_instance()
            registry.increment(name, labels=labels)
            return await func(*args, **kwargs)

        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ==================== Health Check Framework ====================


class HealthStatus(Enum):
    """Component health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """
    System health check framework.

    Aggregates health checks from multiple components.

    Example:
        checker = HealthChecker()

        checker.register("database", lambda: ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY if db.is_connected() else HealthStatus.UNHEALTHY,
        ))

        overall_health = checker.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a health check.

        Args:
            name: Component name
            check_fn: Function that returns ComponentHealth
        """
        with self._lock:
            self._checks[name] = check_fn

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)

    def check(self, name: str) -> Optional[ComponentHealth]:
        """Run a specific health check."""
        with self._lock:
            check_fn = self._checks.get(name)

        if check_fn is None:
            return None

        try:
            return check_fn()
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
            )

    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Dictionary with overall status and individual checks
        """
        results = {}
        overall_status = HealthStatus.HEALTHY

        with self._lock:
            check_names = list(self._checks.keys())

        for name in check_names:
            result = self.check(name)
            if result:
                results[name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "last_check": result.last_check.isoformat(),
                }

                # Update overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    result.status == HealthStatus.DEGRADED
                    and overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "components": results,
            "timestamp": datetime.now().isoformat(),
        }


# ==================== Global Instances ====================


def get_metrics() -> MetricsRegistry:
    """Get global metrics registry."""
    return MetricsRegistry.get_instance()


_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
