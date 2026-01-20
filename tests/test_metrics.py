"""
Tests for metrics collection and observability module.
"""

import asyncio
import pytest
import time

from libs.core.metrics import (
    MetricsRegistry,
    MetricType,
    HistogramBuckets,
    TimerContext,
    HealthChecker,
    HealthStatus,
    ComponentHealth,
    timed,
    counted,
    get_metrics,
    get_health_checker,
)


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        MetricsRegistry.reset()
        yield
        MetricsRegistry.reset()

    def test_singleton(self):
        """Test registry is a singleton."""
        registry1 = MetricsRegistry.get_instance()
        registry2 = MetricsRegistry.get_instance()

        assert registry1 is registry2

    def test_increment_counter(self):
        """Test counter increment."""
        registry = MetricsRegistry.get_instance()

        registry.increment("requests_total")
        assert registry.get_counter("requests_total") == 1

        registry.increment("requests_total", 5)
        assert registry.get_counter("requests_total") == 6

    def test_counter_with_labels(self):
        """Test counter with labels."""
        registry = MetricsRegistry.get_instance()

        registry.increment("api_calls", labels={"exchange": "binance"})
        registry.increment("api_calls", labels={"exchange": "upbit"})
        registry.increment("api_calls", labels={"exchange": "binance"}, value=2)

        assert registry.get_counter("api_calls", labels={"exchange": "binance"}) == 3
        assert registry.get_counter("api_calls", labels={"exchange": "upbit"}) == 1

    def test_set_gauge(self):
        """Test gauge set."""
        registry = MetricsRegistry.get_instance()

        registry.set("active_connections", 5)
        assert registry.get_gauge("active_connections") == 5

        registry.set("active_connections", 3)
        assert registry.get_gauge("active_connections") == 3

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        registry = MetricsRegistry.get_instance()

        registry.set("queue_size", 10, labels={"queue": "orders"})
        registry.set("queue_size", 5, labels={"queue": "signals"})

        assert registry.get_gauge("queue_size", labels={"queue": "orders"}) == 10
        assert registry.get_gauge("queue_size", labels={"queue": "signals"}) == 5

    def test_inc_gauge(self):
        """Test gauge increment/decrement."""
        registry = MetricsRegistry.get_instance()

        registry.set("balance", 100)
        registry.inc_gauge("balance", 50)
        assert registry.get_gauge("balance") == 150

        registry.inc_gauge("balance", -30)
        assert registry.get_gauge("balance") == 120

    def test_observe_histogram(self):
        """Test histogram observation."""
        registry = MetricsRegistry.get_instance()

        for value in [5, 10, 15, 20, 25]:
            registry.observe("response_size", value)

        stats = registry.get_histogram("response_size")
        assert stats is not None
        assert stats["count"] == 5
        assert stats["sum"] == 75
        assert stats["avg"] == 15

    def test_histogram_with_labels(self):
        """Test histogram with labels."""
        registry = MetricsRegistry.get_instance()

        registry.observe("latency", 100, labels={"endpoint": "/ticker"})
        registry.observe("latency", 150, labels={"endpoint": "/ticker"})
        registry.observe("latency", 50, labels={"endpoint": "/balance"})

        ticker_stats = registry.get_histogram("latency", labels={"endpoint": "/ticker"})
        assert ticker_stats["count"] == 2
        assert ticker_stats["avg"] == 125

        balance_stats = registry.get_histogram("latency", labels={"endpoint": "/balance"})
        assert balance_stats["count"] == 1

    def test_custom_histogram_buckets(self):
        """Test histogram with custom buckets."""
        registry = MetricsRegistry.get_instance()

        buckets = [10, 50, 100, 500, 1000]
        for value in [5, 25, 75, 200, 800]:
            registry.observe("custom_metric", value, buckets=buckets)

        stats = registry.get_histogram("custom_metric")
        assert stats["count"] == 5

    def test_timer_context(self):
        """Test timer context manager."""
        registry = MetricsRegistry.get_instance()

        with registry.timer("operation_time"):
            time.sleep(0.01)

        stats = registry.get_timing_stats("operation_time")
        assert stats is not None
        assert stats["count"] == 1
        assert stats["avg"] >= 10  # At least 10ms

    def test_timer_with_labels(self):
        """Test timer with labels."""
        registry = MetricsRegistry.get_instance()

        with registry.timer("api_latency", labels={"exchange": "binance"}):
            time.sleep(0.005)

        stats = registry.get_timing_stats("api_latency", labels={"exchange": "binance"})
        assert stats is not None
        assert stats["count"] == 1

    def test_timing_stats(self):
        """Test timing statistics calculation."""
        registry = MetricsRegistry.get_instance()

        for ms in [10, 20, 30, 40, 50]:
            registry.record_timing("test_timing", ms)

        stats = registry.get_timing_stats("test_timing")
        assert stats["count"] == 5
        assert stats["avg"] == 30
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["p50"] == 30

    def test_get_all_metrics(self):
        """Test getting all metrics."""
        registry = MetricsRegistry.get_instance()

        registry.increment("counter1")
        registry.set("gauge1", 100)
        registry.observe("histogram1", 50)
        registry.record_timing("timing1", 25)

        all_metrics = registry.get_all_metrics()

        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "timings" in all_metrics
        assert "timestamp" in all_metrics

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        registry = MetricsRegistry.get_instance()

        registry.increment("test_counter")
        registry.set("test_gauge", 50)

        registry.clear()

        assert registry.get_counter("test_counter") == 0
        assert registry.get_gauge("test_gauge") == 0


class TestHistogramBuckets:
    """Tests for HistogramBuckets."""

    def test_bucket_initialization(self):
        """Test bucket initialization."""
        buckets = HistogramBuckets(boundaries=[10, 50, 100])

        assert 10 in buckets.counts
        assert 50 in buckets.counts
        assert 100 in buckets.counts
        assert float("inf") in buckets.counts

    def test_observe(self):
        """Test observation recording."""
        buckets = HistogramBuckets(boundaries=[10, 50, 100])

        buckets.observe(5)
        buckets.observe(25)
        buckets.observe(75)
        buckets.observe(150)

        assert buckets.count == 4
        assert buckets.sum == 255

        # 5 <= 10
        assert buckets.counts[10] == 1
        # 5, 25 <= 50
        assert buckets.counts[50] == 2
        # 5, 25, 75 <= 100
        assert buckets.counts[100] == 3
        # All values
        assert buckets.counts[float("inf")] == 4


class TestDecorators:
    """Tests for metric decorators."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        MetricsRegistry.reset()
        yield
        MetricsRegistry.reset()

    def test_timed_decorator_sync(self):
        """Test timed decorator with sync function."""

        @timed("decorated_function")
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

        registry = MetricsRegistry.get_instance()
        stats = registry.get_timing_stats("decorated_function")
        assert stats is not None
        assert stats["count"] == 1

    @pytest.mark.asyncio
    async def test_timed_decorator_async(self):
        """Test timed decorator with async function."""

        @timed("async_function")
        async def async_slow_function():
            await asyncio.sleep(0.01)
            return "async done"

        result = await async_slow_function()
        assert result == "async done"

        registry = MetricsRegistry.get_instance()
        stats = registry.get_timing_stats("async_function")
        assert stats is not None
        assert stats["count"] == 1

    def test_counted_decorator_sync(self):
        """Test counted decorator with sync function."""

        @counted("function_calls")
        def my_function():
            return "called"

        my_function()
        my_function()
        my_function()

        registry = MetricsRegistry.get_instance()
        assert registry.get_counter("function_calls") == 3

    @pytest.mark.asyncio
    async def test_counted_decorator_async(self):
        """Test counted decorator with async function."""

        @counted("async_calls")
        async def async_function():
            return "async called"

        await async_function()
        await async_function()

        registry = MetricsRegistry.get_instance()
        assert registry.get_counter("async_calls") == 2

    def test_decorator_with_labels(self):
        """Test decorators with labels."""

        @counted("api_calls", labels={"exchange": "binance"})
        def api_call():
            pass

        api_call()
        api_call()

        registry = MetricsRegistry.get_instance()
        assert registry.get_counter("api_calls", labels={"exchange": "binance"}) == 2


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_register_check(self):
        """Test registering a health check."""
        checker = HealthChecker()

        def db_check():
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Connected",
            )

        checker.register("database", db_check)
        result = checker.check("database")

        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    def test_check_all(self):
        """Test checking all components."""
        checker = HealthChecker()

        checker.register("db", lambda: ComponentHealth(
            name="db",
            status=HealthStatus.HEALTHY,
        ))
        checker.register("cache", lambda: ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
        ))

        result = checker.check_all()

        assert result["status"] == "healthy"
        assert "db" in result["components"]
        assert "cache" in result["components"]

    def test_degraded_status(self):
        """Test degraded overall status."""
        checker = HealthChecker()

        checker.register("healthy", lambda: ComponentHealth(
            name="healthy",
            status=HealthStatus.HEALTHY,
        ))
        checker.register("degraded", lambda: ComponentHealth(
            name="degraded",
            status=HealthStatus.DEGRADED,
            message="Slow response",
        ))

        result = checker.check_all()
        assert result["status"] == "degraded"

    def test_unhealthy_status(self):
        """Test unhealthy overall status."""
        checker = HealthChecker()

        checker.register("healthy", lambda: ComponentHealth(
            name="healthy",
            status=HealthStatus.HEALTHY,
        ))
        checker.register("unhealthy", lambda: ComponentHealth(
            name="unhealthy",
            status=HealthStatus.UNHEALTHY,
            message="Connection failed",
        ))

        result = checker.check_all()
        assert result["status"] == "unhealthy"

    def test_check_exception_handling(self):
        """Test health check handles exceptions."""
        checker = HealthChecker()

        def failing_check():
            raise RuntimeError("Check failed")

        checker.register("failing", failing_check)
        result = checker.check("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    def test_unregister_check(self):
        """Test unregistering a health check."""
        checker = HealthChecker()

        checker.register("temp", lambda: ComponentHealth(
            name="temp",
            status=HealthStatus.HEALTHY,
        ))
        checker.unregister("temp")

        result = checker.check("temp")
        assert result is None

    def test_component_health_details(self):
        """Test component health with details."""
        checker = HealthChecker()

        checker.register("detailed", lambda: ComponentHealth(
            name="detailed",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"latency_ms": 50, "connections": 10},
        ))

        result = checker.check_all()
        assert result["components"]["detailed"]["details"]["latency_ms"] == 50


class TestGlobalHelpers:
    """Tests for global helper functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        MetricsRegistry.reset()
        yield
        MetricsRegistry.reset()

    def test_get_metrics(self):
        """Test get_metrics returns singleton."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        assert metrics1 is metrics2
        assert isinstance(metrics1, MetricsRegistry)

    def test_get_health_checker(self):
        """Test get_health_checker returns instance."""
        checker = get_health_checker()

        assert checker is not None
        assert isinstance(checker, HealthChecker)


class TestThreadSafety:
    """Tests for thread safety."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset registry before each test."""
        MetricsRegistry.reset()
        yield
        MetricsRegistry.reset()

    def test_concurrent_increments(self):
        """Test concurrent counter increments."""
        import threading

        registry = MetricsRegistry.get_instance()
        num_threads = 10
        increments_per_thread = 100

        def increment_counter():
            for _ in range(increments_per_thread):
                registry.increment("concurrent_counter")

        threads = [
            threading.Thread(target=increment_counter)
            for _ in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread
        assert registry.get_counter("concurrent_counter") == expected

    def test_concurrent_timings(self):
        """Test concurrent timing recordings."""
        import threading

        registry = MetricsRegistry.get_instance()
        num_threads = 5
        timings_per_thread = 20

        def record_timings():
            for i in range(timings_per_thread):
                registry.record_timing("concurrent_timing", float(i))

        threads = [
            threading.Thread(target=record_timings)
            for _ in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = registry.get_timing_stats("concurrent_timing")
        assert stats["count"] == num_threads * timings_per_thread
