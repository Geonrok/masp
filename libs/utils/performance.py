"""
Performance monitoring and optimization utilities.
- Function timing decorator
- Performance metrics collection
- Resource usage tracking
"""
from __future__ import annotations

import functools
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Statistics for function timing."""

    function_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    last_call_time_ms: float = 0.0
    recent_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count

    @property
    def recent_avg_ms(self) -> float:
        """Average of recent calls."""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)

    @property
    def std_dev_ms(self) -> float:
        """Standard deviation of recent calls."""
        if len(self.recent_times) < 2:
            return 0.0
        return statistics.stdev(self.recent_times)

    def record(self, time_ms: float) -> None:
        """Record a timing measurement."""
        self.call_count += 1
        self.total_time_ms += time_ms
        self.last_call_time_ms = time_ms
        self.min_time_ms = min(self.min_time_ms, time_ms)
        self.max_time_ms = max(self.max_time_ms, time_ms)
        self.recent_times.append(time_ms)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "call_count": self.call_count,
            "total_time_ms": round(self.total_time_ms, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "min_time_ms": round(self.min_time_ms, 2) if self.min_time_ms != float("inf") else 0,
            "max_time_ms": round(self.max_time_ms, 2),
            "recent_avg_ms": round(self.recent_avg_ms, 2),
            "std_dev_ms": round(self.std_dev_ms, 2),
        }


class PerformanceMonitor:
    """Global performance monitor for tracking function timings."""

    _instance: Optional["PerformanceMonitor"] = None

    def __new__(cls) -> "PerformanceMonitor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._stats: Dict[str, TimingStats] = {}
        self._lock = threading.RLock()
        self._enabled = True
        self._slow_threshold_ms = 1000.0  # Log warnings for slow calls

        self._initialized = True

    def record(self, function_name: str, time_ms: float) -> None:
        """Record a timing measurement.

        Args:
            function_name: Name of the function
            time_ms: Execution time in milliseconds
        """
        if not self._enabled:
            return

        with self._lock:
            if function_name not in self._stats:
                self._stats[function_name] = TimingStats(function_name=function_name)

            self._stats[function_name].record(time_ms)

        # Log slow calls
        if time_ms > self._slow_threshold_ms:
            logger.warning(
                "[Performance] Slow call: %s took %.2fms",
                function_name,
                time_ms,
            )

    def get_stats(self, function_name: str) -> Optional[TimingStats]:
        """Get stats for a specific function.

        Args:
            function_name: Name of the function

        Returns:
            TimingStats or None
        """
        with self._lock:
            return self._stats.get(function_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all timing statistics.

        Returns:
            Dict of function name to stats dict
        """
        with self._lock:
            return {name: stats.to_dict() for name, stats in self._stats.items()}

    def get_slowest(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n slowest functions by average time.

        Args:
            n: Number of functions to return

        Returns:
            List of stats dicts, sorted by avg time
        """
        with self._lock:
            sorted_stats = sorted(
                self._stats.values(),
                key=lambda s: s.avg_time_ms,
                reverse=True,
            )
            return [s.to_dict() for s in sorted_stats[:n]]

    def get_most_called(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most called functions.

        Args:
            n: Number of functions to return

        Returns:
            List of stats dicts, sorted by call count
        """
        with self._lock:
            sorted_stats = sorted(
                self._stats.values(),
                key=lambda s: s.call_count,
                reverse=True,
            )
            return [s.to_dict() for s in sorted_stats[:n]]

    def clear(self) -> None:
        """Clear all stats."""
        with self._lock:
            self._stats.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable monitoring."""
        self._enabled = enabled

    def set_slow_threshold(self, threshold_ms: float) -> None:
        """Set threshold for slow call warnings."""
        self._slow_threshold_ms = threshold_ms


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance.

    Returns:
        PerformanceMonitor singleton
    """
    return PerformanceMonitor()


# =============================================================================
# Decorators
# =============================================================================


def timed(
    name: Optional[str] = None,
    log_args: bool = False,
    monitor: Optional[PerformanceMonitor] = None,
) -> Callable:
    """Decorator to time function execution.

    Args:
        name: Custom name (defaults to function name)
        log_args: Whether to log function arguments
        monitor: Performance monitor instance (uses global if None)

    Returns:
        Decorated function
    """
    _monitor = monitor or get_performance_monitor()

    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _monitor.record(func_name, elapsed_ms)

                if log_args:
                    logger.debug(
                        "[Timing] %s(%s, %s) took %.2fms",
                        func_name,
                        args,
                        kwargs,
                        elapsed_ms,
                    )

        return wrapper

    return decorator


@contextmanager
def timer(name: str, monitor: Optional[PerformanceMonitor] = None):
    """Context manager for timing code blocks.

    Args:
        name: Name for the timing
        monitor: Performance monitor instance

    Yields:
        None

    Example:
        with timer("database_query"):
            result = db.query(...)
    """
    _monitor = monitor or get_performance_monitor()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _monitor.record(name, elapsed_ms)


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Token bucket rate limiter.

    Thread-safe implementation for API rate limiting.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 1,
        name: str = "default",
    ):
        """Initialize rate limiter.

        Args:
            rate: Requests per second
            burst: Maximum burst size
            name: Limiter name for logging
        """
        self.rate = rate
        self.burst = burst
        self.name = name

        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()

        # Stats
        self._total_requests = 0
        self._throttled_requests = 0

    def acquire(self, tokens: int = 1, timeout: float = 0.0) -> bool:
        """Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (0 = no wait)

        Returns:
            True if tokens acquired, False if rate limited
        """
        start = time.time()

        while True:
            with self._lock:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_requests += 1
                    return True

                # Check timeout
                if timeout <= 0 or (time.time() - start) >= timeout:
                    self._throttled_requests += 1
                    return False

            # Wait for tokens
            wait_time = min(0.1, tokens / self.rate)
            time.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on rate
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "name": self.name,
                "rate": self.rate,
                "burst": self.burst,
                "tokens": self._tokens,
                "total_requests": self._total_requests,
                "throttled_requests": self._throttled_requests,
                "throttle_rate": (
                    self._throttled_requests / self._total_requests
                    if self._total_requests > 0
                    else 0.0
                ),
            }


def rate_limited(
    rate: float,
    burst: int = 1,
    timeout: float = 5.0,
) -> Callable:
    """Decorator for rate limiting function calls.

    Args:
        rate: Requests per second
        burst: Maximum burst size
        timeout: Maximum wait time

    Returns:
        Decorated function
    """
    limiter = RateLimiter(rate=rate, burst=burst)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.acquire(timeout=timeout):
                raise RuntimeError(f"Rate limit exceeded for {func.__name__}")
            return func(*args, **kwargs)

        wrapper.rate_limiter = limiter

        return wrapper

    return decorator


# =============================================================================
# Connection Pool
# =============================================================================


class ConnectionPool:
    """Generic connection pool for reusing expensive connections."""

    def __init__(
        self,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 1,
    ):
        """Initialize pool.

        Args:
            factory: Function to create new connections
            max_size: Maximum pool size
            min_size: Minimum connections to keep
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size

        self._pool: Deque[Any] = deque()
        self._in_use: int = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

        # Pre-create minimum connections
        for _ in range(min_size):
            self._pool.append(factory())

    def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a connection from the pool.

        Args:
            timeout: Maximum wait time

        Returns:
            Connection object

        Raises:
            TimeoutError: If no connection available
        """
        with self._condition:
            start = time.time()

            while True:
                # Try to get from pool
                if self._pool:
                    conn = self._pool.popleft()
                    self._in_use += 1
                    return conn

                # Create new if under limit
                total = len(self._pool) + self._in_use
                if total < self.max_size:
                    conn = self.factory()
                    self._in_use += 1
                    return conn

                # Wait for available connection
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError("No connection available")

                self._condition.wait(remaining)

    def release(self, conn: Any) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection to return
        """
        with self._condition:
            self._in_use -= 1
            self._pool.append(conn)
            self._condition.notify()

    @contextmanager
    def connection(self):
        """Context manager for acquiring/releasing connections.

        Yields:
            Connection object
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "in_use": self._in_use,
                "max_size": self.max_size,
                "min_size": self.min_size,
            }
