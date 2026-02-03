"""Tests for performance and cache utilities."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

# =============================================================================
# TTLCache Tests
# =============================================================================


def test_ttl_cache_basic():
    """Test basic TTLCache get/set."""
    from libs.utils.cache import TTLCache

    cache = TTLCache(default_ttl=60.0)

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_ttl_cache_miss():
    """Test TTLCache miss returns None."""
    from libs.utils.cache import TTLCache

    cache = TTLCache()
    assert cache.get("nonexistent") is None


def test_ttl_cache_expiration():
    """Test TTLCache entry expiration."""
    from libs.utils.cache import TTLCache

    cache = TTLCache(default_ttl=0.1)  # 100ms TTL

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    time.sleep(0.15)  # Wait for expiration
    assert cache.get("key1") is None


def test_ttl_cache_custom_ttl():
    """Test TTLCache with custom TTL per entry."""
    from libs.utils.cache import TTLCache

    cache = TTLCache(default_ttl=60.0)

    cache.set("key1", "value1", ttl=0.1)  # 100ms
    cache.set("key2", "value2", ttl=60.0)  # 60s

    time.sleep(0.15)

    assert cache.get("key1") is None  # Expired
    assert cache.get("key2") == "value2"  # Still valid


def test_ttl_cache_delete():
    """Test TTLCache delete."""
    from libs.utils.cache import TTLCache

    cache = TTLCache()

    cache.set("key1", "value1")
    assert cache.delete("key1") is True
    assert cache.get("key1") is None
    assert cache.delete("key1") is False


def test_ttl_cache_clear():
    """Test TTLCache clear."""
    from libs.utils.cache import TTLCache

    cache = TTLCache()

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    count = cache.clear()
    assert count == 2
    assert len(cache) == 0


def test_ttl_cache_max_size():
    """Test TTLCache max size eviction."""
    from libs.utils.cache import TTLCache

    cache = TTLCache(max_size=3)

    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")
    cache.set("key4", "value4")

    assert len(cache) == 3
    assert cache.get("key1") is None  # Evicted (oldest)


def test_ttl_cache_stats():
    """Test TTLCache statistics."""
    from libs.utils.cache import TTLCache

    cache = TTLCache()

    cache.set("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("key1")  # Hit
    cache.get("missing")  # Miss

    stats = cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1


# =============================================================================
# LRUCache Tests
# =============================================================================


def test_lru_cache_basic():
    """Test basic LRUCache operations."""
    from libs.utils.cache import LRUCache

    cache = LRUCache(max_size=100)

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_lru_cache_eviction():
    """Test LRUCache evicts least recently used."""
    from libs.utils.cache import LRUCache

    cache = LRUCache(max_size=2)

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    # Access key1 (making key2 least recently used)
    cache.get("key1")

    # Add key3, should evict key2
    cache.set("key3", "value3")

    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None  # Evicted
    assert cache.get("key3") == "value3"


def test_lru_cache_stats():
    """Test LRUCache statistics."""
    from libs.utils.cache import LRUCache

    cache = LRUCache()

    cache.set("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("missing")  # Miss

    stats = cache.get_stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 1


# =============================================================================
# Cache Decorator Tests
# =============================================================================


def test_cached_decorator():
    """Test @cached decorator."""
    from libs.utils.cache import cached

    call_count = 0

    @cached(ttl=60.0)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # First call - computes
    result1 = expensive_function(5)
    assert result1 == 10
    assert call_count == 1

    # Second call - from cache
    result2 = expensive_function(5)
    assert result2 == 10
    assert call_count == 1  # Not called again

    # Different argument - computes
    result3 = expensive_function(10)
    assert result3 == 20
    assert call_count == 2


def test_lru_cached_decorator():
    """Test @lru_cached decorator."""
    from libs.utils.cache import lru_cached

    call_count = 0

    @lru_cached(max_size=10)
    def compute(x):
        nonlocal call_count
        call_count += 1
        return x**2

    assert compute(5) == 25
    assert call_count == 1

    assert compute(5) == 25
    assert call_count == 1  # Cached


def test_make_cache_key():
    """Test cache key generation."""
    from libs.utils.cache import make_cache_key

    key1 = make_cache_key(1, 2, a="x", b="y")
    key2 = make_cache_key(1, 2, a="x", b="y")
    key3 = make_cache_key(1, 2, a="x", b="z")

    assert key1 == key2
    assert key1 != key3


# =============================================================================
# Performance Monitor Tests
# =============================================================================


def test_timing_stats():
    """Test TimingStats dataclass."""
    from libs.utils.performance import TimingStats

    stats = TimingStats(function_name="test_func")

    stats.record(10.0)
    stats.record(20.0)
    stats.record(15.0)

    assert stats.call_count == 3
    assert stats.avg_time_ms == 15.0
    assert stats.min_time_ms == 10.0
    assert stats.max_time_ms == 20.0


def test_timing_stats_to_dict():
    """Test TimingStats to_dict."""
    from libs.utils.performance import TimingStats

    stats = TimingStats(function_name="test")
    stats.record(10.0)

    data = stats.to_dict()

    assert data["function_name"] == "test"
    assert data["call_count"] == 1


def test_performance_monitor_singleton():
    """Test PerformanceMonitor is singleton."""
    from libs.utils.performance import PerformanceMonitor

    m1 = PerformanceMonitor()
    m2 = PerformanceMonitor()

    assert m1 is m2


def test_performance_monitor_record():
    """Test PerformanceMonitor recording."""
    from libs.utils.performance import get_performance_monitor

    monitor = get_performance_monitor()
    monitor.clear()

    monitor.record("test_func", 10.0)
    monitor.record("test_func", 20.0)

    stats = monitor.get_stats("test_func")

    assert stats is not None
    assert stats.call_count == 2


def test_performance_monitor_get_slowest():
    """Test PerformanceMonitor get_slowest."""
    from libs.utils.performance import get_performance_monitor

    monitor = get_performance_monitor()
    monitor.clear()

    monitor.record("fast_func", 10.0)
    monitor.record("slow_func", 100.0)

    slowest = monitor.get_slowest(n=1)

    assert len(slowest) == 1
    assert slowest[0]["function_name"] == "slow_func"


def test_timed_decorator():
    """Test @timed decorator."""
    from libs.utils.performance import get_performance_monitor, timed

    monitor = get_performance_monitor()
    monitor.clear()

    @timed(name="decorated_func")
    def slow_func():
        time.sleep(0.01)
        return 42

    result = slow_func()

    assert result == 42

    stats = monitor.get_stats("decorated_func")
    assert stats is not None
    assert stats.call_count == 1
    assert stats.avg_time_ms >= 10  # At least 10ms


def test_timer_context_manager():
    """Test timer context manager."""
    from libs.utils.performance import get_performance_monitor, timer

    monitor = get_performance_monitor()
    monitor.clear()

    with timer("test_block"):
        time.sleep(0.01)

    stats = monitor.get_stats("test_block")
    assert stats is not None
    assert stats.call_count == 1


# =============================================================================
# Rate Limiter Tests
# =============================================================================


def test_rate_limiter_basic():
    """Test basic rate limiter."""
    from libs.utils.performance import RateLimiter

    limiter = RateLimiter(rate=10, burst=5)

    # Should allow burst
    for _ in range(5):
        assert limiter.acquire() is True


def test_rate_limiter_throttle():
    """Test rate limiter throttling."""
    from libs.utils.performance import RateLimiter

    limiter = RateLimiter(rate=1, burst=1)

    assert limiter.acquire() is True
    assert limiter.acquire(timeout=0) is False  # Throttled


def test_rate_limiter_stats():
    """Test rate limiter statistics."""
    from libs.utils.performance import RateLimiter

    limiter = RateLimiter(rate=10, burst=2)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire(timeout=0)  # Throttled

    stats = limiter.get_stats()

    assert stats["total_requests"] == 2
    assert stats["throttled_requests"] == 1


def test_rate_limited_decorator():
    """Test @rate_limited decorator."""
    from libs.utils.performance import rate_limited

    @rate_limited(rate=10, burst=2, timeout=0)
    def api_call():
        return "result"

    # First calls should succeed
    assert api_call() == "result"
    assert api_call() == "result"

    # Should be throttled
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        api_call()


# =============================================================================
# Connection Pool Tests
# =============================================================================


def test_connection_pool_basic():
    """Test basic connection pool."""
    from libs.utils.performance import ConnectionPool

    pool = ConnectionPool(factory=lambda: MagicMock(), max_size=5, min_size=2)

    conn = pool.acquire()
    assert conn is not None

    pool.release(conn)


def test_connection_pool_context_manager():
    """Test connection pool context manager."""
    from libs.utils.performance import ConnectionPool

    pool = ConnectionPool(factory=lambda: MagicMock(), max_size=5)

    with pool.connection() as conn:
        assert conn is not None


def test_connection_pool_stats():
    """Test connection pool statistics."""
    from libs.utils.performance import ConnectionPool

    pool = ConnectionPool(factory=lambda: MagicMock(), max_size=5, min_size=2)

    stats = pool.get_stats()

    assert stats["pool_size"] == 2
    assert stats["max_size"] == 5


# =============================================================================
# Global Cache Tests
# =============================================================================


def test_global_caches():
    """Test global cache instances."""
    from libs.utils.cache import (
        get_api_cache,
        get_config_cache,
        get_market_data_cache,
    )

    market_cache = get_market_data_cache()
    config_cache = get_config_cache()
    api_cache = get_api_cache()

    assert market_cache is not None
    assert config_cache is not None
    assert api_cache is not None

    # They should be different instances
    assert market_cache is not config_cache
    assert config_cache is not api_cache
