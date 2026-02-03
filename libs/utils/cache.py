"""
Caching utilities for performance optimization.
- TTL-based in-memory cache
- LRU cache with size limits
- Cache decorator for functions
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL support."""

    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def hit(self) -> T:
        """Record a cache hit and return value."""
        self.hits += 1
        return self.value


class TTLCache(Generic[T]):
    """Time-To-Live cache with automatic expiration.

    Thread-safe implementation with configurable TTL and max size.
    """

    def __init__(
        self,
        default_ttl: float = 60.0,
        max_size: int = 1000,
        cleanup_interval: float = 60.0,
    ):
        """Initialize cache.

        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of entries
            cleanup_interval: Interval for automatic cleanup
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Stats
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.hit()

    def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None = use default)
        """
        with self._lock:
            # Check size limit
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.default_ttl > 0:
                expires_at = time.time() + self.default_ttl

            self._cache[key] = CacheEntry(
                value=value,
                expires_at=expires_at,
            )

    def delete(self, key: str) -> bool:
        """Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has passed."""
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            self._cleanup()
            self._last_cleanup = now

    def _cleanup(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("[Cache] Cleaned up %d expired entries", len(expired_keys))

        return len(expired_keys)

    def _evict_oldest(self) -> None:
        """Evict oldest entry when at capacity."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at,
        )
        del self._cache[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Stats dict with hits, misses, size, etc.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "default_ttl": self.default_ttl,
            }

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Return number of cache entries."""
        return len(self._cache)


class LRUCache(Generic[T]):
    """Least Recently Used cache with size limit.

    Thread-safe implementation using OrderedDict.
    """

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """Get value and move to end (most recently used).

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

    def set(self, key: str, value: T) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache[key] = value
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = value

                # Evict if over capacity
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


# =============================================================================
# Cache Decorators
# =============================================================================


def make_cache_key(*args, **kwargs) -> str:
    """Create cache key from function arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Hash-based cache key
    """
    key_data = json.dumps(
        {
            "args": [str(a) for a in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
    )
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(
    ttl: float = 60.0,
    max_size: int = 100,
    cache: Optional[TTLCache] = None,
) -> Callable:
    """Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds
        max_size: Maximum cache size
        cache: External cache instance (creates new if None)

    Returns:
        Decorated function
    """
    _cache = cache or TTLCache(default_ttl=ttl, max_size=max_size)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{make_cache_key(*args, **kwargs)}"

            # Try cache first
            result = _cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            _cache.set(key, result)

            return result

        # Attach cache for inspection
        wrapper.cache = _cache
        wrapper.cache_clear = _cache.clear

        return wrapper

    return decorator


def lru_cached(max_size: int = 100) -> Callable:
    """Decorator for LRU caching.

    Args:
        max_size: Maximum cache size

    Returns:
        Decorated function
    """
    _cache = LRUCache(max_size=max_size)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{make_cache_key(*args, **kwargs)}"

            result = _cache.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            _cache.set(key, result)

            return result

        wrapper.cache = _cache
        wrapper.cache_clear = _cache.clear

        return wrapper

    return decorator


# =============================================================================
# Global caches for common use cases
# =============================================================================


# Market data cache (short TTL for real-time data)
_market_data_cache: Optional[TTLCache] = None


def get_market_data_cache() -> TTLCache:
    """Get global market data cache (TTL=5s).

    Returns:
        TTLCache instance
    """
    global _market_data_cache
    if _market_data_cache is None:
        _market_data_cache = TTLCache(default_ttl=5.0, max_size=500)
    return _market_data_cache


# Config cache (longer TTL for configuration data)
_config_cache: Optional[TTLCache] = None


def get_config_cache() -> TTLCache:
    """Get global config cache (TTL=300s).

    Returns:
        TTLCache instance
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = TTLCache(default_ttl=300.0, max_size=100)
    return _config_cache


# API response cache (medium TTL)
_api_cache: Optional[TTLCache] = None


def get_api_cache() -> TTLCache:
    """Get global API response cache (TTL=30s).

    Returns:
        TTLCache instance
    """
    global _api_cache
    if _api_cache is None:
        _api_cache = TTLCache(default_ttl=30.0, max_size=200)
    return _api_cache
