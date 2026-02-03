"""
Thread Safety Utilities

Thread-safe primitives and utilities for concurrent access
in the MASP platform.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeDict(Generic[K, V]):
    """
    Thread-safe dictionary with read-write locking.

    Uses RLock for reentrant access and provides atomic operations.
    """

    def __init__(self, initial: Optional[Dict[K, V]] = None):
        self._data: Dict[K, V] = dict(initial) if initial else {}
        self._lock = threading.RLock()

    def __getitem__(self, key: K) -> V:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            return self._data.get(key, default)

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            return self._data.pop(key, default)

    def setdefault(self, key: K, default: V) -> V:
        with self._lock:
            return self._data.setdefault(key, default)

    def update(self, other: Dict[K, V]) -> None:
        with self._lock:
            self._data.update(other)

    def keys(self) -> List[K]:
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[V]:
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[tuple[K, V]]:
        with self._lock:
            return list(self._data.items())

    def copy(self) -> Dict[K, V]:
        with self._lock:
            return self._data.copy()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def get_and_set(self, key: K, value: V) -> Optional[V]:
        """Atomically get old value and set new value."""
        with self._lock:
            old = self._data.get(key)
            self._data[key] = value
            return old

    def compute_if_absent(self, key: K, compute_fn: Callable[[], V]) -> V:
        """Atomically compute and store value if key is absent."""
        with self._lock:
            if key not in self._data:
                self._data[key] = compute_fn()
            return self._data[key]


class ThreadSafeList(Generic[T]):
    """Thread-safe list with locking."""

    def __init__(self, initial: Optional[List[T]] = None):
        self._data: List[T] = list(initial) if initial else []
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __getitem__(self, index: int) -> T:
        with self._lock:
            return self._data[index]

    def append(self, item: T) -> None:
        with self._lock:
            self._data.append(item)

    def extend(self, items: List[T]) -> None:
        with self._lock:
            self._data.extend(items)

    def pop(self, index: int = -1) -> T:
        with self._lock:
            return self._data.pop(index)

    def remove(self, item: T) -> None:
        with self._lock:
            self._data.remove(item)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def copy(self) -> List[T]:
        with self._lock:
            return self._data.copy()


class ThreadSafeCounter:
    """Thread-safe counter with atomic increment/decrement."""

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def increment(self, amount: int = 1) -> int:
        """Increment and return new value."""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """Decrement and return new value."""
        with self._lock:
            self._value -= amount
            return self._value

    def get_and_reset(self) -> int:
        """Get current value and reset to 0."""
        with self._lock:
            old = self._value
            self._value = 0
            return old

    def compare_and_set(self, expected: int, new_value: int) -> bool:
        """Atomically set value if current equals expected."""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with size limit."""

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: K) -> Optional[V]:
        """Get item from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: K, value: V) -> None:
        """Put item in cache."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, key: K) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


@dataclass
class LockInfo:
    """Information about a held lock."""

    name: str
    holder: str
    acquired_at: float
    timeout: Optional[float] = None


class NamedLockManager:
    """
    Manager for named locks with timeout and deadlock detection.

    Example:
        lock_manager = NamedLockManager()

        with lock_manager.acquire("resource_1", timeout=5.0):
            # exclusive access to resource_1
            pass
    """

    def __init__(self, default_timeout: float = 30.0):
        self._locks: Dict[str, threading.RLock] = {}
        self._lock_info: Dict[str, LockInfo] = {}
        self._manager_lock = threading.Lock()
        self._default_timeout = default_timeout

    def _get_lock(self, name: str) -> threading.RLock:
        """Get or create a named lock."""
        with self._manager_lock:
            if name not in self._locks:
                self._locks[name] = threading.RLock()
            return self._locks[name]

    @contextmanager
    def acquire(
        self,
        name: str,
        timeout: Optional[float] = None,
        holder: str = "",
    ) -> Iterator[None]:
        """
        Context manager for acquiring a named lock.

        Args:
            name: Lock name
            timeout: Acquisition timeout (None = default)
            holder: Identifier of the lock holder

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        lock = self._get_lock(name)
        timeout = timeout if timeout is not None else self._default_timeout
        holder = holder or threading.current_thread().name

        acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire lock '{name}' within {timeout}s")

        try:
            with self._manager_lock:
                self._lock_info[name] = LockInfo(
                    name=name,
                    holder=holder,
                    acquired_at=time.time(),
                    timeout=timeout,
                )
            yield
        finally:
            with self._manager_lock:
                self._lock_info.pop(name, None)
            lock.release()

    def try_acquire(self, name: str, holder: str = "") -> bool:
        """
        Try to acquire a lock without blocking.

        Returns:
            True if lock was acquired
        """
        lock = self._get_lock(name)
        acquired = lock.acquire(blocking=False)

        if acquired:
            with self._manager_lock:
                self._lock_info[name] = LockInfo(
                    name=name,
                    holder=holder or threading.current_thread().name,
                    acquired_at=time.time(),
                )
        return acquired

    def release(self, name: str) -> None:
        """Release a named lock."""
        if name in self._locks:
            with self._manager_lock:
                self._lock_info.pop(name, None)
            try:
                self._locks[name].release()
            except RuntimeError:
                pass  # Lock not held

    def get_held_locks(self) -> List[LockInfo]:
        """Get information about currently held locks."""
        with self._manager_lock:
            return list(self._lock_info.values())

    def detect_long_held_locks(self, threshold_seconds: float = 60.0) -> List[LockInfo]:
        """Find locks held longer than threshold."""
        now = time.time()
        with self._manager_lock:
            return [
                info
                for info in self._lock_info.values()
                if now - info.acquired_at > threshold_seconds
            ]


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    Example:
        limiter = RateLimiter(rate=10, per_seconds=1)  # 10 requests per second

        if limiter.acquire():
            # proceed with request
            pass
    """

    def __init__(
        self,
        rate: float,
        per_seconds: float = 1.0,
        burst: Optional[float] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Number of tokens per time period
            per_seconds: Time period in seconds
            burst: Maximum burst size (default = rate)
        """
        self._rate = rate / per_seconds
        self._burst = burst if burst is not None else rate
        self._tokens = self._burst
        self._last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_update = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            blocking: Wait for tokens if not available

        Returns:
            True if tokens were acquired
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not blocking:
                return False

        # Wait for tokens
        while True:
            wait_time = (tokens - self._tokens) / self._rate
            if wait_time > 0:
                time.sleep(min(wait_time, 0.1))

            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

    @contextmanager
    def limit(self, tokens: float = 1.0) -> Iterator[None]:
        """Context manager for rate limiting."""
        self.acquire(tokens)
        yield

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        with self._lock:
            self._refill()
            return self._tokens


def synchronized(lock: Optional[threading.Lock] = None):
    """
    Decorator to synchronize method execution.

    Args:
        lock: Lock to use (creates new RLock if None)

    Example:
        class MyClass:
            _lock = threading.Lock()

            @synchronized(_lock)
            def critical_section(self):
                pass
    """

    def decorator(func: Callable) -> Callable:
        nonlocal lock
        if lock is None:
            lock = threading.RLock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def read_write_lock():
    """
    Create a read-write lock for multiple readers / single writer.

    Returns:
        Tuple of (read_lock_context, write_lock_context)

    Example:
        read_lock, write_lock = read_write_lock()

        with read_lock():
            # multiple readers allowed
            pass

        with write_lock():
            # exclusive write access
            pass
    """
    readers = 0
    readers_lock = threading.Lock()
    writer_lock = threading.Lock()

    @contextmanager
    def read_lock():
        nonlocal readers
        with readers_lock:
            readers += 1
            if readers == 1:
                writer_lock.acquire()
        try:
            yield
        finally:
            with readers_lock:
                readers -= 1
                if readers == 0:
                    writer_lock.release()

    @contextmanager
    def write_lock():
        writer_lock.acquire()
        try:
            yield
        finally:
            writer_lock.release()

    return read_lock, write_lock


class ThreadLocalState(Generic[T]):
    """
    Thread-local state with default value factory.

    Example:
        state = ThreadLocalState(lambda: {"request_id": None})
        state.get()["request_id"] = "123"
    """

    def __init__(self, factory: Callable[[], T]):
        self._local = threading.local()
        self._factory = factory

    def get(self) -> T:
        """Get thread-local state, creating if needed."""
        if not hasattr(self._local, "value"):
            self._local.value = self._factory()
        return self._local.value

    def set(self, value: T) -> None:
        """Set thread-local state."""
        self._local.value = value

    def clear(self) -> None:
        """Clear thread-local state."""
        if hasattr(self._local, "value"):
            del self._local.value


# Global lock manager instance
_global_lock_manager = NamedLockManager()


def get_lock_manager() -> NamedLockManager:
    """Get the global lock manager instance."""
    return _global_lock_manager
