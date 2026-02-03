"""
Tests for thread safety utilities.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from libs.core.thread_safety import (
    ThreadSafeDict,
    ThreadSafeList,
    ThreadSafeCounter,
    LRUCache,
    NamedLockManager,
    RateLimiter,
    synchronized,
    read_write_lock,
    ThreadLocalState,
    get_lock_manager,
)


class TestThreadSafeDict:
    """Tests for ThreadSafeDict."""

    def test_basic_operations(self):
        """Test basic dict operations."""
        d = ThreadSafeDict()

        d["key"] = "value"
        assert d["key"] == "value"
        assert "key" in d
        assert len(d) == 1

        del d["key"]
        assert "key" not in d

    def test_get_with_default(self):
        """Test get with default value."""
        d = ThreadSafeDict()
        assert d.get("missing", "default") == "default"

    def test_setdefault(self):
        """Test setdefault operation."""
        d = ThreadSafeDict()
        result = d.setdefault("key", "value")
        assert result == "value"
        assert d["key"] == "value"

        result = d.setdefault("key", "other")
        assert result == "value"

    def test_concurrent_access(self):
        """Test concurrent read/write access."""
        d = ThreadSafeDict()
        errors = []

        def writer():
            for i in range(100):
                d[f"key_{i}"] = i

        def reader():
            for i in range(100):
                try:
                    _ = d.get(f"key_{i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)] + [
            threading.Thread(target=reader) for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_get_and_set(self):
        """Test atomic get and set."""
        d = ThreadSafeDict({"key": "old"})
        old = d.get_and_set("key", "new")
        assert old == "old"
        assert d["key"] == "new"

    def test_compute_if_absent(self):
        """Test compute if absent."""
        d = ThreadSafeDict()
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        result1 = d.compute_if_absent("key", compute)
        result2 = d.compute_if_absent("key", compute)

        assert result1 == "computed"
        assert result2 == "computed"
        assert call_count == 1  # Only called once


class TestThreadSafeList:
    """Tests for ThreadSafeList."""

    def test_basic_operations(self):
        """Test basic list operations."""
        lst = ThreadSafeList()

        lst.append(1)
        lst.append(2)
        assert len(lst) == 2
        assert lst[0] == 1

        lst.pop()
        assert len(lst) == 1

    def test_concurrent_append(self):
        """Test concurrent appends."""
        lst = ThreadSafeList()

        def appender():
            for i in range(100):
                lst.append(i)

        threads = [threading.Thread(target=appender) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(lst) == 1000


class TestThreadSafeCounter:
    """Tests for ThreadSafeCounter."""

    def test_increment(self):
        """Test increment operation."""
        counter = ThreadSafeCounter()
        assert counter.increment() == 1
        assert counter.increment(5) == 6
        assert counter.value == 6

    def test_decrement(self):
        """Test decrement operation."""
        counter = ThreadSafeCounter(10)
        assert counter.decrement() == 9
        assert counter.decrement(5) == 4

    def test_concurrent_increment(self):
        """Test concurrent increments."""
        counter = ThreadSafeCounter()

        def incrementer():
            for _ in range(1000):
                counter.increment()

        threads = [threading.Thread(target=incrementer) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == 10000

    def test_compare_and_set(self):
        """Test compare and set operation."""
        counter = ThreadSafeCounter(5)

        assert counter.compare_and_set(5, 10) is True
        assert counter.value == 10

        assert counter.compare_and_set(5, 15) is False
        assert counter.value == 10

    def test_get_and_reset(self):
        """Test get and reset operation."""
        counter = ThreadSafeCounter(42)
        value = counter.get_and_reset()

        assert value == 42
        assert counter.value == 0


class TestLRUCache:
    """Tests for LRUCache."""

    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") is None

    def test_eviction(self):
        """Test LRU eviction."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_access_order(self):
        """Test that access updates LRU order."""
        cache = LRUCache(max_size=2)

        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # Access "a" to make it recent
        cache.put("c", 3)  # Should evict "b", not "a"

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10)

        cache.put("a", 1)
        cache.get("a")  # Hit
        cache.get("a")  # Hit
        cache.get("b")  # Miss

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)


class TestNamedLockManager:
    """Tests for NamedLockManager."""

    def test_basic_locking(self):
        """Test basic lock acquisition."""
        manager = NamedLockManager()

        with manager.acquire("test_lock"):
            # Lock held
            assert len(manager.get_held_locks()) == 1

        assert len(manager.get_held_locks()) == 0

    def test_reentrant_locking(self):
        """Test that same thread can acquire lock multiple times."""
        manager = NamedLockManager()

        with manager.acquire("test_lock"):
            with manager.acquire("test_lock"):
                # Both locks held by same thread
                pass

    def test_timeout(self):
        """Test lock acquisition timeout."""
        manager = NamedLockManager(default_timeout=0.1)
        lock_acquired = threading.Event()

        def holder():
            with manager.acquire("test_lock"):
                lock_acquired.set()
                time.sleep(0.5)

        thread = threading.Thread(target=holder)
        thread.start()
        lock_acquired.wait()

        with pytest.raises(TimeoutError):
            with manager.acquire("test_lock", timeout=0.1):
                pass

        thread.join()

    def test_try_acquire(self):
        """Test non-blocking acquire."""
        manager = NamedLockManager()

        assert manager.try_acquire("test_lock") is True
        manager.release("test_lock")


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_basic_limiting(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(rate=10, per_seconds=1)

        # Should succeed immediately
        for _ in range(10):
            assert limiter.acquire(blocking=False) is True

        # 11th should fail (non-blocking)
        assert limiter.acquire(blocking=False) is False

    def test_refill(self):
        """Test token refill over time."""
        limiter = RateLimiter(rate=10, per_seconds=1)

        # Exhaust tokens
        for _ in range(10):
            limiter.acquire(blocking=False)

        # Wait for refill
        time.sleep(0.2)

        # Should have ~2 tokens refilled
        assert limiter.acquire(blocking=False) is True

    def test_context_manager(self):
        """Test rate limiter context manager."""
        limiter = RateLimiter(rate=100, per_seconds=1)

        with limiter.limit():
            pass  # Should succeed


class TestSynchronizedDecorator:
    """Tests for synchronized decorator."""

    def test_synchronized_method(self):
        """Test synchronized method execution."""
        lock = threading.Lock()
        execution_order = []

        @synchronized(lock)
        def critical_section(id):
            execution_order.append(f"start_{id}")
            time.sleep(0.01)
            execution_order.append(f"end_{id}")

        threads = [
            threading.Thread(target=critical_section, args=(i,)) for i in range(3)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no interleaving
        for i in range(0, len(execution_order), 2):
            start_id = execution_order[i].split("_")[1]
            end_id = execution_order[i + 1].split("_")[1]
            assert start_id == end_id


class TestReadWriteLock:
    """Tests for read-write lock."""

    def test_multiple_readers(self):
        """Test multiple concurrent readers."""
        read_lock, write_lock = read_write_lock()
        active_readers = ThreadSafeCounter()
        max_readers = ThreadSafeCounter()

        def reader():
            with read_lock():
                current = active_readers.increment()
                max_readers.compare_and_set(
                    max_readers.value,
                    max(max_readers.value, current),
                )
                time.sleep(0.01)
                active_readers.decrement()

        threads = [threading.Thread(target=reader) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Multiple readers should have been active simultaneously
        assert max_readers.value > 1

    def test_exclusive_writer(self):
        """Test exclusive writer access."""
        read_lock, write_lock = read_write_lock()
        is_writing = threading.Event()
        readers_during_write = []

        def writer():
            with write_lock():
                is_writing.set()
                time.sleep(0.05)
                is_writing.clear()

        def reader():
            with read_lock():
                readers_during_write.append(is_writing.is_set())

        writer_thread = threading.Thread(target=writer)
        writer_thread.start()

        time.sleep(0.01)  # Let writer acquire lock

        reader_threads = [threading.Thread(target=reader) for _ in range(3)]
        for t in reader_threads:
            t.start()
        for t in reader_threads:
            t.join()
        writer_thread.join()

        # No readers should have been active during write
        assert not any(readers_during_write)


class TestThreadLocalState:
    """Tests for ThreadLocalState."""

    def test_thread_isolation(self):
        """Test that state is isolated per thread."""
        state = ThreadLocalState(lambda: {"value": 0})
        results = {}

        def worker(thread_id):
            state.get()["value"] = thread_id
            time.sleep(0.01)
            results[thread_id] = state.get()["value"]

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should see its own value
        for thread_id, value in results.items():
            assert value == thread_id

    def test_factory_called_per_thread(self):
        """Test that factory is called once per thread."""
        call_count = ThreadSafeCounter()

        def factory():
            call_count.increment()
            return {}

        state = ThreadLocalState(factory)

        def worker():
            state.get()
            state.get()  # Should not call factory again

        threads = [threading.Thread(target=worker) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count.value == 3  # Once per thread


class TestConcurrencyStress:
    """Stress tests for concurrency."""

    def test_high_contention(self):
        """Test under high contention."""
        d = ThreadSafeDict()
        counter = ThreadSafeCounter()

        def worker():
            for i in range(1000):
                d[f"key_{i}"] = counter.increment()

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            for f in as_completed(futures):
                f.result()

        assert counter.value == 20000
