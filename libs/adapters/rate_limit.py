"""
Rate limit utilities.
"""
from __future__ import annotations

import asyncio
import logging
import time
from threading import Lock
from typing import Optional, Union

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate_per_sec = rate_per_sec
        self._capacity = capacity if capacity is not None else rate_per_sec
        self._tokens = self._capacity
        self._updated_at = time.monotonic()
        self._lock = Lock()

    def consume(self, tokens: float = 1.0) -> None:
        while True:
            wait_time = 0.0
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._tokens = min(self._capacity, self._tokens + elapsed * self.rate_per_sec)
                self._updated_at = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                needed = tokens - self._tokens
                if self.rate_per_sec > 0:
                    wait_time = needed / self.rate_per_sec

            if wait_time > 0:
                time.sleep(wait_time)
            else:
                return

    def set_tokens(self, remaining: float) -> None:
        """
        Sync bucket tokens to server-reported remaining.

        Args:
            remaining: Remaining tokens reported by server.
        """
        with self._lock:
            clamped = max(0.0, min(float(remaining), float(self._capacity)))
            self._tokens = clamped
            logger.debug("[TokenBucket] Synced to %s/%s tokens", clamped, self._capacity)

    @property
    def available(self) -> int:
        """Current available tokens."""
        with self._lock:
            return int(self._tokens)

    @property
    def capacity(self) -> float:
        """Bucket capacity."""
        return self._capacity


class AsyncTokenBucket:
    """Async token bucket rate limiter (lazy init)."""

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None):
        self.rate_per_sec = rate_per_sec
        self._capacity = capacity if capacity is not None else rate_per_sec
        self._tokens = float(self._capacity)
        self._updated_at = 0.0
        self._initialized = False
        self._lock = asyncio.Lock()

    def _get_loop_time(self) -> float:
        return asyncio.get_running_loop().time()

    def _ensure_initialized(self) -> None:
        """
        Initialize _updated_at on first access.

        This lazy initialization pattern avoids calling get_running_loop()
        in __init__, which would fail if no event loop is running.
        """
        if not self._initialized:
            self._updated_at = self._get_loop_time()
            self._initialized = True

    async def consume(self, tokens: float = 1.0) -> None:
        while True:
            wait_time = 0.0
            async with self._lock:
                self._ensure_initialized()
                now = self._get_loop_time()
                elapsed = now - self._updated_at
                self._tokens = min(self._capacity, self._tokens + elapsed * self.rate_per_sec)
                self._updated_at = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                needed = tokens - self._tokens
                if self.rate_per_sec > 0:
                    wait_time = needed / self.rate_per_sec

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                return

    async def set_tokens(self, remaining: float) -> None:
        """
        Sync bucket tokens to server-reported remaining.

        Args:
            remaining: Remaining tokens reported by server.
        """
        async with self._lock:
            self._ensure_initialized()
            clamped = max(0.0, min(float(remaining), float(self._capacity)))
            self._tokens = clamped
            logger.debug("[AsyncTokenBucket] Synced to %s/%s tokens", clamped, self._capacity)

    async def available(self) -> int:
        """Current available tokens."""
        async with self._lock:
            self._ensure_initialized()
            return int(self._tokens)

    @property
    def capacity(self) -> float:
        """Bucket capacity."""
        return self._capacity


def create_token_bucket(
    rate_per_sec: float,
    capacity: Optional[float] = None,
    async_mode: bool = False,
) -> Union[TokenBucket, AsyncTokenBucket]:
    """Token bucket factory."""
    if async_mode:
        return AsyncTokenBucket(rate_per_sec=rate_per_sec, capacity=capacity)
    return TokenBucket(rate_per_sec=rate_per_sec, capacity=capacity)
