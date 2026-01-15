"""
Rate limit utilities tests (sync + async).
"""
from __future__ import annotations

import asyncio
import time

from libs.adapters.rate_limit import TokenBucket, AsyncTokenBucket, create_token_bucket


def _run(coro):
    return asyncio.run(coro)


def test_token_bucket_consume_reduces_tokens():
    bucket = TokenBucket(rate_per_sec=10, capacity=5)
    bucket._tokens = 5
    bucket._updated_at = time.monotonic()

    bucket.consume(2)

    assert bucket.available == 3


def test_token_bucket_set_tokens_clamps_high():
    bucket = TokenBucket(rate_per_sec=10, capacity=5)

    bucket.set_tokens(100)

    assert bucket.available == 5


def test_token_bucket_set_tokens_clamps_low():
    bucket = TokenBucket(rate_per_sec=10, capacity=5)

    bucket.set_tokens(-10)

    assert bucket.available == 0


def test_async_token_bucket_consume_reduces_tokens():
    async def _test():
        bucket = AsyncTokenBucket(rate_per_sec=10, capacity=5)
        bucket._tokens = 5
        await bucket.consume(2)
        remaining = await bucket.available()
        assert remaining == 3

    _run(_test())


def test_async_token_bucket_set_tokens_clamps():
    async def _test():
        bucket = AsyncTokenBucket(rate_per_sec=10, capacity=5)
        await bucket.set_tokens(100)
        assert await bucket.available() == 5
        await bucket.set_tokens(-10)
        assert await bucket.available() == 0

    _run(_test())


def test_async_token_bucket_lazy_initialization():
    async def _test():
        bucket = AsyncTokenBucket(rate_per_sec=10, capacity=5)
        assert bucket._initialized is False
        assert bucket._updated_at == 0.0
        await bucket.available()
        assert bucket._initialized is True
        assert bucket._updated_at > 0.0

    _run(_test())


def test_async_token_bucket_available_returns_int():
    async def _test():
        bucket = AsyncTokenBucket(rate_per_sec=10, capacity=5.7)
        result = await bucket.available()
        assert isinstance(result, int)
        assert result == 5

    _run(_test())


def test_create_token_bucket_sync():
    bucket = create_token_bucket(rate_per_sec=10, capacity=5, async_mode=False)

    assert isinstance(bucket, TokenBucket)


def test_create_token_bucket_async():
    bucket = create_token_bucket(rate_per_sec=10, capacity=5, async_mode=True)

    assert isinstance(bucket, AsyncTokenBucket)
