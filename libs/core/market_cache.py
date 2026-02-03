"""
Market data caching mechanism

Provides TTL-based caching to reduce API calls and protect against rate limits.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from libs.adapters.base import MarketQuote


@dataclass
class CachedQuote:
    """Cached quote with TTL"""

    quote: MarketQuote
    cached_at: float = field(default_factory=time.time)
    ttl: float = 5.0  # Default 5 seconds TTL

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.cached_at > self.ttl


class MarketCache:
    """
    Market data cache with TTL support.

    Reduces API calls by caching quotes for a configurable TTL.
    Protects against rate limits during high-frequency access.

    Usage:
        cache = MarketCache(default_ttl=5.0)

        # Check cache first
        quote = cache.get("BTC/KRW")
        if not quote:
            # Cache miss, fetch from API
            quote = adapter.get_quote("BTC/KRW")
            cache.set("BTC/KRW", quote)
    """

    def __init__(self, default_ttl: float = 5.0):
        """
        Initialize market cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 5.0)
        """
        self._cache: Dict[str, CachedQuote] = {}
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0

    def get(self, symbol: str) -> Optional[MarketQuote]:
        """
        Get cached quote if available and not expired.

        Args:
            symbol: Symbol to retrieve

        Returns:
            MarketQuote if cache hit and not expired, None otherwise
        """
        cached = self._cache.get(symbol)

        if cached and not cached.is_expired():
            self._hits += 1
            return cached.quote

        self._misses += 1

        # Clean up expired entry
        if cached:
            del self._cache[symbol]

        return None

    def set(self, symbol: str, quote: MarketQuote, ttl: Optional[float] = None):
        """
        Cache a quote with TTL.

        Args:
            symbol: Symbol to cache
            quote: MarketQuote to cache
            ttl: Time-to-live in seconds (default: use default_ttl)
        """
        self._cache[symbol] = CachedQuote(
            quote=quote,
            cached_at=time.time(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

    def invalidate(self, symbol: str):
        """
        Invalidate (delete) a cached quote.

        Args:
            symbol: Symbol to invalidate
        """
        if symbol in self._cache:
            del self._cache[symbol]

    def clear(self):
        """Clear all cached quotes."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0

        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2),
        }
