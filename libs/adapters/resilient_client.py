"""
Resilient HTTP Client

Provides a standardized HTTP client with built-in resilience patterns
for use across all exchange adapters.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

import aiohttp
import requests

from libs.core.exceptions import (
    APIError,
    NetworkError,
    RateLimitError,
)
from libs.core.metrics import MetricsRegistry, get_metrics
from libs.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Tracks rate limit state for an endpoint."""

    limited_until: Optional[datetime] = None
    consecutive_429s: int = 0
    last_request_time: Optional[datetime] = None
    requests_per_window: int = 0
    window_start: Optional[datetime] = None
    window_size_seconds: int = 60

    def is_limited(self) -> bool:
        """Check if currently rate limited."""
        if self.limited_until is None:
            return False
        return datetime.now() < self.limited_until

    def update_window(self) -> None:
        """Update the request window tracking."""
        now = datetime.now()
        if (
            self.window_start is None
            or (now - self.window_start).total_seconds() > self.window_size_seconds
        ):
            self.window_start = now
            self.requests_per_window = 0
        self.requests_per_window += 1
        self.last_request_time = now

    def set_limited(self, retry_after: int) -> None:
        """Set rate limited state."""
        self.limited_until = datetime.now() + timedelta(seconds=retry_after)
        self.consecutive_429s += 1

    def clear_limit(self) -> None:
        """Clear rate limit state on successful request."""
        self.consecutive_429s = 0


@dataclass
class ClientConfig:
    """Configuration for resilient client."""

    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_backoff: bool = True
    jitter: bool = True

    # Timeout settings
    connect_timeout: float = 10.0
    read_timeout: float = 30.0

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_timeout: float = 30.0
    circuit_success_threshold: int = 2

    # Rate limit settings
    rate_limit_window: int = 60
    max_requests_per_window: Optional[int] = None

    # Retry conditions
    retryable_status_codes: Set[int] = field(
        default_factory=lambda: {408, 429, 500, 502, 503, 504}
    )
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
            TimeoutError,
        )
    )


class ResilientHTTPClient:
    """
    HTTP client with built-in resilience patterns.

    Features:
    - Automatic retries with exponential backoff
    - Circuit breaker protection
    - Rate limit handling
    - Request/response metrics
    - Timeout management

    Example:
        config = ClientConfig(max_retries=3)
        client = ResilientHTTPClient("binance", config)

        async with client:
            response = await client.get("/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
    """

    def __init__(
        self,
        name: str,
        config: Optional[ClientConfig] = None,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize resilient client.

        Args:
            name: Client name (used for circuit breaker and metrics)
            config: Client configuration
            base_url: Base URL for all requests
            headers: Default headers for all requests
        """
        self.name = name
        self.config = config or ClientConfig()
        self.base_url = base_url.rstrip("/")
        self.default_headers = headers or {}

        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_state: Dict[str, RateLimitState] = {}

        # Get or create circuit breaker
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            f"http_client_{name}",
            CircuitBreakerConfig(
                failure_threshold=self.config.circuit_failure_threshold,
                timeout=self.config.circuit_timeout,
                success_threshold=self.config.circuit_success_threshold,
            ),
        )

        # Metrics
        self._metrics = get_metrics()

    async def __aenter__(self) -> "ResilientHTTPClient":
        """Enter async context."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.config.connect_timeout,
                total=self.config.read_timeout,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.default_headers,
            )

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_rate_limit_state(self, endpoint: str) -> RateLimitState:
        """Get rate limit state for endpoint."""
        if endpoint not in self._rate_limit_state:
            self._rate_limit_state[endpoint] = RateLimitState(
                window_size_seconds=self.config.rate_limit_window
            )
        return self._rate_limit_state[endpoint]

    async def _wait_for_rate_limit(self, endpoint: str) -> None:
        """Wait if rate limited."""
        state = self._get_rate_limit_state(endpoint)

        if state.is_limited():
            wait_time = (state.limited_until - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(
                    f"[{self.name}] Rate limited, waiting {wait_time:.1f}s for {endpoint}"
                )
                await asyncio.sleep(wait_time)

        # Check max requests per window
        if self.config.max_requests_per_window:
            state.update_window()
            if state.requests_per_window > self.config.max_requests_per_window:
                # Proactive rate limiting
                wait_time = (
                    state.window_size_seconds
                    - (datetime.now() - state.window_start).total_seconds()
                )
                if wait_time > 0:
                    logger.info(
                        f"[{self.name}] Proactive rate limiting, waiting {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        import random

        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2**attempt)
        else:
            delay = self.config.base_delay

        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            delay *= 0.5 + random.random()

        return delay

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with resilience patterns.

        Args:
            method: HTTP method
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            json: JSON body
            data: Form data
            headers: Additional headers
            timeout: Override default timeout

        Returns:
            Response data as dictionary

        Raises:
            NetworkError: Network-related errors
            RateLimitError: Rate limit exceeded
            APIError: API returned error response
        """
        await self._ensure_session()

        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            raise NetworkError(
                f"Circuit breaker open for {self.name}",
                error_code="CIRCUIT_OPEN",
            )

        # Wait for rate limit
        await self._wait_for_rate_limit(endpoint)

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        request_headers = {**self.default_headers, **(headers or {})}

        last_exception = None
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries + 1):
            try:
                # Record metric
                self._metrics.increment(
                    "http_requests_total",
                    labels={
                        "client": self.name,
                        "method": method,
                        "endpoint": endpoint,
                    },
                )

                async with self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    data=data,
                    headers=request_headers,
                    timeout=aiohttp.ClientTimeout(total=timeout) if timeout else None,
                ) as response:
                    # Record latency
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self._metrics.record_timing(
                        "http_request_duration_ms",
                        duration_ms,
                        labels={"client": self.name, "endpoint": endpoint},
                    )

                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        state = self._get_rate_limit_state(endpoint)
                        state.set_limited(retry_after)

                        self._metrics.increment(
                            "http_rate_limits",
                            labels={"client": self.name, "endpoint": endpoint},
                        )

                        if attempt < self.config.max_retries:
                            logger.warning(
                                f"[{self.name}] Rate limited (429), retrying after {retry_after}s"
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        raise RateLimitError(
                            f"Rate limit exceeded for {endpoint}",
                            retry_after=retry_after,
                        )

                    # Handle retryable status codes
                    if response.status in self.config.retryable_status_codes:
                        if attempt < self.config.max_retries:
                            delay = self._calculate_delay(attempt)
                            logger.warning(
                                f"[{self.name}] Status {response.status}, retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue

                    # Record status
                    self._metrics.increment(
                        "http_responses_total",
                        labels={
                            "client": self.name,
                            "status": str(response.status),
                        },
                    )

                    # Handle client/server errors
                    if response.status >= 400:
                        body = await response.text()
                        self._circuit_breaker.record_failure()

                        raise APIError(
                            f"API error: {response.status}",
                            status_code=response.status,
                            response_body=body,
                        )

                    # Success
                    self._circuit_breaker.record_success()
                    state = self._get_rate_limit_state(endpoint)
                    state.clear_limit()

                    # Parse response
                    content_type = response.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        return await response.json()
                    return {"_raw": await response.text()}

            except self.config.retryable_exceptions as e:
                last_exception = e
                self._circuit_breaker.record_failure()

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"[{self.name}] Request failed ({type(e).__name__}), "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue

        # All retries exhausted
        self._metrics.increment(
            "http_failures_total",
            labels={"client": self.name, "endpoint": endpoint},
        )

        raise NetworkError(
            f"Request failed after {self.config.max_retries} retries: {last_exception}",
            cause=last_exception,
        )

    # Convenience methods

    async def get(
        self,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request("GET", endpoint, **kwargs)

    async def post(
        self,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request("POST", endpoint, **kwargs)

    async def put(
        self,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make PUT request."""
        return await self.request("PUT", endpoint, **kwargs)

    async def delete(
        self,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make DELETE request."""
        return await self.request("DELETE", endpoint, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "name": self.name,
            "circuit_breaker": self._circuit_breaker.get_stats(),
            "rate_limits": {
                endpoint: {
                    "is_limited": state.is_limited(),
                    "consecutive_429s": state.consecutive_429s,
                    "requests_in_window": state.requests_per_window,
                }
                for endpoint, state in self._rate_limit_state.items()
            },
        }


class SyncResilientHTTPClient:
    """
    Synchronous version of resilient HTTP client.

    Uses requests library for synchronous operations.
    """

    def __init__(
        self,
        name: str,
        config: Optional[ClientConfig] = None,
        base_url: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.config = config or ClientConfig()
        self.base_url = base_url.rstrip("/")
        self.default_headers = headers or {}

        self._session = requests.Session()
        self._session.headers.update(self.default_headers)

        self._rate_limit_state: Dict[str, RateLimitState] = {}
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            f"http_client_sync_{name}",
            CircuitBreakerConfig(
                failure_threshold=self.config.circuit_failure_threshold,
                timeout=self.config.circuit_timeout,
                success_threshold=self.config.circuit_success_threshold,
            ),
        )
        self._metrics = get_metrics()

    def close(self) -> None:
        """Close the session."""
        self._session.close()

    def __enter__(self) -> "SyncResilientHTTPClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _get_rate_limit_state(self, endpoint: str) -> RateLimitState:
        if endpoint not in self._rate_limit_state:
            self._rate_limit_state[endpoint] = RateLimitState(
                window_size_seconds=self.config.rate_limit_window
            )
        return self._rate_limit_state[endpoint]

    def _wait_for_rate_limit(self, endpoint: str) -> None:
        state = self._get_rate_limit_state(endpoint)
        if state.is_limited():
            wait_time = (state.limited_until - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"[{self.name}] Rate limited, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

    def _calculate_delay(self, attempt: int) -> float:
        import random

        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2**attempt)
        else:
            delay = self.config.base_delay

        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            delay *= 0.5 + random.random()

        return delay

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make synchronous HTTP request with resilience."""
        if not self._circuit_breaker.allow_request():
            raise NetworkError(
                f"Circuit breaker open for {self.name}",
                error_code="CIRCUIT_OPEN",
            )

        self._wait_for_rate_limit(endpoint)

        url = f"{self.base_url}{endpoint}" if self.base_url else endpoint
        request_headers = {**self.default_headers, **(headers or {})}
        timeout = timeout or self.config.read_timeout

        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                self._metrics.increment(
                    "http_requests_total",
                    labels={"client": self.name, "method": method},
                )

                response = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    data=data,
                    headers=request_headers,
                    timeout=timeout,
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    state = self._get_rate_limit_state(endpoint)
                    state.set_limited(retry_after)

                    if attempt < self.config.max_retries:
                        time.sleep(retry_after)
                        continue

                    raise RateLimitError(
                        f"Rate limit exceeded for {endpoint}",
                        retry_after=retry_after,
                    )

                if response.status_code in self.config.retryable_status_codes:
                    if attempt < self.config.max_retries:
                        delay = self._calculate_delay(attempt)
                        time.sleep(delay)
                        continue

                if response.status_code >= 400:
                    self._circuit_breaker.record_failure()
                    raise APIError(
                        f"API error: {response.status_code}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                self._circuit_breaker.record_success()
                state = self._get_rate_limit_state(endpoint)
                state.clear_limit()

                if "application/json" in response.headers.get("Content-Type", ""):
                    return response.json()
                return {"_raw": response.text}

            except (requests.RequestException, ConnectionError, TimeoutError) as e:
                last_exception = e
                self._circuit_breaker.record_failure()

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
                    continue

        raise NetworkError(
            f"Request failed after {self.config.max_retries} retries: {last_exception}",
            cause=last_exception,
        )

    def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request("PUT", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self.request("DELETE", endpoint, **kwargs)
