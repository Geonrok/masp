"""
Tests for resilient HTTP client.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from libs.adapters.resilient_client import (
    ResilientHTTPClient,
    SyncResilientHTTPClient,
    ClientConfig,
    RateLimitState,
)
from libs.core.exceptions import NetworkError, RateLimitError, APIError
from libs.core.metrics import MetricsRegistry
from libs.core.resilience import CircuitBreakerRegistry


class TestRateLimitState:
    """Tests for RateLimitState."""

    def test_initial_state(self):
        """Test initial state is not limited."""
        state = RateLimitState()
        assert state.is_limited() is False

    def test_set_limited(self):
        """Test setting rate limit."""
        state = RateLimitState()
        state.set_limited(60)

        assert state.is_limited() is True
        assert state.consecutive_429s == 1

    def test_clear_limit(self):
        """Test clearing rate limit."""
        state = RateLimitState()
        state.set_limited(1)
        state.consecutive_429s = 5
        state.clear_limit()

        assert state.consecutive_429s == 0

    def test_update_window(self):
        """Test request window tracking."""
        state = RateLimitState(window_size_seconds=60)
        state.update_window()

        assert state.requests_per_window == 1
        assert state.window_start is not None

        state.update_window()
        assert state.requests_per_window == 2


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClientConfig()

        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.exponential_backoff is True
        assert 429 in config.retryable_status_codes
        assert 500 in config.retryable_status_codes

    def test_custom_values(self):
        """Test custom configuration."""
        config = ClientConfig(
            max_retries=5,
            base_delay=0.5,
            circuit_failure_threshold=10,
        )

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.circuit_failure_threshold == 10


class TestResilientHTTPClient:
    """Tests for async ResilientHTTPClient."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons before each test."""
        MetricsRegistry.reset()
        CircuitBreakerRegistry._breakers.clear()
        yield
        MetricsRegistry.reset()
        CircuitBreakerRegistry._breakers.clear()

    @pytest.fixture
    def client(self):
        """Create test client."""
        config = ClientConfig(
            max_retries=2,
            base_delay=0.01,
            circuit_failure_threshold=3,
        )
        return ResilientHTTPClient("test", config, base_url="https://api.example.com")

    @pytest.mark.asyncio
    async def test_successful_request(self, client):
        """Test successful GET request."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"result": "success"})

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            async with client:
                result = await client.get("/test")

            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_retry_on_500(self, client):
        """Test retry on 500 status code."""
        mock_fail = AsyncMock()
        mock_fail.status = 500
        mock_fail.headers = {}

        mock_success = AsyncMock()
        mock_success.status = 200
        mock_success.headers = {"Content-Type": "application/json"}
        mock_success.json = AsyncMock(return_value={"result": "success"})

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.side_effect = [mock_fail, mock_success]

            async with client:
                result = await client.get("/test")

            assert result == {"result": "success"}
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, client):
        """Test 429 rate limit handling."""
        mock_429 = AsyncMock()
        mock_429.status = 429
        mock_429.headers = {"Retry-After": "1"}

        mock_success = AsyncMock()
        mock_success.status = 200
        mock_success.headers = {"Content-Type": "application/json"}
        mock_success.json = AsyncMock(return_value={"result": "success"})

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.side_effect = [mock_429, mock_success]

            async with client:
                result = await client.get("/test")

            assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, client):
        """Test circuit breaker opens after failures."""
        mock_error = AsyncMock()
        mock_error.status = 500
        mock_error.headers = {}
        mock_error.text = AsyncMock(return_value="Server Error")

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_error

            async with client:
                # Make multiple failing requests to trigger circuit breaker
                for _ in range(3):
                    with pytest.raises(APIError):
                        await client.get("/test")

                # Circuit should now be open
                with pytest.raises(NetworkError) as exc_info:
                    await client.get("/test")

                assert "Circuit breaker" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_api_error_raised(self, client):
        """Test APIError is raised for 4xx responses."""
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.headers = {}
        mock_response.text = AsyncMock(return_value='{"error": "Bad Request"}')

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            async with client:
                with pytest.raises(APIError) as exc_info:
                    await client.get("/test")

                assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_get_stats(self, client):
        """Test getting client statistics."""
        async with client:
            stats = client.get_stats()

        assert stats["name"] == "test"
        assert "circuit_breaker" in stats
        assert "rate_limits" in stats

    @pytest.mark.asyncio
    async def test_convenience_methods(self, client):
        """Test convenience methods (post, put, delete)."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = AsyncMock(return_value={"result": "ok"})

        with patch("aiohttp.ClientSession.request") as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            async with client:
                await client.post("/test", json={"data": "test"})
                await client.put("/test", json={"data": "test"})
                await client.delete("/test")

            assert mock_request.call_count == 3


class TestSyncResilientHTTPClient:
    """Tests for synchronous ResilientHTTPClient."""

    @pytest.fixture(autouse=True)
    def reset_singletons(self):
        """Reset singletons before each test."""
        MetricsRegistry.reset()
        CircuitBreakerRegistry._breakers.clear()
        yield
        MetricsRegistry.reset()
        CircuitBreakerRegistry._breakers.clear()

    @pytest.fixture
    def sync_client(self):
        """Create test sync client."""
        config = ClientConfig(
            max_retries=2,
            base_delay=0.01,
            circuit_failure_threshold=3,
        )
        return SyncResilientHTTPClient(
            "test_sync", config, base_url="https://api.example.com"
        )

    def test_successful_request(self, sync_client):
        """Test successful sync request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"result": "success"}

        with patch.object(sync_client._session, "request", return_value=mock_response):
            with sync_client:
                result = sync_client.get("/test")

            assert result == {"result": "success"}

    def test_retry_on_error(self, sync_client):
        """Test retry on connection error."""
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500

        mock_success_response = MagicMock()
        mock_success_response.status_code = 200
        mock_success_response.headers = {"Content-Type": "application/json"}
        mock_success_response.json.return_value = {"result": "success"}

        with patch.object(
            sync_client._session,
            "request",
            side_effect=[mock_error_response, mock_success_response],
        ):
            with sync_client:
                result = sync_client.get("/test")

            assert result == {"result": "success"}

    def test_api_error(self, sync_client):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.headers = {}

        with patch.object(sync_client._session, "request", return_value=mock_response):
            with sync_client:
                with pytest.raises(APIError) as exc_info:
                    sync_client.get("/test")

                assert exc_info.value.status_code == 404


class TestDelayCalculation:
    """Tests for delay calculation."""

    @pytest.fixture
    def client(self):
        config = ClientConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
            jitter=False,
        )
        return ResilientHTTPClient("test", config)

    def test_exponential_backoff(self, client):
        """Test exponential backoff calculation."""
        assert client._calculate_delay(0) == 1.0
        assert client._calculate_delay(1) == 2.0
        assert client._calculate_delay(2) == 4.0

    def test_max_delay_cap(self, client):
        """Test delay is capped at max_delay."""
        delay = client._calculate_delay(10)
        assert delay == 10.0  # Capped

    def test_linear_delay(self):
        """Test linear delay (no exponential)."""
        config = ClientConfig(
            base_delay=1.0,
            exponential_backoff=False,
            jitter=False,
        )
        client = ResilientHTTPClient("test", config)

        assert client._calculate_delay(0) == 1.0
        assert client._calculate_delay(1) == 1.0
        assert client._calculate_delay(5) == 1.0

    def test_jitter(self):
        """Test jitter adds randomness."""
        config = ClientConfig(
            base_delay=1.0,
            exponential_backoff=False,
            jitter=True,
        )
        client = ResilientHTTPClient("test", config)

        delays = [client._calculate_delay(0) for _ in range(10)]

        # With jitter, delays should vary
        assert len(set(delays)) > 1
        # All delays should be between 0.5 and 1.5
        assert all(0.5 <= d <= 1.5 for d in delays)
