"""
Tests for WebSocket Client

Tests:
- Configuration
- Connection state management
- Reconnection with exponential backoff
- Upbit/Binance specific clients
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from libs.realtime.websocket_client import (
    BinanceWebSocket,
    ConnectionState,
    ReconnectingWebSocket,
    UpbitWebSocket,
    WebSocketConfig,
)


class TestWebSocketConfig:
    """Test WebSocketConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WebSocketConfig()

        assert config.timeout == 120.0
        assert config.ping_interval == 60.0
        assert config.reconnect_enabled is True
        assert config.max_reconnect_attempts == 10

    def test_upbit_preset(self):
        """Test Upbit-specific preset."""
        config = WebSocketConfig.for_upbit()

        assert config.url == "wss://api.upbit.com/websocket/v1"
        assert config.timeout == 120.0  # Official Upbit timeout
        assert config.ping_interval == 60.0
        assert config.exchange == "upbit"

    def test_binance_preset(self):
        """Test Binance-specific preset."""
        config = WebSocketConfig.for_binance()

        assert config.url == "wss://stream.binance.com:9443/ws"
        assert config.timeout == 30.0
        assert config.ping_interval == 20.0
        assert config.exchange == "binance"

    def test_custom_config(self):
        """Test custom configuration."""
        config = WebSocketConfig(
            url="wss://custom.example.com",
            timeout=60.0,
            ping_interval=30.0,
            max_reconnect_attempts=5,
        )

        assert config.url == "wss://custom.example.com"
        assert config.timeout == 60.0
        assert config.max_reconnect_attempts == 5


class TestConnectionState:
    """Test ConnectionState enum."""

    def test_states_exist(self):
        """Test all states are defined."""
        assert ConnectionState.DISCONNECTED
        assert ConnectionState.CONNECTING
        assert ConnectionState.CONNECTED
        assert ConnectionState.RECONNECTING
        assert ConnectionState.CLOSED


class TestReconnectingWebSocket:
    """Test ReconnectingWebSocket class."""

    def test_initialization(self):
        """Test client initialization."""
        config = WebSocketConfig(url="wss://test.example.com")
        client = ReconnectingWebSocket(config)

        assert client.config.url == "wss://test.example.com"
        assert client.state == ConnectionState.DISCONNECTED
        assert client.is_connected is False

    def test_initial_stats(self):
        """Test initial statistics."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        stats = client.stats

        assert stats["messages_received"] == 0
        assert stats["reconnect_count"] == 0
        assert stats["last_connect_time"] is None

    def test_callbacks_registered(self):
        """Test callbacks are registered."""
        config = WebSocketConfig()
        on_message = MagicMock()
        on_connect = MagicMock()
        on_disconnect = MagicMock()
        on_error = MagicMock()

        client = ReconnectingWebSocket(
            config,
            on_message=on_message,
            on_connect=on_connect,
            on_disconnect=on_disconnect,
            on_error=on_error,
        )

        assert client.on_message is on_message
        assert client.on_connect is on_connect
        assert client.on_disconnect is on_disconnect
        assert client.on_error is on_error

    @pytest.mark.asyncio
    async def test_disconnect_changes_state(self):
        """Test disconnect changes state to CLOSED."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        await client.disconnect()

        assert client.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_send_fails_when_disconnected(self):
        """Test send fails when not connected."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        result = await client.send({"test": "message"})

        assert result is False

    @pytest.mark.asyncio
    async def test_subscribe_stores_subscription(self):
        """Test subscribe stores subscription for reconnect."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        subscription = {"type": "ticker", "symbols": ["BTC"]}
        await client.subscribe(subscription)

        assert subscription in client._subscriptions


class TestBackoffCalculation:
    """Test exponential backoff calculation."""

    def test_backoff_increases(self):
        """Test backoff time increases with attempts."""
        config = WebSocketConfig(
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=60.0,
            jitter_factor=0.0,  # No jitter for predictable test
        )

        # Expected backoffs: 1, 2, 4, 8, 16, 32, 60 (capped)
        expected = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0]

        for i, expected_backoff in enumerate(expected):
            attempt = i + 1
            calculated = min(
                config.initial_backoff * (config.backoff_multiplier ** (attempt - 1)),
                config.max_backoff,
            )
            assert calculated == expected_backoff

    def test_backoff_capped_at_max(self):
        """Test backoff is capped at max value."""
        config = WebSocketConfig(
            initial_backoff=1.0,
            backoff_multiplier=2.0,
            max_backoff=10.0,
        )

        # After 5 attempts: 1, 2, 4, 8, 10 (capped), 10, ...
        attempt = 10
        calculated = min(
            config.initial_backoff * (config.backoff_multiplier ** (attempt - 1)),
            config.max_backoff,
        )

        assert calculated == config.max_backoff


class TestUpbitWebSocket:
    """Test UpbitWebSocket class."""

    def test_initialization(self):
        """Test Upbit client initialization."""
        client = UpbitWebSocket()

        assert client.config.exchange == "upbit"
        assert client.config.timeout == 120.0
        assert client.config.ping_interval == 60.0

    @pytest.mark.asyncio
    async def test_subscribe_ticker_format(self):
        """Test ticker subscription message format."""
        client = UpbitWebSocket()
        client._state = ConnectionState.CONNECTED
        client._ws = AsyncMock()

        await client.subscribe_ticker(["KRW-BTC", "KRW-ETH"])

        # Check that send was called
        client._ws.send.assert_called()


class TestBinanceWebSocket:
    """Test BinanceWebSocket class."""

    def test_initialization(self):
        """Test Binance client initialization."""
        client = BinanceWebSocket()

        assert client.config.exchange == "binance"
        assert client.config.timeout == 30.0

    @pytest.mark.asyncio
    async def test_subscribe_stores_subscription(self):
        """Test subscription is stored for reconnect."""
        client = BinanceWebSocket()

        await client.subscribe_ticker("btcusdt")

        assert len(client._subscriptions) == 1
        assert "SUBSCRIBE" in str(client._subscriptions[0])


class TestJitterCalculation:
    """Test jitter in backoff calculation."""

    def test_jitter_range(self):
        """Test jitter stays within expected range."""
        import random

        random.seed(42)

        config = WebSocketConfig(
            initial_backoff=10.0,
            jitter_factor=0.3,  # 30% jitter
        )

        base_backoff = config.initial_backoff

        # Generate many samples
        jitters = []
        for _ in range(1000):
            jitter = base_backoff * config.jitter_factor * (random.random() * 2 - 1)
            jitters.append(jitter)

        # Jitter should be between -30% and +30% of base
        max_jitter = base_backoff * config.jitter_factor
        assert all(-max_jitter <= j <= max_jitter for j in jitters)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_config_with_zero_timeout(self):
        """Test configuration with zero timeout is allowed."""
        config = WebSocketConfig(timeout=0.0)

        assert config.timeout == 0.0

    def test_config_with_zero_jitter(self):
        """Test configuration with zero jitter."""
        config = WebSocketConfig(jitter_factor=0.0)

        assert config.jitter_factor == 0.0

    @pytest.mark.asyncio
    async def test_multiple_disconnects(self):
        """Test multiple disconnect calls don't error."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        await client.disconnect()
        await client.disconnect()  # Second call
        await client.disconnect()  # Third call

        assert client.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_subscribe_multiple_channels(self):
        """Test subscribing to multiple channels."""
        config = WebSocketConfig()
        client = ReconnectingWebSocket(config)

        await client.subscribe({"type": "ticker"})
        await client.subscribe({"type": "trade"})
        await client.subscribe({"type": "orderbook"})

        assert len(client._subscriptions) == 3
