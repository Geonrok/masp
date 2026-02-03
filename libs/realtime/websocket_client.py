"""
Reconnecting WebSocket Client

Features:
- Exponential backoff with jitter for reconnection
- Configurable timeout and ping intervals
- Upbit-specific settings (120s timeout, 60s ping)
- Async/await based

Reference: Upbit API docs, AWS best practices for reconnection
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSED = auto()


@dataclass
class WebSocketConfig:
    """WebSocket client configuration."""

    # Connection settings
    url: str = ""
    timeout: float = 120.0  # Upbit: 120s (official docs)
    ping_interval: float = 60.0  # Send ping every 60s

    # Reconnection settings
    reconnect_enabled: bool = True
    max_reconnect_attempts: int = 10
    initial_backoff: float = 1.0  # Start with 1 second
    max_backoff: float = 60.0  # Cap at 60 seconds
    backoff_multiplier: float = 2.0  # Double each attempt
    jitter_factor: float = 0.3  # 30% random jitter

    # Exchange-specific presets
    exchange: str = "generic"

    @classmethod
    def for_upbit(
        cls, url: str = "wss://api.upbit.com/websocket/v1"
    ) -> "WebSocketConfig":
        """Create Upbit-specific configuration."""
        return cls(
            url=url,
            timeout=120.0,  # Official Upbit timeout
            ping_interval=60.0,
            exchange="upbit",
        )

    @classmethod
    def for_binance(
        cls, url: str = "wss://stream.binance.com:9443/ws"
    ) -> "WebSocketConfig":
        """Create Binance-specific configuration."""
        return cls(
            url=url,
            timeout=30.0,  # Binance is more aggressive
            ping_interval=20.0,
            exchange="binance",
        )


class ReconnectingWebSocket:
    """
    WebSocket client with automatic reconnection.

    Features:
    - Exponential backoff with jitter
    - Configurable ping/pong keepalive
    - Message queue for offline buffering
    - State tracking and callbacks
    """

    def __init__(
        self,
        config: WebSocketConfig,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """
        Initialize WebSocket client.

        Args:
            config: WebSocket configuration
            on_message: Callback for received messages
            on_connect: Callback when connected
            on_disconnect: Callback when disconnected (with reason)
            on_error: Callback for errors
        """
        self.config = config
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error

        self._state = ConnectionState.DISCONNECTED
        self._ws = None
        self._reconnect_attempts = 0
        self._last_ping_time: float = 0
        self._last_pong_time: float = 0
        self._running = False
        self._message_queue: List[Dict[str, Any]] = []
        self._subscriptions: List[Dict[str, Any]] = []

        # Statistics
        self._stats = {
            "messages_received": 0,
            "reconnect_count": 0,
            "total_uptime": 0,
            "last_connect_time": None,
        }

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return self._stats.copy()

    async def connect(self) -> bool:
        """
        Connect to WebSocket server.

        Returns:
            bool: True if connected successfully
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("[WebSocket] Already connected")
            return True

        self._state = ConnectionState.CONNECTING
        self._running = True

        try:
            # Import websockets here to avoid import errors if not installed
            import websockets

            logger.info("[WebSocket] Connecting to %s", self.config.url)

            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    ping_interval=None,  # We handle ping ourselves
                    ping_timeout=None,
                    close_timeout=10,
                ),
                timeout=30.0,
            )

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._last_ping_time = time.time()
            self._last_pong_time = time.time()
            self._stats["last_connect_time"] = time.time()

            logger.info("[WebSocket] Connected successfully")

            if self.on_connect:
                self.on_connect()

            # Resubscribe if reconnecting
            if self._subscriptions:
                await self._resubscribe()

            return True

        except asyncio.TimeoutError:
            logger.error("[WebSocket] Connection timeout")
            self._state = ConnectionState.DISCONNECTED
            return False

        except Exception as e:
            logger.error("[WebSocket] Connection failed: %s", e)
            self._state = ConnectionState.DISCONNECTED
            if self.on_error:
                self.on_error(e)
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._running = False
        self._state = ConnectionState.CLOSED

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("[WebSocket] Error closing connection: %s", e)

        self._ws = None
        logger.info("[WebSocket] Disconnected")

        if self.on_disconnect:
            self.on_disconnect("manual_disconnect")

    async def send(self, message: Union[Dict[str, Any], str]) -> bool:
        """
        Send message to server.

        Args:
            message: Message to send (dict will be JSON-encoded)

        Returns:
            bool: True if sent successfully
        """
        if not self.is_connected:
            logger.warning("[WebSocket] Cannot send: not connected")
            return False

        try:
            if isinstance(message, dict):
                data = json.dumps(message)
            else:
                data = message

            await self._ws.send(data)
            return True

        except Exception as e:
            logger.error("[WebSocket] Send failed: %s", e)
            return False

    async def subscribe(self, subscription: Dict[str, Any]) -> bool:
        """
        Subscribe to a channel/topic.

        Args:
            subscription: Subscription message

        Returns:
            bool: True if subscribed successfully
        """
        self._subscriptions.append(subscription)

        if self.is_connected:
            return await self.send(subscription)

        return True  # Will subscribe on connect

    async def run(self) -> None:
        """
        Run the WebSocket client.

        This is the main loop that:
        - Receives messages
        - Handles ping/pong
        - Reconnects on disconnect
        """
        while self._running:
            if not self.is_connected:
                connected = await self.connect()
                if not connected:
                    await self._handle_reconnect()
                    continue

            try:
                # Set up tasks
                receive_task = asyncio.create_task(self._receive_loop())
                ping_task = asyncio.create_task(self._ping_loop())

                # Wait for either to complete (usually due to disconnect)
                done, pending = await asyncio.wait(
                    [receive_task, ping_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check for errors
                for task in done:
                    if task.exception():
                        raise task.exception()

            except Exception as e:
                logger.error("[WebSocket] Error in main loop: %s", e)
                if self.on_error:
                    self.on_error(e)

            # Handle disconnection
            if self._running and self.config.reconnect_enabled:
                await self._handle_reconnect()
            else:
                break

    async def _receive_loop(self) -> None:
        """Receive messages from server."""
        try:
            async for message in self._ws:
                self._last_pong_time = time.time()  # Any message counts as pong
                self._stats["messages_received"] += 1

                try:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")

                    data = json.loads(message)

                    if self.on_message:
                        self.on_message(data)

                except json.JSONDecodeError:
                    logger.debug("[WebSocket] Non-JSON message: %s", message[:100])

        except Exception as e:
            logger.error("[WebSocket] Receive error: %s", e)
            self._state = ConnectionState.DISCONNECTED
            raise

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        while self.is_connected and self._running:
            try:
                await asyncio.sleep(self.config.ping_interval)

                if not self.is_connected:
                    break

                # Send ping
                current_time = time.time()
                self._last_ping_time = current_time

                # Check if pong was received
                if current_time - self._last_pong_time > self.config.timeout:
                    logger.warning(
                        "[WebSocket] Pong timeout (%.1fs since last pong)",
                        current_time - self._last_pong_time,
                    )
                    self._state = ConnectionState.DISCONNECTED
                    break

                # Send ping frame or ping message depending on exchange
                if self.config.exchange == "upbit":
                    # Upbit expects PING message
                    await self.send("PING")
                else:
                    # Standard WebSocket ping
                    await self._ws.ping()

                logger.debug("[WebSocket] Ping sent")

            except Exception as e:
                logger.error("[WebSocket] Ping error: %s", e)
                self._state = ConnectionState.DISCONNECTED
                break

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff and jitter."""
        if not self.config.reconnect_enabled:
            return

        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(
                "[WebSocket] Max reconnect attempts (%d) reached",
                self.config.max_reconnect_attempts,
            )
            self._state = ConnectionState.CLOSED
            return

        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        self._stats["reconnect_count"] += 1

        # Calculate backoff with exponential increase
        backoff = min(
            self.config.initial_backoff
            * (self.config.backoff_multiplier ** (self._reconnect_attempts - 1)),
            self.config.max_backoff,
        )

        # Add jitter (random variation)
        jitter = backoff * self.config.jitter_factor * (random.random() * 2 - 1)
        wait_time = max(0.1, backoff + jitter)

        logger.info(
            "[WebSocket] Reconnecting in %.1fs (attempt %d/%d)",
            wait_time,
            self._reconnect_attempts,
            self.config.max_reconnect_attempts,
        )

        if self.on_disconnect:
            self.on_disconnect(f"reconnecting_attempt_{self._reconnect_attempts}")

        await asyncio.sleep(wait_time)

    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnection."""
        logger.info(
            "[WebSocket] Resubscribing to %d channels", len(self._subscriptions)
        )

        for sub in self._subscriptions:
            await self.send(sub)
            await asyncio.sleep(0.1)  # Small delay between subscriptions


class UpbitWebSocket(ReconnectingWebSocket):
    """
    Upbit-specific WebSocket client.

    Handles Upbit's specific message format and subscription model.
    """

    def __init__(
        self,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """Initialize Upbit WebSocket."""
        config = WebSocketConfig.for_upbit()
        super().__init__(config, on_message, on_connect, on_disconnect, on_error)

    async def subscribe_ticker(self, symbols: List[str]) -> bool:
        """
        Subscribe to ticker updates.

        Args:
            symbols: List of market codes (e.g., ["KRW-BTC", "KRW-ETH"])

        Returns:
            bool: True if subscribed
        """
        subscription = [
            {"ticket": f"ticker_{int(time.time())}"},
            {"type": "ticker", "codes": symbols},
        ]

        return await self.send(subscription)

    async def subscribe_orderbook(self, symbols: List[str]) -> bool:
        """
        Subscribe to orderbook updates.

        Args:
            symbols: List of market codes

        Returns:
            bool: True if subscribed
        """
        subscription = [
            {"ticket": f"orderbook_{int(time.time())}"},
            {"type": "orderbook", "codes": symbols},
        ]

        return await self.send(subscription)

    async def subscribe_trade(self, symbols: List[str]) -> bool:
        """
        Subscribe to trade updates.

        Args:
            symbols: List of market codes

        Returns:
            bool: True if subscribed
        """
        subscription = [
            {"ticket": f"trade_{int(time.time())}"},
            {"type": "trade", "codes": symbols},
        ]

        return await self.send(subscription)


class BinanceWebSocket(ReconnectingWebSocket):
    """
    Binance-specific WebSocket client.

    Handles Binance's stream format.
    """

    def __init__(
        self,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """Initialize Binance WebSocket."""
        config = WebSocketConfig.for_binance()
        super().__init__(config, on_message, on_connect, on_disconnect, on_error)

    async def subscribe_ticker(self, symbol: str) -> bool:
        """
        Subscribe to ticker updates.

        Args:
            symbol: Symbol in lowercase (e.g., "btcusdt")

        Returns:
            bool: True if subscribed
        """
        subscription = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@ticker"],
            "id": int(time.time()),
        }

        return await self.subscribe(subscription)

    async def subscribe_kline(self, symbol: str, interval: str = "1m") -> bool:
        """
        Subscribe to kline/candlestick updates.

        Args:
            symbol: Symbol in lowercase
            interval: Kline interval (1m, 5m, 1h, etc.)

        Returns:
            bool: True if subscribed
        """
        subscription = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@kline_{interval}"],
            "id": int(time.time()),
        }

        return await self.subscribe(subscription)
