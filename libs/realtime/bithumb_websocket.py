"""
Bithumb WebSocket Client

Connects to Bithumb public WebSocket API for real-time ticker data.
URL: wss://pubwss.bithumb.com/pub/ws

Message formats:
    Subscribe: {"type":"ticker","symbols":["BTC_KRW"],"tickTypes":["24H"]}
    Ticker:    {"type":"ticker","content":{"symbol":"BTC_KRW","closePrice":"50000000",...}}
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from libs.realtime.websocket_client import ReconnectingWebSocket, WebSocketConfig

logger = logging.getLogger(__name__)


class BithumbWebSocket(ReconnectingWebSocket):
    """
    Bithumb-specific WebSocket client.

    Handles Bithumb's public WebSocket ticker/transaction feed.
    """

    def __init__(
        self,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        """Initialize Bithumb WebSocket."""
        config = WebSocketConfig(
            url="wss://pubwss.bithumb.com/pub/ws",
            timeout=120.0,
            ping_interval=60.0,
            exchange="bithumb",
        )
        super().__init__(config, on_message, on_connect, on_disconnect, on_error)

    async def subscribe_ticker(self, symbols: List[str]) -> bool:
        """
        Subscribe to ticker updates.

        Args:
            symbols: List of symbol pairs (e.g., ["BTC_KRW", "ETH_KRW"])

        Returns:
            bool: True if subscribed
        """
        subscription = {
            "type": "ticker",
            "symbols": symbols,
            "tickTypes": ["24H"],
        }

        return await self.send(subscription)

    async def subscribe_transaction(self, symbols: List[str]) -> bool:
        """
        Subscribe to real-time trade (transaction) updates.

        Args:
            symbols: List of symbol pairs (e.g., ["BTC_KRW"])

        Returns:
            bool: True if subscribed
        """
        subscription = {
            "type": "transaction",
            "symbols": symbols,
        }

        return await self.send(subscription)
