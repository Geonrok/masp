"""
Real-time Data Infrastructure

WebSocket client with:
- Exponential backoff reconnection
- Jitter for thundering herd prevention
- Upbit: 120s timeout, 60s ping interval (official docs)
"""

from libs.realtime.websocket_client import (
    ReconnectingWebSocket,
    WebSocketConfig,
    ConnectionState,
)

__all__ = [
    "ReconnectingWebSocket",
    "WebSocketConfig",
    "ConnectionState",
]
