"""
WebSocket real-time stream (stabilized).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    """WebSocket connection manager with safe broadcast."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info("[WS] Client connected. Total: %s", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info("[WS] Client disconnected. Total: %s", len(self.active_connections))

    async def broadcast(self, message: dict) -> None:
        """Broadcast using a snapshot to avoid iteration issues."""
        if not self.active_connections:
            return

        data = json.dumps(message, default=str)

        async with self._lock:
            connections = list(self.active_connections)

        disconnected = []

        for ws in connections:
            try:
                await asyncio.wait_for(ws.send_text(data), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[WS] Send timeout, disconnecting client")
                disconnected.append(ws)
            except Exception as exc:
                logger.warning("[WS] Send failed: %s", exc)
                disconnected.append(ws)

        for ws in disconnected:
            await self.disconnect(ws)

    async def send_price_update(
        self, symbol: str, price: float, change: float = 0
    ) -> None:
        await self.broadcast(
            {
                "type": "price",
                "data": {"symbol": symbol, "price": price, "change_24h": change},
                "ts": datetime.now().isoformat(),
            }
        )

    async def send_signal(
        self, symbol: str, signal: str, price: float, reason: str
    ) -> None:
        await self.broadcast(
            {
                "type": "signal",
                "data": {
                    "symbol": symbol,
                    "signal": signal,
                    "price": price,
                    "reason": reason,
                },
                "ts": datetime.now().isoformat(),
            }
        )


ws_manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with heartbeat support."""
    await ws_manager.connect(websocket)

    ping_interval = 30
    pong_timeout = 75

    async def heartbeat_sender():
        while True:
            await asyncio.sleep(ping_interval)
            try:
                await websocket.send_json(
                    {
                        "type": "ping",
                        "ts": datetime.now().isoformat(),
                    }
                )
            except Exception:
                break

    async def message_receiver():
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=pong_timeout,
                )
                message = json.loads(data)

                if message.get("type") == "pong":
                    continue
                if message.get("type") == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "ts": datetime.now().isoformat(),
                        }
                    )
            except asyncio.TimeoutError:
                logger.warning("[WS] Client timeout, disconnecting")
                break
            except json.JSONDecodeError:
                continue
            except WebSocketDisconnect:
                break
            except Exception as exc:
                logger.error("[WS] Receive error: %s", exc)
                break

    try:
        heartbeat_task = asyncio.create_task(heartbeat_sender())
        receiver_task = asyncio.create_task(message_receiver())

        done, pending = await asyncio.wait(
            [heartbeat_task, receiver_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

    except Exception as exc:
        logger.error("[WS] Connection error: %s", exc)
    finally:
        await ws_manager.disconnect(websocket)
