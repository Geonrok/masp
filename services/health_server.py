"""
MASP Health Check HTTP Server (Phase 4 Final)
All P0 Issues Resolved - 4/4 AI Approved
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Callable, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


class HealthServer:
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8080

    def __init__(
        self,
        scheduler_status_fn: Callable[[], dict],
        host: Optional[str] = None,
        port: Optional[int] = None,
        enable_metrics: bool = False,
        metrics_fn: Optional[Callable[[], bytes]] = None,
    ) -> None:
        self._scheduler_status_fn = scheduler_status_fn
        self._host = host or os.getenv("MASP_HEALTH_HOST", self.DEFAULT_HOST)
        self._port = port or self._get_port_from_env()
        self._enable_metrics = enable_metrics
        self._metrics_fn = metrics_fn
        self._strict_mode = os.getenv("MASP_HEALTH_STRICT", "0") == "1"

        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._start_time = time.time()

    def _get_port_from_env(self) -> int:
        try:
            return int(os.getenv("MASP_HEALTH_PORT", str(self.DEFAULT_PORT)))
        except (ValueError, TypeError):
            logger.warning(
                "[HealthServer] Invalid MASP_HEALTH_PORT, using %d",
                self.DEFAULT_PORT,
            )
            return self.DEFAULT_PORT

    def _sanitize_status(self, status: dict) -> dict:
        try:
            return json.loads(json.dumps(status, default=str))
        except Exception:
            return {"error": "status_serialization_failed"}

    async def _health_handler(self, request: web.Request) -> web.Response:
        """GET /health - Always 200 (P0 Resolved)."""
        try:
            raw_status = self._scheduler_status_fn()
            status = self._sanitize_status(raw_status)
            is_healthy = status.get("running", False)

            response = {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uptime_seconds": round(time.time() - self._start_time, 2),
                "scheduler": status,
            }
            return web.json_response(response, status=200)
        except Exception as exc:
            logger.exception("[HealthServer] Health check failed: %s", exc)
            return web.json_response(
                {"status": "error", "error": str(exc)},
                status=200,
            )

    async def _liveness_handler(self, request: web.Request) -> web.Response:
        """GET /health/live - Always 200."""
        return web.json_response({"status": "alive"}, status=200)

    async def _readiness_handler(self, request: web.Request) -> web.Response:
        """GET /health/ready - 200/503."""
        try:
            status = self._scheduler_status_fn()
            is_ready = status.get("running", False) and status.get("initialized", False)

            if is_ready:
                return web.json_response({"status": "ready"}, status=200)
            return web.json_response(
                {"status": "not_ready", "reason": "scheduler_not_initialized"},
                status=503,
            )
        except Exception as exc:
            return web.json_response(
                {"status": "not_ready", "error": str(exc)},
                status=503,
            )

    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """GET /metrics - 200/501 (P0 Resolved)."""
        if not self._enable_metrics or not self._metrics_fn:
            return web.Response(
                text="Prometheus metrics not enabled. Set MASP_ENABLE_METRICS=1",
                status=501,
                content_type="text/plain",
            )

        try:
            metrics_output = self._metrics_fn()
            return web.Response(
                body=metrics_output,
                content_type="text/plain; version=0.0.4; charset=utf-8",
            )
        except Exception as exc:
            logger.exception("[HealthServer] Metrics failed: %s", exc)
            return web.Response(
                text=f"Error: {exc}",
                status=500,
                content_type="text/plain",
            )

    def _is_port_available(self) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self._host, self._port))
            return True
        except OSError:
            return False

    async def start(self) -> bool:
        if not self._is_port_available():
            msg = f"[HealthServer] Port {self._port} unavailable"
            if self._strict_mode:
                raise RuntimeError(msg)
            logger.error(msg)
            return False

        try:
            self._app = web.Application()
            self._app.router.add_get("/health", self._health_handler)
            self._app.router.add_get("/health/live", self._liveness_handler)
            self._app.router.add_get("/health/ready", self._readiness_handler)
            self._app.router.add_get("/metrics", self._metrics_handler)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()

            site = web.TCPSite(self._runner, self._host, self._port)
            await site.start()

            logger.info("[HealthServer] Started on %s:%d", self._host, self._port)
            return True
        except Exception as exc:
            msg = f"[HealthServer] Start failed: {exc}"
            if self._strict_mode:
                raise RuntimeError(msg) from exc
            logger.exception(msg)
            await self.stop()
            return False

    async def stop(self) -> None:
        if self._runner:
            try:
                await self._runner.cleanup()
            except Exception:
                pass
        self._runner = None
        self._app = None
        logger.info("[HealthServer] Stopped")
