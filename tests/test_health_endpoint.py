"""
Health Endpoint Tests (Phase 4 Final)
9 Test Cases - All P0 Issues Resolved
"""

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.fixture
def mock_scheduler_status():
    return MagicMock(
        return_value={
            "running": True,
            "initialized": True,
            "active_exchanges": ["upbit"],
            "exchange_count": 1,
        }
    )


@pytest.fixture
async def health_app(mock_scheduler_status):
    from services.health_server import HealthServer

    server = HealthServer(
        scheduler_status_fn=mock_scheduler_status,
        port=19999,
        enable_metrics=False,
    )

    app = web.Application()
    app.router.add_get("/health", server._health_handler)
    app.router.add_get("/health/live", server._liveness_handler)
    app.router.add_get("/health/ready", server._readiness_handler)
    app.router.add_get("/metrics", server._metrics_handler)

    return app


@pytest.fixture
async def health_client(health_app):
    server = TestServer(health_app)
    client = TestClient(server)
    await client.start_server()
    yield client
    await client.close()


@pytest.mark.asyncio
async def test_health_always_200_healthy(health_client):
    """TC-H-001: /health returns 200 when healthy."""
    resp = await health_client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_always_200_unhealthy(health_client, mock_scheduler_status):
    """TC-H-002: /health returns 200 even when unhealthy (P0)."""
    mock_scheduler_status.return_value = {"running": False}
    resp = await health_client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_liveness_always_200(health_client):
    """TC-H-003: /health/live always 200."""
    resp = await health_client.get("/health/live")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_readiness_200_when_ready(health_client):
    """TC-H-004: /health/ready 200 when initialized."""
    resp = await health_client.get("/health/ready")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_readiness_503_when_not_ready(health_client, mock_scheduler_status):
    """TC-H-005: /health/ready 503 when not initialized."""
    mock_scheduler_status.return_value = {"running": True, "initialized": False}
    resp = await health_client.get("/health/ready")
    assert resp.status == 503


@pytest.mark.asyncio
async def test_metrics_501_when_disabled(health_client):
    """TC-H-006: /metrics 501 when disabled."""
    resp = await health_client.get("/metrics")
    assert resp.status == 501


def test_default_port():
    """TC-H-007: Default port 8080."""
    from services.health_server import HealthServer

    with patch.dict("os.environ", {}, clear=True):
        server = HealthServer(scheduler_status_fn=lambda: {})
        assert server._port == 8080


def test_env_port_override():
    """TC-H-008: MASP_HEALTH_PORT override."""
    from services.health_server import HealthServer

    with patch.dict("os.environ", {"MASP_HEALTH_PORT": "9090"}):
        server = HealthServer(scheduler_status_fn=lambda: {})
        assert server._port == 9090


def test_default_host_localhost():
    """TC-H-009: Default host 127.0.0.1."""
    from services.health_server import HealthServer

    with patch.dict("os.environ", {}, clear=True):
        server = HealthServer(scheduler_status_fn=lambda: {})
        assert server._host == "127.0.0.1"
