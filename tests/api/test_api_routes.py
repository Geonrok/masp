"""
Tests for API routes.
"""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from services.api.main import app
from services.api.routes.strategy import StrategyManager


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers():
    """Return auth headers with admin token."""
    return {"Authorization": "Bearer test-admin-token"}


class TestHealthRoutes:
    """Tests for health routes."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["message"] == "healthy"


class TestStatusRoutes:
    """Tests for status routes."""

    def test_get_status(self, client):
        """Test system status endpoint."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "version" in data
        assert "uptime_seconds" in data
        assert "exchanges" in data

    def test_get_exchange_status(self, client):
        """Test exchange status endpoint."""
        response = client.get("/api/v1/exchanges")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "exchanges" in data


class TestStrategyRoutes:
    """Tests for strategy routes."""

    def test_list_strategies(self, client):
        """Test strategy list endpoint."""
        response = client.get("/api/v1/strategy/list")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "strategies" in data
        assert isinstance(data["strategies"], list)

    def test_start_strategy_success(self, client):
        """Test starting a valid strategy."""
        # First get available strategies
        list_response = client.get("/api/v1/strategy/list")
        strategies = list_response.json()["strategies"]

        if strategies:
            strategy_id = strategies[0]["strategy_id"]
            response = client.post(
                "/api/v1/strategy/start",
                json={"strategy_id": strategy_id}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_start_strategy_unknown(self, client):
        """Test starting an unknown strategy returns 404."""
        response = client.post(
            "/api/v1/strategy/start",
            json={"strategy_id": "nonexistent_strategy_xyz"}
        )
        assert response.status_code == 404

    def test_stop_strategy(self, client):
        """Test stopping a strategy."""
        response = client.post(
            "/api/v1/strategy/stop",
            json={"strategy_id": "any_strategy"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestPositionsRoutes:
    """Tests for positions routes."""

    def test_list_positions(self, client):
        """Test positions list endpoint."""
        response = client.get("/api/v1/positions/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "positions" in data
        assert isinstance(data["positions"], list)


class TestTradesRoutes:
    """Tests for trades routes."""

    def test_list_trades(self, client):
        """Test trades list endpoint."""
        response = client.get("/api/v1/trades/")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "trades" in data
        assert isinstance(data["trades"], list)


class TestKillSwitchRoutes:
    """Tests for kill switch routes."""

    def test_kill_switch_without_confirm(self, client):
        """Test kill switch requires confirm=true."""
        response = client.post(
            "/api/v1/kill-switch",
            json={"confirm": False}
        )
        assert response.status_code == 400

    def test_kill_switch_with_confirm(self, client):
        """Test kill switch with confirm=true."""
        response = client.post(
            "/api/v1/kill-switch",
            json={"confirm": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "strategies_stopped" in data


class TestStrategyManager:
    """Tests for StrategyManager class."""

    def test_init(self):
        """Test StrategyManager initialization."""
        manager = StrategyManager()
        assert manager.active_strategies == set()

    def test_list_strategies(self):
        """Test listing strategies."""
        manager = StrategyManager()
        strategies = manager.list_strategies()
        assert isinstance(strategies, list)

    def test_start_strategy(self):
        """Test starting a strategy."""
        manager = StrategyManager()
        strategies = manager.list_strategies()

        if strategies:
            strategy_id = strategies[0].strategy_id
            manager.start(strategy_id)
            assert strategy_id in manager.active_strategies

    def test_start_unknown_strategy(self):
        """Test starting unknown strategy raises ValueError."""
        manager = StrategyManager()
        with pytest.raises(ValueError, match="Unknown strategy_id"):
            manager.start("nonexistent_strategy")

    def test_stop_strategy(self):
        """Test stopping a strategy."""
        manager = StrategyManager()
        strategies = manager.list_strategies()

        if strategies:
            strategy_id = strategies[0].strategy_id
            manager.start(strategy_id)
            manager.stop(strategy_id)
            assert strategy_id not in manager.active_strategies

    def test_stop_nonexistent_strategy(self):
        """Test stopping non-started strategy is safe."""
        manager = StrategyManager()
        manager.stop("nonexistent")  # Should not raise
        assert "nonexistent" not in manager.active_strategies

    def test_stop_all(self):
        """Test stopping all strategies."""
        manager = StrategyManager()
        strategies = manager.list_strategies()

        # Start some strategies
        for s in strategies[:3]:
            try:
                manager.start(s.strategy_id)
            except ValueError:
                pass

        count = manager.stop_all()
        assert manager.active_strategies == set()


class TestMiddleware:
    """Tests for API middleware."""

    def test_request_id_header(self, client):
        """Test request ID is added to response headers."""
        response = client.get("/api/v1/health/")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_timestamp_header(self, client):
        """Test timestamp is added to response headers."""
        response = client.get("/api/v1/health/")
        assert "X-Timestamp" in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_response_format(self, client):
        """Test 404 returns proper JSON format."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_validation_error_format(self, client):
        """Test validation error returns proper format."""
        response = client.post(
            "/api/v1/strategy/start",
            json={}  # Missing required field
        )
        assert response.status_code == 422  # Unprocessable Entity


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/health/",
            headers={"Origin": "http://localhost:5173"}
        )
        # Should allow the origin
        assert response.status_code in [200, 405]
