import os
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

os.environ["MASP_ADMIN_TOKEN"] = "test-token-12345"


def test_patch_position_size():
    """position_size_krw update."""
    with patch("services.api.routes.config.ConfigStore") as MockStore:
        mock_instance = MagicMock()
        mock_instance.get.return_value = {"enabled": True, "position_size_krw": 10000}
        mock_instance.update_exchange_atomic.return_value = True
        MockStore.return_value = mock_instance

        from services.api.main import app

        client = TestClient(app)
        resp = client.patch(
            "/api/v1/config/exchanges/upbit",
            json={"position_size_krw": 50000},
            headers={"X-MASP-ADMIN-TOKEN": "test-token-12345"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


def test_patch_invalid_field_rejected():
    """Invalid field should be rejected."""
    with patch("services.api.routes.config.ConfigStore") as MockStore:
        mock_instance = MagicMock()
        mock_instance.get.return_value = {"enabled": True}
        MockStore.return_value = mock_instance

        from services.api.main import app

        client = TestClient(app)
        resp = client.patch(
            "/api/v1/config/exchanges/upbit",
            json={"invalid_field": "value"},
            headers={"X-MASP-ADMIN-TOKEN": "test-token-12345"},
        )
        assert resp.status_code == 400


def test_unauthorized_without_token():
    """Missing token should be forbidden."""
    from services.api.main import app

    client = TestClient(app)
    resp = client.patch("/api/v1/config/exchanges/upbit", json={"enabled": True})
    assert resp.status_code == 403


def test_get_runtime_config():
    """Get runtime config."""
    with patch("services.api.routes.config.ConfigStore") as MockStore:
        mock_instance = MagicMock()
        mock_instance.get.return_value = {"schema_version": 1, "exchanges": {}}
        MockStore.return_value = mock_instance

        from services.api.main import app

        client = TestClient(app)
        resp = client.get(
            "/api/v1/config/runtime",
            headers={"X-MASP-ADMIN-TOKEN": "test-token-12345"},
        )
        assert resp.status_code == 200
