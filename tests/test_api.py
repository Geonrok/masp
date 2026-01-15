"""
API v1 tests for services.api.
Phase 3B-v3: verify POST update.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

api_keys_file = REPO_ROOT / "logs" / "api_keys_test.json"
api_keys_file.parent.mkdir(parents=True, exist_ok=True)
if api_keys_file.exists():
    api_keys_file.unlink()

os.environ["API_KEYS_FILE"] = str(api_keys_file)
os.environ["API_KEY_HMAC_SECRET"] = "test-secret"

from services.api.main import app  # noqa: E402

client = TestClient(app)

def _seed_api_key():
    return client.post(
        "/api/v1/settings/api-keys",
        json={
            "exchange": "upbit",
            "api_key": "abcdefghijklmnop",
            "secret_key": "1234567890abcdef",
        },
    )


def test_status():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "version" in data


def test_strategy_list():
    response = client.get("/api/v1/strategy/list")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "strategies" in data


def test_health():
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_positions_empty():
    response = client.get("/api/v1/positions/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["positions"] == []


def test_trades_empty():
    response = client.get("/api/v1/trades/")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["trades"] == []


def test_api_key_masking():
    response = _seed_api_key()
    assert response.status_code == 200
    data = response.json()

    assert data["masked_key"] == "abc***nop"
    assert "api_key" not in data

    response = client.get("/api/v1/settings/api-keys")
    assert response.status_code == 200
    data = response.json()
    for key in data["keys"]:
        assert "***" in key["masked_key"]


def test_api_key_verify():
    _seed_api_key()
    response = client.post(
        "/api/v1/settings/exchanges/verify",
        json={
            "exchange": "upbit",
            "api_key": "abcdefghijklmnop",
            "secret_key": "1234567890abcdef",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
