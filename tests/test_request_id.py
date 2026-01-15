"""
Request ID propagation tests.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.api.main import app  # noqa: E402

client = TestClient(app)


def test_request_id_in_header():
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    assert len(request_id) == 8


def test_request_id_in_body():
    response = client.get("/api/v1/status")
    data = response.json()
    assert "request_id" in data
    assert len(data["request_id"]) == 8


def test_request_id_in_error_response():
    response = client.post("/api/v1/kill-switch", json={"confirm": False})
    assert response.status_code == 400
    data = response.json()
    assert "request_id" in data
    assert response.headers.get("X-Request-ID") == data["request_id"]


def test_unique_request_ids():
    first = client.get("/api/v1/status")
    second = client.get("/api/v1/status")
    assert first.headers.get("X-Request-ID") != second.headers.get("X-Request-ID")


def test_timestamp_in_response():
    response = client.get("/api/v1/status")
    data = response.json()
    assert "timestamp" in data
    parsed = datetime.fromisoformat(data["timestamp"])
    assert parsed is not None
