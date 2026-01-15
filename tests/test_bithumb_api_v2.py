"""
Unit tests for Bithumb Open API v2 client.
"""

import time
import uuid
from unittest.mock import MagicMock

import jwt
import pytest
import requests

from libs.adapters.bithumb_api_v2 import BithumbAPIV2


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_encode_query_with_list():
    params = {"markets": ["KRW-BTC", "KRW-ETH"]}
    query = BithumbAPIV2._encode_query(params)
    # urlencode converts [] to %5B%5D
    assert "markets%5B%5D=KRW-BTC" in query
    assert "markets%5B%5D=KRW-ETH" in query


def test_query_hash_sha512_length():
    query = "market=KRW-BTC"
    digest = BithumbAPIV2._make_query_hash(query)
    assert len(digest) == 128


def test_jwt_includes_query_hash(monkeypatch):
    monkeypatch.setattr(uuid, "uuid4", lambda: "test-uuid")
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)

    client = BithumbAPIV2("access", "secret", session=MagicMock())
    token = client._generate_jwt({"market": "KRW-BTC"})
    payload = jwt.decode(token, "secret", algorithms=["HS256"])

    assert payload["access_key"] == "access"
    assert payload["nonce"] == "test-uuid"
    assert payload["query_hash_alg"] == "SHA512"

    query = client._encode_query({"market": "KRW-BTC"})
    assert payload["query_hash"] == client._make_query_hash(query)


def test_jwt_includes_empty_query_hash_when_forced(monkeypatch):
    monkeypatch.setattr(uuid, "uuid4", lambda: "test-uuid")
    monkeypatch.setattr(time, "time", lambda: 1700000000.0)
    monkeypatch.setenv("BITHUMB_JWT_INCLUDE_EMPTY_QUERY_HASH", "1")

    client = BithumbAPIV2("access", "secret", session=MagicMock())
    token = client._generate_jwt()
    payload = jwt.decode(token, "secret", algorithms=["HS256"])

    assert payload["query_hash_alg"] == "SHA512"
    assert payload["query_hash"] == client._make_query_hash("")


def test_request_raises_on_api_error():
    session = MagicMock()
    session.get.return_value = DummyResponse({"error": {"message": "bad"}})

    client = BithumbAPIV2("access", "secret", session=session)

    with pytest.raises(ValueError, match="Bithumb API Error"):
        client._request("GET", "/v1/accounts")


def test_request_raises_on_http_error():
    session = MagicMock()
    # Mock response with status_code attribute
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"error": {"name": "jwt_verification", "message": "Invalid"}}
    
    def raise_on_status():
        err = requests.HTTPError("HTTP 401")
        err.response = mock_response
        raise err
    
    mock_response.raise_for_status = raise_on_status
    session.get.return_value = mock_response

    client = BithumbAPIV2("access", "secret", session=session)

    with pytest.raises(PermissionError, match="Bithumb HTTP 401"):
        client._request("GET", "/v1/accounts")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
