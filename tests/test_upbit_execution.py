"""
Upbit spot execution adapter tests.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import jwt
import pytest
import requests
import libs.adapters.real_upbit_spot as upbit_module

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from libs.adapters.real_upbit_spot import UpbitSpotExecution  # noqa: E402


@dataclass
class FakeResponse:
    json_data: Any
    headers: Dict[str, str] | None = None
    status_code: int = 200

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}
        self.text = str(self.json_data) if self.json_data is not None else ""

    def json(self) -> Any:
        if isinstance(self.json_data, Exception):
            raise self.json_data
        return self.json_data


class FakeSession:
    def __init__(self, handler: Callable[..., Any]):
        self._handler = handler
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs):
        call = {"method": method, "url": url, **kwargs}
        self.calls.append(call)
        result = self._handler(call)
        if isinstance(result, Exception):
            raise result
        return result


def _make_adapter(
    monkeypatch,
    session: FakeSession | None = None,
    access: str | None = "test-access",
    secret: str | None = "test-secret",
) -> UpbitSpotExecution:
    monkeypatch.setenv("MASP_ENABLE_LIVE_TRADING", "1")
    adapter = UpbitSpotExecution(access_key=access, secret_key=secret)
    if session is not None:
        adapter._session = session
    return adapter


def test_live_guard_blocks_order(monkeypatch):
    monkeypatch.delenv("MASP_ENABLE_LIVE_TRADING", raising=False)
    adapter = UpbitSpotExecution(access_key="key", secret_key="secret")
    with pytest.raises(RuntimeError):
        adapter.place_order("BTC/KRW", "BUY", 0.1, order_type="MARKET", price=10000)


def test_missing_credentials_raises(monkeypatch):
    monkeypatch.setenv("MASP_ENABLE_LIVE_TRADING", "1")
    monkeypatch.delenv("UPBIT_ACCESS_KEY", raising=False)
    monkeypatch.delenv("UPBIT_SECRET_KEY", raising=False)
    adapter = UpbitSpotExecution()
    with pytest.raises(ValueError):
        adapter.get_balance("KRW")


def test_invalid_side_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(ValueError):
        adapter.place_order("BTC/KRW", "HOLD", 10000, order_type="MARKET")


def test_invalid_order_type_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(ValueError):
        adapter.place_order("BTC/KRW", "BUY", 10000, order_type="STOP")


def test_market_buy_min_amount_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(ValueError):
        adapter.place_order("BTC/KRW", "BUY", 1000, order_type="MARKET")


def test_limit_order_without_price_raises(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(ValueError):
        adapter.place_order("BTC/KRW", "BUY", 0.01, order_type="LIMIT")


def test_invalid_jwt_algorithm_raises(monkeypatch):
    monkeypatch.setenv("UPBIT_JWT_ALG", "HS1024")
    with pytest.raises(ValueError):
        UpbitSpotExecution(access_key="test", secret_key="test")


def test_jwt_default_algorithm_hs512(monkeypatch):
    monkeypatch.delenv("UPBIT_JWT_ALG", raising=False)
    adapter = _make_adapter(monkeypatch)
    token = adapter._build_auth_header({"market": "KRW-BTC"}).split()[1]
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "HS512"


def test_jwt_algorithm_env_override(monkeypatch):
    monkeypatch.setenv("UPBIT_JWT_ALG", "HS256")
    adapter = _make_adapter(monkeypatch)
    token = adapter._build_auth_header({"market": "KRW-BTC"}).split()[1]
    header = jwt.get_unverified_header(token)
    assert header["alg"] == "HS256"


def test_jwt_payload_tracked(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._build_auth_header({"market": "KRW-BTC"})
    payload = adapter._last_jwt_payload
    assert payload is not None
    assert payload["access_key"] == adapter.access_key
    assert payload["query_hash_alg"] == "SHA512"
    assert "nonce" in payload
    assert "query_hash" in payload


def test_query_hash_generation(monkeypatch):
    monkeypatch.delenv("UPBIT_JWT_ALG", raising=False)
    adapter = _make_adapter(monkeypatch)
    params = {"markets": ["KRW-BTC", "KRW-ETH"], "side": "bid"}
    query = adapter._build_query_string(params)
    expected = hashlib.sha512(query.encode()).hexdigest()
    token = adapter._build_auth_header(params).split()[1]
    payload = jwt.decode(token, adapter.secret_key, algorithms=[adapter.jwt_alg])
    assert payload["query_hash"] == expected


def test_rate_limit_uses_order_bucket(monkeypatch):
    calls = {"order": 0, "default": 0}

    def handler(_):
        return FakeResponse(json_data={"uuid": "order-1"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._order_bucket.consume = lambda: calls.__setitem__(
        "order", calls["order"] + 1
    )
    adapter._default_bucket.consume = lambda: calls.__setitem__(
        "default", calls["default"] + 1
    )
    adapter.get_order_status("order-1")
    assert calls["order"] == 1
    assert calls["default"] == 0


def test_rate_limit_uses_default_bucket(monkeypatch):
    calls = {"order": 0, "default": 0}

    def handler(_):
        return FakeResponse(json_data=[])

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._order_bucket.consume = lambda: calls.__setitem__(
        "order", calls["order"] + 1
    )
    adapter._default_bucket.consume = lambda: calls.__setitem__(
        "default", calls["default"] + 1
    )
    adapter.get_all_balances()
    assert calls["default"] == 1
    assert calls["order"] == 0


def test_remaining_req_parsed(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    adapter._update_rate_limit_headers(
        {"Remaining-Req": "group=order; min=10; sec=2"},
        is_order=True,
    )
    assert adapter._rate_limit_info["remaining_req"]["group"] == "order"
    assert adapter._rate_limit_info["remaining_req"]["min"] == 10
    assert adapter._rate_limit_info["remaining_req"]["sec"] == 2


def test_rate_limit_standard_headers_parsed(monkeypatch):
    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-1"},
            headers={
                "X-RateLimit-Limit": "480",
                "X-RateLimit-Remaining": "235",
                "X-RateLimit-Reset": "1610386800",
                "Retry-After": "42",
                "Remaining-Req": "group=order; min=10; sec=5",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._request("GET", "/orders", {}, is_order=True)

    assert adapter._rate_limit_info["standard"]["limit"] == 480
    assert adapter._rate_limit_info["standard"]["remaining"] == 235
    assert adapter._rate_limit_info["standard"]["reset_at"] == 1610386800
    assert adapter._rate_limit_info["remaining_req"]["sec"] == 5
    assert adapter._rate_limit_info["remaining_req"]["group"] == "order"


def test_rate_limit_headers_invalid_values(monkeypatch):
    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-2"},
            headers={
                "X-RateLimit-Limit": "not-a-number",
                "X-RateLimit-Remaining": "235",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._request("GET", "/orders", {}, is_order=True)

    assert adapter._rate_limit_info["standard"] is None


def test_rate_limit_remaining_req_edge_cases(monkeypatch):
    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-3"},
            headers={
                "Remaining-Req": "group=market;min=590;sec=9",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._request("GET", "/markets", {}, is_order=False)

    assert adapter._rate_limit_info["remaining_req"]["sec"] == 9


def test_sync_token_bucket_called_when_flag_enabled(monkeypatch):
    monkeypatch.setenv("MASP_SYNC_TOKEN_BUCKET", "1")

    sync_called = {"count": 0}

    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-1"},
            headers={
                "Remaining-Req": "group=order; min=10; sec=5",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    original_sync = adapter._sync_token_bucket

    def spy_sync(is_order):
        sync_called["count"] += 1
        return original_sync(is_order)

    adapter._sync_token_bucket = spy_sync
    adapter._request("GET", "/orders", {}, is_order=True)

    assert sync_called["count"] == 1


def test_sync_token_bucket_skipped_when_flag_disabled(monkeypatch):
    monkeypatch.delenv("MASP_SYNC_TOKEN_BUCKET", raising=False)

    sync_called = {"count": 0}

    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-1"},
            headers={
                "Remaining-Req": "group=order; min=10; sec=5",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))

    def spy_sync(is_order):
        sync_called["count"] += 1

    adapter._sync_token_bucket = spy_sync
    adapter._request("GET", "/orders", {}, is_order=True)

    assert sync_called["count"] == 0


def test_remaining_req_without_min_field(monkeypatch):
    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-1"},
            headers={
                "Remaining-Req": "group=order; sec=5",
            },
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter._request("GET", "/orders", {}, is_order=True)

    assert adapter._rate_limit_info["remaining_req"]["sec"] == 5
    assert adapter._rate_limit_info["remaining_req"]["min"] is None


def test_token_bucket_set_tokens():
    from libs.adapters.rate_limit import TokenBucket

    bucket = TokenBucket(rate_per_sec=8, capacity=10)
    bucket._tokens = 2

    bucket.set_tokens(8)

    assert bucket.available == 8


def test_token_bucket_set_tokens_caps_at_capacity():
    from libs.adapters.rate_limit import TokenBucket

    bucket = TokenBucket(rate_per_sec=8, capacity=10)

    bucket.set_tokens(100)

    assert bucket.available == 10


def test_market_buy_order_params(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-1", "state": "wait"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    with pytest.raises(ValueError):
        adapter.place_order("BTC/KRW", "BUY", 0.1, order_type="MARKET", price=10000)

    adapter.place_order("BTC/KRW", "BUY", 10000, order_type="MARKET")
    payload = adapter._session.calls[0]["json"]
    assert payload["ord_type"] == "price"
    assert payload["price"] == "10000"


def test_market_sell_order_params(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-2", "state": "wait"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter.place_order("BTC/KRW", "SELL", 0.25, order_type="MARKET")
    payload = adapter._session.calls[0]["json"]
    assert payload["ord_type"] == "market"
    assert payload["volume"] == "0.25"


def test_limit_order_params(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-3", "state": "wait"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter.place_order("BTC/KRW", "BUY", 0.01, order_type="LIMIT", price=1000)
    payload = adapter._session.calls[0]["json"]
    assert payload["ord_type"] == "limit"
    assert payload["price"] == "1000"
    assert payload["volume"] == "0.01"


def test_identifier_uuid_attached(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-4", "state": "wait"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter.place_order("BTC/KRW", "BUY", 10000, order_type="MARKET")
    payload = adapter._session.calls[0]["json"]
    assert "identifier" in payload
    assert isinstance(payload["identifier"], str)


def test_timeout_recovery_uses_identifier(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    def fake_request(method, path, params, is_order):
        raise requests.exceptions.Timeout()

    adapter._request = fake_request  # type: ignore[assignment]
    adapter._get_order_by_identifier = lambda identifier: {"uuid": "recovered", "state": "done"}  # type: ignore[assignment]
    result = adapter.place_order("BTC/KRW", "BUY", 10000, order_type="MARKET")
    assert result.success is True
    assert result.order_id == "recovered"


def test_timeout_recovery_max_attempts_exceeded(monkeypatch):
    adapter = _make_adapter(monkeypatch)
    monkeypatch.setattr(upbit_module.time, "sleep", lambda _: None)

    attempt = {"count": 0}

    def fake_request(method, path, params, is_order):
        raise requests.exceptions.Timeout()

    def fake_lookup(identifier):
        attempt["count"] += 1
        return None

    adapter._request = fake_request  # type: ignore[assignment]
    adapter._get_order_by_identifier = fake_lookup  # type: ignore[assignment]

    result = adapter.place_order("BTC/KRW", "BUY", 10000, order_type="MARKET")
    assert result.success is False
    assert "Timeout" in (result.message or "")
    assert attempt["count"] == adapter.MAX_RECOVERY_ATTEMPTS


def test_get_order_status_calls_order_endpoint(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-5", "state": "wait"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    adapter.get_order_status("order-5")
    call = adapter._session.calls[0]
    assert call["method"] == "GET"
    assert call["params"]["uuid"] == "order-5"


def test_cancel_order_calls_delete(monkeypatch):
    def handler(call):
        return FakeResponse(json_data={"uuid": "order-6"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    ok = adapter.cancel_order("order-6")
    call = adapter._session.calls[0]
    assert ok is True
    assert call["method"] == "DELETE"
    assert call["params"]["uuid"] == "order-6"


def test_get_balance_returns_value(monkeypatch):
    def handler(call):
        return FakeResponse(json_data=[{"currency": "KRW", "balance": "1234.5"}])

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    assert adapter.get_balance("KRW") == 1234.5


def test_get_all_balances_returns_list(monkeypatch):
    balances = [{"currency": "BTC", "balance": "0.1"}]

    def handler(call):
        return FakeResponse(json_data=balances)

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter.get_all_balances()
    assert result == balances


def test_order_result_mapping_done_state(monkeypatch):
    def handler(call):
        return FakeResponse(
            json_data={"uuid": "order-7", "state": "done", "price": "1000"}
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter.place_order("BTC/KRW", "BUY", 0.01, order_type="LIMIT", price=1000)
    assert result.success is True
    assert result.status == "FILLED"


def test_rate_limit_429_backoff_retry_after(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(upbit_module.random, "uniform", lambda a, b: 1.0)

    attempt = {"count": 0}

    def handler(call):
        attempt["count"] += 1
        if attempt["count"] <= 2:
            return FakeResponse(
                json_data={"error": "rate_limited"},
                status_code=429,
                headers={"Retry-After": "1"},
            )
        return FakeResponse(json_data={"uuid": "order-1"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter._request("GET", "/orders", {}, is_order=True)

    assert result == {"uuid": "order-1"}
    assert attempt["count"] == 3
    assert sleep_calls == [1, 1]


def test_rate_limit_429_max_retries_exceeded(monkeypatch):
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: None)

    def handler(call):
        return FakeResponse(
            json_data={"error": "rate_limited"},
            status_code=429,
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))

    with pytest.raises(RuntimeError, match="Rate limited after 3 retries"):
        adapter._request("GET", "/orders", {}, is_order=True)


def test_rate_limit_429_decorrelated_backoff(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(upbit_module.random, "uniform", lambda a, b: a)

    attempt = {"count": 0}

    def handler(call):
        attempt["count"] += 1
        if attempt["count"] <= 3:
            return FakeResponse(
                json_data={"error": "rate_limited"},
                status_code=429,
                headers={},
            )
        return FakeResponse(json_data={"uuid": "order-1"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter._request("GET", "/orders", {}, is_order=True)

    assert result == {"uuid": "order-1"}
    assert sleep_calls == [1.0, 1.0, 1.0]


def test_rate_limit_429_jitter_applied(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(upbit_module.random, "uniform", lambda a, b: 1.05)

    attempt = {"count": 0}

    def handler(call):
        attempt["count"] += 1
        if attempt["count"] == 1:
            return FakeResponse(
                json_data={"error": "rate_limited"},
                status_code=429,
                headers={},
            )
        return FakeResponse(json_data={"uuid": "order-1"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter._request("GET", "/orders", {}, is_order=True)

    assert result == {"uuid": "order-1"}
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(1.05, rel=1e-6)


def test_rate_limit_429_decorrelated_jitter_propagation(monkeypatch):
    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))

    call_count = {"n": 0}

    def fake_uniform(a, b):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return 2.0
        return 4.0

    monkeypatch.setattr(upbit_module.random, "uniform", fake_uniform)

    attempt = {"count": 0}

    def handler(call):
        attempt["count"] += 1
        if attempt["count"] <= 2:
            return FakeResponse(
                json_data={"error": "rate_limited"},
                status_code=429,
                headers={},
            )
        return FakeResponse(json_data={"uuid": "order-1"})

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))
    result = adapter._request("GET", "/orders", {}, is_order=True)

    assert result == {"uuid": "order-1"}
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == pytest.approx(2.0, rel=1e-6)
    assert sleep_calls[1] == pytest.approx(4.0, rel=1e-6)


def test_rate_limit_418_circuit_breaker_fail_fast(monkeypatch):
    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0

    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(upbit_module.time, "time", lambda: 1000.0)

    handler_calls = {"count": 0}

    def handler(call):
        handler_calls["count"] += 1
        return FakeResponse(
            json_data={"error": "ip_banned"},
            status_code=418,
        )

    adapter = _make_adapter(monkeypatch, session=FakeSession(handler))

    with pytest.raises(RuntimeError, match="IP Ban"):
        adapter._request("GET", "/orders", {}, is_order=True)

    assert handler_calls["count"] == 1

    with pytest.raises(RuntimeError, match="Circuit breaker is open"):
        adapter._request("GET", "/orders", {}, is_order=True)

    assert handler_calls["count"] == 1
    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0


def test_market_data_429_while_loop_retry(monkeypatch):
    from libs.adapters.real_upbit_spot import UpbitSpotMarketData

    sleep_calls = []
    monkeypatch.setattr(upbit_module.time, "sleep", lambda s: sleep_calls.append(s))
    monkeypatch.setattr(upbit_module.random, "uniform", lambda a, b: 1.0)

    attempt = {"count": 0}

    class FakeMarketDataResponse:
        def __init__(self, status_code, json_data=None):
            self.status_code = status_code
            self._json_data = json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(response=self)

        def json(self):
            return self._json_data

    class FakeMarketDataSession:
        def get(self, url, params=None, timeout=None):
            attempt["count"] += 1
            if attempt["count"] <= 2:
                return FakeMarketDataResponse(429)
            return FakeMarketDataResponse(
                200,
                [
                    {
                        "trade_price": 50000000,
                        "acc_trade_volume_24h": 1000,
                        "timestamp": "1234567890",
                    }
                ],
            )

    adapter = UpbitSpotMarketData()
    adapter.session = FakeMarketDataSession()

    quote = adapter.get_quote("BTC/KRW")

    assert quote is not None
    assert quote.symbol == "BTC/KRW"
    assert attempt["count"] == 3
    assert len(sleep_calls) == 2
    assert sleep_calls == [1.0, 1.0]


def test_get_quotes_circuit_breaker_fail_fast(monkeypatch):
    from libs.adapters.real_upbit_spot import UpbitSpotMarketData

    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0
    monkeypatch.setattr(upbit_module.time, "time", lambda: 1000.0)
    upbit_module._open_circuit(60)

    adapter = UpbitSpotMarketData()

    result = adapter.get_quotes(["BTC/KRW", "ETH/KRW"])
    assert result == {}

    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0


def test_get_orderbook_circuit_breaker_fail_fast(monkeypatch):
    from libs.adapters.real_upbit_spot import UpbitSpotMarketData

    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0
    monkeypatch.setattr(upbit_module.time, "time", lambda: 1000.0)
    upbit_module._open_circuit(60)

    adapter = UpbitSpotMarketData()

    result = adapter.get_orderbook("BTC/KRW")
    assert result is None

    upbit_module._UPBIT_CIRCUIT_OPEN_UNTIL = 0.0
