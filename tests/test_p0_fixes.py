import pytest
from fastapi import HTTPException

from libs.core.config_store import ConfigStore
from services.api.middleware import auth as auth_module
from services.api.routes import config as config_routes


def test_config_store_rejects_invalid_schema(tmp_path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.set("exchanges.test_exchange.schedule.hour", 99) is False
    assert store.get("exchanges.test_exchange") is None


def test_config_route_rejects_unknown_exchange():
    with pytest.raises(HTTPException) as exc_info:
        config_routes._validate_exchange("kraken")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_auth_timing_safe_comparison(monkeypatch):
    expected = "supersecret"
    monkeypatch.setenv("MASP_ADMIN_TOKEN", expected)

    seen = {}

    def fake_compare_digest(provided, stored):
        seen["args"] = (provided, stored)
        return True

    monkeypatch.setattr(auth_module.secrets, "compare_digest", fake_compare_digest)
    assert await auth_module.verify_admin_token("supersecret") is True
    assert seen["args"] == ("supersecret", expected)
