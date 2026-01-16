import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from libs.core.key_manager import KeyManager


def _set_master_key(monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("MASP_MASTER_KEY", key)
    monkeypatch.delenv("MASP_ALLOW_AUTOGEN_MASTER_KEY", raising=False)
    return key


def test_master_key_required(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("MASP_MASTER_KEY", raising=False)
    monkeypatch.delenv("MASP_ALLOW_AUTOGEN_MASTER_KEY", raising=False)
    with pytest.raises(RuntimeError):
        KeyManager(storage_path=str(tmp_path / "keys.json"))


def test_store_and_get_masked(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    km = KeyManager(storage_path=str(tmp_path / "keys.json"))
    assert km.store_key("upbit", "a64ed4b3zzzz", "secret123") is True
    keys = km.get_keys()
    assert keys["upbit"]["api_key"].startswith("a64ed4b3")
    assert keys["upbit"]["api_key"].endswith("...")
    assert keys["upbit"]["has_secret"] is True


def test_get_raw_key(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    km = KeyManager(storage_path=str(tmp_path / "keys.json"))
    km.store_key("bithumb", "api_foo", "secret_bar")
    raw = km.get_raw_key("bithumb")
    assert raw["api_key"] == "api_foo"
    assert raw["secret_key"] == "secret_bar"


def test_delete_key(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    km = KeyManager(storage_path=str(tmp_path / "keys.json"))
    km.store_key("binance", "api1", "sec1")
    assert km.delete_key("binance") is True
    assert km.delete_key("binance") is False


def test_invalid_exchange(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    km = KeyManager(storage_path=str(tmp_path / "keys.json"))
    with pytest.raises(ValueError):
        km.store_key("invalid", "k", "s")


def test_env_fallback(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    monkeypatch.setenv("UPBIT_API_KEY", "env_api_key")
    monkeypatch.setenv("UPBIT_SECRET_KEY", "env_secret_key")
    km = KeyManager(storage_path=str(tmp_path / "keys.json"))
    raw = km.get_key_with_env_fallback("upbit")
    assert raw["api_key"] == "env_api_key"
    assert raw["secret_key"] == "env_secret_key"


def test_decrypt_error(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    path = tmp_path / "keys.json"
    bad = {
        "schema_version": 1,
        "keys": {
            "upbit": {
                "api_key": "not-a-token",
                "secret_key": "not-a-token",
                "updated_at": "2026-01-16T00:00:00Z",
            }
        },
    }
    path.write_text(json.dumps(bad), encoding="utf-8")
    km = KeyManager(storage_path=str(path))
    keys = km.get_keys()
    assert keys["upbit"]["api_key"] == "[DECRYPT_ERROR]"
    assert keys["upbit"]["has_secret"] is False


def test_store_uses_encryption(monkeypatch, tmp_path: Path):
    _set_master_key(monkeypatch)
    path = tmp_path / "keys.json"
    km = KeyManager(storage_path=str(path))
    km.store_key("binance_futures", "api_plain", "secret_plain")
    raw = json.loads(path.read_text(encoding="utf-8"))
    enc = raw["keys"]["binance_futures"]["api_key"]
    assert enc != "api_plain"
