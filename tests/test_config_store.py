import json
from pathlib import Path

from libs.core.config_store import ConfigStore


def test_config_store_creates_default(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    data = store.get()
    assert path.exists()
    assert data["schema_version"] == 1
    assert isinstance(data["exchanges"], dict)
    assert data["updated_at"]


def test_config_store_get_set_dot_notation(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.get("exchanges.upbit.enabled") is None

    assert store.set("exchanges.upbit.enabled", False) is True
    assert store.get("exchanges.upbit.enabled") is False


def test_config_store_set_nested_value(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.set("exchanges.bithumb.position_size_krw", 12345) is True
    assert store.get("exchanges.bithumb.position_size_krw") == 12345


def test_config_store_updates_timestamp(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    first = store.get()["updated_at"]
    assert store.set("exchanges.upbit.enabled", True) is True
    second = store.get()["updated_at"]
    assert second != ""
    assert second >= first


def test_config_store_atomic_write_format(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.set("telegram.enabled", True) is True
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["telegram"]["enabled"] is True


def test_config_store_get_missing_key_returns_none(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.get("exchanges.unknown.enabled") is None


def test_config_store_overwrite_value(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(path))
    assert store.set("exchanges.upbit.enabled", True) is True
    assert store.set("exchanges.upbit.enabled", False) is True
    assert store.get("exchanges.upbit.enabled") is False


def test_config_store_handles_empty_file(tmp_path: Path):
    path = tmp_path / "runtime_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    store = ConfigStore(path=str(path))
    data = store.get()
    assert data["schema_version"] == 1
