from pathlib import Path

from libs.core.config_store import ConfigStore


def test_load_creates_backup_on_corruption(tmp_path: Path):
    config_path = tmp_path / "runtime_config.json"
    config_path.write_text("{invalid json", encoding="utf-8")

    store = ConfigStore(path=str(config_path))
    data = store.get()

    backup_files = list(tmp_path.glob("*.bad.*"))
    assert len(backup_files) >= 1
    assert data.get("schema_version") == 1


def test_set_strict_mode_rejects_non_dict_path(tmp_path: Path):
    config_path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(config_path))

    assert store.set("custom", "not_a_dict") is True

    result = store.set("custom.child", True, strict=True)
    assert result is False


def test_set_strict_rejects_new_path(tmp_path: Path):
    config_path = tmp_path / "runtime_config.json"
    store = ConfigStore(path=str(config_path))

    result = store.set("new_section.new_key", "value", strict=True)
    assert result is False
