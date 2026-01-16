"""
Runtime config store with atomic writes and file locking.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ScheduleConfig(BaseModel):
    """Schedule settings."""

    hour: int = Field(default=9, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    timezone: str = "Asia/Seoul"
    jitter: int = Field(default=30, ge=0, le=300)


class ExchangeConfig(BaseModel):
    """Exchange config with safe defaults."""

    enabled: bool = True
    strategy: str = "KAMA-TSMOM-Gate"
    symbols: List[str] = Field(default_factory=lambda: ["BTC/KRW"])
    position_size_krw: int = 10000
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)


class RuntimeConfig(BaseModel):
    """Full runtime config."""

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    telegram: Optional[dict] = None
    updated_at: str = ""


class ConfigStore:
    """
    Runtime config store.
    - file lock for concurrent access
    - atomic write for integrity
    - dot-notation access
    """

    def __init__(self, path: str = "storage/runtime_config.json"):
        self._path = Path(path)
        self._lock = FileLock(f"{path}.lock")
        self._ensure_storage_dir()
        self._ensure_file()

    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _default_data(self) -> Dict[str, Any]:
        data = RuntimeConfig().model_dump()
        data["updated_at"] = datetime.now().isoformat()
        return data

    def _ensure_file(self) -> None:
        if self._path.exists() and self._path.stat().st_size > 0:
            return
        self._atomic_write(self._default_data())

    def _atomic_write(self, data: dict) -> bool:
        """Atomic write with temp replace."""
        temp_path = self._path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            temp_path.replace(self._path)
            return True
        except Exception as exc:
            logger.error("[ConfigStore] Atomic write failed: %s", exc)
            if temp_path.exists():
                temp_path.unlink()
            return False

    def _load(self) -> Dict[str, Any]:
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                validated = RuntimeConfig.model_validate(data)
                return validated.model_dump()
        except Exception as exc:
            logger.warning("[ConfigStore] Load failed: %s", exc)
            try:
                cfg_path = getattr(self, "_path", None)
                if cfg_path and cfg_path.exists():
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = cfg_path.with_name(cfg_path.name + f".bad.{ts}")
                    shutil.copy2(cfg_path, backup_path)
                    logger.error(
                        "[ConfigStore] Corrupted config backed up to: %s", backup_path
                    )
            except Exception as backup_exc:
                logger.error("[ConfigStore] Backup failed: %s", backup_exc)
        return self._default_data()

    def get(self, key: str | None = None) -> Any:
        """Get config by dot notation."""
        with self._lock:
            data = self._load()
            if key is None:
                return data
            return self._get_nested(data, key.split("."))

    def set(self, key: str, value: Any, strict: bool = False) -> bool:
        """Set config value by dot notation key."""
        with self._lock:
            data = self._load()
            if strict:
                keys = key.split(".")
                current = data
                for k in keys[:-1]:
                    if k not in current:
                        break
                    if not isinstance(current[k], dict):
                        logger.warning(
                            "[ConfigStore] Strict mode: path '%s' is not a dict", k
                        )
                        return False
                    current = current[k]
            self._set_nested(data, key.split("."), value)
            data["updated_at"] = datetime.now().isoformat()
            try:
                validated = RuntimeConfig.model_validate(data)
                data = validated.model_dump()
            except Exception as exc:
                logger.warning("[ConfigStore] Validation failed: %s", exc)
                return False
            return self._atomic_write(data)

    def _get_nested(self, data: dict, keys: List[str]) -> Any:
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return None
        return data

    def _set_nested(self, data: dict, keys: List[str], value: Any) -> None:
        for k in keys[:-1]:
            if not isinstance(data.get(k), dict):
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value
