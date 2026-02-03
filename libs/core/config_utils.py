"""
Config Utilities for Phase 2C
환경변수 기반 설정 유틸리티 및 고급 설정 관리

Features:
- ConfigHierarchy: 계층적 설정 소스 관리
- ConfigValidator: 설정 유효성 검증
- ConfigWatcher: 설정 파일 핫 리로드
- ConfigDiff: 설정 변경 감지
"""

from __future__ import annotations

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# ==================== Config Source Classes ====================


class ConfigPriority(Enum):
    """Configuration source priority levels."""

    DEFAULT = 0  # Built-in defaults
    FILE = 10  # Configuration files
    ENV = 20  # Environment variables
    RUNTIME = 30  # Runtime overrides (highest priority)


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""

    def __init__(self, priority: ConfigPriority):
        self.priority = priority
        self._cache: Dict[str, Any] = {}
        self._last_load: Optional[datetime] = None

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass

    def invalidate_cache(self) -> None:
        """Invalidate the configuration cache."""
        self._cache.clear()
        self._last_load = None


class DefaultConfigSource(ConfigSource):
    """Configuration source for built-in defaults."""

    def __init__(self, defaults: Dict[str, Any]):
        super().__init__(ConfigPriority.DEFAULT)
        self._defaults = defaults
        self._cache = defaults.copy()

    def load(self) -> Dict[str, Any]:
        return self._defaults.copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self._defaults.get(key, default)


class EnvConfigSource(ConfigSource):
    """Configuration source for environment variables."""

    def __init__(
        self,
        prefix: str = "",
        type_hints: Optional[Dict[str, Type]] = None,
    ):
        super().__init__(ConfigPriority.ENV)
        self.prefix = prefix
        self.type_hints = type_hints or {}

    def load(self) -> Dict[str, Any]:
        """Load all environment variables with prefix."""
        config = {}
        for key, value in os.environ.items():
            if self.prefix and not key.startswith(self.prefix):
                continue

            clean_key = key[len(self.prefix) :] if self.prefix else key
            clean_key = clean_key.lower()  # Convert to lowercase for consistency
            config[clean_key] = self._convert_type(clean_key, value)

        self._cache = config
        self._last_load = datetime.now()
        return config

    def get(self, key: str, default: Any = None) -> Any:
        env_key = f"{self.prefix}{key.upper()}" if self.prefix else key.upper()
        value = os.environ.get(env_key)
        if value is None:
            return default
        return self._convert_type(key, value)

    def _convert_type(self, key: str, value: str) -> Any:
        """Convert string value to appropriate type."""
        target_type = self.type_hints.get(key)
        if target_type is None:
            # Auto-detect type
            if value.lower() in ("true", "false"):
                return value.lower() == "true"
            try:
                return int(value)
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                pass
            return value

        try:
            if target_type == bool:
                return value.lower() in ("true", "1", "yes")
            return target_type(value)
        except (ValueError, TypeError):
            return value


class FileConfigSource(ConfigSource):
    """Configuration source for JSON/YAML files."""

    def __init__(self, file_path: Union[str, Path]):
        super().__init__(ConfigPriority.FILE)
        self.file_path = Path(file_path)
        self._last_modified: Optional[float] = None

    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            return {}

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                if self.file_path.suffix in (".json",):
                    self._cache = json.load(f)
                elif self.file_path.suffix in (".yaml", ".yml"):
                    try:
                        import yaml

                        self._cache = yaml.safe_load(f)
                    except ImportError:
                        self._cache = {}
                else:
                    self._cache = {}

            self._last_modified = self.file_path.stat().st_mtime
            self._last_load = datetime.now()

        except (json.JSONDecodeError, IOError) as e:
            self._cache = {}

        return self._cache.copy()

    def get(self, key: str, default: Any = None) -> Any:
        if not self._cache:
            self.load()
        return self._cache.get(key, default)

    def has_changed(self) -> bool:
        """Check if file has been modified since last load."""
        if not self.file_path.exists():
            return False
        current_mtime = self.file_path.stat().st_mtime
        return self._last_modified is None or current_mtime > self._last_modified


class RuntimeConfigSource(ConfigSource):
    """Configuration source for runtime overrides."""

    def __init__(self):
        super().__init__(ConfigPriority.RUNTIME)
        self._values: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        return self._values.copy()

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a runtime configuration value."""
        self._values[key] = value

    def delete(self, key: str) -> None:
        """Delete a runtime configuration value."""
        self._values.pop(key, None)

    def clear(self) -> None:
        """Clear all runtime configuration values."""
        self._values.clear()


# ==================== Config Hierarchy ====================


@dataclass
class ConfigChange:
    """Represents a configuration change."""

    key: str
    old_value: Any
    new_value: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConfigHierarchy:
    """
    Manages multiple configuration sources with priority-based resolution.

    Example:
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(DefaultConfigSource({"timeout": 30}))
        hierarchy.add_source(EnvConfigSource(prefix="APP_"))
        hierarchy.add_source(FileConfigSource("config.json"))

        # Get value (highest priority wins)
        timeout = hierarchy.get("timeout", default=30)
    """

    def __init__(self):
        self._sources: List[ConfigSource] = []
        self._change_listeners: List[Callable[[ConfigChange], None]] = []
        self._lock = threading.RLock()

    def add_source(self, source: ConfigSource) -> "ConfigHierarchy":
        """Add a configuration source."""
        with self._lock:
            self._sources.append(source)
            # Sort by priority (highest first for lookup)
            self._sources.sort(key=lambda s: s.priority.value, reverse=True)
        return self

    def remove_source(self, source: ConfigSource) -> None:
        """Remove a configuration source."""
        with self._lock:
            self._sources.remove(source)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value from highest priority source."""
        with self._lock:
            for source in self._sources:
                value = source.get(key)
                if value is not None:
                    return value
            return default

    def get_all(self) -> Dict[str, Any]:
        """Get merged configuration from all sources."""
        with self._lock:
            merged = {}
            # Process from lowest to highest priority
            for source in reversed(self._sources):
                merged.update(source.load())
            return merged

    def get_with_source(self, key: str) -> tuple:
        """Get value and its source."""
        with self._lock:
            for source in self._sources:
                value = source.get(key)
                if value is not None:
                    return value, source.__class__.__name__
            return None, None

    def reload(self) -> List[ConfigChange]:
        """Reload all sources and return changes."""
        changes = []
        with self._lock:
            old_config = self.get_all()
            for source in self._sources:
                source.invalidate_cache()
            new_config = self.get_all()

            # Detect changes
            all_keys = set(old_config.keys()) | set(new_config.keys())
            for key in all_keys:
                old_val = old_config.get(key)
                new_val = new_config.get(key)
                if old_val != new_val:
                    _, source_name = self.get_with_source(key)
                    change = ConfigChange(
                        key=key,
                        old_value=old_val,
                        new_value=new_val,
                        source=source_name or "unknown",
                    )
                    changes.append(change)
                    self._notify_listeners(change)

        return changes

    def add_change_listener(
        self,
        listener: Callable[[ConfigChange], None],
    ) -> None:
        """Add a listener for configuration changes."""
        self._change_listeners.append(listener)

    def _notify_listeners(self, change: ConfigChange) -> None:
        """Notify all listeners of a change."""
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception:
                pass  # Don't let listener errors break config


# ==================== Config Validator ====================


@dataclass
class ValidationRule:
    """A configuration validation rule."""

    key: str
    validator: Callable[[Any], bool]
    message: str
    required: bool = False


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ConfigValidator:
    """
    Validates configuration values against rules.

    Example:
        validator = ConfigValidator()
        validator.add_rule("port", lambda v: 1 <= v <= 65535, "Port must be 1-65535")
        validator.add_rule("timeout", lambda v: v > 0, "Timeout must be positive", required=True)

        result = validator.validate(config)
        if not result.is_valid:
            print(result.errors)
    """

    def __init__(self):
        self._rules: List[ValidationRule] = []

    def add_rule(
        self,
        key: str,
        validator: Callable[[Any], bool],
        message: str,
        required: bool = False,
    ) -> "ConfigValidator":
        """Add a validation rule."""
        self._rules.append(ValidationRule(key, validator, message, required))
        return self

    def require(self, key: str, message: Optional[str] = None) -> "ConfigValidator":
        """Add a required field rule."""
        msg = message or f"'{key}' is required"
        self._rules.append(
            ValidationRule(key, lambda v: v is not None, msg, required=True)
        )
        return self

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against all rules."""
        errors = []
        warnings = []

        for rule in self._rules:
            value = config.get(rule.key)

            if value is None:
                if rule.required:
                    errors.append(f"Missing required config: {rule.key}")
                continue

            try:
                if not rule.validator(value):
                    if rule.required:
                        errors.append(rule.message)
                    else:
                        warnings.append(rule.message)
            except Exception as e:
                errors.append(f"Validation error for '{rule.key}': {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


# ==================== Config Watcher ====================


class ConfigWatcher:
    """
    Watches configuration files for changes and triggers reload.

    Example:
        watcher = ConfigWatcher(hierarchy)
        watcher.add_file("config.json")
        watcher.start()

        # Later...
        watcher.stop()
    """

    def __init__(
        self,
        hierarchy: ConfigHierarchy,
        poll_interval: float = 5.0,
    ):
        self.hierarchy = hierarchy
        self.poll_interval = poll_interval
        self._files: List[Path] = []
        self._file_mtimes: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_file(self, file_path: Union[str, Path]) -> "ConfigWatcher":
        """Add a file to watch."""
        path = Path(file_path)
        with self._lock:
            if path not in self._files:
                self._files.append(path)
                if path.exists():
                    self._file_mtimes[str(path)] = path.stat().st_mtime
        return self

    def start(self) -> None:
        """Start watching for changes."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop watching for changes."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 1)
            self._thread = None

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                self._check_files()
            except Exception:
                pass  # Don't let errors stop the watcher
            time.sleep(self.poll_interval)

    def _check_files(self) -> None:
        """Check for file changes."""
        with self._lock:
            changed = False
            for path in self._files:
                if not path.exists():
                    continue

                current_mtime = path.stat().st_mtime
                last_mtime = self._file_mtimes.get(str(path), 0)

                if current_mtime > last_mtime:
                    self._file_mtimes[str(path)] = current_mtime
                    changed = True

            if changed:
                self.hierarchy.reload()


# ==================== Config Diff ====================


class ConfigDiff:
    """
    Computes and represents differences between configurations.

    Example:
        old_config = {"a": 1, "b": 2}
        new_config = {"a": 1, "b": 3, "c": 4}

        diff = ConfigDiff.compute(old_config, new_config)
        print(diff.added)    # {"c": 4}
        print(diff.removed)  # {}
        print(diff.changed)  # {"b": (2, 3)}
    """

    def __init__(self):
        self.added: Dict[str, Any] = {}
        self.removed: Dict[str, Any] = {}
        self.changed: Dict[str, tuple] = {}  # key -> (old, new)
        self.unchanged: Set[str] = set()

    @classmethod
    def compute(
        cls,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> "ConfigDiff":
        """Compute diff between two configurations."""
        diff = cls()

        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            in_old = key in old_config
            in_new = key in new_config

            if in_old and in_new:
                if old_config[key] == new_config[key]:
                    diff.unchanged.add(key)
                else:
                    diff.changed[key] = (old_config[key], new_config[key])
            elif in_old:
                diff.removed[key] = old_config[key]
            else:
                diff.added[key] = new_config[key]

        return diff

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> str:
        """Get a human-readable summary."""
        parts = []
        if self.added:
            parts.append(f"Added: {list(self.added.keys())}")
        if self.removed:
            parts.append(f"Removed: {list(self.removed.keys())}")
        if self.changed:
            parts.append(f"Changed: {list(self.changed.keys())}")
        return "; ".join(parts) if parts else "No changes"


def get_trading_limits() -> dict:
    """
    환경변수에서 Trading Limits 로드

    Returns:
        dict: Trading limits configuration
    """
    return {
        "max_order_value_krw": int(os.getenv("MAX_ORDER_VALUE_KRW", "1000000")),
        "max_position_pct": float(os.getenv("MAX_POSITION_PCT", "0.10")),
        "max_daily_loss_krw": int(os.getenv("MAX_DAILY_LOSS_KRW", "100000")),
        "adapter_mode": os.getenv("ADAPTER_MODE", "paper"),
    }


def is_live_mode() -> bool:
    """
    실거래 모드 여부 확인

    Returns:
        bool: True if live mode, False if paper mode
    """
    return os.getenv("ADAPTER_MODE", "paper").lower() == "live"


def validate_api_keys(exchange: str) -> bool:
    """
    환경변수 API 키 유효성 검증

    Args:
        exchange: "upbit" | "bithumb" | "binance"

    Returns:
        bool: True if all required keys are set
    """
    if exchange.lower() == "upbit":
        access = os.getenv("UPBIT_ACCESS_KEY", "")
        secret = os.getenv("UPBIT_SECRET_KEY", "")
        return bool(access and secret and access != "your_access_key_here")

    elif exchange.lower() == "bithumb":
        api_key = os.getenv("BITHUMB_API_KEY", "")
        secret = os.getenv("BITHUMB_SECRET_KEY", "")
        return bool(api_key and secret and api_key != "your_api_key_here")

    elif exchange.lower() == "binance":
        api_key = os.getenv("BINANCE_API_KEY", "")
        secret = os.getenv("BINANCE_API_SECRET", "")
        return bool(api_key and secret and api_key != "your_api_key_here")

    return False


def get_adapter_mode_description() -> str:
    """
    현재 Adapter Mode 설명

    Returns:
        str: Mode description with warnings
    """
    if is_live_mode():
        return "⚠️ LIVE MODE - 실거래 활성화! 실제 자금이 사용됩니다."
    else:
        return "✅ PAPER MODE - 모의 거래 (안전)"


def print_config_summary():
    """환경변수 기반 설정 요약 출력"""
    print("=" * 60)
    print("Configuration Summary (from .env)")
    print("=" * 60)

    limits = get_trading_limits()
    print(f"\nTrading Limits:")
    print(f"  Max Order Value: {limits['max_order_value_krw']:,} KRW")
    print(f"  Max Position: {limits['max_position_pct']*100:.0f}%")
    print(f"  Max Daily Loss: {limits['max_daily_loss_krw']:,} KRW")

    print(f"\nAdapter Mode:")
    print(f"  {get_adapter_mode_description()}")

    print(f"\nAPI Keys:")
    for exchange in ["upbit", "bithumb", "binance"]:
        status = "✅ SET" if validate_api_keys(exchange) else "❌ NOT SET"
        print(f"  {exchange.capitalize()}: {status}")

    print("=" * 60)


if __name__ == "__main__":
    print_config_summary()
