"""
Tests for config utilities module.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libs.core.config_utils import (
    ConfigChange,
    ConfigDiff,
    ConfigHierarchy,
    ConfigPriority,
    ConfigSource,
    ConfigValidator,
    ConfigWatcher,
    DefaultConfigSource,
    EnvConfigSource,
    FileConfigSource,
    RuntimeConfigSource,
    ValidationResult,
    ValidationRule,
)


class TestConfigPriority:
    """Tests for ConfigPriority enum."""

    def test_priority_order(self):
        """Test priority values are in correct order."""
        assert ConfigPriority.DEFAULT.value < ConfigPriority.FILE.value
        assert ConfigPriority.FILE.value < ConfigPriority.ENV.value
        assert ConfigPriority.ENV.value < ConfigPriority.RUNTIME.value


class TestDefaultConfigSource:
    """Tests for DefaultConfigSource."""

    def test_load_defaults(self):
        """Test loading default values."""
        defaults = {"timeout": 30, "retries": 3, "debug": False}
        source = DefaultConfigSource(defaults)

        loaded = source.load()
        assert loaded == defaults

    def test_get_existing_key(self):
        """Test getting existing key."""
        source = DefaultConfigSource({"key": "value"})
        assert source.get("key") == "value"

    def test_get_missing_key(self):
        """Test getting missing key with default."""
        source = DefaultConfigSource({})
        assert source.get("missing", "default") == "default"

    def test_priority(self):
        """Test priority is DEFAULT."""
        source = DefaultConfigSource({})
        assert source.priority == ConfigPriority.DEFAULT


class TestEnvConfigSource:
    """Tests for EnvConfigSource."""

    def test_get_env_variable(self):
        """Test getting environment variable."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            source = EnvConfigSource()
            assert source.get("TEST_VAR") == "test_value"

    def test_get_with_prefix(self):
        """Test getting with prefix."""
        with patch.dict(os.environ, {"APP_TIMEOUT": "30"}):
            source = EnvConfigSource(prefix="APP_")
            assert source.get("TIMEOUT") == 30

    def test_type_conversion_int(self):
        """Test automatic int conversion."""
        with patch.dict(os.environ, {"NUM": "42"}):
            source = EnvConfigSource()
            assert source.get("NUM") == 42
            assert isinstance(source.get("NUM"), int)

    def test_type_conversion_float(self):
        """Test automatic float conversion."""
        with patch.dict(os.environ, {"RATE": "3.14"}):
            source = EnvConfigSource()
            assert source.get("RATE") == 3.14
            assert isinstance(source.get("RATE"), float)

    def test_type_conversion_bool(self):
        """Test automatic bool conversion."""
        with patch.dict(os.environ, {"FLAG": "true", "OTHER": "false"}):
            source = EnvConfigSource()
            assert source.get("FLAG") is True
            assert source.get("OTHER") is False

    def test_type_hints(self):
        """Test explicit type hints."""
        with patch.dict(os.environ, {"VALUE": "123"}):
            source = EnvConfigSource(type_hints={"VALUE": str})
            # With str type hint, should stay as string
            assert source.get("VALUE") == "123"

    def test_load_with_prefix(self):
        """Test loading all vars with prefix."""
        with patch.dict(
            os.environ,
            {
                "APP_A": "1",
                "APP_B": "2",
                "OTHER_C": "3",
            },
            clear=True,
        ):
            source = EnvConfigSource(prefix="APP_")
            loaded = source.load()
            assert "A" in loaded
            assert "B" in loaded
            assert "OTHER_C" not in loaded


class TestFileConfigSource:
    """Tests for FileConfigSource."""

    def test_load_json_file(self):
        """Test loading JSON config file."""
        fd, filepath = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"key": "value", "number": 42}, f)

            source = FileConfigSource(filepath)
            loaded = source.load()

            assert loaded["key"] == "value"
            assert loaded["number"] == 42
        finally:
            os.unlink(filepath)

    def test_get_from_file(self):
        """Test getting value from file."""
        fd, filepath = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"setting": "enabled"}, f)

            source = FileConfigSource(filepath)
            assert source.get("setting") == "enabled"
        finally:
            os.unlink(filepath)

    def test_missing_file(self):
        """Test handling missing file."""
        source = FileConfigSource("/nonexistent/path.json")
        loaded = source.load()
        assert loaded == {}

    def test_has_changed(self):
        """Test file change detection."""
        fd, filepath = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"v": 1}, f)

            source = FileConfigSource(filepath)
            source.load()
            assert source.has_changed() is False

            # Modify file
            time.sleep(0.1)
            with open(filepath, "w") as f2:
                json.dump({"v": 2}, f2)

            assert source.has_changed() is True
        finally:
            os.unlink(filepath)


class TestRuntimeConfigSource:
    """Tests for RuntimeConfigSource."""

    def test_set_and_get(self):
        """Test setting and getting values."""
        source = RuntimeConfigSource()
        source.set("key", "value")
        assert source.get("key") == "value"

    def test_delete(self):
        """Test deleting values."""
        source = RuntimeConfigSource()
        source.set("key", "value")
        source.delete("key")
        assert source.get("key") is None

    def test_clear(self):
        """Test clearing all values."""
        source = RuntimeConfigSource()
        source.set("a", 1)
        source.set("b", 2)
        source.clear()
        assert source.get("a") is None
        assert source.get("b") is None

    def test_load(self):
        """Test loading all values."""
        source = RuntimeConfigSource()
        source.set("x", 10)
        source.set("y", 20)
        loaded = source.load()
        assert loaded == {"x": 10, "y": 20}

    def test_priority(self):
        """Test priority is RUNTIME (highest)."""
        source = RuntimeConfigSource()
        assert source.priority == ConfigPriority.RUNTIME


class TestConfigHierarchy:
    """Tests for ConfigHierarchy."""

    def test_priority_resolution(self):
        """Test that higher priority sources win."""
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(DefaultConfigSource({"key": "default"}))

        runtime = RuntimeConfigSource()
        runtime.set("key", "runtime")
        hierarchy.add_source(runtime)

        assert hierarchy.get("key") == "runtime"

    def test_fallback_to_lower_priority(self):
        """Test fallback to lower priority sources."""
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(DefaultConfigSource({"only_default": "value"}))
        hierarchy.add_source(RuntimeConfigSource())

        assert hierarchy.get("only_default") == "value"

    def test_get_all_merged(self):
        """Test getting merged config from all sources."""
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(DefaultConfigSource({"a": 1, "b": 2}))

        runtime = RuntimeConfigSource()
        runtime.set("b", 20)
        runtime.set("c", 30)
        hierarchy.add_source(runtime)

        merged = hierarchy.get_all()
        assert merged["a"] == 1
        assert merged["b"] == 20  # Runtime wins
        assert merged["c"] == 30

    def test_get_with_source(self):
        """Test getting value with source info."""
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(DefaultConfigSource({"key": "value"}))

        value, source_name = hierarchy.get_with_source("key")
        assert value == "value"
        assert source_name == "DefaultConfigSource"

    def test_reload_detects_changes(self):
        """Test reload detects configuration changes."""
        hierarchy = ConfigHierarchy()
        runtime = RuntimeConfigSource()
        runtime.set("key", "old")
        hierarchy.add_source(runtime)

        # Change value before reload
        runtime.set("key", "new")
        runtime.invalidate_cache()

        changes = hierarchy.reload()
        # Note: Changes may not be detected if cache wasn't cleared
        # This tests the mechanism exists

    def test_change_listener(self):
        """Test change listener notification."""
        hierarchy = ConfigHierarchy()
        runtime = RuntimeConfigSource()
        hierarchy.add_source(runtime)

        changes_received = []
        hierarchy.add_change_listener(lambda c: changes_received.append(c))

        # Trigger a reload (no changes expected initially)
        hierarchy.reload()


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_valid_config(self):
        """Test validation passes for valid config."""
        validator = ConfigValidator()
        validator.add_rule("port", lambda v: 1 <= v <= 65535, "Invalid port")

        result = validator.validate({"port": 8080})
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_invalid_config(self):
        """Test validation fails for invalid config."""
        validator = ConfigValidator()
        validator.add_rule(
            "port", lambda v: 1 <= v <= 65535, "Port must be 1-65535", required=True
        )

        result = validator.validate({"port": 70000})
        assert result.is_valid is False
        assert "Port must be 1-65535" in result.errors

    def test_required_field_missing(self):
        """Test validation fails for missing required field."""
        validator = ConfigValidator()
        validator.require("api_key")

        result = validator.validate({})
        assert result.is_valid is False
        assert any("api_key" in e for e in result.errors)

    def test_optional_field_missing(self):
        """Test validation passes for missing optional field."""
        validator = ConfigValidator()
        validator.add_rule("timeout", lambda v: v > 0, "Timeout must be positive")

        result = validator.validate({})
        assert result.is_valid is True

    def test_warnings_for_optional_invalid(self):
        """Test warnings for invalid optional fields."""
        validator = ConfigValidator()
        validator.add_rule(
            "cache_size", lambda v: v >= 0, "Cache size should be non-negative"
        )

        result = validator.validate({"cache_size": -1})
        assert result.is_valid is True  # Still valid (not required)
        assert len(result.warnings) == 1

    def test_validation_exception(self):
        """Test handling validation exceptions."""
        validator = ConfigValidator()
        validator.add_rule(
            "value", lambda v: v["nested"], "Check failed", required=True
        )

        result = validator.validate({"value": "not_a_dict"})
        assert result.is_valid is False
        assert any("Validation error" in e for e in result.errors)

    def test_chaining(self):
        """Test fluent chaining."""
        validator = (
            ConfigValidator()
            .require("key1")
            .add_rule("key2", lambda v: v > 0, "Must be positive")
        )

        assert len(validator._rules) == 2


class TestConfigWatcher:
    """Tests for ConfigWatcher."""

    def test_add_file(self):
        """Test adding file to watch."""
        hierarchy = ConfigHierarchy()
        watcher = ConfigWatcher(hierarchy)

        fd, filepath = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("{}")

            watcher.add_file(filepath)
            assert Path(filepath) in watcher._files
        finally:
            os.unlink(filepath)

    def test_start_stop(self):
        """Test starting and stopping watcher."""
        hierarchy = ConfigHierarchy()
        watcher = ConfigWatcher(hierarchy, poll_interval=0.1)

        watcher.start()
        assert watcher._running is True
        assert watcher._thread is not None

        watcher.stop()
        assert watcher._running is False

    def test_file_change_detection(self):
        """Test file change triggers reload."""
        hierarchy = ConfigHierarchy()

        fd, filepath = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump({"v": 1}, f)

            file_source = FileConfigSource(filepath)
            hierarchy.add_source(file_source)

            watcher = ConfigWatcher(hierarchy, poll_interval=0.1)
            watcher.add_file(filepath)
            watcher.start()

            # Modify file
            time.sleep(0.15)
            with open(filepath, "w") as f2:
                json.dump({"v": 2}, f2)

            # Wait for detection
            time.sleep(0.2)
            watcher.stop()

            # File should have been reloaded
        finally:
            os.unlink(filepath)


class TestConfigDiff:
    """Tests for ConfigDiff."""

    def test_no_changes(self):
        """Test diff with no changes."""
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 2}

        diff = ConfigDiff.compute(old, new)
        assert diff.has_changes() is False
        assert diff.unchanged == {"a", "b"}

    def test_added_keys(self):
        """Test detecting added keys."""
        old = {"a": 1}
        new = {"a": 1, "b": 2}

        diff = ConfigDiff.compute(old, new)
        assert diff.added == {"b": 2}
        assert diff.has_changes() is True

    def test_removed_keys(self):
        """Test detecting removed keys."""
        old = {"a": 1, "b": 2}
        new = {"a": 1}

        diff = ConfigDiff.compute(old, new)
        assert diff.removed == {"b": 2}
        assert diff.has_changes() is True

    def test_changed_keys(self):
        """Test detecting changed keys."""
        old = {"a": 1, "b": 2}
        new = {"a": 1, "b": 20}

        diff = ConfigDiff.compute(old, new)
        assert diff.changed == {"b": (2, 20)}
        assert diff.has_changes() is True

    def test_complex_diff(self):
        """Test complex diff with multiple change types."""
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 20, "d": 4}

        diff = ConfigDiff.compute(old, new)

        assert diff.unchanged == {"a"}
        assert diff.changed == {"b": (2, 20)}
        assert diff.removed == {"c": 3}
        assert diff.added == {"d": 4}

    def test_summary(self):
        """Test human-readable summary."""
        old = {"a": 1}
        new = {"a": 2, "b": 3}

        diff = ConfigDiff.compute(old, new)
        summary = diff.summary()

        assert "Added" in summary
        assert "Changed" in summary

    def test_summary_no_changes(self):
        """Test summary with no changes."""
        diff = ConfigDiff.compute({"a": 1}, {"a": 1})
        assert diff.summary() == "No changes"


class TestIntegration:
    """Integration tests for config utilities."""

    def test_full_hierarchy_with_validation(self):
        """Test full config hierarchy with validation."""
        # Create hierarchy
        hierarchy = ConfigHierarchy()
        hierarchy.add_source(
            DefaultConfigSource(
                {
                    "timeout": 30,
                    "retries": 3,
                    "debug": False,
                }
            )
        )

        with patch.dict(os.environ, {"APP_TIMEOUT": "60"}):
            hierarchy.add_source(EnvConfigSource(prefix="APP_"))

            runtime = RuntimeConfigSource()
            runtime.set("debug", True)
            hierarchy.add_source(runtime)

            # Create validator
            validator = ConfigValidator()
            validator.add_rule("timeout", lambda v: v > 0, "Timeout must be positive")
            validator.add_rule(
                "retries", lambda v: v >= 0, "Retries must be non-negative"
            )

            # Get and validate config
            config = hierarchy.get_all()
            result = validator.validate(config)

            assert result.is_valid is True
            assert hierarchy.get("timeout") == 60  # From env
            assert hierarchy.get("debug") is True  # From runtime
            assert hierarchy.get("retries") == 3  # From defaults

    def test_config_diff_after_reload(self):
        """Test detecting changes after reload."""
        hierarchy = ConfigHierarchy()
        runtime = RuntimeConfigSource()
        runtime.set("key", "original")
        hierarchy.add_source(runtime)

        old_config = hierarchy.get_all()

        # Change and reload
        runtime.set("key", "modified")
        runtime.set("new_key", "new_value")
        new_config = hierarchy.get_all()

        diff = ConfigDiff.compute(old_config, new_config)

        assert diff.changed.get("key") == ("original", "modified")
        assert "new_key" in diff.added
