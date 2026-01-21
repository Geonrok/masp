"""
Tests for startup API key validation.
"""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from libs.core.startup_validator import (
    validate_api_keys,
    validate_startup,
    check_exchange_ready,
    get_missing_keys,
    get_enabled_exchanges,
    _is_placeholder,
    _mask_key,
    _validate_key,
    EXCHANGE_KEY_CONFIGS,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_mask_key_empty(self):
        """Test masking empty key."""
        assert _mask_key("") == "(empty)"
        assert _mask_key(None) == "(empty)"

    def test_mask_key_short(self):
        """Test masking short key."""
        assert _mask_key("abc") == "****"
        assert _mask_key("12345678") == "****"

    def test_mask_key_long(self):
        """Test masking long key."""
        result = _mask_key("abcdefghijklmnop")
        assert result == "abcd...mnop"

    def test_is_placeholder_true(self):
        """Test detecting placeholder values."""
        placeholders = [
            "your-api-key-here",
            "your_secret_key",
            "xxx-xxx-xxx",
            "placeholder-key",
            "example-key-123",
            "test-key-abc",
            "dummy-value",
            "changeme",
        ]
        for p in placeholders:
            assert _is_placeholder(p) is True, f"Should detect {p} as placeholder"

    def test_is_placeholder_false(self):
        """Test real values are not detected as placeholders."""
        real_values = [
            "abc123def456ghi789jkl012",
            "fj49fj29fj39fj49fj59fj69",
            "REAL_API_KEY_VALUE_12345",
        ]
        for v in real_values:
            assert _is_placeholder(v) is False, f"Should not detect {v} as placeholder"


class TestValidateKey:
    """Tests for single key validation."""

    def test_validate_key_missing_required(self):
        """Test validation of missing required key."""
        result = _validate_key("TEST_KEY", None, required=True)
        assert result.is_valid is False
        assert "not set" in result.error

    def test_validate_key_missing_optional(self):
        """Test validation of missing optional key."""
        result = _validate_key("TEST_KEY", None, required=False)
        assert result.is_valid is True
        assert result.warning is not None

    def test_validate_key_placeholder(self):
        """Test validation rejects placeholder values."""
        result = _validate_key("TEST_KEY", "your-api-key-here", required=True)
        assert result.is_valid is False
        assert "placeholder" in result.error

    def test_validate_key_too_short(self):
        """Test validation rejects too short keys."""
        result = _validate_key("TEST_KEY", "abc123", required=True, min_length=16)
        assert result.is_valid is False
        assert "too short" in result.error

    def test_validate_key_valid(self):
        """Test validation accepts valid key."""
        result = _validate_key(
            "TEST_KEY",
            "valid_api_key_that_is_long_enough",
            required=True,
            min_length=16,
        )
        assert result.is_valid is True
        assert result.error is None


class TestGetEnabledExchanges:
    """Tests for get_enabled_exchanges function."""

    def test_get_enabled_exchanges_from_config(self, tmp_path):
        """Test reading enabled exchanges from config."""
        config = {
            "exchanges": {
                "upbit": {"enabled": True},
                "bithumb": {"enabled": False},
                "binance_spot": {"enabled": True},
            }
        }
        config_path = tmp_path / "schedule_config.json"
        config_path.write_text(json.dumps(config))

        enabled = get_enabled_exchanges(str(config_path))

        assert "upbit" in enabled
        assert "bithumb" not in enabled
        assert "binance_spot" in enabled

    def test_get_enabled_exchanges_missing_file(self, tmp_path):
        """Test with missing config file."""
        enabled = get_enabled_exchanges(str(tmp_path / "nonexistent.json"))
        assert enabled == set()

    def test_get_enabled_exchanges_invalid_json(self, tmp_path):
        """Test with invalid JSON."""
        config_path = tmp_path / "schedule_config.json"
        config_path.write_text("not valid json")

        enabled = get_enabled_exchanges(str(config_path))
        assert enabled == set()


class TestValidateApiKeys:
    """Tests for validate_api_keys function."""

    def test_validate_no_exchanges(self):
        """Test validation with no enabled exchanges."""
        result = validate_api_keys(exchanges=set())
        assert result.is_valid is True
        assert result.exchanges_validated == 0

    def test_validate_unknown_exchange(self):
        """Test validation with unknown exchange."""
        result = validate_api_keys(exchanges={"unknown_exchange"})
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "Unknown exchange" in result.warnings[0]

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "valid_upbit_access_key_123456789012345678",
        "UPBIT_SECRET_KEY": "valid_upbit_secret_key_123456789012345678",
    })
    def test_validate_upbit_valid_keys(self):
        """Test validation with valid Upbit keys."""
        result = validate_api_keys(exchanges={"upbit"})
        assert result.is_valid is True
        assert result.exchanges_valid == 1
        assert result.exchanges_invalid == 0

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "",
        "UPBIT_SECRET_KEY": "",
    }, clear=True)
    def test_validate_upbit_missing_keys(self):
        """Test validation with missing Upbit keys."""
        # Clear the env vars
        env = dict(os.environ)
        env.pop("UPBIT_ACCESS_KEY", None)
        env.pop("UPBIT_SECRET_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            result = validate_api_keys(exchanges={"upbit"})
            assert result.is_valid is False
            assert result.exchanges_invalid == 1

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "your-upbit-api-key",
        "UPBIT_SECRET_KEY": "your-upbit-secret",
    })
    def test_validate_upbit_placeholder_keys(self):
        """Test validation rejects placeholder keys."""
        result = validate_api_keys(exchanges={"upbit"})
        assert result.is_valid is False
        assert "placeholder" in str(result.errors).lower()

    @patch.dict(os.environ, {
        "BINANCE_API_KEY": "valid_binance_api_key_12345678901234567890123456789012345678901234567890123456789012",
        "BINANCE_API_SECRET": "valid_binance_secret_1234567890123456789012345678901234567890123456789012345678901",
    })
    def test_validate_binance_spot_valid_keys(self):
        """Test validation with valid Binance keys."""
        result = validate_api_keys(exchanges={"binance_spot"})
        assert result.is_valid is True

    @patch.dict(os.environ, {
        "BINANCE_API_KEY": "short_key",
        "BINANCE_API_SECRET": "short_secret",
    })
    def test_validate_binance_short_keys(self):
        """Test validation rejects too short Binance keys."""
        result = validate_api_keys(exchanges={"binance_spot"})
        assert result.is_valid is False
        assert "too short" in str(result.errors).lower()


class TestCheckExchangeReady:
    """Tests for check_exchange_ready function."""

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "valid_upbit_access_key_123456789012345678",
        "UPBIT_SECRET_KEY": "valid_upbit_secret_key_123456789012345678",
    })
    def test_exchange_ready_true(self):
        """Test exchange is ready when keys are valid."""
        assert check_exchange_ready("upbit") is True

    def test_exchange_ready_unknown(self):
        """Test unknown exchange returns False."""
        assert check_exchange_ready("unknown_exchange") is False

    @patch.dict(os.environ, {}, clear=True)
    def test_exchange_ready_missing_keys(self):
        """Test exchange is not ready when keys are missing."""
        # Clear all related env vars
        env = dict(os.environ)
        for key in list(env.keys()):
            if "UPBIT" in key:
                env.pop(key)

        with patch.dict(os.environ, env, clear=True):
            assert check_exchange_ready("upbit") is False


class TestGetMissingKeys:
    """Tests for get_missing_keys function."""

    def test_get_missing_keys_unknown_exchange(self):
        """Test unknown exchange returns empty list."""
        assert get_missing_keys("unknown_exchange") == []

    @patch.dict(os.environ, {}, clear=True)
    def test_get_missing_keys_all_missing(self):
        """Test all keys missing."""
        env = dict(os.environ)
        for key in list(env.keys()):
            if "UPBIT" in key:
                env.pop(key)

        with patch.dict(os.environ, env, clear=True):
            missing = get_missing_keys("upbit")
            assert "UPBIT_ACCESS_KEY" in missing
            assert "UPBIT_SECRET_KEY" in missing

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "valid_upbit_access_key_123456789012345678",
        "UPBIT_SECRET_KEY": "valid_upbit_secret_key_123456789012345678",
    })
    def test_get_missing_keys_none_missing(self):
        """Test no keys missing."""
        missing = get_missing_keys("upbit")
        assert missing == []


class TestValidateStartup:
    """Tests for validate_startup function."""

    def test_validate_startup_no_exchanges(self, tmp_path):
        """Test startup validation with no enabled exchanges."""
        config = {"exchanges": {}}
        config_path = tmp_path / "schedule_config.json"
        config_path.write_text(json.dumps(config))

        result = validate_startup(
            schedule_config_path=str(config_path),
            strict=False,
        )
        assert result is True

    @patch.dict(os.environ, {
        "UPBIT_ACCESS_KEY": "valid_upbit_access_key_123456789012345678",
        "UPBIT_SECRET_KEY": "valid_upbit_secret_key_123456789012345678",
    })
    def test_validate_startup_valid_config(self, tmp_path):
        """Test startup validation with valid config."""
        config = {
            "exchanges": {
                "upbit": {"enabled": True},
            }
        }
        config_path = tmp_path / "schedule_config.json"
        config_path.write_text(json.dumps(config))

        result = validate_startup(
            schedule_config_path=str(config_path),
            strict=False,
        )
        assert result is True


class TestExchangeKeyConfigs:
    """Tests for exchange key configurations."""

    def test_all_exchanges_have_required_keys(self):
        """Test all exchange configs have required keys."""
        for exchange, config in EXCHANGE_KEY_CONFIGS.items():
            assert config.exchange == exchange
            assert len(config.required_env_vars) > 0
            assert config.min_key_length > 0

    def test_upbit_config(self):
        """Test Upbit key configuration."""
        config = EXCHANGE_KEY_CONFIGS["upbit"]
        assert "UPBIT_ACCESS_KEY" in config.required_env_vars
        assert "UPBIT_SECRET_KEY" in config.required_env_vars
        assert config.min_key_length == 36  # UUID format

    def test_binance_config(self):
        """Test Binance key configuration."""
        config = EXCHANGE_KEY_CONFIGS["binance_spot"]
        assert "BINANCE_API_KEY" in config.required_env_vars
        assert "BINANCE_API_SECRET" in config.required_env_vars
        assert config.min_key_length == 64

    def test_ebest_config(self):
        """Test eBest key configuration."""
        config = EXCHANGE_KEY_CONFIGS["ebest_kospi"]
        assert "EBEST_APP_KEY" in config.required_env_vars
        assert "EBEST_APP_SECRET" in config.required_env_vars
        assert "EBEST_ACCOUNT_NO" in config.optional_env_vars
