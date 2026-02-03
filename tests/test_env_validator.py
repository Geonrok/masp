"""
Tests for environment variable validation.
"""

import os

import pytest

from libs.core.env_validator import (
    EnvironmentValidator,
    EnvVarSpec,
    EnvVarType,
    ValidationResult,
    create_masp_validator,
    get_env,
)
from libs.core.exceptions import ConfigurationError


class TestEnvVarSpec:
    """Tests for EnvVarSpec."""

    def test_default_values(self):
        """Test default specification values."""
        spec = EnvVarSpec(name="TEST_VAR")
        assert spec.var_type == EnvVarType.STRING
        assert spec.required is True
        assert spec.default is None
        assert spec.sensitive is False


class TestEnvironmentValidator:
    """Tests for EnvironmentValidator."""

    def test_register_and_validate(self):
        """Test basic registration and validation."""
        validator = EnvironmentValidator()
        validator.register("TEST_VAR", EnvVarType.STRING, required=True)

        result = validator.validate({"TEST_VAR": "hello"})
        assert result.valid
        assert result.values["TEST_VAR"] == "hello"

    def test_required_missing(self):
        """Test missing required variable."""
        validator = EnvironmentValidator()
        validator.register("REQUIRED_VAR", required=True)

        result = validator.validate({})
        assert not result.valid
        assert any("required" in e.lower() for e in result.errors)

    def test_optional_with_default(self):
        """Test optional variable with default."""
        validator = EnvironmentValidator()
        validator.register("OPT_VAR", required=False, default="default_value")

        result = validator.validate({})
        assert result.valid
        assert result.values["OPT_VAR"] == "default_value"

    def test_int_type(self):
        """Test integer type parsing."""
        validator = EnvironmentValidator()
        validator.register("PORT", EnvVarType.INT)

        result = validator.validate({"PORT": "8080"})
        assert result.valid
        assert result.values["PORT"] == 8080

    def test_float_type(self):
        """Test float type parsing."""
        validator = EnvironmentValidator()
        validator.register("RATE", EnvVarType.FLOAT)

        result = validator.validate({"RATE": "3.14"})
        assert result.valid
        assert result.values["RATE"] == pytest.approx(3.14)

    def test_bool_type(self):
        """Test boolean type parsing."""
        validator = EnvironmentValidator()
        validator.register("DEBUG", EnvVarType.BOOL, required=False)

        for true_val in ["true", "True", "1", "yes", "on"]:
            result = validator.validate({"DEBUG": true_val})
            assert result.values["DEBUG"] is True

        for false_val in ["false", "False", "0", "no", "off"]:
            result = validator.validate({"DEBUG": false_val})
            assert result.values["DEBUG"] is False

    def test_url_type(self):
        """Test URL type validation."""
        validator = EnvironmentValidator()
        validator.register("API_URL", EnvVarType.URL)

        result = validator.validate({"API_URL": "https://api.example.com/"})
        assert result.valid
        assert (
            result.values["API_URL"] == "https://api.example.com"
        )  # Trailing slash removed

        result = validator.validate({"API_URL": "not-a-url"})
        assert not result.valid

    def test_api_key_type(self):
        """Test API key validation."""
        validator = EnvironmentValidator()
        validator.register("API_KEY", EnvVarType.API_KEY)

        # Valid key
        result = validator.validate({"API_KEY": "abcdefghijklmnopqrstuvwxyz"})
        assert result.valid

        # Too short
        result = validator.validate({"API_KEY": "short"})
        assert not result.valid

        # Placeholder
        result = validator.validate({"API_KEY": "your_api_key_here_12345"})
        assert not result.valid

    def test_list_type(self):
        """Test list type parsing."""
        validator = EnvironmentValidator()
        validator.register("HOSTS", EnvVarType.LIST, required=False)

        result = validator.validate({"HOSTS": "host1, host2, host3"})
        assert result.valid
        assert result.values["HOSTS"] == ["host1", "host2", "host3"]

    def test_json_type(self):
        """Test JSON type parsing."""
        validator = EnvironmentValidator()
        validator.register("CONFIG", EnvVarType.JSON, required=False)

        result = validator.validate({"CONFIG": '{"key": "value", "num": 42}'})
        assert result.valid
        assert result.values["CONFIG"] == {"key": "value", "num": 42}

    def test_path_type(self):
        """Test path type with expansion."""
        validator = EnvironmentValidator()
        validator.register("DATA_DIR", EnvVarType.PATH, required=False)

        result = validator.validate({"DATA_DIR": "~/data"})
        assert result.valid
        assert "~" not in result.values["DATA_DIR"]  # Should be expanded

    def test_pattern_validation(self):
        """Test custom pattern validation."""
        validator = EnvironmentValidator()
        validator.register("CODE", pattern=r"^[A-Z]{3}$")

        result = validator.validate({"CODE": "ABC"})
        assert result.valid

        result = validator.validate({"CODE": "abc"})
        assert not result.valid

        result = validator.validate({"CODE": "ABCD"})
        assert not result.valid

    def test_length_validation(self):
        """Test min/max length validation."""
        validator = EnvironmentValidator()
        validator.register("PASSWORD", min_length=8, max_length=20)

        result = validator.validate({"PASSWORD": "validpassword"})
        assert result.valid

        result = validator.validate({"PASSWORD": "short"})
        assert not result.valid

        result = validator.validate({"PASSWORD": "a" * 25})
        assert not result.valid

    def test_range_validation(self):
        """Test min/max value validation."""
        validator = EnvironmentValidator()
        validator.register("PORT", EnvVarType.INT, min_value=1, max_value=65535)

        result = validator.validate({"PORT": "8080"})
        assert result.valid

        result = validator.validate({"PORT": "0"})
        assert not result.valid

        result = validator.validate({"PORT": "70000"})
        assert not result.valid

    def test_allowed_values(self):
        """Test allowed values validation."""
        validator = EnvironmentValidator()
        validator.register("ENV", allowed_values=["dev", "staging", "prod"])

        result = validator.validate({"ENV": "dev"})
        assert result.valid

        result = validator.validate({"ENV": "invalid"})
        assert not result.valid

    def test_custom_validator(self):
        """Test custom validator function."""
        validator = EnvironmentValidator()
        validator.register(
            "EVEN_NUMBER",
            EnvVarType.INT,
            validator=lambda x: x % 2 == 0,
        )

        result = validator.validate({"EVEN_NUMBER": "4"})
        assert result.valid

        result = validator.validate({"EVEN_NUMBER": "3"})
        assert not result.valid

    def test_deprecated_warning(self):
        """Test deprecated variable warning."""
        validator = EnvironmentValidator()
        validator.register(
            "OLD_VAR",
            deprecated=True,
            replacement="NEW_VAR",
            required=False,
        )

        result = validator.validate({"OLD_VAR": "value"})
        assert result.valid
        assert any("deprecated" in w.lower() for w in result.warnings)

    def test_multiple_errors(self):
        """Test multiple validation errors."""
        validator = EnvironmentValidator()
        validator.register("VAR1", required=True)
        validator.register("VAR2", required=True)
        validator.register("VAR3", required=True)

        result = validator.validate({})
        assert not result.valid
        assert len(result.errors) == 3

    def test_fluent_api(self):
        """Test fluent registration API."""
        validator = (
            EnvironmentValidator()
            .register("VAR1", required=True)
            .register("VAR2", required=True)
            .register("VAR3", required=False, default="default")
        )

        result = validator.validate({"VAR1": "a", "VAR2": "b"})
        assert result.valid


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_raise_if_invalid(self):
        """Test raise_if_invalid method."""
        result = ValidationResult(valid=False, errors=["error1", "error2"])

        with pytest.raises(ConfigurationError) as exc_info:
            result.raise_if_invalid()

        assert "error1" in str(exc_info.value)

    def test_valid_does_not_raise(self):
        """Test that valid result doesn't raise."""
        result = ValidationResult(valid=True)
        result.raise_if_invalid()  # Should not raise


class TestMASPValidator:
    """Tests for MASP-specific validator."""

    def test_create_validator(self):
        """Test MASP validator creation."""
        validator = create_masp_validator()
        assert len(validator._specs) > 0

    def test_env_variable_registration(self):
        """Test that expected variables are registered."""
        validator = create_masp_validator()

        expected_vars = [
            "MASP_ENV",
            "MASP_LOG_LEVEL",
            "UPBIT_ACCESS_KEY",
            "BITHUMB_API_KEY",
            "BINANCE_API_KEY",
            "SLACK_WEBHOOK_URL",
        ]

        for var in expected_vars:
            assert var in validator._specs

    def test_validate_with_minimal_env(self):
        """Test validation with minimal environment."""
        validator = create_masp_validator()

        # All MASP vars are optional, so empty env should be valid
        result = validator.validate({})
        assert result.valid


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_env_with_default(self):
        """Test get_env with default value."""
        # Clear any existing value
        os.environ.pop("TEST_NONEXISTENT_VAR", None)

        value = get_env("TEST_NONEXISTENT_VAR", default="fallback")
        assert value == "fallback"

    def test_get_env_with_type(self):
        """Test get_env with type conversion."""
        os.environ["TEST_INT_VAR"] = "42"
        try:
            value = get_env("TEST_INT_VAR", var_type=EnvVarType.INT)
            assert value == 42
        finally:
            del os.environ["TEST_INT_VAR"]


class TestDocumentation:
    """Tests for documentation generation."""

    def test_generate_documentation(self):
        """Test documentation generation."""
        validator = EnvironmentValidator()
        validator.register(
            "REQUIRED_VAR", required=True, description="A required variable"
        )
        validator.register(
            "OPTIONAL_VAR", required=False, default="default", description="Optional"
        )

        docs = validator.get_documentation()

        assert "# Environment Variables" in docs
        assert "REQUIRED_VAR" in docs
        assert "OPTIONAL_VAR" in docs
        assert "Required" in docs
        assert "Optional" in docs
