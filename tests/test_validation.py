"""
Tests for input validation and sanitization.
"""

import pytest

from libs.core.validation import (
    ValidationResult,
    FieldValidator,
    Validator,
    OrderValidator,
    PriceValidator,
    ConfigValidator,
    validate_json_response,
    sanitize_string,
    sanitize_numeric,
    sanitize_symbol,
)
from libs.core.exceptions import ValidationError


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self):
        """Test valid result."""
        result = ValidationResult(valid=True, sanitized_value=42)
        assert result.valid
        assert bool(result)
        assert result.sanitized_value == 42

    def test_invalid_result(self):
        """Test invalid result."""
        result = ValidationResult(valid=False, errors=["error1", "error2"])
        assert not result.valid
        assert not bool(result)
        assert len(result.errors) == 2

    def test_raise_if_invalid(self):
        """Test raise_if_invalid."""
        result = ValidationResult(valid=False, errors=["test error"])

        with pytest.raises(ValidationError) as exc_info:
            result.raise_if_invalid("test context")

        assert "test error" in str(exc_info.value)

    def test_raise_if_valid(self):
        """Test raise_if_invalid with valid result."""
        result = ValidationResult(valid=True)
        result.raise_if_invalid()  # Should not raise


class TestFieldValidator:
    """Tests for FieldValidator."""

    def test_required_field(self):
        """Test required field validation."""
        validator = FieldValidator(name="test", required=True)

        result = validator.validate(None)
        assert not result.valid
        assert "required" in result.errors[0]

        result = validator.validate("value")
        assert result.valid

    def test_optional_field(self):
        """Test optional field validation."""
        validator = FieldValidator(name="test", required=False, default="default")

        result = validator.validate(None)
        assert result.valid
        assert result.sanitized_value == "default"

    def test_type_validation(self):
        """Test type validation."""
        validator = FieldValidator(name="test", field_type=int)

        result = validator.validate("123")
        assert result.valid
        assert result.sanitized_value == 123

        result = validator.validate("not a number")
        assert not result.valid

    def test_range_validation(self):
        """Test range validation."""
        validator = FieldValidator(name="test", min_value=0, max_value=100)

        result = validator.validate(50)
        assert result.valid

        result = validator.validate(-1)
        assert not result.valid

        result = validator.validate(101)
        assert not result.valid

    def test_length_validation(self):
        """Test length validation."""
        validator = FieldValidator(name="test", min_length=2, max_length=5)

        result = validator.validate("abc")
        assert result.valid

        result = validator.validate("a")
        assert not result.valid

        result = validator.validate("toolong")
        assert not result.valid

    def test_pattern_validation(self):
        """Test pattern validation."""
        validator = FieldValidator(name="test", pattern=r"^[A-Z]{3}$")

        result = validator.validate("ABC")
        assert result.valid

        result = validator.validate("abc")
        assert not result.valid

        result = validator.validate("ABCD")
        assert not result.valid

    def test_allowed_values(self):
        """Test allowed values validation."""
        validator = FieldValidator(name="test", allowed_values=["a", "b", "c"])

        result = validator.validate("a")
        assert result.valid

        result = validator.validate("d")
        assert not result.valid

    def test_custom_validator(self):
        """Test custom validator."""
        validator = FieldValidator(
            name="test",
            custom_validator=lambda x: x % 2 == 0,
        )

        result = validator.validate(4)
        assert result.valid

        result = validator.validate(3)
        assert not result.valid

    def test_sanitizer(self):
        """Test sanitizer function."""
        validator = FieldValidator(
            name="test",
            sanitizer=lambda x: x.upper(),
        )

        result = validator.validate("hello")
        assert result.valid
        assert result.sanitized_value == "HELLO"


class TestValidator:
    """Tests for multi-field Validator."""

    def test_multiple_fields(self):
        """Test validation of multiple fields."""
        validator = Validator("test")
        validator.add_field("name", required=True, field_type=str)
        validator.add_field("age", required=True, field_type=int, min_value=0)

        result = validator.validate({"name": "John", "age": 25})
        assert result.valid

        result = validator.validate({"name": "John"})
        assert not result.valid  # Missing age

    def test_validate_or_raise(self):
        """Test validate_or_raise."""
        validator = Validator("test")
        validator.add_field("value", required=True)

        data = validator.validate_or_raise({"value": "test"})
        assert data["value"] == "test"

        with pytest.raises(ValidationError):
            validator.validate_or_raise({})


class TestOrderValidator:
    """Tests for OrderValidator."""

    def test_valid_order(self):
        """Test valid order validation."""
        result = OrderValidator.validate_order(
            symbol="BTCUSDT",
            side="buy",
            quantity=1.5,
            price=50000,
            exchange="binance",
        )

        assert result["symbol"] == "BTCUSDT"
        assert result["side"] == "buy"
        assert result["quantity"] == 1.5

    def test_invalid_symbol(self):
        """Test invalid symbol."""
        with pytest.raises(ValidationError):
            OrderValidator.validate_order(
                symbol="invalid symbol!",
                side="buy",
                quantity=1.0,
                exchange="binance",
            )

    def test_invalid_quantity(self):
        """Test invalid quantity."""
        with pytest.raises(ValidationError):
            OrderValidator.validate_order(
                symbol="BTCUSDT",
                side="buy",
                quantity=-1.0,
                exchange="binance",
            )

    def test_invalid_side(self):
        """Test invalid side."""
        with pytest.raises(ValidationError):
            OrderValidator.validate_order(
                symbol="BTCUSDT",
                side="invalid",
                quantity=1.0,
            )

    def test_upbit_symbol_format(self):
        """Test Upbit symbol format."""
        result = OrderValidator.validate_order(
            symbol="KRW-BTC",
            side="buy",
            quantity=1.0,
            exchange="upbit",
        )
        assert result["symbol"] == "KRW-BTC"

    def test_ebest_symbol_format(self):
        """Test eBest symbol format (6 digits)."""
        result = OrderValidator.validate_order(
            symbol="005930",
            side="buy",
            quantity=10,
            exchange="ebest",
        )
        assert result["symbol"] == "005930"


class TestPriceValidator:
    """Tests for PriceValidator."""

    def test_valid_ohlcv(self):
        """Test valid OHLCV data."""
        result = PriceValidator.validate_ohlcv({
            "open": 100,
            "high": 110,
            "low": 90,
            "close": 105,
            "volume": 1000,
        })
        assert result.valid

    def test_invalid_ohlcv_high_low(self):
        """Test OHLCV with high < low."""
        result = PriceValidator.validate_ohlcv({
            "open": 100,
            "high": 90,  # Invalid: high < low
            "low": 110,
            "close": 100,
        })
        assert not result.valid
        assert any("high" in e for e in result.errors)

    def test_valid_quote(self):
        """Test valid quote data."""
        result = PriceValidator.validate_quote({
            "bid": 99,
            "ask": 101,
            "last": 100,
        })
        assert result.valid

    def test_crossed_market_warning(self):
        """Test crossed market warning."""
        result = PriceValidator.validate_quote({
            "bid": 101,
            "ask": 99,  # Crossed: bid > ask
        })
        assert result.valid  # Still valid but with warning
        assert any("crossed" in w for w in result.warnings)


class TestConfigValidator:
    """Tests for ConfigValidator."""

    def test_valid_api_key(self):
        """Test valid API key."""
        result = ConfigValidator.validate_api_key(
            "abcdefghijklmnopqrstuvwxyz123456"
        )
        assert result.valid

    def test_empty_api_key(self):
        """Test empty API key."""
        result = ConfigValidator.validate_api_key(None)
        assert not result.valid

    def test_placeholder_api_key(self):
        """Test placeholder API key detection."""
        result = ConfigValidator.validate_api_key("your_api_key_here")
        assert not result.valid
        assert any("placeholder" in e for e in result.errors)

    def test_valid_url(self):
        """Test valid URL."""
        result = ConfigValidator.validate_url("https://api.example.com")
        assert result.valid
        assert result.sanitized_value == "https://api.example.com"

    def test_invalid_url(self):
        """Test invalid URL."""
        result = ConfigValidator.validate_url("not a url")
        assert not result.valid


class TestValidateJsonResponse:
    """Tests for validate_json_response."""

    def test_valid_dict(self):
        """Test valid dict input."""
        result = validate_json_response({"key": "value"})
        assert result == {"key": "value"}

    def test_valid_json_string(self):
        """Test valid JSON string input."""
        result = validate_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_string(self):
        """Test invalid JSON string."""
        with pytest.raises(ValidationError) as exc_info:
            validate_json_response("not json")
        assert "Invalid JSON" in str(exc_info.value)

    def test_required_fields(self):
        """Test required fields check."""
        result = validate_json_response(
            {"a": 1, "b": 2},
            required_fields=["a", "b"],
        )
        assert result == {"a": 1, "b": 2}

        with pytest.raises(ValidationError) as exc_info:
            validate_json_response(
                {"a": 1},
                required_fields=["a", "b"],
            )
        assert "Missing" in str(exc_info.value)

    def test_non_dict_input(self):
        """Test non-dict input."""
        with pytest.raises(ValidationError):
            validate_json_response([1, 2, 3])


class TestSanitizeString:
    """Tests for sanitize_string."""

    def test_basic_sanitization(self):
        """Test basic string sanitization."""
        result = sanitize_string("  hello  ")
        assert result == "hello"

    def test_max_length(self):
        """Test max length truncation."""
        result = sanitize_string("a" * 100, max_length=10)
        assert len(result) == 10

    def test_null_bytes(self):
        """Test null byte removal."""
        result = sanitize_string("hello\x00world")
        assert result == "helloworld"

    def test_none_input(self):
        """Test None input."""
        result = sanitize_string(None)
        assert result is None


class TestSanitizeNumeric:
    """Tests for sanitize_numeric."""

    def test_basic_conversion(self):
        """Test basic numeric conversion."""
        assert sanitize_numeric("123.45") == 123.45
        assert sanitize_numeric(123) == 123.0

    def test_comma_handling(self):
        """Test comma in number string."""
        assert sanitize_numeric("1,234.56") == 1234.56

    def test_min_max_clamping(self):
        """Test min/max value clamping."""
        assert sanitize_numeric(150, max_value=100) == 100
        assert sanitize_numeric(-10, min_value=0) == 0

    def test_precision(self):
        """Test decimal precision."""
        assert sanitize_numeric(123.456789, precision=2) == 123.46

    def test_invalid_input(self):
        """Test invalid input with default."""
        assert sanitize_numeric("not a number", default=0) == 0


class TestSanitizeSymbol:
    """Tests for sanitize_symbol."""

    def test_basic_sanitization(self):
        """Test basic symbol sanitization."""
        assert sanitize_symbol("btcusdt") == "BTCUSDT"
        assert sanitize_symbol("  BTC  ") == "BTC"

    def test_special_char_removal(self):
        """Test special character removal."""
        assert sanitize_symbol("BTC@USDT!") == "BTCUSDT"

    def test_empty_symbol(self):
        """Test empty symbol rejection."""
        with pytest.raises(ValidationError):
            sanitize_symbol("")

    def test_all_special_chars(self):
        """Test symbol with only special chars."""
        with pytest.raises(ValidationError):
            sanitize_symbol("@#$%")
