"""
Input Validation and Data Sanitization

Centralized validation utilities for the MASP platform.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, Callable

from libs.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of validation."""

    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_value: Any = None

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self, context: str = "") -> None:
        """Raise ValidationError if not valid."""
        if not self.valid:
            msg = "; ".join(self.errors)
            if context:
                msg = f"{context}: {msg}"
            raise ValidationError(message=msg, error_code="VALIDATION_FAILED")


@dataclass
class FieldValidator:
    """Validator for a single field."""

    name: str
    required: bool = True
    field_type: Optional[Type] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    default: Any = None

    def validate(self, value: Any) -> ValidationResult:
        """Validate a single value."""
        errors = []
        warnings = []
        sanitized = value

        # Required check
        if value is None or value == "":
            if self.required:
                errors.append(f"{self.name} is required")
                return ValidationResult(False, errors)
            return ValidationResult(True, sanitized_value=self.default)

        # Type check
        if self.field_type:
            if not isinstance(value, self.field_type):
                try:
                    sanitized = self.field_type(value)
                except (ValueError, TypeError):
                    errors.append(
                        f"{self.name} must be {self.field_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # Range check (for numeric types)
        if self.min_value is not None or self.max_value is not None:
            try:
                num_value = float(sanitized)
                if self.min_value is not None and num_value < self.min_value:
                    errors.append(
                        f"{self.name} must be >= {self.min_value}, got {num_value}"
                    )
                if self.max_value is not None and num_value > self.max_value:
                    errors.append(
                        f"{self.name} must be <= {self.max_value}, got {num_value}"
                    )
            except (ValueError, TypeError):
                errors.append(f"{self.name} must be numeric for range check")

        # Length check (for strings/lists)
        if self.min_length is not None or self.max_length is not None:
            try:
                length = len(sanitized)
                if self.min_length is not None and length < self.min_length:
                    errors.append(
                        f"{self.name} length must be >= {self.min_length}, got {length}"
                    )
                if self.max_length is not None and length > self.max_length:
                    errors.append(
                        f"{self.name} length must be <= {self.max_length}, got {length}"
                    )
            except TypeError:
                pass

        # Pattern check
        if self.pattern and isinstance(sanitized, str):
            if not re.match(self.pattern, sanitized):
                errors.append(f"{self.name} does not match pattern {self.pattern}")

        # Allowed values check
        if self.allowed_values is not None:
            if sanitized not in self.allowed_values:
                errors.append(
                    f"{self.name} must be one of {self.allowed_values}, got {sanitized}"
                )

        # Custom validator
        if self.custom_validator and not errors:
            try:
                if not self.custom_validator(sanitized):
                    errors.append(f"{self.name} failed custom validation")
            except Exception as e:
                errors.append(f"{self.name} custom validation error: {e}")

        # Sanitizer
        if self.sanitizer and not errors:
            try:
                sanitized = self.sanitizer(sanitized)
            except Exception as e:
                warnings.append(f"{self.name} sanitization warning: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized,
        )


class Validator:
    """
    Multi-field validator for data objects.

    Example:
        validator = Validator()
        validator.add_field("symbol", required=True, pattern=r"^[A-Z]{2,10}$")
        validator.add_field("quantity", required=True, field_type=float, min_value=0)
        validator.add_field("price", required=True, field_type=float, min_value=0)

        result = validator.validate({"symbol": "BTC", "quantity": 1.5, "price": 50000})
    """

    def __init__(self, name: str = "Validator"):
        self.name = name
        self.fields: Dict[str, FieldValidator] = {}

    def add_field(
        self,
        name: str,
        required: bool = True,
        field_type: Optional[Type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_values: Optional[List[Any]] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        sanitizer: Optional[Callable[[Any], Any]] = None,
        default: Any = None,
    ) -> "Validator":
        """Add a field validator."""
        self.fields[name] = FieldValidator(
            name=name,
            required=required,
            field_type=field_type,
            min_value=min_value,
            max_value=max_value,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern,
            allowed_values=allowed_values,
            custom_validator=custom_validator,
            sanitizer=sanitizer,
            default=default,
        )
        return self

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate all fields in data."""
        all_errors = []
        all_warnings = []
        sanitized_data = {}

        for field_name, validator in self.fields.items():
            value = data.get(field_name)
            result = validator.validate(value)

            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            sanitized_data[field_name] = result.sanitized_value

        return ValidationResult(
            valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            sanitized_value=sanitized_data,
        )

    def validate_or_raise(self, data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Validate and return sanitized data or raise ValidationError."""
        result = self.validate(data)
        if not result.valid:
            result.raise_if_invalid(context or self.name)
        return result.sanitized_value


# Pre-built validators for common use cases

class OrderValidator:
    """Validator for order parameters."""

    # Symbol patterns for different exchanges
    SYMBOL_PATTERNS = {
        "default": r"^[A-Za-z0-9\-_/]{1,20}$",
        "binance": r"^[A-Z0-9]{2,20}$",
        "upbit": r"^[A-Z]+-[A-Z]+$",
        "bithumb": r"^[A-Z]+_[A-Z]+$",
        "ebest": r"^[0-9]{6}$",
    }

    @classmethod
    def create(cls, exchange: str = "default") -> Validator:
        """Create order validator for specific exchange."""
        pattern = cls.SYMBOL_PATTERNS.get(exchange, cls.SYMBOL_PATTERNS["default"])

        validator = Validator(f"OrderValidator({exchange})")
        validator.add_field(
            "symbol",
            required=True,
            field_type=str,
            pattern=pattern,
            sanitizer=lambda x: x.upper() if isinstance(x, str) else x,
        )
        validator.add_field(
            "side",
            required=True,
            allowed_values=["buy", "sell", "BUY", "SELL"],
            sanitizer=lambda x: x.lower() if isinstance(x, str) else x,
        )
        validator.add_field(
            "quantity",
            required=True,
            field_type=float,
            min_value=0,
            custom_validator=lambda x: x > 0,
        )
        validator.add_field(
            "price",
            required=False,
            field_type=float,
            min_value=0,
        )
        validator.add_field(
            "order_type",
            required=False,
            allowed_values=["market", "limit", "MARKET", "LIMIT"],
            default="market",
            sanitizer=lambda x: x.lower() if isinstance(x, str) else x,
        )

        return validator

    @classmethod
    def validate_order(
        cls,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        exchange: str = "default",
    ) -> Dict[str, Any]:
        """Validate order parameters and return sanitized values."""
        validator = cls.create(exchange)
        return validator.validate_or_raise(
            {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "order_type": order_type,
            },
            context=f"Order validation ({exchange})",
        )


class PriceValidator:
    """Validator for price data."""

    @staticmethod
    def validate_ohlcv(data: Dict[str, Any]) -> ValidationResult:
        """Validate OHLCV data."""
        validator = Validator("OHLCV")
        validator.add_field("open", required=True, field_type=float, min_value=0)
        validator.add_field("high", required=True, field_type=float, min_value=0)
        validator.add_field("low", required=True, field_type=float, min_value=0)
        validator.add_field("close", required=True, field_type=float, min_value=0)
        validator.add_field("volume", required=False, field_type=float, min_value=0, default=0)

        result = validator.validate(data)

        # Additional cross-field validation
        if result.valid:
            sanitized = result.sanitized_value
            if sanitized["high"] < sanitized["low"]:
                result.errors.append("high must be >= low")
                result.valid = False
            if sanitized["high"] < sanitized["open"]:
                result.errors.append("high must be >= open")
                result.valid = False
            if sanitized["high"] < sanitized["close"]:
                result.errors.append("high must be >= close")
                result.valid = False
            if sanitized["low"] > sanitized["open"]:
                result.errors.append("low must be <= open")
                result.valid = False
            if sanitized["low"] > sanitized["close"]:
                result.errors.append("low must be <= close")
                result.valid = False

        return result

    @staticmethod
    def validate_quote(data: Dict[str, Any]) -> ValidationResult:
        """Validate quote data."""
        validator = Validator("Quote")
        validator.add_field("bid", required=False, field_type=float, min_value=0)
        validator.add_field("ask", required=False, field_type=float, min_value=0)
        validator.add_field("last", required=False, field_type=float, min_value=0)
        validator.add_field("volume", required=False, field_type=float, min_value=0)

        result = validator.validate(data)

        # Cross-field validation
        if result.valid:
            sanitized = result.sanitized_value
            if sanitized.get("bid") and sanitized.get("ask"):
                if sanitized["bid"] > sanitized["ask"]:
                    result.warnings.append("bid > ask (crossed market)")

        return result


class ConfigValidator:
    """Validator for configuration data."""

    @staticmethod
    def validate_api_key(key: Optional[str], name: str = "api_key") -> ValidationResult:
        """Validate API key format."""
        if not key:
            return ValidationResult(False, [f"{name} is required"])

        errors = []
        warnings = []

        if len(key) < 16:
            warnings.append(f"{name} seems too short ({len(key)} chars)")

        if len(key) > 256:
            errors.append(f"{name} is too long ({len(key)} chars)")

        # Check for common placeholder patterns
        placeholder_patterns = [
            r"^your[_-]?",
            r"^xxx+$",
            r"^test[_-]?key",
            r"^\*+$",
            r"^placeholder",
        ]
        for pattern in placeholder_patterns:
            if re.match(pattern, key, re.IGNORECASE):
                errors.append(f"{name} appears to be a placeholder")
                break

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=key.strip() if key else None,
        )

    @staticmethod
    def validate_url(url: Optional[str], name: str = "url") -> ValidationResult:
        """Validate URL format."""
        if not url:
            return ValidationResult(False, [f"{name} is required"])

        errors = []

        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, url, re.IGNORECASE):
            errors.append(f"{name} is not a valid URL")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            sanitized_value=url.rstrip("/") if url else None,
        )


def validate_json_response(
    data: Any,
    required_fields: Optional[List[str]] = None,
    context: str = "JSON response",
) -> Dict[str, Any]:
    """
    Validate and parse JSON response data.

    Args:
        data: Response data (dict, list, or JSON string)
        required_fields: List of required field names
        context: Context for error messages

    Returns:
        Validated data dictionary

    Raises:
        ValidationError: If validation fails
    """
    import json

    # Parse if string
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(
                message=f"{context}: Invalid JSON - {e}",
                error_code="INVALID_JSON",
            )

    # Must be dict
    if not isinstance(data, dict):
        raise ValidationError(
            message=f"{context}: Expected dict, got {type(data).__name__}",
            error_code="INVALID_TYPE",
        )

    # Check required fields
    if required_fields:
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValidationError(
                message=f"{context}: Missing required fields: {missing}",
                error_code="MISSING_FIELDS",
            )

    return data


def sanitize_string(
    value: Optional[str],
    max_length: int = 1000,
    strip: bool = True,
    remove_null: bool = True,
) -> Optional[str]:
    """
    Sanitize a string value.

    Args:
        value: Input string
        max_length: Maximum allowed length
        strip: Strip whitespace
        remove_null: Remove null bytes

    Returns:
        Sanitized string
    """
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    if strip:
        value = value.strip()

    if remove_null:
        value = value.replace("\x00", "")

    if len(value) > max_length:
        value = value[:max_length]

    return value


def sanitize_numeric(
    value: Any,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    precision: Optional[int] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Sanitize a numeric value.

    Args:
        value: Input value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        precision: Decimal precision
        default: Default value if conversion fails

    Returns:
        Sanitized float or default
    """
    try:
        if isinstance(value, str):
            # Handle common number formats
            value = value.strip().replace(",", "")

        result = float(value)

        if min_value is not None:
            result = max(result, min_value)

        if max_value is not None:
            result = min(result, max_value)

        if precision is not None:
            result = round(result, precision)

        return result

    except (ValueError, TypeError):
        return default


def sanitize_symbol(
    symbol: str,
    exchange: str = "default",
    uppercase: bool = True,
) -> str:
    """
    Sanitize a trading symbol.

    Args:
        symbol: Raw symbol string
        exchange: Exchange name for format rules
        uppercase: Convert to uppercase

    Returns:
        Sanitized symbol
    """
    if not symbol:
        raise ValidationError("Symbol is required", error_code="EMPTY_SYMBOL")

    # Remove whitespace and special chars
    symbol = re.sub(r"[^\w\-_/]", "", symbol.strip())

    if uppercase:
        symbol = symbol.upper()

    if not symbol:
        raise ValidationError("Symbol contains no valid characters", error_code="INVALID_SYMBOL")

    return symbol
