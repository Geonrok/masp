"""
Environment Variable Validation

Validates required environment variables at startup
with type checking and format validation.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from libs.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EnvVarType(Enum):
    """Environment variable types."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    URL = "url"
    PATH = "path"
    API_KEY = "api_key"
    SECRET = "secret"
    JSON = "json"
    LIST = "list"


@dataclass
class EnvVarSpec:
    """Specification for an environment variable."""

    name: str
    var_type: EnvVarType = EnvVarType.STRING
    required: bool = True
    default: Any = None
    description: str = ""
    pattern: Optional[str] = None  # Regex pattern
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    sensitive: bool = False  # Don't log value
    deprecated: bool = False
    replacement: Optional[str] = None  # Replacement var if deprecated
    validator: Optional[Callable[[Any], bool]] = None


@dataclass
class ValidationResult:
    """Result of environment validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise ConfigurationError if validation failed."""
        if not self.valid:
            raise ConfigurationError(
                message=f"Environment validation failed: {'; '.join(self.errors)}",
                error_code="ENV_VALIDATION_FAILED",
            )


class EnvironmentValidator:
    """
    Validates environment variables against specifications.

    Example:
        validator = EnvironmentValidator()
        validator.register("DATABASE_URL", EnvVarType.URL, required=True)
        validator.register("PORT", EnvVarType.INT, default=8080)
        validator.register("DEBUG", EnvVarType.BOOL, default=False)

        result = validator.validate()
        result.raise_if_invalid()
    """

    # Common patterns
    PATTERNS = {
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "api_key": r"^[A-Za-z0-9\-_]{16,}$",
        "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "hostname": r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$",
    }

    def __init__(self):
        self._specs: Dict[str, EnvVarSpec] = {}

    def register(
        self,
        name: str,
        var_type: EnvVarType = EnvVarType.STRING,
        required: bool = True,
        default: Any = None,
        description: str = "",
        pattern: Optional[str] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allowed_values: Optional[List[Any]] = None,
        sensitive: bool = False,
        deprecated: bool = False,
        replacement: Optional[str] = None,
        validator: Optional[Callable[[Any], bool]] = None,
    ) -> "EnvironmentValidator":
        """Register an environment variable specification."""
        self._specs[name] = EnvVarSpec(
            name=name,
            var_type=var_type,
            required=required,
            default=default,
            description=description,
            pattern=pattern,
            min_length=min_length,
            max_length=max_length,
            min_value=min_value,
            max_value=max_value,
            allowed_values=allowed_values,
            sensitive=sensitive,
            deprecated=deprecated,
            replacement=replacement,
            validator=validator,
        )
        return self

    def register_spec(self, spec: EnvVarSpec) -> "EnvironmentValidator":
        """Register a pre-built specification."""
        self._specs[spec.name] = spec
        return self

    def validate(self, env: Optional[Dict[str, str]] = None) -> ValidationResult:
        """
        Validate all registered environment variables.

        Args:
            env: Environment dict (defaults to os.environ)

        Returns:
            ValidationResult
        """
        env = env if env is not None else dict(os.environ)
        errors = []
        warnings = []
        values = {}

        for name, spec in self._specs.items():
            raw_value = env.get(name)

            # Check deprecated
            if spec.deprecated:
                if raw_value:
                    msg = f"{name} is deprecated"
                    if spec.replacement:
                        msg += f", use {spec.replacement} instead"
                    warnings.append(msg)

            # Check required
            if raw_value is None or raw_value == "":
                if spec.required:
                    errors.append(f"{name} is required but not set")
                    continue
                values[name] = spec.default
                continue

            # Parse and validate
            try:
                parsed = self._parse_value(raw_value, spec)
                validation_errors = self._validate_value(parsed, spec)
                errors.extend(validation_errors)

                if not validation_errors:
                    values[name] = parsed
                    self._log_value(name, parsed, spec)

            except Exception as e:
                errors.append(f"{name}: {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            values=values,
        )

    def _parse_value(self, raw: str, spec: EnvVarSpec) -> Any:
        """Parse raw string value to appropriate type."""
        if spec.var_type == EnvVarType.STRING:
            return raw

        elif spec.var_type == EnvVarType.INT:
            return int(raw)

        elif spec.var_type == EnvVarType.FLOAT:
            return float(raw)

        elif spec.var_type == EnvVarType.BOOL:
            return raw.lower() in ("true", "1", "yes", "on")

        elif spec.var_type == EnvVarType.URL:
            return raw.rstrip("/")

        elif spec.var_type == EnvVarType.PATH:
            return os.path.expanduser(os.path.expandvars(raw))

        elif spec.var_type in (EnvVarType.API_KEY, EnvVarType.SECRET):
            return raw

        elif spec.var_type == EnvVarType.JSON:
            import json

            return json.loads(raw)

        elif spec.var_type == EnvVarType.LIST:
            return [item.strip() for item in raw.split(",") if item.strip()]

        return raw

    def _validate_value(self, value: Any, spec: EnvVarSpec) -> List[str]:
        """Validate parsed value against spec."""
        errors = []

        # Pattern check
        if spec.pattern and isinstance(value, str):
            if not re.match(spec.pattern, value):
                errors.append(f"{spec.name} does not match pattern {spec.pattern}")

        # Type-specific patterns
        if spec.var_type == EnvVarType.URL:
            if not re.match(self.PATTERNS["url"], value, re.IGNORECASE):
                errors.append(f"{spec.name} is not a valid URL")

        elif spec.var_type == EnvVarType.API_KEY:
            if len(value) < 16:
                errors.append(f"{spec.name} appears too short for an API key")
            # Check for placeholder values
            placeholders = ["your", "xxx", "test", "placeholder", "***"]
            if any(p in value.lower() for p in placeholders):
                errors.append(f"{spec.name} appears to be a placeholder value")

        elif spec.var_type == EnvVarType.PATH:
            # Optionally check if path exists
            pass

        # Length checks
        if spec.min_length is not None:
            if hasattr(value, "__len__") and len(value) < spec.min_length:
                errors.append(f"{spec.name} length must be >= {spec.min_length}")

        if spec.max_length is not None:
            if hasattr(value, "__len__") and len(value) > spec.max_length:
                errors.append(f"{spec.name} length must be <= {spec.max_length}")

        # Numeric range checks
        if spec.min_value is not None:
            try:
                if float(value) < spec.min_value:
                    errors.append(f"{spec.name} must be >= {spec.min_value}")
            except (ValueError, TypeError):
                pass

        if spec.max_value is not None:
            try:
                if float(value) > spec.max_value:
                    errors.append(f"{spec.name} must be <= {spec.max_value}")
            except (ValueError, TypeError):
                pass

        # Allowed values
        if spec.allowed_values is not None:
            if value not in spec.allowed_values:
                errors.append(
                    f"{spec.name} must be one of {spec.allowed_values}, got {value}"
                )

        # Custom validator
        if spec.validator:
            try:
                if not spec.validator(value):
                    errors.append(f"{spec.name} failed custom validation")
            except Exception as e:
                errors.append(f"{spec.name} validation error: {e}")

        return errors

    def _log_value(self, name: str, value: Any, spec: EnvVarSpec) -> None:
        """Log validated value (masking sensitive data)."""
        if spec.sensitive:
            display = (
                f"{str(value)[:4]}...{str(value)[-4:]}"
                if len(str(value)) > 8
                else "****"
            )
        else:
            display = value

        logger.debug(f"[EnvValidator] {name} = {display}")

    def get_documentation(self) -> str:
        """Generate documentation for all registered variables."""
        lines = ["# Environment Variables", ""]

        required = [s for s in self._specs.values() if s.required]
        optional = [s for s in self._specs.values() if not s.required]

        if required:
            lines.append("## Required")
            lines.append("")
            for spec in required:
                lines.append(self._format_spec(spec))
            lines.append("")

        if optional:
            lines.append("## Optional")
            lines.append("")
            for spec in optional:
                lines.append(self._format_spec(spec))

        return "\n".join(lines)

    def _format_spec(self, spec: EnvVarSpec) -> str:
        """Format a spec for documentation."""
        parts = [f"- **{spec.name}**"]

        if spec.var_type != EnvVarType.STRING:
            parts.append(f" ({spec.var_type.value})")

        if spec.description:
            parts.append(f": {spec.description}")

        if spec.default is not None:
            parts.append(f" [default: {spec.default}]")

        if spec.deprecated:
            parts.append(" ⚠️ DEPRECATED")
            if spec.replacement:
                parts.append(f" - use {spec.replacement}")

        return "".join(parts)


# Pre-built validators for MASP


def create_masp_validator() -> EnvironmentValidator:
    """Create validator with MASP-specific environment variables."""
    validator = EnvironmentValidator()

    # Core settings
    validator.register(
        "MASP_ENV",
        EnvVarType.STRING,
        required=False,
        default="development",
        allowed_values=["development", "staging", "production"],
        description="Deployment environment",
    )

    validator.register(
        "MASP_LOG_LEVEL",
        EnvVarType.STRING,
        required=False,
        default="INFO",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        description="Logging level",
    )

    # API Keys (all optional - depends on which exchanges are used)
    validator.register(
        "UPBIT_ACCESS_KEY",
        EnvVarType.API_KEY,
        required=False,
        sensitive=True,
        description="Upbit API access key",
    )

    validator.register(
        "UPBIT_SECRET_KEY",
        EnvVarType.SECRET,
        required=False,
        sensitive=True,
        description="Upbit API secret key",
    )

    validator.register(
        "BITHUMB_API_KEY",
        EnvVarType.API_KEY,
        required=False,
        sensitive=True,
        description="Bithumb API key",
    )

    validator.register(
        "BITHUMB_SECRET_KEY",
        EnvVarType.SECRET,
        required=False,
        sensitive=True,
        description="Bithumb API secret",
    )

    validator.register(
        "BINANCE_API_KEY",
        EnvVarType.API_KEY,
        required=False,
        sensitive=True,
        description="Binance API key",
    )

    validator.register(
        "BINANCE_SECRET_KEY",
        EnvVarType.SECRET,
        required=False,
        sensitive=True,
        description="Binance API secret",
    )

    validator.register(
        "EBEST_APP_KEY",
        EnvVarType.API_KEY,
        required=False,
        sensitive=True,
        description="eBest/LS Securities app key",
    )

    validator.register(
        "EBEST_APP_SECRET",
        EnvVarType.SECRET,
        required=False,
        sensitive=True,
        description="eBest/LS Securities app secret",
    )

    # Notifications
    validator.register(
        "SLACK_WEBHOOK_URL",
        EnvVarType.URL,
        required=False,
        sensitive=True,
        description="Slack webhook URL for alerts",
    )

    validator.register(
        "TELEGRAM_BOT_TOKEN",
        EnvVarType.SECRET,
        required=False,
        sensitive=True,
        description="Telegram bot token",
    )

    validator.register(
        "TELEGRAM_CHAT_ID",
        EnvVarType.STRING,
        required=False,
        description="Telegram chat ID for alerts",
    )

    # Storage
    validator.register(
        "MASP_DATA_DIR",
        EnvVarType.PATH,
        required=False,
        default="./storage",
        description="Data storage directory",
    )

    return validator


def validate_environment(raise_on_error: bool = True) -> ValidationResult:
    """
    Validate MASP environment variables.

    Args:
        raise_on_error: Raise exception on validation failure

    Returns:
        ValidationResult
    """
    validator = create_masp_validator()
    result = validator.validate()

    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"[EnvValidator] {warning}")

    if not result.valid:
        for error in result.errors:
            logger.error(f"[EnvValidator] {error}")

        if raise_on_error:
            result.raise_if_invalid()

    return result


def require_env(name: str, var_type: EnvVarType = EnvVarType.STRING) -> Any:
    """
    Get a required environment variable or raise error.

    Args:
        name: Variable name
        var_type: Expected type

    Returns:
        Parsed value

    Raises:
        ConfigurationError: If variable is not set
    """
    validator = EnvironmentValidator()
    validator.register(name, var_type, required=True)
    result = validator.validate()
    result.raise_if_invalid()
    return result.values[name]


def get_env(
    name: str,
    default: Any = None,
    var_type: EnvVarType = EnvVarType.STRING,
) -> Any:
    """
    Get an optional environment variable with default.

    Args:
        name: Variable name
        default: Default value
        var_type: Expected type

    Returns:
        Parsed value or default
    """
    validator = EnvironmentValidator()
    validator.register(name, var_type, required=False, default=default)
    result = validator.validate()
    return result.values.get(name, default)
