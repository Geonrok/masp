"""
Startup API Key Validation.

Validates that required API keys are configured for enabled exchanges
before allowing the service to start.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from libs.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ExchangeKeyConfig:
    """Configuration for exchange API key requirements."""

    exchange: str
    required_env_vars: List[str]
    optional_env_vars: List[str] = field(default_factory=list)
    description: str = ""
    min_key_length: int = 16


# Exchange API key requirements mapping
EXCHANGE_KEY_CONFIGS: Dict[str, ExchangeKeyConfig] = {
    "upbit": ExchangeKeyConfig(
        exchange="upbit",
        required_env_vars=["UPBIT_ACCESS_KEY", "UPBIT_SECRET_KEY"],
        description="Upbit (Korean Crypto Exchange)",
        min_key_length=36,  # Upbit keys are UUID format
    ),
    "bithumb": ExchangeKeyConfig(
        exchange="bithumb",
        required_env_vars=["BITHUMB_API_KEY", "BITHUMB_SECRET_KEY"],
        description="Bithumb (Korean Crypto Exchange)",
    ),
    "binance_spot": ExchangeKeyConfig(
        exchange="binance_spot",
        required_env_vars=["BINANCE_API_KEY", "BINANCE_API_SECRET"],
        description="Binance Spot (Global Crypto Exchange)",
        min_key_length=64,  # Binance keys are 64 chars
    ),
    "binance_futures": ExchangeKeyConfig(
        exchange="binance_futures",
        required_env_vars=["BINANCE_API_KEY", "BINANCE_API_SECRET"],
        description="Binance Futures (Global Crypto Exchange)",
        min_key_length=64,
    ),
    "ebest_kospi": ExchangeKeyConfig(
        exchange="ebest_kospi",
        required_env_vars=["EBEST_APP_KEY", "EBEST_APP_SECRET"],
        optional_env_vars=["EBEST_ACCOUNT_NO", "EBEST_ACCOUNT_PWD"],
        description="eBest/LS Securities (KOSPI)",
    ),
    "ebest_kosdaq": ExchangeKeyConfig(
        exchange="ebest_kosdaq",
        required_env_vars=["EBEST_APP_KEY", "EBEST_APP_SECRET"],
        optional_env_vars=["EBEST_ACCOUNT_NO", "EBEST_ACCOUNT_PWD"],
        description="eBest/LS Securities (KOSDAQ)",
    ),
}

# Placeholder patterns that indicate unconfigured keys
PLACEHOLDER_PATTERNS = [
    "your-",
    "your_",
    "xxx",
    "placeholder",
    "example",
    "test-key",
    "dummy",
    "changeme",
]


@dataclass
class KeyValidationResult:
    """Result of a single key validation."""

    key_name: str
    is_valid: bool
    error: Optional[str] = None
    warning: Optional[str] = None
    masked_value: Optional[str] = None


@dataclass
class ExchangeValidationResult:
    """Result of validating an exchange's keys."""

    exchange: str
    enabled: bool
    is_valid: bool
    key_results: List[KeyValidationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class StartupValidationResult:
    """Overall startup validation result."""

    is_valid: bool
    exchanges_validated: int = 0
    exchanges_valid: int = 0
    exchanges_invalid: int = 0
    exchange_results: List[ExchangeValidationResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        """Raise ConfigurationError if validation failed."""
        if not self.is_valid:
            error_msg = "API key validation failed:\n" + "\n".join(
                f"  - {err}" for err in self.errors
            )
            raise ConfigurationError(
                message=error_msg,
                error_code="API_KEY_VALIDATION_FAILED",
            )

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"API Key Validation: {'PASSED' if self.is_valid else 'FAILED'}",
            f"  Exchanges validated: {self.exchanges_validated}",
            f"  Valid: {self.exchanges_valid}",
            f"  Invalid: {self.exchanges_invalid}",
        ]

        if self.errors:
            lines.append("  Errors:")
            for err in self.errors:
                lines.append(f"    - {err}")

        if self.warnings:
            lines.append("  Warnings:")
            for warn in self.warnings:
                lines.append(f"    - {warn}")

        return "\n".join(lines)


def _mask_key(value: str) -> str:
    """Mask API key for logging."""
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _is_placeholder(value: str) -> bool:
    """Check if value appears to be a placeholder."""
    value_lower = value.lower()
    return any(pattern in value_lower for pattern in PLACEHOLDER_PATTERNS)


def _validate_key(
    key_name: str,
    value: Optional[str],
    required: bool = True,
    min_length: int = 16,
) -> KeyValidationResult:
    """Validate a single API key."""
    if value is None or value == "":
        if required:
            return KeyValidationResult(
                key_name=key_name,
                is_valid=False,
                error=f"{key_name} is not set",
            )
        return KeyValidationResult(
            key_name=key_name,
            is_valid=True,
            warning=f"{key_name} is not set (optional)",
        )

    # Check for placeholder
    if _is_placeholder(value):
        return KeyValidationResult(
            key_name=key_name,
            is_valid=False,
            error=f"{key_name} appears to be a placeholder value",
            masked_value=_mask_key(value),
        )

    # Check minimum length
    if len(value) < min_length:
        return KeyValidationResult(
            key_name=key_name,
            is_valid=False,
            error=f"{key_name} is too short (min {min_length} chars)",
            masked_value=_mask_key(value),
        )

    return KeyValidationResult(
        key_name=key_name,
        is_valid=True,
        masked_value=_mask_key(value),
    )


def _validate_exchange_keys(
    exchange: str,
    config: ExchangeKeyConfig,
) -> ExchangeValidationResult:
    """Validate API keys for a single exchange."""
    result = ExchangeValidationResult(
        exchange=exchange,
        enabled=True,
        is_valid=True,
    )

    # Validate required keys
    for key_name in config.required_env_vars:
        value = os.getenv(key_name)
        key_result = _validate_key(
            key_name=key_name,
            value=value,
            required=True,
            min_length=config.min_key_length,
        )
        result.key_results.append(key_result)

        if not key_result.is_valid:
            result.is_valid = False
            if key_result.error:
                result.errors.append(key_result.error)

    # Validate optional keys (warnings only)
    for key_name in config.optional_env_vars:
        value = os.getenv(key_name)
        key_result = _validate_key(
            key_name=key_name,
            value=value,
            required=False,
            min_length=config.min_key_length,
        )
        result.key_results.append(key_result)

        if key_result.warning:
            result.warnings.append(key_result.warning)

    return result


def get_enabled_exchanges(
    schedule_config_path: str = "config/schedule_config.json",
) -> Set[str]:
    """
    Get list of enabled exchanges from schedule config.

    Args:
        schedule_config_path: Path to schedule_config.json

    Returns:
        Set of enabled exchange names
    """
    config_path = Path(schedule_config_path)
    enabled = set()

    if not config_path.exists():
        logger.warning(
            "[StartupValidator] Schedule config not found: %s", schedule_config_path
        )
        return enabled

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        exchanges = config.get("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if exchange_config.get("enabled", False):
                enabled.add(exchange_name)

        logger.debug("[StartupValidator] Enabled exchanges: %s", enabled)
        return enabled

    except Exception as e:
        logger.error("[StartupValidator] Failed to read schedule config: %s", e)
        return enabled


def validate_api_keys(
    exchanges: Optional[Set[str]] = None,
    schedule_config_path: str = "config/schedule_config.json",
    raise_on_error: bool = False,
) -> StartupValidationResult:
    """
    Validate API keys for enabled exchanges.

    Args:
        exchanges: Set of exchange names to validate. If None, reads from schedule_config.
        schedule_config_path: Path to schedule_config.json
        raise_on_error: Raise ConfigurationError if validation fails

    Returns:
        StartupValidationResult with validation details

    Raises:
        ConfigurationError: If raise_on_error=True and validation fails
    """
    if exchanges is None:
        exchanges = get_enabled_exchanges(schedule_config_path)

    result = StartupValidationResult(is_valid=True)

    if not exchanges:
        logger.info("[StartupValidator] No exchanges enabled, skipping validation")
        return result

    logger.info("[StartupValidator] Validating API keys for: %s", exchanges)

    for exchange in exchanges:
        # Get key config for this exchange
        key_config = EXCHANGE_KEY_CONFIGS.get(exchange)

        if key_config is None:
            # Unknown exchange, warn but don't fail
            result.warnings.append(f"Unknown exchange: {exchange} (no key validation)")
            continue

        exchange_result = _validate_exchange_keys(exchange, key_config)
        result.exchange_results.append(exchange_result)
        result.exchanges_validated += 1

        if exchange_result.is_valid:
            result.exchanges_valid += 1
            logger.info(
                "[StartupValidator] ✓ %s (%s) - keys valid",
                exchange,
                key_config.description,
            )
        else:
            result.exchanges_invalid += 1
            result.is_valid = False
            for err in exchange_result.errors:
                result.errors.append(f"[{exchange}] {err}")
            logger.error(
                "[StartupValidator] ✗ %s (%s) - validation failed",
                exchange,
                key_config.description,
            )

        # Collect warnings
        for warn in exchange_result.warnings:
            result.warnings.append(f"[{exchange}] {warn}")

    # Log summary
    logger.info(
        "[StartupValidator] Validation complete: %d/%d exchanges valid",
        result.exchanges_valid,
        result.exchanges_validated,
    )

    if raise_on_error:
        result.raise_if_invalid()

    return result


def validate_startup(
    schedule_config_path: str = "config/schedule_config.json",
    strict: bool = False,
) -> bool:
    """
    Validate all startup requirements including API keys.

    Args:
        schedule_config_path: Path to schedule_config.json
        strict: If True, raise exception on validation failure

    Returns:
        True if validation passed, False otherwise

    Raises:
        ConfigurationError: If strict=True and validation fails
    """
    logger.info("[StartupValidator] Running startup validation...")

    # Validate API keys
    api_result = validate_api_keys(
        schedule_config_path=schedule_config_path,
        raise_on_error=strict,
    )

    if not api_result.is_valid:
        logger.error("[StartupValidator] Startup validation FAILED")
        logger.error(api_result.summary())
        return False

    # Log warnings
    for warning in api_result.warnings:
        logger.warning("[StartupValidator] %s", warning)

    logger.info("[StartupValidator] Startup validation PASSED")
    return True


def check_exchange_ready(exchange: str) -> bool:
    """
    Quick check if a specific exchange has valid API keys.

    Args:
        exchange: Exchange name (e.g., 'upbit', 'binance_spot')

    Returns:
        True if exchange keys are valid
    """
    key_config = EXCHANGE_KEY_CONFIGS.get(exchange)
    if key_config is None:
        return False

    result = _validate_exchange_keys(exchange, key_config)
    return result.is_valid


def get_missing_keys(exchange: str) -> List[str]:
    """
    Get list of missing/invalid keys for an exchange.

    Args:
        exchange: Exchange name

    Returns:
        List of missing key names
    """
    key_config = EXCHANGE_KEY_CONFIGS.get(exchange)
    if key_config is None:
        return []

    missing = []
    result = _validate_exchange_keys(exchange, key_config)

    for key_result in result.key_results:
        if not key_result.is_valid:
            missing.append(key_result.key_name)

    return missing
