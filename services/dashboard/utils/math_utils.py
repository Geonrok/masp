"""Safe math utilities for dashboard components."""

from __future__ import annotations

import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return default
        return float(value)
    try:
        result = float(str(value).strip())
        if not math.isfinite(result):
            return default
        return result
    except (ValueError, TypeError):
        return default


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    result = numerator / denominator
    if not math.isfinite(result):
        return default
    return result


def safe_sqrt(value: float, default: float = 0.0) -> float:
    """
    Safely compute square root.

    Args:
        value: Value to compute sqrt of
        default: Default value if sqrt fails

    Returns:
        Square root or default
    """
    if value < 0 or not math.isfinite(value):
        return default
    result = math.sqrt(value)
    if not math.isfinite(result):
        return default
    return result


def safe_percent(value: float, total: float, default: float = 0.0) -> float:
    """
    Safely compute percentage.

    Args:
        value: Value
        total: Total value
        default: Default value if computation fails

    Returns:
        Percentage or default
    """
    return safe_div(value * 100, total, default)


def format_krw(value: float, include_sign: bool = False) -> str:
    """
    Format value as KRW currency string.

    Args:
        value: Value to format
        include_sign: Whether to include +/- sign

    Returns:
        Formatted string
    """
    safe_value = safe_float(value)
    if include_sign:
        return f"₩{safe_value:+,.0f}"
    return f"₩{safe_value:,.0f}"


def format_percent(value: float, decimals: int = 2, include_sign: bool = False) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format (as decimal, e.g., 0.05 for 5%)
        decimals: Number of decimal places
        include_sign: Whether to include +/- sign

    Returns:
        Formatted percentage string
    """
    safe_value = safe_float(value) * 100
    if include_sign:
        return f"{safe_value:+.{decimals}f}%"
    return f"{safe_value:.{decimals}f}%"
