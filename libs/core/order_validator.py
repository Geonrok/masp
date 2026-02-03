"""
Order Validator - Pre-trade validation logic

Ensures orders comply with risk limits and safety requirements.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from libs.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Order validation result."""

    valid: bool
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PositionLimits:
    """
    Configurable position limits.

    Supports both KRW (Korean Won) and USDT for different markets.
    """

    # Per-position limits
    max_position_pct: float = 0.10  # 10% of total equity
    max_position_value_krw: float = 10_000_000  # 10M KRW
    max_position_value_usdt: float = 10_000  # 10K USDT

    # Order limits
    min_order_value_krw: float = 5_000  # 5K KRW
    min_order_value_usdt: float = 5  # 5 USDT
    max_order_value_krw: float = 10_000_000  # 10M KRW
    max_order_value_usdt: float = 10_000  # 10K USDT

    # Portfolio limits
    max_total_positions: int = 20  # Maximum number of open positions
    max_concentration_pct: float = 0.50  # Max 50% in single asset

    # Daily limits
    max_daily_orders: int = 100  # Maximum orders per day
    max_daily_volume_krw: float = 100_000_000  # 100M KRW daily
    max_daily_volume_usdt: float = 100_000  # 100K USDT daily

    # Leverage limits (for futures)
    max_leverage: int = 10  # Maximum leverage
    default_leverage: int = 1  # Default leverage

    @classmethod
    def from_env(cls) -> "PositionLimits":
        """
        Load limits from environment variables.

        Environment variables:
            MASP_MAX_POSITION_PCT: Max position size as percentage
            MASP_MAX_POSITION_VALUE_KRW: Max position value in KRW
            MASP_MAX_POSITION_VALUE_USDT: Max position value in USDT
            MASP_MAX_TOTAL_POSITIONS: Max number of positions
            MASP_MAX_CONCENTRATION_PCT: Max concentration in single asset
            MASP_MAX_DAILY_ORDERS: Max orders per day
            MASP_MAX_LEVERAGE: Max leverage for futures
        """
        return cls(
            max_position_pct=float(os.getenv("MASP_MAX_POSITION_PCT", "0.10")),
            max_position_value_krw=float(
                os.getenv("MASP_MAX_POSITION_VALUE_KRW", "10000000")
            ),
            max_position_value_usdt=float(
                os.getenv("MASP_MAX_POSITION_VALUE_USDT", "10000")
            ),
            min_order_value_krw=float(os.getenv("MASP_MIN_ORDER_VALUE_KRW", "5000")),
            min_order_value_usdt=float(os.getenv("MASP_MIN_ORDER_VALUE_USDT", "5")),
            max_order_value_krw=float(
                os.getenv("MASP_MAX_ORDER_VALUE_KRW", "10000000")
            ),
            max_order_value_usdt=float(os.getenv("MASP_MAX_ORDER_VALUE_USDT", "10000")),
            max_total_positions=int(os.getenv("MASP_MAX_TOTAL_POSITIONS", "20")),
            max_concentration_pct=float(
                os.getenv("MASP_MAX_CONCENTRATION_PCT", "0.50")
            ),
            max_daily_orders=int(os.getenv("MASP_MAX_DAILY_ORDERS", "100")),
            max_daily_volume_krw=float(
                os.getenv("MASP_MAX_DAILY_VOLUME_KRW", "100000000")
            ),
            max_daily_volume_usdt=float(
                os.getenv("MASP_MAX_DAILY_VOLUME_USDT", "100000")
            ),
            max_leverage=int(os.getenv("MASP_MAX_LEVERAGE", "10")),
            default_leverage=int(os.getenv("MASP_DEFAULT_LEVERAGE", "1")),
        )


class OrderValidator:
    """
    Order validation logic.

    Validates orders against:
    - Kill-switch status
    - Position size limits (configurable)
    - Maximum order value
    - Balance requirements
    - Portfolio concentration
    - Daily limits

    Example:
        limits = PositionLimits.from_env()
        validator = OrderValidator(config, limits=limits)
        result = validator.validate(symbol, "BUY", 0.1, 50000000, balance, equity)
    """

    # Legacy class constants for backward compatibility
    MAX_POSITION_PCT = 0.10
    MAX_ORDER_VALUE_KRW = 10_000_000
    MIN_ORDER_VALUE_KRW = 5_000

    def __init__(
        self,
        config: Config,
        limits: Optional[PositionLimits] = None,
        quote_currency: str = "KRW",
    ):
        """
        Initialize order validator.

        Args:
            config: Configuration object
            limits: Position limits (uses defaults if not provided)
            quote_currency: "KRW" or "USDT"
        """
        self.config = config
        self.limits = limits or PositionLimits.from_env()
        self.quote_currency = quote_currency.upper()

        # Daily tracking
        self._daily_order_count = 0
        self._daily_volume = 0.0
        self._current_positions: Dict[str, float] = {}

        logger.info(
            "[OrderValidator] Initialized: max_position=%.1f%%, max_value=%s %s",
            self.limits.max_position_pct * 100,
            (
                f"{self.limits.max_position_value_krw:,.0f}"
                if self.quote_currency == "KRW"
                else f"{self.limits.max_position_value_usdt:,.0f}"
            ),
            self.quote_currency,
        )

    def validate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        balance: float,
        total_equity: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate an order.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Order price (current or limit)
            balance: Current balance
            total_equity: Total account equity
            current_positions: Optional dict of current positions {symbol: value}

        Returns:
            ValidationResult with validation status
        """
        warnings = []

        # 1. Kill-Switch check
        if self.config.is_kill_switch_active():
            return ValidationResult(
                False, "Kill-Switch is active - all trading blocked"
            )

        # 2. Order value check
        order_value = quantity * price

        # Get appropriate limits based on quote currency
        is_usdt = self.quote_currency == "USDT"
        min_order = (
            self.limits.min_order_value_usdt
            if is_usdt
            else self.limits.min_order_value_krw
        )
        max_order = (
            self.limits.max_order_value_usdt
            if is_usdt
            else self.limits.max_order_value_krw
        )
        max_position = (
            self.limits.max_position_value_usdt
            if is_usdt
            else self.limits.max_position_value_krw
        )
        max_daily_volume = (
            self.limits.max_daily_volume_usdt
            if is_usdt
            else self.limits.max_daily_volume_krw
        )
        currency = self.quote_currency

        if order_value < min_order:
            return ValidationResult(
                False,
                f"Order value too small: {order_value:,.0f} {currency} (min: {min_order:,.0f})",
            )

        if order_value > max_order:
            return ValidationResult(
                False,
                f"Order exceeds max value: {order_value:,.0f} {currency} (max: {max_order:,.0f})",
            )

        # 3. Position size limit (% of equity)
        # Check cumulative position (existing + new order) against limit
        max_position_by_pct = total_equity * self.limits.max_position_pct
        effective_max = min(max_position_by_pct, max_position)

        existing_value = 0.0
        if current_positions and side.upper() == "BUY":
            existing_value = current_positions.get(symbol, 0.0)

        total_position_value = existing_value + order_value

        if total_position_value > effective_max:
            return ValidationResult(
                False,
                f"Order exceeds position limit: {total_position_value:,.0f} > {effective_max:,.0f} {currency} "
                f"({self.limits.max_position_pct*100:.0f}% of equity, "
                f"existing: {existing_value:,.0f}, order: {order_value:,.0f})",
            )

        # 4. Portfolio concentration check
        if current_positions and side.upper() == "BUY":
            existing_value = current_positions.get(symbol, 0)
            new_total = existing_value + order_value
            max_concentration = total_equity * self.limits.max_concentration_pct

            if new_total > max_concentration:
                return ValidationResult(
                    False,
                    f"Position would exceed concentration limit: "
                    f"{new_total:,.0f} > {max_concentration:,.0f} {currency} "
                    f"({self.limits.max_concentration_pct*100:.0f}%)",
                )

        # 5. Total positions check
        if current_positions and side.upper() == "BUY":
            if symbol not in current_positions:
                if len(current_positions) >= self.limits.max_total_positions:
                    return ValidationResult(
                        False,
                        f"Maximum positions reached: {len(current_positions)} "
                        f"(max: {self.limits.max_total_positions})",
                    )

        # 6. Daily order count check
        if self._daily_order_count >= self.limits.max_daily_orders:
            return ValidationResult(
                False,
                f"Daily order limit reached: {self._daily_order_count} "
                f"(max: {self.limits.max_daily_orders})",
            )

        # 7. Daily volume check
        if self._daily_volume + order_value > max_daily_volume:
            return ValidationResult(
                False,
                f"Daily volume limit would be exceeded: "
                f"{self._daily_volume + order_value:,.0f} > {max_daily_volume:,.0f} {currency}",
            )

        # 8. Balance check (BUY only)
        if side.upper() == "BUY":
            # Include estimated fee (0.05%)
            cost_with_fee = order_value * 1.0005
            if cost_with_fee > balance:
                return ValidationResult(
                    False,
                    f"Insufficient balance: need {cost_with_fee:,.0f}, have {balance:,.0f}",
                )

        # Warnings
        if order_value > max_position_by_pct * 0.8:
            warnings.append(
                f"Order is close to position limit ({order_value/max_position_by_pct*100:.0f}%)"
            )

        if self._daily_order_count > self.limits.max_daily_orders * 0.8:
            warnings.append(
                f"Approaching daily order limit ({self._daily_order_count}/{self.limits.max_daily_orders})"
            )

        # All checks passed
        return ValidationResult(True, warnings=warnings)

    def record_order(self, order_value: float) -> None:
        """
        Record an executed order for daily tracking.

        Args:
            order_value: Value of the executed order
        """
        self._daily_order_count += 1
        self._daily_volume += order_value

    def reset_daily_counters(self) -> None:
        """Reset daily tracking counters (call at start of each trading day)."""
        self._daily_order_count = 0
        self._daily_volume = 0.0
        logger.info("[OrderValidator] Daily counters reset")

    def update_positions(self, positions: Dict[str, float]) -> None:
        """
        Update current positions for tracking.

        Args:
            positions: Dict of symbol -> position value
        """
        self._current_positions = positions.copy()

    def get_stats(self) -> Dict:
        """Get current validator statistics."""
        return {
            "daily_order_count": self._daily_order_count,
            "daily_volume": self._daily_volume,
            "position_count": len(self._current_positions),
            "limits": {
                "max_position_pct": self.limits.max_position_pct,
                "max_total_positions": self.limits.max_total_positions,
                "max_daily_orders": self.limits.max_daily_orders,
                "quote_currency": self.quote_currency,
            },
        }

    def validate_quick(self, kill_switch_only: bool = False) -> ValidationResult:
        """
        Quick validation (kill-switch check only).

        Args:
            kill_switch_only: If True, only check kill-switch

        Returns:
            ValidationResult
        """
        if self.config.is_kill_switch_active():
            return ValidationResult(False, "Kill-Switch is active")

        return ValidationResult(True)
