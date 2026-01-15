"""
Order Validator - Pre-trade validation logic

Ensures orders comply with risk limits and safety requirements.
"""

from dataclasses import dataclass
from typing import Optional
from libs.core.config import Config


@dataclass
class ValidationResult:
    """Order validation result"""
    valid: bool
    reason: Optional[str] = None


class OrderValidator:
    """
    Order validation logic.
    
    Validates orders against:
    - Kill-switch status
    - Position size limits
    - Maximum order value
    - Balance requirements
    """
    
    MAX_POSITION_PCT = 0.10  # 10% of total equity
    MAX_ORDER_VALUE_KRW = 10_000_000  # 10M KRW
    MIN_ORDER_VALUE_KRW = 5_000  # 5K KRW minimum
    
    def __init__(self, config: Config):
        """
        Initialize order validator.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def validate(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        balance: float,
        total_equity: float
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
        
        Returns:
            ValidationResult with validation status
        """
        # 1. Kill-Switch check
        if self.config.is_kill_switch_active():
            return ValidationResult(False, "Kill-Switch is active - all trading blocked")
        
        # 2. Order value check
        order_value = quantity * price
        
        if order_value < self.MIN_ORDER_VALUE_KRW:
            return ValidationResult(
                False,
                f"Order value too small: {order_value:,.0f} KRW (min: {self.MIN_ORDER_VALUE_KRW:,.0f})"
            )
        
        if order_value > self.MAX_ORDER_VALUE_KRW:
            return ValidationResult(
                False,
                f"Order exceeds max value: {order_value:,.0f} KRW (max: {self.MAX_ORDER_VALUE_KRW:,.0f})"
            )
        
        # 3. Position size limit (% of equity)
        max_position_value = total_equity * self.MAX_POSITION_PCT
        if order_value > max_position_value:
            return ValidationResult(
                False,
                f"Order exceeds {self.MAX_POSITION_PCT*100:.0f}% of equity: "
                f"{order_value:,.0f} > {max_position_value:,.0f}"
            )
        
        # 4. Balance check (BUY only)
        if side.upper() == "BUY":
            # Include estimated fee (0.05%)
            cost_with_fee = order_value * 1.0005
            if cost_with_fee > balance:
                return ValidationResult(
                    False,
                    f"Insufficient balance: need {cost_with_fee:,.0f}, have {balance:,.0f}"
                )
        
        # All checks passed
        return ValidationResult(True)
    
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
