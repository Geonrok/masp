"""
Circuit Breaker for Automatic Trading Halt.

Monitors portfolio drawdown and automatically halts trading when
risk thresholds are exceeded.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


class TriggerReason(Enum):
    """Reasons for circuit breaker trigger."""

    DRAWDOWN = "drawdown"  # Max drawdown exceeded
    DAILY_LOSS = "daily_loss"  # Daily loss limit
    CONSECUTIVE_LOSSES = "consecutive_losses"  # Too many losses in a row
    VOLATILITY = "volatility"  # Extreme volatility
    MANUAL = "manual"  # Manual trigger
    SYSTEM_ERROR = "system_error"  # System errors


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    All percentages are in decimal form (e.g., 0.10 = 10%).
    """

    # Drawdown triggers
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    warning_drawdown_pct: float = 0.05  # 5% warning threshold

    # Daily loss triggers
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    warning_daily_loss_pct: float = 0.02  # 2% warning

    # Consecutive losses
    max_consecutive_losses: int = 5  # Max losses in a row
    warning_consecutive_losses: int = 3  # Warning threshold

    # Recovery settings
    cooldown_minutes: int = 60  # Time before attempting recovery
    recovery_check_minutes: int = 15  # How often to check for recovery
    auto_recover: bool = False  # Auto-recover when conditions improve

    # Kill switch integration
    kill_switch_file: Optional[str] = None  # Path to kill switch file
    create_kill_switch_on_trigger: bool = True  # Create file when triggered

    # Notifications
    notify_on_warning: bool = True
    notify_on_trigger: bool = True
    notification_callback: Optional[Callable[[str, dict], None]] = None

    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        """Load config from environment variables."""
        return cls(
            max_drawdown_pct=float(os.getenv("MASP_MAX_DRAWDOWN_PCT", "0.10")),
            warning_drawdown_pct=float(os.getenv("MASP_WARNING_DRAWDOWN_PCT", "0.05")),
            max_daily_loss_pct=float(os.getenv("MASP_MAX_DAILY_LOSS_PCT", "0.03")),
            warning_daily_loss_pct=float(
                os.getenv("MASP_WARNING_DAILY_LOSS_PCT", "0.02")
            ),
            max_consecutive_losses=int(os.getenv("MASP_MAX_CONSECUTIVE_LOSSES", "5")),
            cooldown_minutes=int(os.getenv("MASP_CIRCUIT_COOLDOWN_MIN", "60")),
            recovery_check_minutes=int(os.getenv("MASP_RECOVERY_CHECK_MIN", "15")),
            auto_recover=os.getenv("MASP_AUTO_RECOVER", "0") == "1",
            kill_switch_file=os.getenv("MASP_KILL_SWITCH_FILE"),
            create_kill_switch_on_trigger=os.getenv("MASP_CREATE_KILL_SWITCH", "1")
            == "1",
        )


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    trigger_reason: Optional[TriggerReason] = None
    trigger_time: Optional[datetime] = None
    trigger_value: Optional[float] = None

    # Tracking metrics
    peak_equity: float = 0.0
    current_equity: float = 0.0
    drawdown_pct: float = 0.0

    daily_start_equity: float = 0.0
    daily_loss_pct: float = 0.0

    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # History
    trigger_history: List[dict] = field(default_factory=list)
    last_check_time: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker for automatic trading halt.

    Monitors portfolio metrics and automatically halts trading when
    risk thresholds are exceeded.

    Example:
        config = CircuitBreakerConfig(max_drawdown_pct=0.10)
        breaker = CircuitBreaker(config)

        # Update with new equity
        breaker.update_equity(1_000_000)
        breaker.update_equity(950_000)  # 5% drawdown - warning

        # Check if trading is allowed
        if breaker.is_trading_allowed():
            # Execute trade
            ...

        # Record trade result
        breaker.record_trade(profit=-10_000)
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        initial_equity: float = 0.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
            initial_equity: Initial account equity
        """
        self.config = config or CircuitBreakerConfig.from_env()
        self.state = CircuitBreakerState()
        self._lock = threading.Lock()

        if initial_equity > 0:
            self.state.peak_equity = initial_equity
            self.state.current_equity = initial_equity
            self.state.daily_start_equity = initial_equity

        logger.info(
            "[CircuitBreaker] Initialized: max_drawdown=%.1f%%, max_daily_loss=%.1f%%",
            self.config.max_drawdown_pct * 100,
            self.config.max_daily_loss_pct * 100,
        )

    def update_equity(self, equity: float) -> CircuitState:
        """
        Update current equity and check thresholds.

        Args:
            equity: Current account equity

        Returns:
            Current circuit state
        """
        with self._lock:
            self.state.current_equity = equity

            # Update peak
            if equity > self.state.peak_equity:
                self.state.peak_equity = equity

            # Calculate drawdown
            if self.state.peak_equity > 0:
                drawdown = self.state.peak_equity - equity
                self.state.drawdown_pct = drawdown / self.state.peak_equity
            else:
                self.state.drawdown_pct = 0.0

            # Calculate daily loss
            if self.state.daily_start_equity > 0:
                daily_loss = self.state.daily_start_equity - equity
                self.state.daily_loss_pct = max(
                    0, daily_loss / self.state.daily_start_equity
                )
            else:
                self.state.daily_loss_pct = 0.0

            self.state.last_check_time = datetime.now(timezone.utc)

            # Check thresholds
            self._check_thresholds()

            return self.state.state

    def record_trade(self, profit: float) -> None:
        """
        Record a trade result.

        Args:
            profit: Trade profit (negative for loss)
        """
        with self._lock:
            self.state.total_trades += 1

            if profit >= 0:
                self.state.winning_trades += 1
                self.state.consecutive_losses = 0
            else:
                self.state.losing_trades += 1
                self.state.consecutive_losses += 1

                # Check consecutive losses
                if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                    self._trigger(
                        TriggerReason.CONSECUTIVE_LOSSES,
                        self.state.consecutive_losses,
                    )

    def _check_thresholds(self) -> None:
        """Check all thresholds and trigger if needed."""
        # Skip if already triggered
        if self.state.state == CircuitState.OPEN:
            return

        # Drawdown check
        if self.state.drawdown_pct >= self.config.max_drawdown_pct:
            self._trigger(TriggerReason.DRAWDOWN, self.state.drawdown_pct)
            return

        if (
            self.state.drawdown_pct >= self.config.warning_drawdown_pct
            and self.config.notify_on_warning
        ):
            self._warn("drawdown", self.state.drawdown_pct)

        # Daily loss check
        if self.state.daily_loss_pct >= self.config.max_daily_loss_pct:
            self._trigger(TriggerReason.DAILY_LOSS, self.state.daily_loss_pct)
            return

        if (
            self.state.daily_loss_pct >= self.config.warning_daily_loss_pct
            and self.config.notify_on_warning
        ):
            self._warn("daily_loss", self.state.daily_loss_pct)

    def _trigger(self, reason: TriggerReason, value: float) -> None:
        """Trigger the circuit breaker."""
        self.state.state = CircuitState.OPEN
        self.state.trigger_reason = reason
        self.state.trigger_time = datetime.now(timezone.utc)
        self.state.trigger_value = value

        trigger_info = {
            "reason": reason.value,
            "value": value,
            "time": self.state.trigger_time.isoformat(),
            "equity": self.state.current_equity,
            "drawdown_pct": self.state.drawdown_pct,
            "daily_loss_pct": self.state.daily_loss_pct,
        }
        self.state.trigger_history.append(trigger_info)

        logger.critical(
            "[CircuitBreaker] TRIGGERED: %s=%.2f%% - Trading halted",
            reason.value,
            value * 100,
        )

        # Create kill switch file
        if self.config.create_kill_switch_on_trigger and self.config.kill_switch_file:
            self._create_kill_switch()

        # Send notification
        if self.config.notify_on_trigger and self.config.notification_callback:
            try:
                self.config.notification_callback(
                    "circuit_breaker_triggered", trigger_info
                )
            except Exception as e:
                logger.error("[CircuitBreaker] Notification failed: %s", e)

    def _warn(self, metric: str, value: float) -> None:
        """Send warning notification."""
        logger.warning(
            "[CircuitBreaker] WARNING: %s=%.2f%% approaching limit",
            metric,
            value * 100,
        )

        if self.config.notification_callback:
            try:
                self.config.notification_callback(
                    "circuit_breaker_warning",
                    {"metric": metric, "value": value},
                )
            except Exception as e:
                logger.error("[CircuitBreaker] Warning notification failed: %s", e)

    def _create_kill_switch(self) -> None:
        """Create kill switch file."""
        if not self.config.kill_switch_file:
            return

        try:
            path = Path(self.config.kill_switch_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"CIRCUIT_BREAKER_TRIGGERED\n"
                f"Time: {self.state.trigger_time.isoformat()}\n"
                f"Reason: {self.state.trigger_reason.value}\n"
                f"Value: {self.state.trigger_value}\n"
            )
            logger.info("[CircuitBreaker] Kill switch file created: %s", path)
        except Exception as e:
            logger.error("[CircuitBreaker] Failed to create kill switch: %s", e)

    def is_trading_allowed(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed, False if circuit is open
        """
        return self.state.state != CircuitState.OPEN

    def reset(self, force: bool = False) -> bool:
        """
        Reset the circuit breaker.

        Args:
            force: Force reset even if conditions aren't met

        Returns:
            True if reset was successful
        """
        with self._lock:
            if self.state.state == CircuitState.CLOSED:
                return True

            if not force:
                # Check if cooldown has passed
                if self.state.trigger_time:
                    elapsed = datetime.now(timezone.utc) - self.state.trigger_time
                    if elapsed < timedelta(minutes=self.config.cooldown_minutes):
                        logger.warning(
                            "[CircuitBreaker] Cooldown not complete: %s remaining",
                            timedelta(minutes=self.config.cooldown_minutes) - elapsed,
                        )
                        return False

            self.state.state = CircuitState.CLOSED
            self.state.trigger_reason = None
            self.state.consecutive_losses = 0

            # Remove kill switch file
            if self.config.kill_switch_file:
                try:
                    path = Path(self.config.kill_switch_file)
                    if path.exists():
                        path.unlink()
                        logger.info("[CircuitBreaker] Kill switch file removed")
                except Exception as e:
                    logger.error("[CircuitBreaker] Failed to remove kill switch: %s", e)

            logger.info("[CircuitBreaker] Reset - Trading resumed")
            return True

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        with self._lock:
            self.state.daily_start_equity = self.state.current_equity
            self.state.daily_loss_pct = 0.0
            logger.info("[CircuitBreaker] Daily counters reset")

    def manual_trigger(self, reason: str = "Manual trigger") -> None:
        """
        Manually trigger the circuit breaker.

        Args:
            reason: Reason for manual trigger
        """
        with self._lock:
            self._trigger(TriggerReason.MANUAL, 0.0)
            logger.warning("[CircuitBreaker] Manual trigger: %s", reason)

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "state": self.state.state.value,
            "is_trading_allowed": self.is_trading_allowed(),
            "trigger_reason": (
                self.state.trigger_reason.value if self.state.trigger_reason else None
            ),
            "trigger_time": (
                self.state.trigger_time.isoformat() if self.state.trigger_time else None
            ),
            "metrics": {
                "current_equity": self.state.current_equity,
                "peak_equity": self.state.peak_equity,
                "drawdown_pct": self.state.drawdown_pct,
                "daily_loss_pct": self.state.daily_loss_pct,
                "consecutive_losses": self.state.consecutive_losses,
            },
            "limits": {
                "max_drawdown_pct": self.config.max_drawdown_pct,
                "max_daily_loss_pct": self.config.max_daily_loss_pct,
                "max_consecutive_losses": self.config.max_consecutive_losses,
            },
            "trade_stats": {
                "total": self.state.total_trades,
                "wins": self.state.winning_trades,
                "losses": self.state.losing_trades,
            },
            "trigger_history_count": len(self.state.trigger_history),
        }


def create_circuit_breaker(
    initial_equity: float = 0.0,
    max_drawdown_pct: float = 0.10,
    max_daily_loss_pct: float = 0.03,
) -> CircuitBreaker:
    """
    Convenience function to create a circuit breaker.

    Args:
        initial_equity: Initial account equity
        max_drawdown_pct: Maximum drawdown percentage (default 10%)
        max_daily_loss_pct: Maximum daily loss percentage (default 3%)

    Returns:
        Configured CircuitBreaker instance
    """
    config = CircuitBreakerConfig(
        max_drawdown_pct=max_drawdown_pct,
        max_daily_loss_pct=max_daily_loss_pct,
    )
    return CircuitBreaker(config, initial_equity)
