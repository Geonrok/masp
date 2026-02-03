"""
Drawdown Guard for MASP

Monitors portfolio drawdown and enforces risk limits:
- Daily loss limit
- Weekly loss limit
- Maximum drawdown limit
- Automatic trading halt when limits breached
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Risk status levels."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    HALTED = "halted"


@dataclass
class RiskState:
    """Current risk state of the portfolio."""

    status: RiskStatus
    current_drawdown: float
    daily_pnl: float
    weekly_pnl: float
    message: str
    can_trade: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeRecord:
    """Record of a completed trade for P&L tracking."""

    symbol: str
    pnl: float
    timestamp: datetime
    side: str  # "buy" or "sell"


class DrawdownGuard:
    """
    Portfolio Drawdown Protection.

    Monitors:
    - Daily P&L against daily loss limit
    - Weekly P&L against weekly loss limit
    - Peak-to-trough drawdown against maximum drawdown limit

    Automatically halts trading when limits are breached.
    """

    def __init__(
        self,
        daily_loss_limit: float = 0.03,
        weekly_loss_limit: float = 0.07,
        max_drawdown_limit: float = 0.15,
        warning_threshold: float = 0.7,
        timezone: str = "Asia/Seoul",
    ):
        """
        Initialize Drawdown Guard.

        Args:
            daily_loss_limit: Maximum daily loss as fraction of capital (default 3%)
            weekly_loss_limit: Maximum weekly loss as fraction of capital (default 7%)
            max_drawdown_limit: Maximum drawdown from peak (default 15%)
            warning_threshold: Fraction of limit to trigger warning (default 70%)
            timezone: Timezone for daily/weekly reset
        """
        if not 0 < daily_loss_limit <= 0.2:
            raise ValueError("daily_loss_limit must be between 0 and 0.2")
        if not 0 < weekly_loss_limit <= 0.3:
            raise ValueError("weekly_loss_limit must be between 0 and 0.3")
        if not 0 < max_drawdown_limit <= 0.5:
            raise ValueError("max_drawdown_limit must be between 0 and 0.5")

        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.warning_threshold = warning_threshold
        self.tz = ZoneInfo(timezone)

        # State tracking
        self.peak_capital: float = 0.0
        self.current_capital: float = 0.0
        self.trade_history: List[TradeRecord] = []
        self.is_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.halt_until: Optional[datetime] = None

        logger.info(
            f"[DrawdownGuard] Initialized: daily={daily_loss_limit:.1%}, "
            f"weekly={weekly_loss_limit:.1%}, max_dd={max_drawdown_limit:.1%}"
        )

    def initialize(self, starting_capital: float) -> None:
        """Initialize with starting capital."""
        if starting_capital <= 0:
            raise ValueError("Starting capital must be positive")
        self.peak_capital = starting_capital
        self.current_capital = starting_capital
        self.trade_history.clear()
        self.is_halted = False
        self.halt_reason = None
        logger.info(
            f"[DrawdownGuard] Initialized with capital: {starting_capital:,.0f}"
        )

    def record_trade(self, symbol: str, pnl: float, side: str = "buy") -> None:
        """Record a completed trade."""
        record = TradeRecord(
            symbol=symbol,
            pnl=pnl,
            timestamp=datetime.now(self.tz),
            side=side,
        )
        self.trade_history.append(record)
        self.current_capital += pnl

        # Update peak if new high
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        logger.debug(f"[DrawdownGuard] Trade recorded: {symbol} P&L={pnl:+,.0f}")

    def _get_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        now = datetime.now(self.tz)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        daily_pnl = sum(t.pnl for t in self.trade_history if t.timestamp >= today_start)
        return daily_pnl

    def _get_weekly_pnl(self) -> float:
        """Calculate this week's P&L (Monday to now)."""
        now = datetime.now(self.tz)
        # Get Monday of current week
        days_since_monday = now.weekday()
        week_start = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        weekly_pnl = sum(t.pnl for t in self.trade_history if t.timestamp >= week_start)
        return weekly_pnl

    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def check_risk(self) -> RiskState:
        """
        Check current risk state.

        Returns:
            RiskState with current status and whether trading is allowed
        """
        # Check if halt period has expired
        if self.is_halted and self.halt_until:
            if datetime.now(self.tz) >= self.halt_until:
                self.is_halted = False
                self.halt_reason = None
                self.halt_until = None
                logger.info("[DrawdownGuard] Halt period expired, trading resumed")

        # If still halted, return halted state
        if self.is_halted:
            return RiskState(
                status=RiskStatus.HALTED,
                current_drawdown=self._get_current_drawdown(),
                daily_pnl=self._get_daily_pnl(),
                weekly_pnl=self._get_weekly_pnl(),
                message=f"Trading halted: {self.halt_reason}",
                can_trade=False,
            )

        # Calculate metrics
        daily_pnl = self._get_daily_pnl()
        weekly_pnl = self._get_weekly_pnl()
        current_dd = self._get_current_drawdown()

        # Calculate limits in absolute terms
        daily_limit_abs = self.peak_capital * self.daily_loss_limit
        weekly_limit_abs = self.peak_capital * self.weekly_loss_limit

        # Check daily loss limit
        if daily_pnl < 0 and abs(daily_pnl) >= daily_limit_abs:
            self._halt_trading(
                reason=f"Daily loss limit breached ({abs(daily_pnl):,.0f} >= {daily_limit_abs:,.0f})",
                duration_hours=24,
            )
            return RiskState(
                status=RiskStatus.HALTED,
                current_drawdown=current_dd,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                message=f"Daily loss limit breached: {abs(daily_pnl):,.0f}",
                can_trade=False,
            )

        # Check weekly loss limit
        if weekly_pnl < 0 and abs(weekly_pnl) >= weekly_limit_abs:
            self._halt_trading(
                reason=f"Weekly loss limit breached ({abs(weekly_pnl):,.0f} >= {weekly_limit_abs:,.0f})",
                duration_hours=168,  # Until next Monday
            )
            return RiskState(
                status=RiskStatus.HALTED,
                current_drawdown=current_dd,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                message=f"Weekly loss limit breached: {abs(weekly_pnl):,.0f}",
                can_trade=False,
            )

        # Check max drawdown
        if current_dd >= self.max_drawdown_limit:
            self._halt_trading(
                reason=f"Max drawdown breached ({current_dd:.1%} >= {self.max_drawdown_limit:.1%})",
                duration_hours=None,  # Manual reset required
            )
            return RiskState(
                status=RiskStatus.HALTED,
                current_drawdown=current_dd,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                message=f"Max drawdown breached: {current_dd:.1%}",
                can_trade=False,
            )

        # Check warning thresholds
        daily_warning = (
            daily_pnl < 0 and abs(daily_pnl) >= daily_limit_abs * self.warning_threshold
        )
        weekly_warning = (
            weekly_pnl < 0
            and abs(weekly_pnl) >= weekly_limit_abs * self.warning_threshold
        )
        dd_warning = current_dd >= self.max_drawdown_limit * self.warning_threshold

        if daily_warning or weekly_warning or dd_warning:
            warnings = []
            if daily_warning:
                warnings.append(f"daily loss at {abs(daily_pnl)/daily_limit_abs:.0%}")
            if weekly_warning:
                warnings.append(
                    f"weekly loss at {abs(weekly_pnl)/weekly_limit_abs:.0%}"
                )
            if dd_warning:
                warnings.append(f"drawdown at {current_dd/self.max_drawdown_limit:.0%}")

            return RiskState(
                status=RiskStatus.WARNING,
                current_drawdown=current_dd,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                message=f"Warning: {', '.join(warnings)}",
                can_trade=True,
            )

        # All clear
        return RiskState(
            status=RiskStatus.NORMAL,
            current_drawdown=current_dd,
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            message="All risk metrics within limits",
            can_trade=True,
        )

    def _halt_trading(self, reason: str, duration_hours: Optional[int]) -> None:
        """Halt trading for specified duration."""
        self.is_halted = True
        self.halt_reason = reason
        if duration_hours:
            self.halt_until = datetime.now(self.tz) + timedelta(hours=duration_hours)
        else:
            self.halt_until = None  # Manual reset required
        logger.warning(f"[DrawdownGuard] TRADING HALTED: {reason}")

    def reset_halt(self) -> None:
        """Manually reset trading halt (use with caution)."""
        if self.is_halted:
            logger.warning("[DrawdownGuard] Manual halt reset - use with caution!")
            self.is_halted = False
            self.halt_reason = None
            self.halt_until = None

    def can_trade(self) -> bool:
        """Quick check if trading is allowed."""
        return self.check_risk().can_trade

    def get_metrics(self) -> dict:
        """Get current risk metrics as dictionary."""
        state = self.check_risk()
        return {
            "status": state.status.value,
            "can_trade": state.can_trade,
            "current_drawdown": state.current_drawdown,
            "daily_pnl": state.daily_pnl,
            "weekly_pnl": state.weekly_pnl,
            "peak_capital": self.peak_capital,
            "current_capital": self.current_capital,
            "daily_limit": self.daily_loss_limit,
            "weekly_limit": self.weekly_loss_limit,
            "max_drawdown_limit": self.max_drawdown_limit,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
            "message": state.message,
        }
