"""
MASP Risk Management Module

Provides position sizing, drawdown protection, and risk metrics calculation.

Veto System (v2.0):
4-Layer hierarchical veto for trade filtering:
1. Kill Switch - Manual emergency stop
2. Market Structure - ADX, Choppiness Index
3. On-Chain - Exchange inflow analysis
4. Derivatives - Funding rate analysis

Stop Loss Manager:
Position-level exit logic:
- Fixed percentage stop loss / take profit
- Trailing stop
- ATR-based dynamic stop
- Time-based stop (maximum holding period)
"""

from libs.risk.drawdown_guard import DrawdownGuard, RiskState, RiskStatus
from libs.risk.position_sizer import (
    FixedFractionalSizer,
    KellyCriterionSizer,
    PositionSizer,
    VolatilityBasedSizer,
)
from libs.risk.stop_loss_manager import (
    ATRBasedStop,
    CompositeStopManager,
    ExitReason,
    ExitSignal,
    FixedPercentageStop,
    Position,
    StopLossStrategy,
    TimeBasedStop,
    TrailingStop,
    create_default_stop_manager,
)
from libs.risk.veto_manager import (
    VetoConfig,
    VetoLevel,
    VetoManager,
    VetoResult,
    calculate_adx,
    calculate_choppiness_index,
    calculate_funding_rate_signal,
)

__all__ = [
    # Position Sizing
    "PositionSizer",
    "FixedFractionalSizer",
    "KellyCriterionSizer",
    "VolatilityBasedSizer",
    # Drawdown Guard
    "DrawdownGuard",
    "RiskStatus",
    "RiskState",
    # Veto System (v2.0)
    "VetoManager",
    "VetoConfig",
    "VetoResult",
    "VetoLevel",
    "calculate_adx",
    "calculate_choppiness_index",
    "calculate_funding_rate_signal",
    # Stop Loss Manager
    "StopLossStrategy",
    "FixedPercentageStop",
    "TrailingStop",
    "ATRBasedStop",
    "TimeBasedStop",
    "CompositeStopManager",
    "ExitSignal",
    "ExitReason",
    "Position",
    "create_default_stop_manager",
]
