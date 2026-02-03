"""
Simplified Backtest Engine

Provides basic backtesting functionality for strategy evaluation.
Phase 2B: Simplified version for quick validation.
Phase 3+: Enhanced with advanced features.
"""

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# [CRITICAL PATCH] Minimum sample size for statistical validity (same as performance.py)
MIN_SAMPLE_SIZE = 30


@dataclass
class BacktestResult:
    """Backtest execution result"""

    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    initial_capital: float
    final_capital: float


class BacktestEngine:
    """
    Simplified backtesting engine.

    Evaluates strategy performance on historical data.
    """

    def __init__(self, initial_capital: float = 10_000_000):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital (default: 10M KRW)
        """
        self.initial_capital = initial_capital
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = []

    def run_simple(self, signals: List[str], prices: List[float]) -> BacktestResult:
        """
        Run simplified backtest with signal/price lists.

        Args:
            signals: List of signals ("BUY", "SELL", "HOLD")
            prices: List of prices at each signal

        Returns:
            BacktestResult
        """
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0

        for i, (signal, price) in enumerate(zip(signals, prices)):
            if signal == "BUY" and position == 0:
                # Enter position (10% of capital)
                position_value = capital * 0.1
                position = position_value / price
                entry_price = price
                capital -= position_value

            elif signal == "SELL" and position > 0:
                # Exit position
                exit_value = position * price
                pnl = exit_value - (position * entry_price)
                capital += exit_value

                self.trades.append(
                    {
                        "entry_price": entry_price,
                        "exit_price": price,
                        "quantity": position,
                        "pnl": pnl,
                        "pnl_pct": (price - entry_price) / entry_price * 100,
                    }
                )

                position = 0.0

            # Record equity
            equity = capital + (position * price if position > 0 else 0)
            self.equity_curve.append(equity)

        # Calculate metrics
        return self._calculate_metrics()

    def _calculate_metrics(self) -> BacktestResult:
        """Calculate performance metrics"""
        if not self.trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0,
                total_pnl_pct=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                win_rate=0,
                profit_factor=0,
                avg_trade_pnl=0,
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
            )

        winning = [t for t in self.trades if t["pnl"] > 0]
        losing = [t for t in self.trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in self.trades)
        final_capital = (
            self.equity_curve[-1] if self.equity_curve else self.initial_capital
        )

        # Sharpe Ratio (annualized)
        # [CRITICAL PATCH] Minimum sample guard
        returns = [t["pnl_pct"] / 100 for t in self.trades]  # Convert to decimal

        if len(returns) < MIN_SAMPLE_SIZE:
            logger.warning(
                f"[BACKTEST] Sharpe calculation with {len(returns)} trades. "
                f"Minimum {MIN_SAMPLE_SIZE} recommended for statistical validity."
            )

        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            if std_return > 0:
                sharpe = (avg_return / std_return) * (252**0.5)  # Annualized
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max Drawdown
        peak = self.equity_curve[0]
        max_dd = 0
        max_dd_amount = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak if peak > 0 else 0
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_amount = dd

        # Profit Factor
        gross_profits = sum(t["pnl"] for t in winning)
        gross_losses = abs(sum(t["pnl"] for t in losing))
        profit_factor = (
            gross_profits / gross_losses if gross_losses > 0 else float("inf")
        )

        return BacktestResult(
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / self.initial_capital * 100,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd_amount,
            max_drawdown_pct=max_dd * 100,
            win_rate=len(winning) / len(self.trades) * 100 if self.trades else 0,
            profit_factor=profit_factor,
            avg_trade_pnl=total_pnl / len(self.trades),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
        )

    def reset(self):
        """Reset backtest state"""
        self.trades = []
        self.equity_curve = []
