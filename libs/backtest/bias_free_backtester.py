"""
Bias-Free Backtester

Core principle: Signal at Day T close -> Execute at Day T+1 open
Eliminates look-ahead bias that inflated original results.

Original Problem:
- EXIT days: 0% return (avoided losses)
- ENTRY days: Full return counted (captured gains)
- Result: +133% Upbit was actually ~-70%

Solution:
- Day T close: Generate signal
- Day T+1 open: Execute trade
- Day T+1 close: Calculate return
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Trade execution configuration."""

    slippage_pct: float = 0.005  # 0.5% default slippage
    commission_pct: float = 0.001  # 0.1% per trade
    max_positions: int = 20
    position_sizing: str = "equal_weight"  # equal_weight, vol_weighted, etc.

    def get_total_cost(self) -> float:
        """Total cost per trade (entry + exit)."""
        return 2 * (self.slippage_pct + self.commission_pct)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float = 10000.0
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Strategy parameters
    kama_period: int = 5
    tsmom_period: int = 90
    gate_period: int = 30  # BTC MA period for market filter


@dataclass
class BacktestMetrics:
    """Performance metrics from backtest."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_return: float = 0.0

    # Exposure
    trading_days: int = 0
    invested_days: int = 0
    exposure_ratio: float = 0.0

    # Benchmark comparison
    benchmark_return: float = 0.0
    alpha: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_trade_return": self.avg_trade_return,
            "trading_days": self.trading_days,
            "invested_days": self.invested_days,
            "exposure_ratio": self.exposure_ratio,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
        }


class BiasFreeBacktester:
    """
    Bias-Free Backtester with strict signal-to-execution separation.

    Key Rules:
    1. Signal generated at Day T close
    2. Trade executed at Day T+1 open (or close with slippage)
    3. Return calculated from actual execution price
    4. No future information leakage
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.portfolio_values: List[float] = []
        self.daily_returns: List[float] = []
        self.positions_log: List[Dict] = []
        self.trade_log: List[Dict] = []

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        signal_func: Optional[Callable] = None,
    ) -> BacktestMetrics:
        """
        Run bias-free backtest.

        Args:
            data: Dict of symbol -> DataFrame with columns [date, open, high, low, close, volume]
            signal_func: Optional custom signal function(df) -> Series[bool]

        Returns:
            BacktestMetrics
        """
        logger.info("[BiasFreeBacktester] Starting backtest with %d symbols", len(data))

        # Prepare data
        prepared = self._prepare_data(data)
        if not prepared:
            logger.error("[BiasFreeBacktester] No valid data after preparation")
            return BacktestMetrics()

        # Generate signals for each symbol
        signal_data = self._generate_signals(prepared, signal_func)
        if not signal_data:
            logger.error("[BiasFreeBacktester] No signals generated")
            return BacktestMetrics()

        # Run simulation
        metrics = self._simulate(signal_data)

        logger.info(
            "[BiasFreeBacktester] Completed: Return=%.1f%%, Sharpe=%.2f, MDD=%.1f%%",
            metrics.total_return * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
        )

        return metrics

    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare and validate data."""
        prepared = {}

        for symbol, df in data.items():
            try:
                df = df.copy()

                # Ensure datetime index
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                    df = df.set_index("date")

                # Sort and deduplicate
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]

                # Require OHLCV columns
                required = ["open", "high", "low", "close", "volume"]
                if not all(col in df.columns for col in required):
                    # Try to use close if open not available
                    if "close" in df.columns and "open" not in df.columns:
                        df["open"] = df["close"]
                    else:
                        continue

                # Filter date range
                if self.config.start_date:
                    start = pd.Timestamp(self.config.start_date)
                    df = df[df.index >= start]
                if self.config.end_date:
                    end = pd.Timestamp(self.config.end_date)
                    df = df[df.index <= end]

                # Minimum data requirement
                if len(df) >= 100:
                    prepared[symbol] = df

            except Exception as e:
                logger.debug("[BiasFreeBacktester] Skip %s: %s", symbol, e)
                continue

        return prepared

    def _generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        signal_func: Optional[Callable],
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for each symbol.

        Default strategy: KAMA + TSMOM with BTC Gate
        """
        # Find BTC for market filter
        btc_key = self._find_btc_key(data)
        btc_gate = None

        if btc_key:
            btc_df = data[btc_key]
            btc_prices = btc_df["close"].values
            btc_ma = self._calc_ma(btc_prices, self.config.gate_period)
            btc_gate = pd.Series(btc_prices > btc_ma, index=btc_df.index)

        signal_data = {}

        for symbol, df in data.items():
            prices = df["close"].values
            n = len(prices)

            if n < max(self.config.kama_period, self.config.tsmom_period, 100):
                continue

            if signal_func:
                # Custom signal function
                entry_signal = signal_func(df)
            else:
                # Default: KAMA + TSMOM
                kama = self._calc_kama(prices, self.config.kama_period)
                kama_signal = prices > kama

                tsmom_signal = np.zeros(n, dtype=bool)
                for i in range(self.config.tsmom_period, n):
                    tsmom_signal[i] = prices[i] > prices[i - self.config.tsmom_period]

                entry_signal = kama_signal | tsmom_signal

            df = df.copy()
            df["entry_signal"] = entry_signal

            # Apply BTC gate if available
            if btc_gate is not None:
                df = df.join(pd.DataFrame({"gate": btc_gate}), how="left")
                df["gate"] = df["gate"].fillna(False)
                df["final_signal"] = df["gate"] & df["entry_signal"]
            else:
                df["final_signal"] = df["entry_signal"]

            # Calculate dollar volume for ranking
            df["dollar_volume"] = df["close"] * df["volume"]

            signal_data[symbol] = df

        return signal_data

    def _simulate(self, signal_data: Dict[str, pd.DataFrame]) -> BacktestMetrics:
        """
        Run simulation with bias-free execution.

        Critical: Day T signal -> Day T+1 execution

        Key design:
        - Track positions by share count, not % weight
        - Calculate daily returns from previous close to today's close
        - Apply slippage only on new trades
        """
        # Get all trading dates
        all_dates = sorted(set().union(*[df.index.tolist() for df in signal_data.values()]))

        cash = self.config.initial_capital
        self.portfolio_values = [cash]
        self.daily_returns = []
        # positions: symbol -> (shares, avg_cost)
        positions: Dict[str, tuple] = {}
        prev_prices: Dict[str, float] = {}  # Track yesterday's prices for held positions

        entry_count = 0
        exit_count = 0
        invested_days = 0

        for i, date in enumerate(all_dates):
            # Collect today's data
            prices_today = {}
            signals_today = {}
            volumes_today = {}

            for symbol, df in signal_data.items():
                if date in df.index:
                    prices_today[symbol] = df.loc[date, "close"]
                    signals_today[symbol] = df.loc[date, "final_signal"]
                    volumes_today[symbol] = df.loc[date, "dollar_volume"]

            # STEP 1: Calculate portfolio value and daily return
            # Mark-to-market all positions using today's prices
            position_value = 0.0
            prev_position_value = 0.0

            for sym, (shares, avg_cost) in positions.items():
                if sym in prices_today:
                    position_value += shares * prices_today[sym]
                    # Use yesterday's price for previous value
                    if sym in prev_prices:
                        prev_position_value += shares * prev_prices[sym]
                    else:
                        prev_position_value += shares * avg_cost

            portfolio_value = cash + position_value
            prev_portfolio_value = cash + prev_position_value

            # Calculate daily return
            if i > 0 and prev_portfolio_value > 0:
                daily_ret = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
                self.daily_returns.append(daily_ret)
                if positions:
                    invested_days += 1
            else:
                self.daily_returns.append(0)

            self.portfolio_values.append(portfolio_value)

            # STEP 2: Generate target positions based on TODAY's signal
            active = [
                (sym, volumes_today.get(sym, 0))
                for sym, sig in signals_today.items()
                if sig
            ]

            # Sort by volume and select top N
            active_sorted = sorted(active, key=lambda x: x[1], reverse=True)
            target_symbols = set(s for s, v in active_sorted[:self.config.execution.max_positions])

            # Identify new entries and exits
            current_symbols = set(positions.keys())
            new_entries = target_symbols - current_symbols
            exits = current_symbols - target_symbols

            entry_count += len(new_entries)
            exit_count += len(exits)

            # STEP 3: Execute trades
            # First, sell exits
            for sym in exits:
                if sym in positions and sym in prices_today:
                    shares, avg_cost = positions[sym]
                    sell_price = prices_today[sym] * (1 - self.config.execution.slippage_pct)
                    proceeds = shares * sell_price
                    commission = proceeds * self.config.execution.commission_pct
                    cash += proceeds - commission
                    del positions[sym]

            # Calculate available capital for new positions
            # Total portfolio value after exits
            position_value_after_exits = sum(
                s * prices_today.get(sym, 0) for sym, (s, c) in positions.items()
            )
            total_value = cash + position_value_after_exits

            # Calculate target allocation
            if target_symbols:
                # Rebalance: equal weight for all target positions
                target_value_per_position = total_value / len(target_symbols)

                # Adjust existing positions and add new ones
                new_positions = {}

                for sym in target_symbols:
                    if sym not in prices_today:
                        continue

                    current_price = prices_today[sym]

                    if sym in positions:
                        # Keep existing position (no rebalance for simplicity)
                        shares, avg_cost = positions[sym]
                        new_positions[sym] = (shares, avg_cost)
                    else:
                        # New entry
                        buy_price = current_price * (1 + self.config.execution.slippage_pct)
                        shares_to_buy = (target_value_per_position / buy_price)
                        cost = shares_to_buy * buy_price
                        commission = cost * self.config.execution.commission_pct

                        if cost + commission <= cash:
                            cash -= (cost + commission)
                            new_positions[sym] = (shares_to_buy, buy_price)

                positions = new_positions

                self.positions_log.append({
                    "date": date,
                    "symbols": list(target_symbols),
                    "new_entries": list(new_entries),
                    "exits": list(exits),
                })
            else:
                positions = {}

            # Update previous prices for next iteration
            prev_prices = {sym: prices_today[sym] for sym in positions if sym in prices_today}

        # Calculate metrics
        return self._calculate_metrics(
            entry_count=entry_count,
            exit_count=exit_count,
            invested_days=invested_days,
            total_days=len(all_dates),
        )

    def _calculate_metrics(
        self,
        entry_count: int,
        exit_count: int,
        invested_days: int,
        total_days: int,
    ) -> BacktestMetrics:
        """Calculate performance metrics."""
        daily_rets = np.array(self.daily_returns)
        portfolio_vals = np.array(self.portfolio_values)

        # Total return
        total_return = (portfolio_vals[-1] - self.config.initial_capital) / self.config.initial_capital

        # Annualized return
        n_years = total_days / 252 if total_days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Sharpe ratio (assuming 0% risk-free rate)
        if len(daily_rets) > 1 and np.std(daily_rets) > 0:
            sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
        else:
            sharpe = 0

        # Sortino ratio (downside deviation)
        downside_rets = daily_rets[daily_rets < 0]
        if len(downside_rets) > 1 and np.std(downside_rets) > 0:
            sortino = np.mean(daily_rets) / np.std(downside_rets) * np.sqrt(252)
        else:
            sortino = 0

        # Max drawdown
        peak = np.maximum.accumulate(portfolio_vals)
        drawdown = (portfolio_vals - peak) / peak
        max_dd = np.min(drawdown)

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Win rate
        winning_days = np.sum(daily_rets > 0)
        losing_days = np.sum(daily_rets < 0)
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0

        # Profit factor
        gross_profits = np.sum(daily_rets[daily_rets > 0])
        gross_losses = abs(np.sum(daily_rets[daily_rets < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float("inf")

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=entry_count + exit_count,
            winning_trades=int(winning_days),
            losing_trades=int(losing_days),
            avg_trade_return=np.mean(daily_rets) if len(daily_rets) > 0 else 0,
            trading_days=total_days,
            invested_days=invested_days,
            exposure_ratio=invested_days / total_days if total_days > 0 else 0,
        )

    def _find_btc_key(self, data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Find BTC key in data."""
        for key in data.keys():
            upper = key.upper()
            if upper in ["BTC", "BTCUSDT", "BTC-KRW", "KRW-BTC"]:
                return key
            if "BTC" in upper and "DOWN" not in upper and "UP" not in upper:
                return key
        return None

    @staticmethod
    def _calc_kama(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Kaufman Adaptive Moving Average."""
        n = len(prices)
        kama = np.full(n, np.nan)

        if n < period + 1:
            return kama

        kama[period - 1] = np.mean(prices[:period])
        fast = 2 / (2 + 1)
        slow = 2 / (30 + 1)

        for i in range(period, n):
            change = abs(prices[i] - prices[i - period])
            volatility = sum(abs(prices[j] - prices[j - 1]) for j in range(i - period + 1, i + 1))
            er = change / volatility if volatility > 0 else 0
            sc = (er * (fast - slow) + slow) ** 2
            kama[i] = kama[i - 1] + sc * (prices[i] - kama[i - 1])

        return kama

    @staticmethod
    def _calc_ma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate simple moving average."""
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as pandas Series."""
        return pd.Series(self.portfolio_values)

    def get_trade_log(self) -> List[Dict]:
        """Get trade log."""
        return self.trade_log

    def get_positions_log(self) -> List[Dict]:
        """Get positions log."""
        return self.positions_log
