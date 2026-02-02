"""
Backtesting engine for KOSPI stocks.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from libs.backtesting.metrics import calculate_metrics, format_metrics

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position."""
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    direction: int  # 1 for long, -1 for short
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    def close(self, exit_date: datetime, exit_price: float):
        self.exit_date = exit_date
        self.exit_price = exit_price

    @property
    def pnl(self) -> float:
        """Calculate P&L."""
        if self.exit_price is None:
            return 0
        return self.direction * self.shares * (self.exit_price - self.entry_price)

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.exit_price is None:
            return 0
        return self.direction * (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100_000_000  # 1억원
    commission_rate: float = 0.00015  # 0.015% per trade
    slippage_rate: float = 0.001  # 0.1% slippage
    tax_rate: float = 0.0023  # 0.23% transaction tax (on sell only)
    max_position_size: float = 0.1  # Max 10% per position
    max_positions: int = 20  # Max number of concurrent positions
    allow_short: bool = False  # Allow short selling


@dataclass
class BacktestResult:
    """Backtest results."""
    strategy_name: str
    config: BacktestConfig
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: List[Position]
    trades: pd.DataFrame
    metrics: Dict
    start_date: datetime
    end_date: datetime

    def __str__(self):
        return format_metrics(self.metrics)


class BacktestEngine:
    """
    Vectorized backtesting engine for KOSPI stocks.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run_single_stock(
        self,
        data: pd.DataFrame,
        signal_func: Callable[[pd.DataFrame], pd.Series],
        strategy_name: str = "Strategy",
        direction: int = 1,  # 1=long, -1=short
    ) -> BacktestResult:
        """
        Run backtest on a single stock.

        Args:
            data: DataFrame with OHLCV data
            signal_func: Function that takes data and returns signal series (1=buy, -1=sell, 0=hold)
            strategy_name: Name of strategy
            direction: 1 for long-only, -1 for short-only

        Returns:
            BacktestResult
        """
        data = data.copy().reset_index(drop=True)
        data['Date'] = pd.to_datetime(data['Date'])

        # Generate signals
        signals = signal_func(data)
        data['Signal'] = signals

        # Apply direction
        if direction == -1:
            data['Signal'] = -data['Signal']

        # Calculate position (1 = in position, 0 = out)
        data['Position'] = 0
        in_position = False

        for i in range(len(data)):
            if not in_position and data.loc[i, 'Signal'] == 1:
                in_position = True
            elif in_position and data.loc[i, 'Signal'] == -1:
                in_position = False
            data.loc[i, 'Position'] = 1 if in_position else 0

        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Strategy_Returns'] = data['Returns'] * data['Position'].shift(1)

        # Apply transaction costs
        data['Position_Change'] = data['Position'].diff().abs()

        # Commission on both entry and exit
        data['Commission'] = data['Position_Change'] * self.config.commission_rate

        # Tax only on exit (when position changes from 1 to 0)
        data['Tax'] = 0.0
        data.loc[(data['Position'].shift(1) == 1) & (data['Position'] == 0), 'Tax'] = self.config.tax_rate

        # Slippage
        data['Slippage'] = data['Position_Change'] * self.config.slippage_rate

        # Net returns
        data['Net_Returns'] = data['Strategy_Returns'] - data['Commission'] - data['Tax'] - data['Slippage']
        data['Net_Returns'] = data['Net_Returns'].fillna(0)

        # Equity curve
        data['Equity'] = self.config.initial_capital * (1 + data['Net_Returns']).cumprod()

        # Extract trades
        positions, trades_df = self._extract_trades(data, direction)

        # Calculate metrics
        returns_series = data.set_index('Date')['Net_Returns']
        benchmark_returns = data.set_index('Date')['Returns']
        metrics = calculate_metrics(returns_series, benchmark_returns)

        return BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            equity_curve=data.set_index('Date')['Equity'],
            daily_returns=returns_series,
            positions=positions,
            trades=trades_df,
            metrics=metrics,
            start_date=data['Date'].iloc[0],
            end_date=data['Date'].iloc[-1],
        )

    def run_universe(
        self,
        data_dict: Dict[str, pd.DataFrame],
        signal_func: Callable[[pd.DataFrame], pd.Series],
        strategy_name: str = "Strategy",
        direction: int = 1,
        rebalance_freq: str = 'M',  # M=monthly, W=weekly, D=daily
        top_n: int = 10,  # Number of stocks to hold
        rank_by: str = 'signal_strength',  # Ranking criteria
    ) -> BacktestResult:
        """
        Run backtest on a universe of stocks with periodic rebalancing.

        Args:
            data_dict: Dict of ticker -> DataFrame
            signal_func: Signal generation function
            strategy_name: Strategy name
            direction: Trading direction
            rebalance_freq: Rebalancing frequency
            top_n: Number of stocks to hold
            rank_by: Ranking method for stock selection

        Returns:
            BacktestResult
        """
        # Get common date range
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df['Date'].tolist())
        all_dates = sorted(all_dates)

        if not all_dates:
            raise ValueError("No data available")

        # Create price matrix
        price_matrix = pd.DataFrame(index=all_dates)
        signal_matrix = pd.DataFrame(index=all_dates)

        for ticker, df in data_dict.items():
            df = df.set_index('Date')
            price_matrix[ticker] = df['Close']

            # Generate signals
            df_reset = df.reset_index()
            signals = signal_func(df_reset)
            signal_series = pd.Series(signals.values, index=df.index)
            signal_matrix[ticker] = signal_series

        price_matrix = price_matrix.ffill()
        signal_matrix = signal_matrix.fillna(0)

        # Calculate returns
        returns_matrix = price_matrix.pct_change()

        # Generate rebalance dates
        dates_df = pd.DataFrame(index=all_dates)
        dates_df.index = pd.to_datetime(dates_df.index)

        if rebalance_freq == 'M':
            rebalance_mask = dates_df.index.is_month_end
        elif rebalance_freq == 'W':
            rebalance_mask = dates_df.index.dayofweek == 4  # Friday
        else:
            rebalance_mask = [True] * len(dates_df)

        rebalance_dates = dates_df.index[rebalance_mask].tolist()

        # Portfolio simulation
        portfolio_returns = []
        current_holdings = {}

        for i, date in enumerate(all_dates):
            if i == 0:
                portfolio_returns.append(0)
                continue

            date_dt = pd.to_datetime(date)

            # Rebalance if needed
            if date_dt in rebalance_dates or not current_holdings:
                # Get signals for this date
                day_signals = signal_matrix.loc[date]

                # Filter valid signals
                if direction == 1:
                    candidates = day_signals[day_signals > 0]
                else:
                    candidates = day_signals[day_signals < 0]
                    candidates = -candidates  # Convert to positive for ranking

                # Select top N
                if len(candidates) > 0:
                    selected = candidates.nlargest(min(top_n, len(candidates)))
                    current_holdings = {t: 1.0 / len(selected) for t in selected.index}
                else:
                    current_holdings = {}

            # Calculate portfolio return
            if current_holdings:
                day_return = sum(
                    weight * returns_matrix.loc[date, ticker]
                    for ticker, weight in current_holdings.items()
                    if ticker in returns_matrix.columns and pd.notna(returns_matrix.loc[date, ticker])
                )
            else:
                day_return = 0

            portfolio_returns.append(day_return * direction)

        # Apply costs (simplified for portfolio)
        returns_series = pd.Series(portfolio_returns, index=all_dates)
        returns_series = returns_series - self.config.commission_rate / 20  # Amortize costs

        # Equity curve
        equity = self.config.initial_capital * (1 + returns_series).cumprod()

        # Metrics
        returns_series.index = pd.to_datetime(returns_series.index)
        metrics = calculate_metrics(returns_series)

        return BacktestResult(
            strategy_name=strategy_name,
            config=self.config,
            equity_curve=pd.Series(equity.values, index=pd.to_datetime(all_dates)),
            daily_returns=returns_series,
            positions=[],
            trades=pd.DataFrame(),
            metrics=metrics,
            start_date=all_dates[0],
            end_date=all_dates[-1],
        )

    def _extract_trades(self, data: pd.DataFrame, direction: int) -> Tuple[List[Position], pd.DataFrame]:
        """Extract individual trades from backtest data."""
        positions = []
        trades_list = []

        current_pos = None

        for i in range(1, len(data)):
            prev_pos = data.loc[i - 1, 'Position']
            curr_pos = data.loc[i, 'Position']

            # Entry
            if prev_pos == 0 and curr_pos == 1:
                current_pos = Position(
                    ticker=data.get('Ticker', ['UNKNOWN'] * len(data))[i] if 'Ticker' in data.columns else 'STOCK',
                    entry_date=data.loc[i, 'Date'],
                    entry_price=data.loc[i, 'Close'],
                    shares=int(self.config.initial_capital * self.config.max_position_size / data.loc[i, 'Close']),
                    direction=direction,
                )

            # Exit
            elif prev_pos == 1 and curr_pos == 0 and current_pos is not None:
                current_pos.close(data.loc[i, 'Date'], data.loc[i, 'Close'])
                positions.append(current_pos)

                trades_list.append({
                    'entry_date': current_pos.entry_date,
                    'exit_date': current_pos.exit_date,
                    'entry_price': current_pos.entry_price,
                    'exit_price': current_pos.exit_price,
                    'shares': current_pos.shares,
                    'direction': 'LONG' if current_pos.direction == 1 else 'SHORT',
                    'pnl': current_pos.pnl,
                    'return_pct': current_pos.return_pct,
                    'holding_days': (current_pos.exit_date - current_pos.entry_date).days,
                })

                current_pos = None

        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()
        return positions, trades_df


def quick_backtest(
    data: pd.DataFrame,
    signal_func: Callable[[pd.DataFrame], pd.Series],
    strategy_name: str = "Strategy",
    direction: int = 1,
    initial_capital: float = 100_000_000,
) -> BacktestResult:
    """
    Quick backtest helper function.

    Args:
        data: OHLCV DataFrame
        signal_func: Signal function
        strategy_name: Strategy name
        direction: 1=long, -1=short
        initial_capital: Starting capital

    Returns:
        BacktestResult
    """
    config = BacktestConfig(initial_capital=initial_capital)
    engine = BacktestEngine(config)
    return engine.run_single_stock(data, signal_func, strategy_name, direction)
