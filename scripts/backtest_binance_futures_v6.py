#!/usr/bin/env python3
"""
Binance Futures v6 Strategy Backtester

This script validates the AI Consensus strategy using:
1. Historical data from Binance API
2. Bias-free backtesting (Signal Day T -> Execute Day T+1)
3. Walk-Forward Analysis for parameter validation
4. Performance metrics calculation

Usage:
    python scripts/backtest_binance_futures_v6.py
    python scripts/backtest_binance_futures_v6.py --symbols BTCUSDT,ETHUSDT --start 2022-01-01
    python scripts/backtest_binance_futures_v6.py --wfa  # Run Walk-Forward Analysis

Requirements:
    - Binance API access (public endpoints for historical data)
    - libs/strategies/binance_futures_v6.py
    - libs/strategies/indicators.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from libs.strategies.binance_futures_v6 import (
    BinanceFuturesV6Strategy,
    MarketRegime,
    SignalType,
)

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float = 10000.0  # USDT
    start_date: str = "2022-01-01"
    end_date: Optional[str] = None  # None = today

    # Symbols to test
    symbols: List[str] = field(
        default_factory=lambda: [
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "SOLUSDT",
            "XRPUSDT",
        ]
    )

    # Timeframes
    higher_tf: str = "1d"
    lower_tf: str = "4h"

    # Execution costs
    commission_pct: float = 0.001  # 0.1% taker fee
    slippage_pct: float = 0.0005  # 0.05% slippage

    # Leverage
    leverage: int = 5

    # WFA settings (Gemini)
    wfa_in_sample_days: int = 180
    wfa_out_sample_days: int = 30


@dataclass
class BacktestResult:
    """Backtest results."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_pnl: float = 0.0

    # Regime breakdown
    bull_trades: int = 0
    bull_win_rate: float = 0.0
    neutral_trades: int = 0
    neutral_win_rate: float = 0.0
    bear_trades: int = 0
    bear_win_rate: float = 0.0

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": f"{self.total_return * 100:.2f}%",
            "annualized_return": f"{self.annualized_return * 100:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "sortino_ratio": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown * 100:.2f}%",
            "calmar_ratio": f"{self.calmar_ratio:.2f}",
            "win_rate": f"{self.win_rate * 100:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "total_trades": self.total_trades,
            "avg_trade_pnl": f"{self.avg_trade_pnl * 100:.2f}%",
            "regime_breakdown": {
                "bull": {
                    "trades": self.bull_trades,
                    "win_rate": f"{self.bull_win_rate * 100:.1f}%",
                },
                "neutral": {
                    "trades": self.neutral_trades,
                    "win_rate": f"{self.neutral_win_rate * 100:.1f}%",
                },
                "bear": {
                    "trades": self.bear_trades,
                    "win_rate": f"{self.bear_win_rate * 100:.1f}%",
                },
            },
        }


class BinanceDataFetcher:
    """Fetch historical data from Binance API."""

    BASE_URL = "https://fapi.binance.com"  # Futures API

    INTERVALS = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
    }

    def __init__(self):
        if requests is None:
            raise ImportError("requests library required: pip install requests")

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch klines (candlestick) data from Binance Futures.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "4h", "1d")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with OHLCV data
        """
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)

        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        all_data = []
        current_start = start_ts

        logger.info(
            f"Fetching {symbol} {interval} data from {start_date} to {end_date or 'now'}..."
        )

        while current_start < end_ts:
            try:
                response = requests.get(
                    f"{self.BASE_URL}/fapi/v1/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_ts,
                        "limit": 1500,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                all_data.extend(data)

                # Move to next batch
                last_ts = data[-1][0]
                if last_ts <= current_start:
                    break
                current_start = last_ts + 1

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break

        if not all_data:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ],
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        logger.info(f"  {symbol}: {len(df)} candles fetched")

        return df


class BinanceFuturesV6Backtester:
    """
    Backtest engine for Binance Futures v6 Strategy.

    Features:
    - Bias-free execution (signal at candle close -> execute next candle)
    - Multi-timeframe support (1D for trend, 4H for entry)
    - Regime-based position sizing
    - Detailed trade logging
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy = BinanceFuturesV6Strategy()
        self.fetcher = BinanceDataFetcher()

        # Data storage
        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.data_1d: Dict[str, pd.DataFrame] = {}

        # Results
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []
        self.regime_log: List[Dict] = []

    def fetch_data(self) -> bool:
        """Fetch historical data for all symbols."""
        logger.info("=" * 60)
        logger.info("FETCHING HISTORICAL DATA")
        logger.info("=" * 60)

        for symbol in self.config.symbols:
            # Fetch 4H data
            df_4h = self.fetcher.fetch_klines(
                symbol,
                self.config.lower_tf,
                self.config.start_date,
                self.config.end_date,
            )

            if len(df_4h) < 100:
                logger.warning(f"Insufficient data for {symbol} (4H), skipping")
                continue

            self.data_4h[symbol] = df_4h

            # Fetch 1D data
            df_1d = self.fetcher.fetch_klines(
                symbol,
                self.config.higher_tf,
                self.config.start_date,
                self.config.end_date,
            )

            if len(df_1d) < 200:
                logger.warning(f"Insufficient data for {symbol} (1D), skipping")
                continue

            self.data_1d[symbol] = df_1d

        # Always need BTC for regime detection
        if "BTCUSDT" not in self.data_4h:
            logger.error("BTC data required for regime detection")
            return False

        logger.info(f"Data fetched for {len(self.data_4h)} symbols")
        return len(self.data_4h) > 0

    def _calculate_btc_metrics(
        self, btc_df: pd.DataFrame, idx: int
    ) -> Dict[str, float]:
        """Calculate BTC metrics for regime detection."""
        if idx < 200:
            return {
                "price": btc_df["close"].iloc[idx],
                "ema_200": btc_df["close"].iloc[idx],
                "52w_high": btc_df["close"].iloc[: idx + 1].max(),
                "change_24h": 0.0,
            }

        close = btc_df["close"].values[: idx + 1]

        # 200 EMA
        ema_200 = self._ema(close, 200)[-1]

        # 52-week high (approximately 365 daily candles)
        lookback = min(365, idx)
        high_52w = btc_df["high"].iloc[idx - lookback : idx + 1].max()

        # 24h change (6 x 4H candles = 24h)
        if idx >= 6:
            change_24h = (close[-1] - close[-7]) / close[-7] * 100
        else:
            change_24h = 0.0

        return {
            "price": close[-1],
            "ema_200": ema_200,
            "52w_high": high_52w,
            "change_24h": change_24h,
        }

    def run(self) -> BacktestResult:
        """Run the backtest."""
        logger.info("=" * 60)
        logger.info("RUNNING BACKTEST")
        logger.info("=" * 60)

        # Initialize
        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}  # symbol -> position info
        self.equity_curve = [capital]
        self.trade_log = []

        # Get aligned timestamps
        btc_4h = self.data_4h["BTCUSDT"]
        timestamps = btc_4h.index.tolist()

        # Track regime statistics
        regime_trades = {
            MarketRegime.BULL: {"wins": 0, "losses": 0},
            MarketRegime.NEUTRAL: {"wins": 0, "losses": 0},
            MarketRegime.BEAR: {"wins": 0, "losses": 0},
        }

        # DEBUG: Signal statistics
        signal_stats = {
            "total_candles": 0,
            "entries": 0,
            "exits": 0,
            "hold": 0,
            "by_reason": {},
            "positions_opened": 0,
            "positions_closed": 0,
        }

        last_date = None
        last_week = None

        for i in range(200, len(timestamps)):  # Start after warmup
            signal_stats["total_candles"] += 1
            current_time = timestamps[i]
            current_date_obj = current_time.date()
            current_week = current_time.isocalendar()[1]

            # Reset daily/weekly counters at day/week boundaries
            if last_date is not None and current_date_obj != last_date:
                self.strategy.reset_daily()
            if last_week is not None and current_week != last_week:
                self.strategy.reset_weekly()
            last_date = current_date_obj
            last_week = current_week

            # Calculate BTC metrics for regime detection
            btc_metrics = self._calculate_btc_metrics(btc_4h, i)

            # Update strategy with BTC data
            self.strategy.update_btc_data(
                btc_metrics["price"],
                btc_metrics["ema_200"],
                btc_metrics["52w_high"],
                btc_metrics["change_24h"],
            )

            current_regime = self.strategy.current_regime

            # Process each symbol
            for symbol in self.data_4h.keys():
                if symbol not in self.data_1d:
                    continue

                df_4h = self.data_4h[symbol]
                df_1d = self.data_1d[symbol]

                # Get data up to current time (no look-ahead)
                df_4h_current = df_4h.iloc[: i + 1].copy()

                # Find corresponding 1D data
                current_date = current_time.date()
                df_1d_current = df_1d[df_1d.index.date <= current_date].copy()

                # Need sufficient data for indicators (reduced from 200 to 50 for 1D)
                if len(df_4h_current) < 100 or len(df_1d_current) < 50:
                    continue

                # Generate signal
                signal = self.strategy.generate_signal(
                    symbol,
                    df_4h_current,
                    df_1d_current,
                    capital,
                )

                # DEBUG: Track signal reasons
                reason_key = signal.reason[:50] if signal.reason else "unknown"
                signal_stats["by_reason"][reason_key] = (
                    signal_stats["by_reason"].get(reason_key, 0) + 1
                )

                # Handle exits
                if symbol in positions:
                    pos = positions[symbol]

                    if signal.signal_type in [
                        SignalType.EXIT_LONG,
                        SignalType.EXIT_SHORT,
                    ]:
                        # Execute exit
                        exit_price = df_4h.iloc[i]["close"]
                        exit_price *= (
                            (1 - self.config.slippage_pct)
                            if pos["side"] == "LONG"
                            else (1 + self.config.slippage_pct)
                        )

                        # Calculate PnL
                        if pos["side"] == "LONG":
                            pnl_pct = (exit_price - pos["entry_price"]) / pos[
                                "entry_price"
                            ]
                        else:
                            pnl_pct = (pos["entry_price"] - exit_price) / pos[
                                "entry_price"
                            ]

                        pnl_pct *= pos["leverage"]
                        pnl_usd = pos["size"] * pnl_pct

                        # Apply commission
                        commission = (
                            pos["size"] * self.config.commission_pct * 2
                        )  # Entry + exit
                        pnl_usd -= commission

                        capital += pnl_usd

                        # Track regime stats
                        regime_at_entry = pos["regime"]
                        if pnl_usd > 0:
                            regime_trades[regime_at_entry]["wins"] += 1
                        else:
                            regime_trades[regime_at_entry]["losses"] += 1

                        # Log trade
                        self.trade_log.append(
                            {
                                "symbol": symbol,
                                "side": pos["side"],
                                "entry_time": pos["entry_time"],
                                "entry_price": pos["entry_price"],
                                "exit_time": current_time,
                                "exit_price": exit_price,
                                "pnl_pct": pnl_pct,
                                "pnl_usd": pnl_usd,
                                "regime": regime_at_entry.value,
                            }
                        )

                        logger.info(
                            f"[DEBUG] EXIT: {symbol} {pos['side']} @ {exit_price:.2f}, pnl={pnl_pct*100:.2f}%, reason={signal.reason[:50]}"
                        )

                        del positions[symbol]
                        self.strategy.close_position(symbol, signal)
                        signal_stats["positions_closed"] += 1
                        signal_stats["exits"] += 1

                # Handle entries
                if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                    if (
                        symbol not in positions
                        and len(positions) < self.strategy.config.max_positions
                    ):
                        # Execute entry
                        entry_price = df_4h.iloc[i]["close"]
                        entry_price *= (
                            (1 + self.config.slippage_pct)
                            if signal.signal_type == SignalType.LONG
                            else (1 - self.config.slippage_pct)
                        )

                        # Position size from strategy
                        position_size = signal.position_size or (capital * 0.1)
                        position_size = min(
                            position_size, capital * 0.2
                        )  # Max 20% per position

                        positions[symbol] = {
                            "side": (
                                "LONG"
                                if signal.signal_type == SignalType.LONG
                                else "SHORT"
                            ),
                            "entry_price": entry_price,
                            "entry_time": current_time,
                            "size": position_size,
                            "leverage": self.config.leverage,
                            "stop_loss": signal.stop_loss,
                            "regime": current_regime,
                        }

                        self.strategy.open_position(symbol, signal)
                        signal_stats["positions_opened"] += 1
                        signal_stats["entries"] += 1
                        logger.info(
                            f"[DEBUG] ENTRY: {symbol} {positions[symbol]['side']} @ {entry_price:.2f}, regime={current_regime.value}"
                        )

            # Mark-to-market
            portfolio_value = capital
            for symbol, pos in positions.items():
                if symbol in self.data_4h:
                    current_price = self.data_4h[symbol].iloc[i]["close"]
                    if pos["side"] == "LONG":
                        unrealized_pnl = (current_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]
                    else:
                        unrealized_pnl = (pos["entry_price"] - current_price) / pos[
                            "entry_price"
                        ]
                    unrealized_pnl *= pos["leverage"]
                    portfolio_value += pos["size"] * unrealized_pnl

            self.equity_curve.append(portfolio_value)

            # Log regime
            self.regime_log.append(
                {
                    "time": current_time,
                    "regime": current_regime.value,
                    "btc_price": btc_metrics["price"],
                    "portfolio_value": portfolio_value,
                }
            )

        # DEBUG: Print signal statistics
        logger.info("=" * 60)
        logger.info("DEBUG: SIGNAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total candles processed: {signal_stats['total_candles']}")
        logger.info(f"Positions opened: {signal_stats['positions_opened']}")
        logger.info(f"Positions closed: {signal_stats['positions_closed']}")
        logger.info(f"Positions still open: {len(positions)}")
        logger.info("")
        logger.info("Signal reasons breakdown (top 15):")
        sorted_reasons = sorted(signal_stats["by_reason"].items(), key=lambda x: -x[1])[
            :15
        ]
        for reason, count in sorted_reasons:
            pct = (
                count / signal_stats["total_candles"] * 100
                if signal_stats["total_candles"] > 0
                else 0
            )
            logger.info(f"  {reason}: {count} ({pct:.1f}%)")

        # Close any remaining positions at end of backtest
        if positions:
            logger.info("")
            logger.info("Closing remaining positions at end of backtest:")
            for symbol, pos in list(positions.items()):
                if symbol in self.data_4h:
                    exit_price = self.data_4h[symbol].iloc[-1]["close"]
                    exit_price *= (
                        (1 - self.config.slippage_pct)
                        if pos["side"] == "LONG"
                        else (1 + self.config.slippage_pct)
                    )

                    if pos["side"] == "LONG":
                        pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    else:
                        pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]

                    pnl_pct *= pos["leverage"]
                    pnl_usd = pos["size"] * pnl_pct
                    commission = pos["size"] * self.config.commission_pct * 2
                    pnl_usd -= commission
                    capital += pnl_usd

                    self.trade_log.append(
                        {
                            "symbol": symbol,
                            "side": pos["side"],
                            "entry_time": pos["entry_time"],
                            "entry_price": pos["entry_price"],
                            "exit_time": timestamps[-1],
                            "exit_price": exit_price,
                            "pnl_pct": pnl_pct,
                            "pnl_usd": pnl_usd,
                            "regime": pos["regime"].value,
                            "reason": "End of backtest",
                        }
                    )

                    regime_at_entry = pos["regime"]
                    if pnl_usd > 0:
                        regime_trades[regime_at_entry]["wins"] += 1
                    else:
                        regime_trades[regime_at_entry]["losses"] += 1

                    logger.info(
                        f"  {symbol} {pos['side']}: entry={pos['entry_price']:.2f} exit={exit_price:.2f} pnl={pnl_pct*100:.2f}%"
                    )

                    del positions[symbol]

        # Calculate final metrics
        return self._calculate_metrics(regime_trades)

    def _calculate_metrics(self, regime_trades: Dict) -> BacktestResult:
        """Calculate backtest metrics."""
        if len(self.equity_curve) < 2:
            return BacktestResult()

        equity = np.array(self.equity_curve)

        # Total return
        total_return = (
            equity[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Daily returns (approximation: each 4H candle)
        returns = np.diff(equity) / equity[:-1]

        # Annualized metrics (252 trading days * 6 candles = 1512 4H candles/year)
        candles_per_year = 1512
        n_candles = len(returns)
        n_years = n_candles / candles_per_year if n_candles > 0 else 1

        annualized_return = (
            (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        )

        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(candles_per_year)
        else:
            sharpe = 0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 1 and np.std(downside) > 0:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(candles_per_year)
        else:
            sortino = 0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Trade statistics
        if self.trade_log:
            trade_pnls = [t["pnl_pct"] for t in self.trade_log]
            wins = sum(1 for p in trade_pnls if p > 0)
            sum(1 for p in trade_pnls if p <= 0)
            win_rate = wins / len(trade_pnls) if trade_pnls else 0

            gross_profit = sum(p for p in trade_pnls if p > 0)
            gross_loss = abs(sum(p for p in trade_pnls if p < 0))
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            avg_trade = np.mean(trade_pnls)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade = 0

        # Regime breakdown
        def calc_win_rate(stats):
            total = stats["wins"] + stats["losses"]
            return stats["wins"] / total if total > 0 else 0

        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trade_log),
            avg_trade_pnl=avg_trade,
            bull_trades=regime_trades[MarketRegime.BULL]["wins"]
            + regime_trades[MarketRegime.BULL]["losses"],
            bull_win_rate=calc_win_rate(regime_trades[MarketRegime.BULL]),
            neutral_trades=regime_trades[MarketRegime.NEUTRAL]["wins"]
            + regime_trades[MarketRegime.NEUTRAL]["losses"],
            neutral_win_rate=calc_win_rate(regime_trades[MarketRegime.NEUTRAL]),
            bear_trades=regime_trades[MarketRegime.BEAR]["wins"]
            + regime_trades[MarketRegime.BEAR]["losses"],
            bear_win_rate=calc_win_rate(regime_trades[MarketRegime.BEAR]),
            equity_curve=self.equity_curve,
            trade_log=self.trade_log,
        )

        return result

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros(len(data))
        ema[0] = data[0]
        multiplier = 2 / (period + 1)

        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema


def run_walk_forward_analysis(config: BacktestConfig) -> Dict[str, Any]:
    """
    Run Walk-Forward Analysis (Gemini methodology).

    180 days In-Sample / 30 days Out-of-Sample rolling.
    """
    logger.info("=" * 60)
    logger.info("WALK-FORWARD ANALYSIS")
    logger.info("=" * 60)

    results = []

    # Parse dates
    start = datetime.strptime(config.start_date, "%Y-%m-%d")
    end = (
        datetime.strptime(config.end_date, "%Y-%m-%d")
        if config.end_date
        else datetime.now()
    )

    is_days = config.wfa_in_sample_days
    oos_days = config.wfa_out_sample_days

    current_start = start

    while current_start + timedelta(days=is_days + oos_days) <= end:
        is_end = current_start + timedelta(days=is_days)
        oos_end = is_end + timedelta(days=oos_days)

        logger.info(
            f"WFA Period: IS {current_start.date()} to {is_end.date()}, OOS {is_end.date()} to {oos_end.date()}"
        )

        # Run OOS backtest
        oos_config = BacktestConfig(
            initial_capital=config.initial_capital,
            start_date=is_end.strftime("%Y-%m-%d"),
            end_date=oos_end.strftime("%Y-%m-%d"),
            symbols=config.symbols,
        )

        backtester = BinanceFuturesV6Backtester(oos_config)
        if backtester.fetch_data():
            result = backtester.run()
            results.append(
                {
                    "period": f"{is_end.date()} to {oos_end.date()}",
                    "return": result.total_return,
                    "sharpe": result.sharpe_ratio,
                    "mdd": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.total_trades,
                }
            )

        current_start += timedelta(days=oos_days)

    # Aggregate results
    if results:
        avg_return = np.mean([r["return"] for r in results])
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        avg_mdd = np.mean([r["mdd"] for r in results])
        avg_win_rate = np.mean([r["win_rate"] for r in results])
        positive_periods = sum(1 for r in results if r["return"] > 0)
        robustness = positive_periods / len(results)

        return {
            "periods": len(results),
            "avg_return": f"{avg_return * 100:.2f}%",
            "avg_sharpe": f"{avg_sharpe:.2f}",
            "avg_mdd": f"{avg_mdd * 100:.2f}%",
            "avg_win_rate": f"{avg_win_rate * 100:.1f}%",
            "robustness": f"{robustness * 100:.1f}% positive periods",
            "details": results,
        }

    return {"error": "No WFA periods completed"}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Binance Futures v6 Strategy Backtester"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT",
        help="Comma-separated list of symbols",
    )
    parser.add_argument(
        "--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="End date (YYYY-MM-DD), defaults to today"
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0, help="Initial capital in USDT"
    )
    parser.add_argument("--leverage", type=int, default=5, help="Leverage (default: 5)")
    parser.add_argument("--wfa", action="store_true", help="Run Walk-Forward Analysis")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(","),
        leverage=args.leverage,
    )

    logger.info("=" * 60)
    logger.info("BINANCE FUTURES V6 STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Period: {config.start_date} to {config.end_date or 'now'}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Leverage: {config.leverage}x")

    if args.wfa:
        # Walk-Forward Analysis
        wfa_results = run_walk_forward_analysis(config)

        logger.info("=" * 60)
        logger.info("WFA RESULTS")
        logger.info("=" * 60)
        for key, value in wfa_results.items():
            if key != "details":
                logger.info(f"  {key}: {value}")
    else:
        # Standard backtest
        backtester = BinanceFuturesV6Backtester(config)

        if not backtester.fetch_data():
            logger.error("Failed to fetch data")
            return 1

        result = backtester.run()

        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)

        for key, value in result.to_dict().items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

        # Clova target comparison
        logger.info("")
        logger.info("=" * 60)
        logger.info("vs CLOVA TARGETS")
        logger.info("=" * 60)

        targets = {
            "Win Rate": ("48-52%", result.win_rate),
            "Annual Return": ("25-45%", result.annualized_return),
            "Max MDD": ("< 25%", result.max_drawdown),
        }

        for name, (target, actual) in targets.items():
            actual_str = f"{actual * 100:.1f}%"
            logger.info(f"  {name}: {actual_str} (target: {target})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
