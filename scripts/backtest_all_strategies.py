#!/usr/bin/env python3
"""
Multi-Strategy Backtester - All Symbols
Tests 6 different strategies across all available symbols

Strategies:
1. Funding Rate Based (Long when negative, Short when positive)
2. Momentum/Trend Following (Breakout + ADX)
3. Funding Arbitrage / Carry Trade
4. BTC Dominance Rotation
5. Volatility Based (Mean Reversion in low vol, Breakout in high vol)
6. Short-Only Strategy (Inverse of Long-Only)

Target: Find strategies that work across ALL symbols
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA LOADER
# ============================================================================
class DataLoader:
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        tf_map = {
            "4h": "binance_futures_4h",
            "1d": "binance_futures_1d",
            "1h": "binance_futures_1h",
        }
        folder = tf_map.get(timeframe, "binance_futures_4h")

        for file_path in [
            self.data_root / folder / f"{symbol}.csv",
            self.data_root / folder / f"{symbol.replace('USDT', '')}.csv",
        ]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                return df[["open", "high", "low", "close", "volume"]].sort_index()
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str) -> pd.DataFrame:
        for filename in [f"{symbol}_funding_full.csv", f"{symbol}_funding.csv"]:
            file_path = self.data_root / "binance_funding_rate" / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                cols = [
                    c
                    for c in df.columns
                    if "funding" in c.lower() or "rate" in c.lower()
                ]
                if cols:
                    df = df.rename(columns={cols[0]: "funding_rate"})
                return df[["funding_rate"]] if "funding_rate" in df.columns else df
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        for filename in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            file_path = self.data_root / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                if "close" in df.columns:
                    df = df.rename(columns={"close": "value"})
                return df[["value"]] if "value" in df.columns else df
        return pd.DataFrame()

    def load_btc_dominance(self) -> pd.DataFrame:
        """Load BTC dominance data (from coingecko or calculate from market caps)"""
        # Try to load pre-computed dominance
        file_path = self.data_root / "coingecko" / "btc_dominance.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"], index_col="datetime")
            return df

        # Calculate from BTC vs total market (simplified - use BTC/ETH ratio as proxy)
        btc = self.load_ohlcv("BTCUSDT", "1d")
        eth = self.load_ohlcv("ETHUSDT", "1d")
        if not btc.empty and not eth.empty:
            df = pd.DataFrame(index=btc.index)
            df["btc_eth_ratio"] = btc["close"] / eth["close"]
            df["dominance"] = df["btc_eth_ratio"].rolling(30).mean()
            return df
        return pd.DataFrame()


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Calculate ADX (Average Directional Index)"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = calc_atr(high, low, close, period)
    plus_di = 100 * calc_ema(plus_dm, period) / atr
    minus_di = 100 * calc_ema(minus_dm, period) / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = calc_ema(dx, period)
    return adx


def calc_bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = calc_sma(series, period)
    std = series.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, sma, lower


def calc_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate historical volatility (annualized)"""
    returns = series.pct_change()
    return returns.rolling(period).std() * np.sqrt(365)


# ============================================================================
# BASE BACKTEST ENGINE
# ============================================================================
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int  # 1=Long, -1=Short
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    total_return: float
    mdd: float
    win_rate: float
    profit_factor: float
    num_trades: int
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None


class BaseBacktester:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        leverage: int = 3,
        position_size_pct: float = 0.25,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.leverage = leverage
        self.position_size_pct = position_size_pct

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame,  # columns: signal (1=long, -1=short, 0=flat), exit (bool)
        symbol: str,
        strategy_name: str,
    ) -> BacktestResult:
        capital = self.initial_capital
        position = None
        trades: List[Trade] = []
        equity = [capital]
        equity_dates = [df.index[0]]

        for i in range(1, len(df)):
            current_time = df.index[i]
            current_price = df["close"].iloc[i]
            signal = signals["signal"].iloc[i] if i < len(signals) else 0
            exit_signal = (
                signals["exit"].iloc[i]
                if i < len(signals) and "exit" in signals.columns
                else False
            )

            # Exit logic
            if position is not None:
                should_exit = exit_signal or (
                    signal != 0 and signal != position.direction
                )

                if should_exit:
                    exit_price = current_price * (
                        1 - self.slippage_pct * position.direction
                    )
                    pnl_pct = (
                        exit_price / position.entry_price - 1
                    ) * position.direction
                    pnl_pct -= self.commission_pct * 2  # Entry + exit commission
                    pnl = capital * self.position_size_pct * self.leverage * pnl_pct

                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    trades.append(position)

                    capital += pnl
                    position = None

            # Entry logic
            if position is None and signal != 0:
                entry_price = current_price * (1 + self.slippage_pct * signal)
                position = Trade(
                    entry_time=current_time,
                    entry_price=entry_price,
                    direction=signal,
                    size=capital * self.position_size_pct * self.leverage / entry_price,
                )

            equity.append(
                capital
                if position is None
                else capital + self._unrealized_pnl(position, current_price)
            )
            equity_dates.append(current_time)

        # Close any open position
        if position is not None:
            exit_price = df["close"].iloc[-1]
            pnl_pct = (exit_price / position.entry_price - 1) * position.direction
            pnl_pct -= self.commission_pct * 2
            position.exit_time = df.index[-1]
            position.exit_price = exit_price
            position.pnl = capital * self.position_size_pct * self.leverage * pnl_pct
            position.pnl_pct = pnl_pct
            trades.append(position)
            capital += position.pnl

        equity_series = pd.Series(equity, index=equity_dates)

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            total_return=(capital / self.initial_capital - 1) * 100,
            mdd=self._calc_mdd(equity_series),
            win_rate=self._calc_win_rate(trades),
            profit_factor=self._calc_profit_factor(trades),
            num_trades=len(trades),
            sharpe_ratio=self._calc_sharpe(equity_series),
            trades=trades,
            equity_curve=equity_series,
        )

    def _unrealized_pnl(self, position: Trade, current_price: float) -> float:
        pnl_pct = (current_price / position.entry_price - 1) * position.direction
        return self.initial_capital * self.position_size_pct * self.leverage * pnl_pct

    def _calc_mdd(self, equity: pd.Series) -> float:
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100
        return dd.min()

    def _calc_win_rate(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades) * 100

    def _calc_profit_factor(self, trades: List[Trade]) -> float:
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        return (
            gross_profit / gross_loss
            if gross_loss > 0
            else float("inf") if gross_profit > 0 else 0.0
        )

    def _calc_sharpe(self, equity: pd.Series) -> float:
        returns = equity.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252 * 6)  # 4h bars


# ============================================================================
# STRATEGY 1: FUNDING RATE BASED
# ============================================================================
class FundingRateStrategy:
    """
    Logic:
    - Negative funding rate = Shorts paying Longs = Crowded shorts = GO LONG
    - Positive funding rate = Longs paying Shorts = Crowded longs = GO SHORT
    - Use funding rate extreme + trend confirmation
    """

    def __init__(
        self,
        funding_threshold: float = 0.0005,  # 0.05% per 8h = ~0.5% daily
        trend_period: int = 50,
        holding_period: int = 18,  # ~3 days in 4h bars
    ):
        self.funding_threshold = funding_threshold
        self.trend_period = trend_period
        self.holding_period = holding_period

    def generate_signals(self, df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if funding.empty:
            return signals

        # Merge funding data
        merged = df.copy()
        merged["funding"] = funding["funding_rate"].reindex(df.index, method="ffill")
        merged["funding_ma"] = merged["funding"].rolling(6).mean()  # ~2 days average

        # Trend filter
        merged["ema"] = calc_ema(merged["close"], self.trend_period)
        merged["trend"] = np.where(merged["close"] > merged["ema"], 1, -1)

        bars_since_entry = 0
        in_position = False

        for i in range(self.trend_period, len(merged)):
            if in_position:
                bars_since_entry += 1
                if bars_since_entry >= self.holding_period:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    in_position = False
                    bars_since_entry = 0
                continue

            funding_val = merged["funding_ma"].iloc[i]
            trend = merged["trend"].iloc[i]

            # Strong negative funding in uptrend = LONG
            if funding_val < -self.funding_threshold and trend == 1:
                signals.iloc[i, signals.columns.get_loc("signal")] = 1
                in_position = True
                bars_since_entry = 0
            # Strong positive funding in downtrend = SHORT
            elif funding_val > self.funding_threshold and trend == -1:
                signals.iloc[i, signals.columns.get_loc("signal")] = -1
                in_position = True
                bars_since_entry = 0

        return signals


# ============================================================================
# STRATEGY 2: MOMENTUM / TREND FOLLOWING
# ============================================================================
class MomentumStrategy:
    """
    Logic:
    - Enter on breakout of N-period high/low
    - ADX > 25 confirms trend strength
    - Trail stop using ATR
    """

    def __init__(
        self,
        breakout_period: int = 20,
        adx_threshold: float = 25,
        atr_multiplier: float = 2.0,
    ):
        self.breakout_period = breakout_period
        self.adx_threshold = adx_threshold
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        # Calculate indicators
        df = df.copy()
        df["highest"] = df["high"].rolling(self.breakout_period).max()
        df["lowest"] = df["low"].rolling(self.breakout_period).min()
        df["adx"] = calc_adx(df["high"], df["low"], df["close"])
        df["atr"] = calc_atr(df["high"], df["low"], df["close"])

        position = 0
        entry_price = 0
        trail_stop = 0

        for i in range(self.breakout_period + 14, len(df)):
            close = df["close"].iloc[i]
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            prev_highest = df["highest"].iloc[i - 1]
            prev_lowest = df["lowest"].iloc[i - 1]
            adx = df["adx"].iloc[i]
            atr = df["atr"].iloc[i]

            # Exit on trailing stop
            if position == 1 and low < trail_stop:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and high > trail_stop:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            # Entry on breakout with ADX confirmation
            if position == 0 and adx > self.adx_threshold:
                if high > prev_highest:  # Breakout up
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                    entry_price = close
                    trail_stop = close - self.atr_multiplier * atr
                elif low < prev_lowest:  # Breakout down
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
                    entry_price = close
                    trail_stop = close + self.atr_multiplier * atr

            # Update trailing stop
            if position == 1:
                new_stop = close - self.atr_multiplier * atr
                trail_stop = max(trail_stop, new_stop)
            elif position == -1:
                new_stop = close + self.atr_multiplier * atr
                trail_stop = min(trail_stop, new_stop)

        return signals


# ============================================================================
# STRATEGY 3: FUNDING ARBITRAGE / CARRY TRADE
# ============================================================================
class CarryTradeStrategy:
    """
    Logic:
    - When funding is extremely positive, GO SHORT to collect funding
    - When funding is extremely negative, GO LONG to collect funding
    - Pure carry, no directional bias (hold through price moves)
    """

    def __init__(
        self,
        funding_extreme: float = 0.001,  # 0.1% per 8h = extreme
        min_holding_periods: int = 9,  # Hold at least 3 funding periods (24h)
        max_holding_periods: int = 63,  # Max 1 week
    ):
        self.funding_extreme = funding_extreme
        self.min_holding_periods = min_holding_periods
        self.max_holding_periods = max_holding_periods

    def generate_signals(self, df: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if funding.empty:
            return signals

        merged = df.copy()
        merged["funding"] = funding["funding_rate"].reindex(df.index, method="ffill")
        merged["funding_ma"] = merged["funding"].rolling(3).mean()

        bars_since_entry = 0
        in_position = False
        position_direction = 0

        for i in range(10, len(merged)):
            funding_val = merged["funding_ma"].iloc[i]

            if in_position:
                bars_since_entry += 1

                # Exit conditions
                should_exit = False
                if bars_since_entry >= self.max_holding_periods:
                    should_exit = True
                elif bars_since_entry >= self.min_holding_periods:
                    # Exit when funding normalizes
                    if position_direction == 1 and funding_val > 0:
                        should_exit = True
                    elif position_direction == -1 and funding_val < 0:
                        should_exit = True

                if should_exit:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    in_position = False
                    position_direction = 0
                continue

            # Entry on extreme funding
            if funding_val < -self.funding_extreme:
                signals.iloc[i, signals.columns.get_loc("signal")] = (
                    1  # Long to collect funding
                )
                in_position = True
                position_direction = 1
                bars_since_entry = 0
            elif funding_val > self.funding_extreme:
                signals.iloc[i, signals.columns.get_loc("signal")] = (
                    -1
                )  # Short to collect funding
                in_position = True
                position_direction = -1
                bars_since_entry = 0

        return signals


# ============================================================================
# STRATEGY 4: BTC DOMINANCE ROTATION
# ============================================================================
class DominanceRotationStrategy:
    """
    Logic:
    - Rising BTC dominance = Risk-off, favor BTC, avoid alts
    - Falling BTC dominance = Risk-on, favor alts over BTC
    - For BTC: Long when dominance rising, Short when falling
    - For Alts: Long when dominance falling, Short when rising
    """

    def __init__(
        self,
        dominance_period: int = 20,
        trend_period: int = 50,
    ):
        self.dominance_period = dominance_period
        self.trend_period = trend_period

    def generate_signals(
        self, df: pd.DataFrame, dominance: pd.DataFrame, is_btc: bool = False
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if dominance.empty:
            return signals

        merged = df.copy()
        dom_col = (
            "dominance" if "dominance" in dominance.columns else dominance.columns[0]
        )
        merged["dominance"] = dominance[dom_col].reindex(df.index, method="ffill")
        merged["dom_ma"] = merged["dominance"].rolling(self.dominance_period).mean()
        merged["dom_rising"] = merged["dom_ma"] > merged["dom_ma"].shift(5)

        # Price trend
        merged["ema"] = calc_ema(merged["close"], self.trend_period)
        merged["uptrend"] = merged["close"] > merged["ema"]

        position = 0

        for i in range(self.trend_period, len(merged)):
            dom_rising = merged["dom_rising"].iloc[i]
            uptrend = merged["uptrend"].iloc[i]

            if is_btc:
                # BTC: Long when dominance rising + uptrend
                if dom_rising and uptrend and position <= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif not dom_rising and not uptrend and position >= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
                elif (dom_rising and not uptrend) or (not dom_rising and uptrend):
                    if position != 0:
                        signals.iloc[i, signals.columns.get_loc("exit")] = True
                        position = 0
            else:
                # Alts: Long when dominance falling + uptrend
                if not dom_rising and uptrend and position <= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif dom_rising and not uptrend and position >= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
                elif (not dom_rising and not uptrend) or (dom_rising and uptrend):
                    if position != 0:
                        signals.iloc[i, signals.columns.get_loc("exit")] = True
                        position = 0

        return signals


# ============================================================================
# STRATEGY 5: VOLATILITY BASED
# ============================================================================
class VolatilityStrategy:
    """
    Logic:
    - Low volatility regime: Mean reversion (buy dips, sell rips)
    - High volatility regime: Trend following (breakout)
    - Volatility measured by ATR percentile
    """

    def __init__(
        self,
        vol_period: int = 20,
        vol_lookback: int = 100,
        low_vol_threshold: float = 30,  # Below 30th percentile
        high_vol_threshold: float = 70,  # Above 70th percentile
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        self.vol_period = vol_period
        self.vol_lookback = vol_lookback
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df.copy()
        df["atr"] = calc_atr(df["high"], df["low"], df["close"], self.vol_period)
        df["atr_pct"] = (
            df["atr"]
            .rolling(self.vol_lookback)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        )

        upper, middle, lower = calc_bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_middle"] = middle

        df["highest"] = df["high"].rolling(20).max()
        df["lowest"] = df["low"].rolling(20).min()

        position = 0

        for i in range(self.vol_lookback, len(df)):
            close = df["close"].iloc[i]
            atr_pct = df["atr_pct"].iloc[i]
            bb_upper = df["bb_upper"].iloc[i]
            bb_lower = df["bb_lower"].iloc[i]
            bb_middle = df["bb_middle"].iloc[i]
            highest = df["highest"].iloc[i - 1]
            lowest = df["lowest"].iloc[i - 1]

            if atr_pct < self.low_vol_threshold:
                # Low vol: Mean reversion
                if close < bb_lower and position <= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif close > bb_upper and position >= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
                elif abs(close - bb_middle) / bb_middle < 0.01 and position != 0:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0

            elif atr_pct > self.high_vol_threshold:
                # High vol: Breakout
                if df["high"].iloc[i] > highest and position <= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif df["low"].iloc[i] < lowest and position >= 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

            else:
                # Neutral vol: Exit positions
                if position != 0:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0

        return signals


# ============================================================================
# STRATEGY 6: SHORT-ONLY (Inverse of Long-Only Fear)
# ============================================================================
class ShortOnlyStrategy:
    """
    Logic (Inverse of v12 Long-Only):
    - ONLY SHORT positions
    - Enter short on Greed in confirmed downtrends
    - Exit on Fear or technical stops

    Hypothesis: If Long-Only works on fear in uptrends,
    Short-Only might work on greed in downtrends
    """

    def __init__(
        self,
        greed_entry: int = 70,
        fear_exit: int = 30,
        trend_deviation: float = 0.01,
        trailing_stop_pct: float = 0.08,
        take_profit_pct: float = 0.15,
    ):
        self.greed_entry = greed_entry
        self.fear_exit = fear_exit
        self.trend_deviation = trend_deviation
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct

    def generate_signals(
        self, df: pd.DataFrame, fear_greed: pd.DataFrame
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if fear_greed.empty:
            return signals

        merged = df.copy()
        fg_col = "value" if "value" in fear_greed.columns else fear_greed.columns[0]
        merged["fg"] = fear_greed[fg_col].reindex(df.index, method="ffill")

        merged["ema50"] = calc_ema(merged["close"], 50)
        merged["ema200"] = calc_ema(merged["close"], 200)

        position = 0
        entry_price = 0
        highest_since_entry = 0

        for i in range(200, len(merged)):
            close = merged["close"].iloc[i]
            high = merged["high"].iloc[i]
            fg = merged["fg"].iloc[i]
            ema50 = merged["ema50"].iloc[i]
            ema200 = merged["ema200"].iloc[i]
            ema200_prev = merged["ema200"].iloc[i - 1]

            # Check downtrend
            ema200_falling = ema200 < ema200_prev
            price_below_ema200 = close < ema200 * (1 - self.trend_deviation)
            ema_aligned = ema50 < ema200
            is_downtrend = price_below_ema200 and (ema200_falling or ema_aligned)

            if position == -1:
                highest_since_entry = max(highest_since_entry, high)

                # Exit conditions
                pnl_pct = (entry_price - close) / entry_price  # Short PnL
                trailing_triggered = (
                    highest_since_entry - close
                ) / highest_since_entry < -self.trailing_stop_pct
                take_profit_hit = pnl_pct >= self.take_profit_pct
                fear_exit = fg <= self.fear_exit

                if trailing_triggered or take_profit_hit or fear_exit:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                continue

            # Entry: Downtrend + Greed
            if position == 0 and is_downtrend and fg >= self.greed_entry:
                signals.iloc[i, signals.columns.get_loc("signal")] = -1
                position = -1
                entry_price = close
                highest_since_entry = high

        return signals


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run_all_strategies():
    loader = DataLoader()
    backtester = BaseBacktester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        leverage=3,
        position_size_pct=0.25,
    )

    # Symbols to test (ones with funding rate data)
    symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "LINKUSDT",
        "DOTUSDT",
        "LTCUSDT",
        "AVAXUSDT",
        "ATOMUSDT",
        "UNIUSDT",
        "MATICUSDT",
        "NEARUSDT",
    ]

    # Load common data
    fear_greed = loader.load_fear_greed()
    btc_dominance = loader.load_btc_dominance()

    # Initialize strategies
    strategies = {
        "1_FundingRate": FundingRateStrategy(),
        "2_Momentum": MomentumStrategy(),
        "3_CarryTrade": CarryTradeStrategy(),
        "4_Dominance": DominanceRotationStrategy(),
        "5_Volatility": VolatilityStrategy(),
        "6_ShortOnly": ShortOnlyStrategy(),
    }

    all_results: Dict[str, List[BacktestResult]] = {name: [] for name in strategies}

    logger.info("=" * 70)
    logger.info("MULTI-STRATEGY BACKTEST - ALL SYMBOLS")
    logger.info("=" * 70)
    logger.info(f"Period: 2020-01-01 ~ Present")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Strategies: {list(strategies.keys())}")
    logger.info("=" * 70)

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*50}")

        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            logger.warning(f"  No data for {symbol}")
            continue

        df = df[df.index >= "2020-01-01"]
        if len(df) < 500:
            logger.warning(f"  Insufficient data for {symbol}")
            continue

        funding = loader.load_funding_rate(symbol)
        is_btc = symbol == "BTCUSDT"

        for strat_name, strategy in strategies.items():
            try:
                if strat_name == "1_FundingRate":
                    signals = strategy.generate_signals(df, funding)
                elif strat_name == "2_Momentum":
                    signals = strategy.generate_signals(df)
                elif strat_name == "3_CarryTrade":
                    signals = strategy.generate_signals(df, funding)
                elif strat_name == "4_Dominance":
                    signals = strategy.generate_signals(df, btc_dominance, is_btc)
                elif strat_name == "5_Volatility":
                    signals = strategy.generate_signals(df)
                elif strat_name == "6_ShortOnly":
                    signals = strategy.generate_signals(df, fear_greed)
                else:
                    continue

                result = backtester.run(df, signals, symbol, strat_name)
                all_results[strat_name].append(result)

                status = "✓" if result.profit_factor > 1.0 else "✗"
                logger.info(
                    f"  {status} {strat_name}: PF={result.profit_factor:.2f}, "
                    f"Ret={result.total_return:+.1f}%, Trades={result.num_trades}"
                )

            except Exception as e:
                logger.error(f"  Error in {strat_name} for {symbol}: {e}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY SUMMARY")
    logger.info("=" * 70)

    summary_data = []
    for strat_name, results in all_results.items():
        if not results:
            continue

        profitable = sum(1 for r in results if r.profit_factor > 1.0)
        avg_pf = np.mean(
            [r.profit_factor for r in results if r.profit_factor < float("inf")]
        )
        avg_ret = np.mean([r.total_return for r in results])
        avg_trades = np.mean([r.num_trades for r in results])
        avg_wr = np.mean([r.win_rate for r in results])

        summary_data.append(
            {
                "Strategy": strat_name,
                "Profitable": f"{profitable}/{len(results)}",
                "Avg_PF": avg_pf,
                "Avg_Return": avg_ret,
                "Avg_WR": avg_wr,
                "Avg_Trades": avg_trades,
            }
        )

        logger.info(f"\n{strat_name}:")
        logger.info(
            f"  Profitable symbols: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)"
        )
        logger.info(f"  Average PF: {avg_pf:.2f}")
        logger.info(f"  Average Return: {avg_ret:+.1f}%")
        logger.info(f"  Average Win Rate: {avg_wr:.1f}%")
        logger.info(f"  Average Trades: {avg_trades:.0f}")

    # Rank strategies
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY RANKING (by profitable symbol %)")
    logger.info("=" * 70)

    summary_df = pd.DataFrame(summary_data)
    summary_df["Prof_Pct"] = summary_df["Profitable"].apply(
        lambda x: int(x.split("/")[0]) / int(x.split("/")[1]) * 100
    )
    summary_df = summary_df.sort_values("Prof_Pct", ascending=False)

    for i, row in summary_df.iterrows():
        logger.info(
            f"  {row['Strategy']}: {row['Profitable']} profitable, "
            f"PF={row['Avg_PF']:.2f}, Ret={row['Avg_Return']:+.1f}%"
        )

    # Best strategy per symbol
    logger.info("\n" + "=" * 70)
    logger.info("BEST STRATEGY PER SYMBOL")
    logger.info("=" * 70)

    for symbol in symbols:
        best_pf = 0
        best_strat = None
        for strat_name, results in all_results.items():
            for r in results:
                if (
                    r.symbol == symbol
                    and r.profit_factor > best_pf
                    and r.profit_factor < float("inf")
                ):
                    best_pf = r.profit_factor
                    best_strat = strat_name
        if best_strat:
            logger.info(f"  {symbol}: {best_strat} (PF={best_pf:.2f})")

    return all_results


if __name__ == "__main__":
    results = run_all_strategies()
