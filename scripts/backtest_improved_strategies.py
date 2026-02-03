#!/usr/bin/env python3
"""
Improved Strategy Backtester
Based on initial findings:
1. Dominance + Volatility showed promise
2. Need to improve generalization across all symbols
3. Try ensemble approach combining multiple signals

New strategies to test:
A. Improved Dominance (with volatility filter)
B. Improved Volatility (with trend filter)
C. Ensemble: Dominance + Volatility + Momentum
D. Mean Reversion with RSI
E. Multi-Timeframe Momentum
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
# DATA LOADER (same as before)
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

    def load_btc_data(self) -> pd.DataFrame:
        return self.load_ohlcv("BTCUSDT", "4h")


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
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr = calc_atr(high, low, close, period)
    plus_di = 100 * calc_ema(plus_dm, period) / atr
    minus_di = 100 * calc_ema(minus_dm, period) / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return calc_ema(dx, period)


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = calc_sma(series, period)
    std = series.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std


# ============================================================================
# BACKTEST ENGINE
# ============================================================================
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int
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


class Backtester:
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
        self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str, strategy_name: str
    ) -> BacktestResult:
        capital = self.initial_capital
        position = None
        trades: List[Trade] = []
        equity = [capital]

        for i in range(1, len(df)):
            current_price = df["close"].iloc[i]
            signal = signals["signal"].iloc[i] if i < len(signals) else 0
            exit_signal = (
                signals["exit"].iloc[i]
                if i < len(signals) and "exit" in signals.columns
                else False
            )

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
                    pnl_pct -= self.commission_pct * 2
                    pnl = capital * self.position_size_pct * self.leverage * pnl_pct
                    position.exit_time = df.index[i]
                    position.exit_price = exit_price
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct
                    trades.append(position)
                    capital += pnl
                    position = None

            if position is None and signal != 0:
                entry_price = current_price * (1 + self.slippage_pct * signal)
                position = Trade(
                    entry_time=df.index[i],
                    entry_price=entry_price,
                    direction=signal,
                    size=capital * self.position_size_pct * self.leverage / entry_price,
                )

            equity.append(capital if position is None else capital)

        if position is not None:
            exit_price = df["close"].iloc[-1]
            pnl_pct = (exit_price / position.entry_price - 1) * position.direction
            pnl_pct -= self.commission_pct * 2
            position.pnl = capital * self.position_size_pct * self.leverage * pnl_pct
            trades.append(position)
            capital += position.pnl

        equity_series = pd.Series(equity)
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
        )

    def _calc_mdd(self, equity: pd.Series) -> float:
        peak = equity.expanding().max()
        dd = (equity - peak) / peak * 100
        return dd.min()

    def _calc_win_rate(self, trades: List[Trade]) -> float:
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.pnl > 0) / len(trades) * 100

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
        return returns.mean() / returns.std() * np.sqrt(252 * 6)


# ============================================================================
# STRATEGY A: IMPROVED RSI MEAN REVERSION
# ============================================================================
class RSIMeanReversionStrategy:
    """
    Simple RSI mean reversion with trend filter
    - Long when RSI < 30 AND price above EMA200 (oversold in uptrend)
    - Short when RSI > 70 AND price below EMA200 (overbought in downtrend)
    - Exit when RSI crosses 50
    """

    def __init__(
        self, rsi_oversold: int = 30, rsi_overbought: int = 70, rsi_exit: int = 50
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit = rsi_exit

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df.copy()
        df["rsi"] = calc_rsi(df["close"], 14)
        df["ema200"] = calc_ema(df["close"], 200)

        position = 0
        for i in range(200, len(df)):
            rsi = df["rsi"].iloc[i]
            close = df["close"].iloc[i]
            ema200 = df["ema200"].iloc[i]

            # Exit on RSI crossing 50
            if position == 1 and rsi >= self.rsi_exit:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and rsi <= self.rsi_exit:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            # Entry
            if position == 0:
                if rsi < self.rsi_oversold and close > ema200:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif rsi > self.rsi_overbought and close < ema200:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

        return signals


# ============================================================================
# STRATEGY B: BTC CORRELATION STRATEGY
# ============================================================================
class BTCCorrelationStrategy:
    """
    Trade altcoins based on BTC momentum and correlation
    - When BTC is trending up strongly, go long on alts that follow BTC
    - When BTC is weak, go short on alts
    - Use correlation to filter signals
    """

    def __init__(
        self,
        lookback: int = 20,
        corr_threshold: float = 0.5,
        btc_momentum_period: int = 10,
    ):
        self.lookback = lookback
        self.corr_threshold = corr_threshold
        self.btc_momentum_period = btc_momentum_period

    def generate_signals(self, df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if btc_df.empty:
            return signals

        merged = df.copy()
        merged["btc_close"] = btc_df["close"].reindex(df.index, method="ffill")
        merged["btc_ret"] = merged["btc_close"].pct_change(self.btc_momentum_period)
        merged["alt_ret"] = merged["close"].pct_change(self.btc_momentum_period)

        # Rolling correlation
        merged["corr"] = (
            merged["alt_ret"].rolling(self.lookback).corr(merged["btc_ret"])
        )

        # BTC trend
        merged["btc_ema50"] = calc_ema(merged["btc_close"], 50)
        merged["btc_uptrend"] = merged["btc_close"] > merged["btc_ema50"]

        # Alt trend
        merged["ema50"] = calc_ema(merged["close"], 50)

        position = 0
        for i in range(100, len(merged)):
            corr = merged["corr"].iloc[i]
            btc_ret = merged["btc_ret"].iloc[i]
            btc_uptrend = merged["btc_uptrend"].iloc[i]
            close = merged["close"].iloc[i]
            ema50 = merged["ema50"].iloc[i]

            if pd.isna(corr):
                continue

            # Exit when correlation breaks down or trend reverses
            if position != 0:
                if abs(corr) < self.corr_threshold * 0.7:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    continue
                if position == 1 and close < ema50 * 0.97:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    continue
                if position == -1 and close > ema50 * 1.03:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    continue

            # Entry
            if position == 0 and corr > self.corr_threshold:
                if btc_uptrend and btc_ret > 0.03:  # BTC up >3% in 10 bars
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif not btc_uptrend and btc_ret < -0.03:  # BTC down >3%
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

        return signals


# ============================================================================
# STRATEGY C: MULTI-TIMEFRAME TREND
# ============================================================================
class MultiTimeframeTrendStrategy:
    """
    Align multiple timeframe trends
    - Daily trend (EMA50 vs EMA200)
    - 4H trend (EMA20 vs EMA50)
    - Entry when both align
    """

    def __init__(self):
        pass

    def generate_signals(
        self, df: pd.DataFrame, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if df_daily.empty:
            return signals

        # 4H indicators
        df = df.copy()
        df["ema20"] = calc_ema(df["close"], 20)
        df["ema50"] = calc_ema(df["close"], 50)
        df["trend_4h"] = np.where(df["ema20"] > df["ema50"], 1, -1)

        # Daily indicators
        df_daily = df_daily.copy()
        df_daily["ema50_d"] = calc_ema(df_daily["close"], 50)
        df_daily["ema200_d"] = calc_ema(df_daily["close"], 200)
        df_daily["trend_daily"] = np.where(
            df_daily["ema50_d"] > df_daily["ema200_d"], 1, -1
        )

        # Merge daily to 4h
        df["trend_daily"] = df_daily["trend_daily"].reindex(df.index, method="ffill")

        position = 0
        for i in range(200, len(df)):
            trend_4h = df["trend_4h"].iloc[i]
            trend_daily = df["trend_daily"].iloc[i]
            close = df["close"].iloc[i]
            df["ema50"].iloc[i]

            # Exit on trend break
            if position == 1 and (trend_4h == -1 or trend_daily == -1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and (trend_4h == 1 or trend_daily == 1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            # Entry when both trends align
            if position == 0:
                if trend_4h == 1 and trend_daily == 1 and close > df["ema20"].iloc[i]:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif (
                    trend_4h == -1 and trend_daily == -1 and close < df["ema20"].iloc[i]
                ):
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

        return signals


# ============================================================================
# STRATEGY D: ENSEMBLE STRATEGY
# ============================================================================
class EnsembleStrategy:
    """
    Combine multiple signals:
    - RSI
    - Trend (EMA crossover)
    - BTC momentum
    - Volatility regime

    Only trade when >= 3 signals agree
    """

    def __init__(self, min_signals: int = 3):
        self.min_signals = min_signals

    def generate_signals(
        self, df: pd.DataFrame, btc_df: pd.DataFrame, fear_greed: pd.DataFrame
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df.copy()

        # Signal 1: RSI
        df["rsi"] = calc_rsi(df["close"], 14)
        df["rsi_signal"] = np.where(df["rsi"] < 40, 1, np.where(df["rsi"] > 60, -1, 0))

        # Signal 2: EMA trend
        df["ema20"] = calc_ema(df["close"], 20)
        df["ema50"] = calc_ema(df["close"], 50)
        df["trend_signal"] = np.where(df["ema20"] > df["ema50"], 1, -1)

        # Signal 3: BTC momentum
        if not btc_df.empty:
            df["btc_close"] = btc_df["close"].reindex(df.index, method="ffill")
            df["btc_mom"] = df["btc_close"].pct_change(20)
            df["btc_signal"] = np.where(
                df["btc_mom"] > 0.05, 1, np.where(df["btc_mom"] < -0.05, -1, 0)
            )
        else:
            df["btc_signal"] = 0

        # Signal 4: Fear & Greed
        if not fear_greed.empty:
            fg_col = "value" if "value" in fear_greed.columns else fear_greed.columns[0]
            df["fg"] = fear_greed[fg_col].reindex(df.index, method="ffill")
            df["fg_signal"] = np.where(df["fg"] < 30, 1, np.where(df["fg"] > 70, -1, 0))
        else:
            df["fg_signal"] = 0

        # Signal 5: Price vs EMA200
        df["ema200"] = calc_ema(df["close"], 200)
        df["ema200_signal"] = np.where(df["close"] > df["ema200"], 1, -1)

        # Aggregate
        df["total_long"] = (
            (df["rsi_signal"] == 1).astype(int)
            + (df["trend_signal"] == 1).astype(int)
            + (df["btc_signal"] == 1).astype(int)
            + (df["fg_signal"] == 1).astype(int)
            + (df["ema200_signal"] == 1).astype(int)
        )
        df["total_short"] = (
            (df["rsi_signal"] == -1).astype(int)
            + (df["trend_signal"] == -1).astype(int)
            + (df["btc_signal"] == -1).astype(int)
            + (df["fg_signal"] == -1).astype(int)
            + (df["ema200_signal"] == -1).astype(int)
        )

        position = 0
        bars_in_position = 0

        for i in range(200, len(df)):
            total_long = df["total_long"].iloc[i]
            total_short = df["total_short"].iloc[i]

            if position != 0:
                bars_in_position += 1
                # Exit when signals reverse or timeout
                if position == 1 and total_short >= self.min_signals:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    bars_in_position = 0
                elif position == -1 and total_long >= self.min_signals:
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    bars_in_position = 0
                elif bars_in_position > 60:  # Max 10 days
                    signals.iloc[i, signals.columns.get_loc("exit")] = True
                    position = 0
                    bars_in_position = 0
                continue

            if position == 0:
                if total_long >= self.min_signals:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                    bars_in_position = 0
                elif total_short >= self.min_signals:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
                    bars_in_position = 0

        return signals


# ============================================================================
# STRATEGY E: BREAKOUT WITH VOLUME
# ============================================================================
class BreakoutVolumeStrategy:
    """
    Classic breakout strategy with volume confirmation
    - Breakout of N-period high/low
    - Volume > 1.5x average
    - Trail stop with ATR
    """

    def __init__(
        self,
        breakout_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_multiplier: float = 2.0,
    ):
        self.breakout_period = breakout_period
        self.volume_multiplier = volume_multiplier
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df.copy()
        df["highest"] = df["high"].rolling(self.breakout_period).max().shift(1)
        df["lowest"] = df["low"].rolling(self.breakout_period).min().shift(1)
        df["vol_ma"] = df["volume"].rolling(20).mean()
        df["vol_spike"] = df["volume"] > df["vol_ma"] * self.volume_multiplier
        df["atr"] = calc_atr(df["high"], df["low"], df["close"], 14)

        position = 0
        trail_stop = 0

        for i in range(self.breakout_period + 20, len(df)):
            high = df["high"].iloc[i]
            low = df["low"].iloc[i]
            close = df["close"].iloc[i]
            highest = df["highest"].iloc[i]
            lowest = df["lowest"].iloc[i]
            vol_spike = df["vol_spike"].iloc[i]
            atr = df["atr"].iloc[i]

            # Exit on trailing stop
            if position == 1 and low < trail_stop:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and high > trail_stop:
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            # Entry on breakout with volume
            if position == 0 and vol_spike:
                if high > highest:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                    trail_stop = close - self.atr_multiplier * atr
                elif low < lowest:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1
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
# MAIN
# ============================================================================
def run_improved_strategies():
    loader = DataLoader()
    backtester = Backtester(
        initial_capital=10000,
        commission_pct=0.001,
        slippage_pct=0.0005,
        leverage=3,
        position_size_pct=0.25,
    )

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
        "NEARUSDT",
    ]

    fear_greed = loader.load_fear_greed()
    btc_df = loader.load_btc_data()

    strategies = {
        "A_RSIMeanRev": RSIMeanReversionStrategy(),
        "B_BTCCorr": BTCCorrelationStrategy(),
        "C_MultiTF": MultiTimeframeTrendStrategy(),
        "D_Ensemble": EnsembleStrategy(min_signals=3),
        "E_BreakoutVol": BreakoutVolumeStrategy(),
    }

    all_results: Dict[str, List[BacktestResult]] = {name: [] for name in strategies}

    logger.info("=" * 70)
    logger.info("IMPROVED STRATEGY BACKTEST")
    logger.info("=" * 70)

    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {symbol}")
        logger.info(f"{'='*50}")

        df_4h = loader.load_ohlcv(symbol, "4h")
        df_1d = loader.load_ohlcv(symbol, "1d")

        if df_4h.empty:
            logger.warning(f"  No 4h data for {symbol}")
            continue

        df_4h = df_4h[df_4h.index >= "2020-01-01"]
        if len(df_4h) < 500:
            logger.warning(f"  Insufficient data for {symbol}")
            continue

        is_btc = symbol == "BTCUSDT"

        for strat_name, strategy in strategies.items():
            try:
                if strat_name == "A_RSIMeanRev":
                    signals = strategy.generate_signals(df_4h)
                elif strat_name == "B_BTCCorr":
                    if is_btc:
                        # Skip BTC correlation for BTC itself
                        continue
                    signals = strategy.generate_signals(df_4h, btc_df)
                elif strat_name == "C_MultiTF":
                    signals = strategy.generate_signals(df_4h, df_1d)
                elif strat_name == "D_Ensemble":
                    signals = strategy.generate_signals(df_4h, btc_df, fear_greed)
                elif strat_name == "E_BreakoutVol":
                    signals = strategy.generate_signals(df_4h)
                else:
                    continue

                result = backtester.run(df_4h, signals, symbol, strat_name)
                all_results[strat_name].append(result)

                status = "✓" if result.profit_factor > 1.0 else "✗"
                logger.info(
                    f"  {status} {strat_name}: PF={result.profit_factor:.2f}, "
                    f"Ret={result.total_return:+.1f}%, WR={result.win_rate:.0f}%, Trades={result.num_trades}"
                )

            except Exception as e:
                logger.error(f"  Error {strat_name}: {e}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("IMPROVED STRATEGY SUMMARY")
    logger.info("=" * 70)

    for strat_name, results in all_results.items():
        if not results:
            continue

        profitable = sum(1 for r in results if r.profit_factor > 1.0)
        avg_pf = np.mean(
            [r.profit_factor for r in results if r.profit_factor < float("inf")]
        )
        avg_ret = np.mean([r.total_return for r in results])
        avg_wr = np.mean([r.win_rate for r in results])
        avg_trades = np.mean([r.num_trades for r in results])

        logger.info(f"\n{strat_name}:")
        logger.info(
            f"  Profitable: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)"
        )
        logger.info(f"  Avg PF: {avg_pf:.2f}")
        logger.info(f"  Avg Return: {avg_ret:+.1f}%")
        logger.info(f"  Avg WR: {avg_wr:.1f}%")
        logger.info(f"  Avg Trades: {avg_trades:.0f}")

    # Find best overall
    logger.info("\n" + "=" * 70)
    logger.info("RANKING BY PROFITABLE %")
    logger.info("=" * 70)

    rankings = []
    for strat_name, results in all_results.items():
        if results:
            profitable = sum(1 for r in results if r.profit_factor > 1.0)
            pct = profitable / len(results) * 100
            avg_pf = np.mean(
                [r.profit_factor for r in results if r.profit_factor < float("inf")]
            )
            rankings.append((strat_name, profitable, len(results), pct, avg_pf))

    rankings.sort(key=lambda x: (-x[3], -x[4]))
    for name, prof, total, pct, avg_pf in rankings:
        logger.info(f"  {name}: {prof}/{total} ({pct:.0f}%), Avg PF={avg_pf:.2f}")

    return all_results


if __name__ == "__main__":
    results = run_improved_strategies()
