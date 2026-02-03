#!/usr/bin/env python3
"""
Comprehensive Strategy Test - ALL Strategies x ALL Symbols
Tests every strategy on every available symbol to find the best combinations.

Strategies:
1. Multi-Timeframe Trend
2. RSI Mean Reversion
3. Momentum Breakout
4. Volatility Adaptive
5. Funding Rate
6. BTC Correlation
7. Ensemble
8. Long-Only Fear (v12)
9. EMA Crossover Simple
10. Bollinger Mean Reversion
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma = calc_sma(series, period)
    std_val = series.rolling(period).std()
    return sma + std * std_val, sma, sma - std * std_val


# ============================================================================
# DATA LOADER
# ============================================================================
class DataLoader:
    def __init__(self):
        self.data_root = DATA_ROOT
        self._cache = {}

    def get_all_symbols(self) -> List[str]:
        folder = self.data_root / "binance_futures_1d"
        symbols = []
        for f in folder.iterdir():
            if f.suffix == ".csv":
                symbol = f.stem
                if not symbol.endswith("USDT"):
                    symbol = symbol + "USDT"
                symbols.append(symbol)
        return sorted(symbols)

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")

        for suffix in [symbol, symbol.replace("USDT", "")]:
            file_path = self.data_root / folder / f"{suffix}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if "datetime" in df.columns:
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df = df.set_index("datetime")
                    elif "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df.set_index("timestamp")

                    required = ["open", "high", "low", "close", "volume"]
                    if all(c in df.columns for c in required):
                        df = df[required].sort_index()
                        self._cache[cache_key] = df
                        return df.copy()
                except:
                    pass
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str) -> pd.DataFrame:
        for fn in [f"{symbol}_funding_full.csv", f"{symbol}_funding.csv"]:
            fp = self.data_root / "binance_funding_rate" / fn
            if fp.exists():
                try:
                    df = pd.read_csv(fp)
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
                        return df[["funding_rate"]]
                except:
                    pass
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.data_root / fn
            if fp.exists():
                df = pd.read_csv(fp)
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


# ============================================================================
# BACKTESTER
# ============================================================================
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int
    pnl: float = 0.0


@dataclass
class Result:
    symbol: str
    strategy: str
    pf: float
    ret: float
    wr: float
    mdd: float
    trades: int
    sharpe: float


class Backtester:
    def __init__(self, commission=0.001, slippage=0.0005, leverage=3, size=0.25):
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.size = size

    def run(
        self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str, strategy: str
    ) -> Optional[Result]:
        if len(df) < 200 or signals["signal"].abs().sum() == 0:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            sig = signals["signal"].iloc[i] if i < len(signals) else 0
            exit_sig = (
                signals["exit"].iloc[i]
                if i < len(signals) and "exit" in signals.columns
                else False
            )

            if position:
                if exit_sig or (sig != 0 and sig != position.direction):
                    exit_p = price * (1 - self.slippage * position.direction)
                    pnl_pct = (
                        exit_p / position.entry_price - 1
                    ) * position.direction - self.commission * 2
                    pnl = capital * self.size * self.leverage * pnl_pct
                    position.pnl = pnl
                    trades.append(position)
                    capital += pnl
                    position = None

            if not position and sig != 0:
                entry_p = price * (1 + self.slippage * sig)
                position = Trade(df.index[i], entry_p, sig)

            equity.append(capital)

        if position:
            pnl_pct = (
                df["close"].iloc[-1] / position.entry_price - 1
            ) * position.direction - self.commission * 2
            position.pnl = capital * self.size * self.leverage * pnl_pct
            trades.append(position)
            capital += position.pnl

        if len(trades) < 5:
            return None

        equity = pd.Series(equity)
        peak = equity.expanding().max()
        mdd = ((equity - peak) / peak).min() * 100

        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100

        gp = sum(t.pnl for t in trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        rets = equity.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252 * 6) if rets.std() > 0 else 0

        return Result(
            symbol,
            strategy,
            min(pf, 999),
            (capital / init - 1) * 100,
            wr,
            mdd,
            len(trades),
            sharpe,
        )


# ============================================================================
# STRATEGIES
# ============================================================================
class Strategy1_MultiTF:
    """Multi-Timeframe Trend"""

    name = "1_MultiTF"

    def generate(
        self, df_4h: pd.DataFrame, df_1d: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if df_1d.empty or len(df_4h) < 300:
            return signals

        df = df_4h.copy()
        df["ema20"] = calc_ema(df["close"], 20)
        df["ema50"] = calc_ema(df["close"], 50)
        df["trend_4h"] = np.where(df["ema20"] > df["ema50"], 1, -1)

        df_d = df_1d.copy()
        df_d["ema50"] = calc_ema(df_d["close"], 50)
        df_d["ema200"] = calc_ema(df_d["close"], 200)
        df_d["trend_d"] = np.where(df_d["ema50"] > df_d["ema200"], 1, -1)
        df["trend_d"] = df_d["trend_d"].reindex(df.index, method="ffill")

        pos = 0
        for i in range(250, len(df)):
            t4h, td = df["trend_4h"].iloc[i], df["trend_d"].iloc[i]
            if pd.isna(td):
                continue

            if pos == 1 and (t4h == -1 or td == -1):
                signals.iloc[i, 1] = True
                pos = 0
            elif pos == -1 and (t4h == 1 or td == 1):
                signals.iloc[i, 1] = True
                pos = 0

            if pos == 0:
                if t4h == 1 and td == 1:
                    signals.iloc[i, 0] = 1
                    pos = 1
                elif t4h == -1 and td == -1:
                    signals.iloc[i, 0] = -1
                    pos = -1
        return signals


class Strategy2_RSIMeanRev:
    """RSI Mean Reversion"""

    name = "2_RSIMeanRev"

    def generate(self, df_4h: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        df["rsi"] = calc_rsi(df["close"], 14)
        df["ema200"] = calc_ema(df["close"], 200)

        pos = 0
        for i in range(200, len(df)):
            rsi = df["rsi"].iloc[i]
            close = df["close"].iloc[i]
            ema200 = df["ema200"].iloc[i]

            if pos == 1 and rsi >= 50:
                signals.iloc[i, 1] = True
                pos = 0
            elif pos == -1 and rsi <= 50:
                signals.iloc[i, 1] = True
                pos = 0

            if pos == 0:
                if rsi < 30 and close > ema200:
                    signals.iloc[i, 0] = 1
                    pos = 1
                elif rsi > 70 and close < ema200:
                    signals.iloc[i, 0] = -1
                    pos = -1
        return signals


class Strategy3_Momentum:
    """Momentum Breakout with ADX"""

    name = "3_Momentum"

    def generate(self, df_4h: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        df["highest"] = df["high"].rolling(20).max().shift(1)
        df["lowest"] = df["low"].rolling(20).min().shift(1)
        df["adx"] = calc_adx(df["high"], df["low"], df["close"])
        df["atr"] = calc_atr(df["high"], df["low"], df["close"])

        pos = 0
        entry_price = 0
        trail = 0

        for i in range(50, len(df)):
            h, l, c = df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
            highest, lowest = df["highest"].iloc[i], df["lowest"].iloc[i]
            adx, atr = df["adx"].iloc[i], df["atr"].iloc[i]

            if pos == 1 and l < trail:
                signals.iloc[i, 1] = True
                pos = 0
            elif pos == -1 and h > trail:
                signals.iloc[i, 1] = True
                pos = 0

            if pos == 0 and adx > 25:
                if h > highest:
                    signals.iloc[i, 0] = 1
                    pos = 1
                    trail = c - 2 * atr
                elif l < lowest:
                    signals.iloc[i, 0] = -1
                    pos = -1
                    trail = c + 2 * atr

            if pos == 1:
                trail = max(trail, c - 2 * atr)
            elif pos == -1:
                trail = min(trail, c + 2 * atr)
        return signals


class Strategy4_Volatility:
    """Volatility Adaptive"""

    name = "4_Volatility"

    def generate(self, df_4h: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        df["atr"] = calc_atr(df["high"], df["low"], df["close"], 20)
        df["atr_pct"] = (
            df["atr"]
            .rolling(100)
            .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
        )

        bb_upper, bb_mid, bb_lower = calc_bollinger(df["close"], 20, 2)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = bb_upper, bb_mid, bb_lower
        df["highest"] = df["high"].rolling(20).max().shift(1)
        df["lowest"] = df["low"].rolling(20).min().shift(1)

        pos = 0
        for i in range(120, len(df)):
            c = df["close"].iloc[i]
            atr_pct = df["atr_pct"].iloc[i]

            if pd.isna(atr_pct):
                continue

            if atr_pct < 30:  # Low vol: mean reversion
                if pos == 0:
                    if c < df["bb_lower"].iloc[i]:
                        signals.iloc[i, 0] = 1
                        pos = 1
                    elif c > df["bb_upper"].iloc[i]:
                        signals.iloc[i, 0] = -1
                        pos = -1
                elif (
                    pos != 0
                    and abs(c - df["bb_mid"].iloc[i]) / df["bb_mid"].iloc[i] < 0.01
                ):
                    signals.iloc[i, 1] = True
                    pos = 0

            elif atr_pct > 70:  # High vol: breakout
                if pos == 0:
                    if df["high"].iloc[i] > df["highest"].iloc[i]:
                        signals.iloc[i, 0] = 1
                        pos = 1
                    elif df["low"].iloc[i] < df["lowest"].iloc[i]:
                        signals.iloc[i, 0] = -1
                        pos = -1
            else:
                if pos != 0:
                    signals.iloc[i, 1] = True
                    pos = 0
        return signals


class Strategy5_FundingRate:
    """Funding Rate Based"""

    name = "5_FundingRate"

    def generate(
        self, df_4h: pd.DataFrame, funding: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if funding.empty:
            return signals

        df = df_4h.copy()
        df["funding"] = funding["funding_rate"].reindex(df.index, method="ffill")
        df["funding_ma"] = df["funding"].rolling(6).mean()
        df["ema50"] = calc_ema(df["close"], 50)
        df["trend"] = np.where(df["close"] > df["ema50"], 1, -1)

        pos = 0
        bars = 0

        for i in range(60, len(df)):
            if pos != 0:
                bars += 1
                if bars >= 18:
                    signals.iloc[i, 1] = True
                    pos = 0
                    bars = 0
                continue

            fund = df["funding_ma"].iloc[i]
            trend = df["trend"].iloc[i]

            if pd.isna(fund):
                continue

            if fund < -0.0005 and trend == 1:
                signals.iloc[i, 0] = 1
                pos = 1
                bars = 0
            elif fund > 0.0005 and trend == -1:
                signals.iloc[i, 0] = -1
                pos = -1
                bars = 0
        return signals


class Strategy6_BTCCorr:
    """BTC Correlation"""

    name = "6_BTCCorr"

    def generate(
        self, df_4h: pd.DataFrame, btc_df: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if btc_df.empty:
            return signals

        df = df_4h.copy()
        df["btc"] = btc_df["close"].reindex(df.index, method="ffill")
        df["btc_ret"] = df["btc"].pct_change(10)
        df["alt_ret"] = df["close"].pct_change(10)
        df["corr"] = df["alt_ret"].rolling(20).corr(df["btc_ret"])
        df["btc_ema"] = calc_ema(df["btc"], 50)
        df["btc_up"] = df["btc"] > df["btc_ema"]
        df["ema50"] = calc_ema(df["close"], 50)

        pos = 0
        for i in range(100, len(df)):
            corr = df["corr"].iloc[i]
            btc_ret = df["btc_ret"].iloc[i]
            btc_up = df["btc_up"].iloc[i]
            c = df["close"].iloc[i]
            ema = df["ema50"].iloc[i]

            if pd.isna(corr):
                continue

            if pos != 0:
                if abs(corr) < 0.35:
                    signals.iloc[i, 1] = True
                    pos = 0
                elif pos == 1 and c < ema * 0.97:
                    signals.iloc[i, 1] = True
                    pos = 0
                elif pos == -1 and c > ema * 1.03:
                    signals.iloc[i, 1] = True
                    pos = 0
                continue

            if pos == 0 and corr > 0.5:
                if btc_up and btc_ret > 0.03:
                    signals.iloc[i, 0] = 1
                    pos = 1
                elif not btc_up and btc_ret < -0.03:
                    signals.iloc[i, 0] = -1
                    pos = -1
        return signals


class Strategy7_Ensemble:
    """Ensemble of Multiple Signals"""

    name = "7_Ensemble"

    def generate(
        self,
        df_4h: pd.DataFrame,
        btc_df: pd.DataFrame,
        fear_greed: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        df["rsi"] = calc_rsi(df["close"], 14)
        df["rsi_sig"] = np.where(df["rsi"] < 40, 1, np.where(df["rsi"] > 60, -1, 0))

        df["ema20"] = calc_ema(df["close"], 20)
        df["ema50"] = calc_ema(df["close"], 50)
        df["trend_sig"] = np.where(df["ema20"] > df["ema50"], 1, -1)

        df["ema200"] = calc_ema(df["close"], 200)
        df["ema200_sig"] = np.where(df["close"] > df["ema200"], 1, -1)

        if not btc_df.empty:
            df["btc"] = btc_df["close"].reindex(df.index, method="ffill")
            df["btc_mom"] = df["btc"].pct_change(20)
            df["btc_sig"] = np.where(
                df["btc_mom"] > 0.05, 1, np.where(df["btc_mom"] < -0.05, -1, 0)
            )
        else:
            df["btc_sig"] = 0

        if not fear_greed.empty:
            fg_col = "value" if "value" in fear_greed.columns else fear_greed.columns[0]
            df["fg"] = fear_greed[fg_col].reindex(df.index, method="ffill")
            df["fg_sig"] = np.where(df["fg"] < 30, 1, np.where(df["fg"] > 70, -1, 0))
        else:
            df["fg_sig"] = 0

        df["long_score"] = (
            (df["rsi_sig"] == 1).astype(int)
            + (df["trend_sig"] == 1).astype(int)
            + (df["btc_sig"] == 1).astype(int)
            + (df["fg_sig"] == 1).astype(int)
            + (df["ema200_sig"] == 1).astype(int)
        )
        df["short_score"] = (
            (df["rsi_sig"] == -1).astype(int)
            + (df["trend_sig"] == -1).astype(int)
            + (df["btc_sig"] == -1).astype(int)
            + (df["fg_sig"] == -1).astype(int)
            + (df["ema200_sig"] == -1).astype(int)
        )

        pos = 0
        bars = 0
        for i in range(200, len(df)):
            ls, ss = df["long_score"].iloc[i], df["short_score"].iloc[i]

            if pos != 0:
                bars += 1
                if (pos == 1 and ss >= 3) or (pos == -1 and ls >= 3) or bars > 60:
                    signals.iloc[i, 1] = True
                    pos = 0
                    bars = 0
                continue

            if ls >= 3:
                signals.iloc[i, 0] = 1
                pos = 1
                bars = 0
            elif ss >= 3:
                signals.iloc[i, 0] = -1
                pos = -1
                bars = 0
        return signals


class Strategy8_LongOnlyFear:
    """Long-Only Fear Based (v12)"""

    name = "8_LongOnlyFear"

    def generate(
        self, df_4h: pd.DataFrame, fear_greed: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if fear_greed.empty:
            return signals

        df = df_4h.copy()
        fg_col = "value" if "value" in fear_greed.columns else fear_greed.columns[0]
        df["fg"] = fear_greed[fg_col].reindex(df.index, method="ffill")
        df["ema50"] = calc_ema(df["close"], 50)
        df["ema200"] = calc_ema(df["close"], 200)

        pos = 0
        entry_price = 0
        highest = 0

        for i in range(200, len(df)):
            c = df["close"].iloc[i]
            h = df["high"].iloc[i]
            fg = df["fg"].iloc[i]
            ema50 = df["ema50"].iloc[i]
            ema200 = df["ema200"].iloc[i]
            ema200_prev = df["ema200"].iloc[i - 1]

            if pd.isna(fg):
                continue

            uptrend = (c > ema200 * 1.01) and (ema200 > ema200_prev or ema50 > ema200)

            if pos == 1:
                highest = max(highest, h)
                pnl = (c - entry_price) / entry_price
                trail_hit = (highest - c) / highest > 0.08
                tp_hit = pnl >= 0.15
                greed_exit = fg >= 70

                if trail_hit or tp_hit or greed_exit:
                    signals.iloc[i, 1] = True
                    pos = 0
                continue

            if pos == 0 and uptrend and fg <= 35:
                signals.iloc[i, 0] = 1
                pos = 1
                entry_price = c
                highest = h
        return signals


class Strategy9_EMASimple:
    """Simple EMA Crossover"""

    name = "9_EMASimple"

    def generate(self, df_4h: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        df["ema9"] = calc_ema(df["close"], 9)
        df["ema21"] = calc_ema(df["close"], 21)
        df["ema50"] = calc_ema(df["close"], 50)

        pos = 0
        for i in range(60, len(df)):
            ema9 = df["ema9"].iloc[i]
            ema21 = df["ema21"].iloc[i]
            ema50 = df["ema50"].iloc[i]
            ema9_prev = df["ema9"].iloc[i - 1]
            ema21_prev = df["ema21"].iloc[i - 1]

            # Golden cross
            if ema9_prev <= ema21_prev and ema9 > ema21 and df["close"].iloc[i] > ema50:
                if pos != 1:
                    if pos == -1:
                        signals.iloc[i, 1] = True
                    signals.iloc[i, 0] = 1
                    pos = 1
            # Death cross
            elif (
                ema9_prev >= ema21_prev and ema9 < ema21 and df["close"].iloc[i] < ema50
            ):
                if pos != -1:
                    if pos == 1:
                        signals.iloc[i, 1] = True
                    signals.iloc[i, 0] = -1
                    pos = -1
        return signals


class Strategy10_BollingerMR:
    """Bollinger Band Mean Reversion"""

    name = "10_BollingerMR"

    def generate(self, df_4h: pd.DataFrame, **kwargs) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        df = df_4h.copy()
        bb_upper, bb_mid, bb_lower = calc_bollinger(df["close"], 20, 2)
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = bb_upper, bb_mid, bb_lower
        df["rsi"] = calc_rsi(df["close"], 14)

        pos = 0
        for i in range(30, len(df)):
            c = df["close"].iloc[i]
            rsi = df["rsi"].iloc[i]

            if pos == 1:
                if c >= df["bb_mid"].iloc[i] or rsi > 60:
                    signals.iloc[i, 1] = True
                    pos = 0
            elif pos == -1:
                if c <= df["bb_mid"].iloc[i] or rsi < 40:
                    signals.iloc[i, 1] = True
                    pos = 0

            if pos == 0:
                if c < df["bb_lower"].iloc[i] and rsi < 35:
                    signals.iloc[i, 0] = 1
                    pos = 1
                elif c > df["bb_upper"].iloc[i] and rsi > 65:
                    signals.iloc[i, 0] = -1
                    pos = -1
        return signals


# ============================================================================
# MAIN
# ============================================================================
def main():
    loader = DataLoader()
    backtester = Backtester()

    strategies = [
        Strategy1_MultiTF(),
        Strategy2_RSIMeanRev(),
        Strategy3_Momentum(),
        Strategy4_Volatility(),
        Strategy5_FundingRate(),
        Strategy6_BTCCorr(),
        Strategy7_Ensemble(),
        Strategy8_LongOnlyFear(),
        Strategy9_EMASimple(),
        Strategy10_BollingerMR(),
    ]

    all_symbols = loader.get_all_symbols()
    logger.info(f"Total symbols: {len(all_symbols)}")
    logger.info(f"Strategies: {[s.name for s in strategies]}")

    # Load common data
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")
    fear_greed = loader.load_fear_greed()

    results: List[Result] = []
    strategy_stats = defaultdict(
        lambda: {"profitable": 0, "total": 0, "pfs": [], "rets": []}
    )

    logger.info("\n" + "=" * 70)
    logger.info("COMPREHENSIVE TEST - ALL STRATEGIES x ALL SYMBOLS")
    logger.info("=" * 70)

    for idx, symbol in enumerate(all_symbols):
        if (idx + 1) % 100 == 0:
            logger.info(f"Progress: {idx + 1}/{len(all_symbols)}...")

        df_4h = loader.load_ohlcv(symbol, "4h")
        df_1d = loader.load_ohlcv(symbol, "1d")

        if df_4h.empty or len(df_4h) < 500:
            continue

        df_4h = df_4h[df_4h.index >= "2021-01-01"]
        if len(df_4h) < 300:
            continue

        funding = loader.load_funding_rate(symbol)
        is_btc = symbol == "BTCUSDT"

        for strategy in strategies:
            try:
                name = strategy.name

                # Skip BTC correlation for BTC itself
                if name == "6_BTCCorr" and is_btc:
                    continue

                if name == "1_MultiTF":
                    sigs = strategy.generate(df_4h, df_1d)
                elif name == "5_FundingRate":
                    sigs = strategy.generate(df_4h, funding)
                elif name == "6_BTCCorr":
                    sigs = strategy.generate(df_4h, btc_df)
                elif name == "7_Ensemble":
                    sigs = strategy.generate(df_4h, btc_df, fear_greed)
                elif name == "8_LongOnlyFear":
                    sigs = strategy.generate(df_4h, fear_greed)
                else:
                    sigs = strategy.generate(df_4h)

                result = backtester.run(df_4h, sigs, symbol, name)

                if result and result.trades >= 10:
                    results.append(result)
                    strategy_stats[name]["total"] += 1
                    if result.pf > 1.0:
                        strategy_stats[name]["profitable"] += 1
                    if result.pf < 999:
                        strategy_stats[name]["pfs"].append(result.pf)
                    strategy_stats[name]["rets"].append(result.ret)

            except Exception as e:
                continue

    # ========================================================================
    # RESULTS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 70)

    summary = []
    for name in sorted(strategy_stats.keys()):
        stats = strategy_stats[name]
        if stats["total"] == 0:
            continue

        prof_pct = stats["profitable"] / stats["total"] * 100
        avg_pf = np.mean(stats["pfs"]) if stats["pfs"] else 0
        avg_ret = np.mean(stats["rets"]) if stats["rets"] else 0

        summary.append(
            {
                "strategy": name,
                "profitable": stats["profitable"],
                "total": stats["total"],
                "prof_pct": prof_pct,
                "avg_pf": avg_pf,
                "avg_ret": avg_ret,
            }
        )

        logger.info(f"\n{name}:")
        logger.info(
            f"  Profitable: {stats['profitable']}/{stats['total']} ({prof_pct:.1f}%)"
        )
        logger.info(f"  Avg PF: {avg_pf:.2f}")
        logger.info(f"  Avg Return: {avg_ret:+.1f}%")

    # Ranking
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY RANKING (by profitable %)")
    logger.info("=" * 70)

    summary = sorted(summary, key=lambda x: (-x["prof_pct"], -x["avg_pf"]))
    for i, s in enumerate(summary, 1):
        logger.info(
            f"  {i}. {s['strategy']}: {s['profitable']}/{s['total']} ({s['prof_pct']:.1f}%), "
            f"PF={s['avg_pf']:.2f}, Ret={s['avg_ret']:+.1f}%"
        )

    # ========================================================================
    # BEST STRATEGY PER SYMBOL
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TOP SYMBOLS (Best performing across strategies)")
    logger.info("=" * 70)

    # Group by symbol, find best strategy
    symbol_best = {}
    for r in results:
        if r.symbol not in symbol_best or r.pf > symbol_best[r.symbol].pf:
            symbol_best[r.symbol] = r

    # Sort by PF
    top_symbols = sorted(symbol_best.values(), key=lambda x: -x.pf)[:50]

    logger.info(f"\nTop 50 symbols with best single-strategy performance:\n")
    for r in top_symbols:
        logger.info(
            f"  {r.symbol:14s} {r.strategy:16s} PF={r.pf:5.2f} Ret={r.ret:+8.1f}% "
            f"WR={r.wr:4.0f}% MDD={r.mdd:6.1f}% Trades={r.trades:3d}"
        )

    # ========================================================================
    # SYMBOLS PROFITABLE IN MULTIPLE STRATEGIES
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ROBUST SYMBOLS (Profitable in 5+ strategies)")
    logger.info("=" * 70)

    symbol_profit_count = defaultdict(
        lambda: {"count": 0, "strategies": [], "avg_pf": []}
    )
    for r in results:
        if r.pf > 1.0:
            symbol_profit_count[r.symbol]["count"] += 1
            symbol_profit_count[r.symbol]["strategies"].append(r.strategy)
            symbol_profit_count[r.symbol]["avg_pf"].append(r.pf)

    robust = [(s, d) for s, d in symbol_profit_count.items() if d["count"] >= 5]
    robust = sorted(robust, key=lambda x: -x[1]["count"])

    logger.info(f"\nFound {len(robust)} robust symbols:\n")
    for symbol, data in robust[:30]:
        avg_pf = np.mean(data["avg_pf"])
        logger.info(
            f"  {symbol:14s} Profitable in {data['count']:2d} strategies, Avg PF={avg_pf:.2f}"
        )
        logger.info(f"    Strategies: {', '.join(data['strategies'])}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    results_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "strategy": r.strategy,
                "profit_factor": r.pf,
                "return": r.ret,
                "win_rate": r.wr,
                "mdd": r.mdd,
                "trades": r.trades,
                "sharpe": r.sharpe,
            }
            for r in results
        ]
    )

    output_path = Path(
        "E:/투자/Multi-Asset Strategy Platform/data/backtests/comprehensive_all_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # ========================================================================
    # FINAL RECOMMENDATION
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RECOMMENDATION")
    logger.info("=" * 70)

    best_strategy = summary[0] if summary else None
    if best_strategy:
        logger.info(f"\nBest Overall Strategy: {best_strategy['strategy']}")
        logger.info(f"  - {best_strategy['prof_pct']:.1f}% of symbols profitable")
        logger.info(f"  - Average PF: {best_strategy['avg_pf']:.2f}")

    if robust:
        logger.info(f"\nMost Robust Symbols (trade these with any strategy):")
        for symbol, data in robust[:10]:
            logger.info(f"  - {symbol} (profitable in {data['count']} strategies)")

    return results, summary, robust


if __name__ == "__main__":
    results, summary, robust = main()
