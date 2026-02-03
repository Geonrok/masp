#!/usr/bin/env python3
"""
Binance Futures v7 Hybrid Strategy Backtester

Key Improvements over v6:
1. Regime-based switching: Trend-Following in trends, Mean-Reversion in ranges
2. Only enter on EXTREME sentiment (Fear/Greed <25 or >75)
3. RSI for mean-reversion signals in range markets
4. Better risk management with regime-adaptive stops

Data sources (E:/data/crypto_ohlcv):
- OHLCV data (4H, 1D)
- Fear & Greed Index (2018-2025)
- Funding Rate (2023-2025)
- VIX
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float = 10000.0
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None

    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    # Execution
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    leverage: int = 3
    max_positions: int = 2
    position_size_pct: float = 0.08  # 8% per position


@dataclass
class AlternativeData:
    """Alternative data for enhanced signals."""

    fear_greed: Optional[pd.DataFrame] = None
    funding_rate: Dict[str, pd.DataFrame] = field(default_factory=dict)
    vix: Optional[pd.DataFrame] = None


class LocalDataLoader:
    """Load data from local CSV files."""

    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        """Load OHLCV data from local files."""
        tf_map = {
            "4h": "binance_futures_4h",
            "1d": "binance_futures_1d",
            "1h": "binance_futures_1h",
        }

        folder = tf_map.get(timeframe, "binance_futures_4h")
        file_paths = [
            self.data_root / folder / f"{symbol}.csv",
            self.data_root / folder / f"{symbol.replace('USDT', '')}.csv",
        ]

        for file_path in file_paths:
            if file_path.exists():
                df = pd.read_csv(file_path)

                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")

                df = df[["open", "high", "low", "close", "volume"]]
                df = df.sort_index()
                logger.info(f"Loaded {symbol} {timeframe}: {len(df)} candles")
                return df

        logger.warning(f"No data found for {symbol} {timeframe}")
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        """Load Fear & Greed Index."""
        file_path = self.data_root / "FEAR_GREED_INDEX.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp")
            df = df.rename(columns={"close": "value"})
            logger.info(
                f"Loaded Fear & Greed: {len(df)} records ({df.index[0]} to {df.index[-1]})"
            )
            return df

        logger.warning("Fear & Greed data not found")
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str) -> pd.DataFrame:
        """Load funding rate data."""
        file_path = self.data_root / "binance_funding_rate" / f"{symbol}_funding.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.set_index("datetime")
            logger.info(f"Loaded {symbol} funding rate: {len(df)} records")
            return df

        return pd.DataFrame()

    def load_vix(self) -> pd.DataFrame:
        """Load VIX data."""
        file_path = self.data_root / "macro" / "VIX.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.set_index("datetime")
            logger.info(f"Loaded VIX: {len(df)} records")
            return df

        return pd.DataFrame()


class HybridStrategy:
    """
    Hybrid Strategy: Trend-Following + Mean-Reversion

    Key Logic:
    1. Detect regime (TREND vs RANGE) using ADX
    2. In TREND regime: Follow Supertrend direction
    3. In RANGE regime: Mean-revert at RSI extremes
    4. ONLY enter on extreme Fear/Greed (contrarian timing)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Indicator periods
        self.supertrend_period = 10
        self.supertrend_factor = 3.0
        self.ema_fast = 20
        self.ema_slow = 50
        self.atr_period = 14
        self.adx_period = 14
        self.rsi_period = 14

        # Regime thresholds
        self.adx_trend_threshold = 25  # Above = TREND regime
        self.adx_range_threshold = 20  # Below = RANGE regime

        # Fear & Greed thresholds (only trade at extremes)
        self.fg_extreme_fear = 25  # Contrarian buy signal
        self.fg_extreme_greed = 75  # Contrarian sell signal
        self.fg_no_trade_low = 35  # No trade zone
        self.fg_no_trade_high = 65  # No trade zone

        # RSI thresholds (for mean-reversion)
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # VIX threshold
        self.vix_risk_off = 35

        # Funding rate threshold
        self.funding_extreme = 0.0005

    def calculate_supertrend(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Supertrend indicator."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(self.supertrend_period).mean().values

        hl2 = (high + low) / 2
        upper_band = hl2 + self.supertrend_factor * atr
        lower_band = hl2 - self.supertrend_factor * atr

        supertrend = np.zeros(len(close))
        direction = np.ones(len(close))

        for i in range(1, len(close)):
            if close[i] > upper_band[i - 1]:
                direction[i] = 1
            elif close[i] < lower_band[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]

            if direction[i] == 1:
                supertrend[i] = (
                    max(lower_band[i], supertrend[i - 1])
                    if direction[i - 1] == 1
                    else lower_band[i]
                )
            else:
                supertrend[i] = (
                    min(upper_band[i], supertrend[i - 1])
                    if direction[i - 1] == -1
                    else upper_band[i]
                )

        return supertrend, direction

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros(len(data))
        ema[0] = data[0]
        mult = 2 / (period + 1)

        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1]

        return ema

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ATR."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]

        return pd.Series(tr).rolling(period).mean().values

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ADX."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))

        for i in range(1, len(high)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
            minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

        atr = self.calculate_atr(df, period)

        plus_di = (
            100
            * pd.Series(plus_dm).rolling(period).mean().values
            / np.maximum(atr, 1e-10)
        )
        minus_di = (
            100
            * pd.Series(minus_dm).rolling(period).mean().values
            / np.maximum(atr, 1e-10)
        )

        dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
        adx = pd.Series(dx).rolling(period).mean().values

        return adx

    def calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values

        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_regime(self, adx: float) -> str:
        """Detect market regime."""
        if adx >= self.adx_trend_threshold:
            return "TREND"
        elif adx <= self.adx_range_threshold:
            return "RANGE"
        return "TRANSITION"

    def get_fear_greed_zone(self, fg: float) -> str:
        """Get Fear & Greed zone."""
        if fg <= self.fg_extreme_fear:
            return "EXTREME_FEAR"  # Strong contrarian BUY
        elif fg >= self.fg_extreme_greed:
            return "EXTREME_GREED"  # Strong contrarian SELL
        elif self.fg_no_trade_low <= fg <= self.fg_no_trade_high:
            return "NEUTRAL"  # No trade
        elif fg < self.fg_no_trade_low:
            return "FEAR"  # Moderate fear
        else:
            return "GREED"  # Moderate greed

    def generate_signal(
        self,
        df_4h: pd.DataFrame,
        fear_greed: Optional[float] = None,
        funding_rate: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate trading signal with hybrid approach."""

        if len(df_4h) < 100:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        close = df_4h["close"].values
        current_price = close[-1]

        # Calculate indicators
        supertrend, st_direction = self.calculate_supertrend(df_4h)
        ema_fast = self.calculate_ema(close, self.ema_fast)
        ema_slow = self.calculate_ema(close, self.ema_slow)
        atr = self.calculate_atr(df_4h, self.atr_period)
        adx = self.calculate_adx(df_4h, self.adx_period)
        rsi = self.calculate_rsi(close, self.rsi_period)

        # Current values
        current_st_dir = st_direction[-1]
        current_adx = adx[-1]
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        ema_bullish = ema_fast[-1] > ema_slow[-1]

        # Detect regime
        regime = self.detect_regime(current_adx)

        # Get Fear & Greed zone
        fg_zone = "UNKNOWN"
        if fear_greed is not None:
            fg_zone = self.get_fear_greed_zone(fear_greed)

        # VIX filter
        risk_off = False
        if vix is not None and vix > self.vix_risk_off:
            risk_off = True

        # Funding rate signal
        funding_signal = "NEUTRAL"
        if funding_rate is not None:
            if funding_rate > self.funding_extreme:
                funding_signal = "HIGH"  # Too many longs
            elif funding_rate < -self.funding_extreme:
                funding_signal = "LOW"  # Too many shorts

        # Build reasons list
        reasons = [
            f"Regime={regime}",
            f"ADX={current_adx:.1f}",
            f"RSI={current_rsi:.1f}",
        ]

        if fear_greed is not None:
            reasons.append(f"FG={fear_greed:.0f}({fg_zone})")

        # === Signal Generation Logic ===
        signal = "HOLD"
        position_mult = 1.0
        stop_mult = 2.0  # Default stop multiplier

        # NO TRADE CONDITIONS
        if risk_off:
            reasons.append("VIX_RISK_OFF")
            return {
                "signal": "HOLD",
                "reason": "VIX risk-off",
                "reasons": reasons,
                "price": current_price,
                "atr": current_atr,
                "regime": regime,
            }

        if fg_zone == "NEUTRAL":
            reasons.append("FG_NEUTRAL_ZONE")
            return {
                "signal": "HOLD",
                "reason": "Neutral sentiment zone",
                "reasons": reasons,
                "price": current_price,
                "atr": current_atr,
                "regime": regime,
            }

        # === TREND REGIME: Follow the trend ===
        if regime == "TREND":
            # Entry only on extreme sentiment (contrarian timing in trend direction)
            if fg_zone == "EXTREME_FEAR":
                # Extreme fear in uptrend = buy the dip
                if current_st_dir == 1 and ema_bullish:
                    signal = "LONG"
                    position_mult = 1.0
                    stop_mult = 2.5  # Wider stops in trend
                    reasons.append("TREND_LONG_FEAR_DIP")

            elif fg_zone == "EXTREME_GREED":
                # Extreme greed in downtrend = sell the rally
                if current_st_dir == -1 and not ema_bullish:
                    signal = "SHORT"
                    position_mult = 1.0
                    stop_mult = 2.5
                    reasons.append("TREND_SHORT_GREED_RALLY")

            elif fg_zone == "FEAR":
                # Moderate fear - require stronger confirmation
                if current_st_dir == 1 and ema_bullish and current_rsi < 40:
                    signal = "LONG"
                    position_mult = 0.7
                    stop_mult = 2.0
                    reasons.append("TREND_LONG_MODERATE_FEAR")

            elif fg_zone == "GREED":
                # Moderate greed - require stronger confirmation
                if current_st_dir == -1 and not ema_bullish and current_rsi > 60:
                    signal = "SHORT"
                    position_mult = 0.7
                    stop_mult = 2.0
                    reasons.append("TREND_SHORT_MODERATE_GREED")

        # === RANGE REGIME: Mean reversion ===
        elif regime == "RANGE":
            # Only trade at RSI extremes + sentiment confirmation
            if fg_zone == "EXTREME_FEAR":
                # Extreme fear + RSI oversold = strong buy
                if current_rsi < self.rsi_oversold:
                    signal = "LONG"
                    position_mult = 1.0
                    stop_mult = 1.5  # Tighter stops in range
                    reasons.append("RANGE_LONG_RSI_FEAR")
                elif current_rsi < 40:
                    signal = "LONG"
                    position_mult = 0.5
                    stop_mult = 1.5
                    reasons.append("RANGE_LONG_FEAR")

            elif fg_zone == "EXTREME_GREED":
                # Extreme greed + RSI overbought = strong sell
                if current_rsi > self.rsi_overbought:
                    signal = "SHORT"
                    position_mult = 1.0
                    stop_mult = 1.5
                    reasons.append("RANGE_SHORT_RSI_GREED")
                elif current_rsi > 60:
                    signal = "SHORT"
                    position_mult = 0.5
                    stop_mult = 1.5
                    reasons.append("RANGE_SHORT_GREED")

            elif fg_zone == "FEAR":
                # Moderate fear + oversold RSI
                if current_rsi < self.rsi_oversold - 5:  # RSI < 25
                    signal = "LONG"
                    position_mult = 0.5
                    stop_mult = 1.5
                    reasons.append("RANGE_LONG_OVERSOLD")

            elif fg_zone == "GREED":
                # Moderate greed + overbought RSI
                if current_rsi > self.rsi_overbought + 5:  # RSI > 75
                    signal = "SHORT"
                    position_mult = 0.5
                    stop_mult = 1.5
                    reasons.append("RANGE_SHORT_OVERBOUGHT")

        # Funding rate adjustment
        if signal == "LONG" and funding_signal == "HIGH":
            position_mult *= 0.7  # Reduce size - too many longs
            reasons.append("FR_HIGH_REDUCE")
        elif signal == "SHORT" and funding_signal == "LOW":
            position_mult *= 0.7  # Reduce size - too many shorts
            reasons.append("FR_LOW_REDUCE")

        # Calculate stop loss
        if signal == "LONG":
            stop_loss = current_price - stop_mult * current_atr
        elif signal == "SHORT":
            stop_loss = current_price + stop_mult * current_atr
        else:
            stop_loss = None

        return {
            "signal": signal,
            "reasons": reasons,
            "price": current_price,
            "stop_loss": stop_loss,
            "atr": current_atr,
            "adx": current_adx,
            "rsi": current_rsi,
            "position_mult": position_mult,
            "stop_mult": stop_mult,
            "regime": regime,
            "fg_zone": fg_zone,
            "fear_greed": fear_greed,
            "funding_rate": funding_rate,
        }


class HybridBacktester:
    """Hybrid strategy backtester."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = LocalDataLoader()
        self.strategy = HybridStrategy(config)

        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.alt_data = AlternativeData()

        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def load_data(self) -> bool:
        """Load all data."""
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        for symbol in self.config.symbols:
            df_4h = self.loader.load_ohlcv(symbol, "4h")
            if len(df_4h) > 100:
                self.data_4h[symbol] = df_4h

        if not self.data_4h:
            logger.error("No OHLCV data loaded")
            return False

        self.alt_data.fear_greed = self.loader.load_fear_greed()
        self.alt_data.vix = self.loader.load_vix()

        for symbol in self.config.symbols:
            fr = self.loader.load_funding_rate(symbol)
            if len(fr) > 0:
                self.alt_data.funding_rate[symbol] = fr

        return True

    def get_fear_greed_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get Fear & Greed at timestamp."""
        if self.alt_data.fear_greed is None or self.alt_data.fear_greed.empty:
            return None

        fg = self.alt_data.fear_greed
        date = timestamp.date()

        mask = fg.index.date <= date
        if mask.any():
            idx = fg.index[mask][-1]
            return (
                fg.loc[idx, "value"] if "value" in fg.columns else fg.loc[idx].iloc[0]
            )

        return None

    def get_funding_rate_at(
        self, symbol: str, timestamp: pd.Timestamp
    ) -> Optional[float]:
        """Get funding rate at timestamp."""
        if symbol not in self.alt_data.funding_rate:
            return None

        fr = self.alt_data.funding_rate[symbol]
        mask = fr.index <= timestamp

        if mask.any():
            return fr.loc[fr.index[mask][-1], "fundingRate"]

        return None

    def get_vix_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get VIX at timestamp."""
        if self.alt_data.vix is None or self.alt_data.vix.empty:
            return None

        vix = self.alt_data.vix
        date = timestamp.date()

        mask = vix.index.date <= date
        if mask.any():
            idx = vix.index[mask][-1]
            return vix.loc[idx, "close"]

        return None

    def run(self) -> Dict[str, Any]:
        """Run backtest."""
        logger.info("=" * 60)
        logger.info("RUNNING HYBRID STRATEGY BACKTEST")
        logger.info("=" * 60)

        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = (
            pd.Timestamp(self.config.end_date)
            if self.config.end_date
            else pd.Timestamp.now()
        )

        if "BTCUSDT" not in self.data_4h:
            logger.error("BTC data required")
            return {}

        btc_4h = self.data_4h["BTCUSDT"]
        btc_4h = btc_4h[(btc_4h.index >= start_dt) & (btc_4h.index <= end_dt)]
        timestamps = btc_4h.index.tolist()

        if len(timestamps) < 200:
            logger.error("Insufficient data")
            return {}

        logger.info(f"Period: {timestamps[0]} to {timestamps[-1]}")
        logger.info(f"Candles: {len(timestamps)}")

        # Initialize
        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        self.equity_curve = [capital]
        self.trade_log = []

        stats = {
            "signals_generated": 0,
            "long_signals": 0,
            "short_signals": 0,
            "trades_opened": 0,
            "trades_closed": 0,
            "wins": 0,
            "losses": 0,
            "trend_trades": 0,
            "range_trades": 0,
        }

        for i in range(100, len(timestamps)):
            current_time = timestamps[i]

            # Get alternative data
            fear_greed = self.get_fear_greed_at(current_time)
            vix = self.get_vix_at(current_time)

            for symbol in self.data_4h.keys():
                df_4h = self.data_4h[symbol]
                df_4h_current = df_4h[df_4h.index <= current_time].tail(200)

                if len(df_4h_current) < 100:
                    continue

                funding_rate = self.get_funding_rate_at(symbol, current_time)
                current_price = df_4h_current["close"].iloc[-1]

                # Check exit conditions first
                if symbol in positions:
                    pos = positions[symbol]
                    exit_signal = False
                    exit_reason = ""

                    if pos["side"] == "LONG":
                        pos["highest"] = max(pos["highest"], current_price)

                        # Trailing stop (adaptive based on regime)
                        trailing_stop = pos["highest"] - pos["atr"] * pos["stop_mult"]

                        # Take profit at 3x ATR gain
                        take_profit = pos["entry_price"] + pos["atr"] * 3.0
                        profit_pct = (current_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]

                        # Breakeven after 1.5% profit
                        if profit_pct > 0.015:
                            trailing_stop = max(trailing_stop, pos["entry_price"])

                        if current_price < pos["stop_loss"]:
                            exit_signal = True
                            exit_reason = "stop_loss"
                        elif current_price < trailing_stop:
                            exit_signal = True
                            exit_reason = "trailing_stop"
                        elif current_price >= take_profit:
                            exit_signal = True
                            exit_reason = "take_profit"

                        if exit_signal:
                            exit_price = current_price * (1 - self.config.slippage_pct)
                            pnl_pct = (
                                (exit_price - pos["entry_price"])
                                / pos["entry_price"]
                                * pos["leverage"]
                            )
                            pnl_usd = (
                                pos["size"] * pnl_pct
                                - pos["size"] * self.config.commission_pct * 2
                            )

                            capital += pnl_usd

                            self.trade_log.append(
                                {
                                    "symbol": symbol,
                                    "side": "LONG",
                                    "entry_time": pos["entry_time"],
                                    "entry_price": pos["entry_price"],
                                    "exit_time": current_time,
                                    "exit_price": exit_price,
                                    "pnl_pct": pnl_pct,
                                    "pnl_usd": pnl_usd,
                                    "exit_reason": exit_reason,
                                    "regime": pos.get("regime", "UNKNOWN"),
                                }
                            )

                            stats["trades_closed"] += 1
                            if pnl_usd > 0:
                                stats["wins"] += 1
                            else:
                                stats["losses"] += 1

                            del positions[symbol]
                            continue

                    else:  # SHORT
                        pos["lowest"] = min(pos["lowest"], current_price)

                        trailing_stop = pos["lowest"] + pos["atr"] * pos["stop_mult"]
                        take_profit = pos["entry_price"] - pos["atr"] * 3.0
                        profit_pct = (pos["entry_price"] - current_price) / pos[
                            "entry_price"
                        ]

                        if profit_pct > 0.015:
                            trailing_stop = min(trailing_stop, pos["entry_price"])

                        if current_price > pos["stop_loss"]:
                            exit_signal = True
                            exit_reason = "stop_loss"
                        elif current_price > trailing_stop:
                            exit_signal = True
                            exit_reason = "trailing_stop"
                        elif current_price <= take_profit:
                            exit_signal = True
                            exit_reason = "take_profit"

                        if exit_signal:
                            exit_price = current_price * (1 + self.config.slippage_pct)
                            pnl_pct = (
                                (pos["entry_price"] - exit_price)
                                / pos["entry_price"]
                                * pos["leverage"]
                            )
                            pnl_usd = (
                                pos["size"] * pnl_pct
                                - pos["size"] * self.config.commission_pct * 2
                            )

                            capital += pnl_usd

                            self.trade_log.append(
                                {
                                    "symbol": symbol,
                                    "side": "SHORT",
                                    "entry_time": pos["entry_time"],
                                    "entry_price": pos["entry_price"],
                                    "exit_time": current_time,
                                    "exit_price": exit_price,
                                    "pnl_pct": pnl_pct,
                                    "pnl_usd": pnl_usd,
                                    "exit_reason": exit_reason,
                                    "regime": pos.get("regime", "UNKNOWN"),
                                }
                            )

                            stats["trades_closed"] += 1
                            if pnl_usd > 0:
                                stats["wins"] += 1
                            else:
                                stats["losses"] += 1

                            del positions[symbol]
                            continue

                # Generate new signal
                if (
                    symbol not in positions
                    and len(positions) < self.config.max_positions
                ):
                    signal_result = self.strategy.generate_signal(
                        df_4h_current,
                        fear_greed=fear_greed,
                        funding_rate=funding_rate,
                        vix=vix,
                    )

                    stats["signals_generated"] += 1

                    if signal_result["signal"] in ["LONG", "SHORT"]:
                        if signal_result["signal"] == "LONG":
                            stats["long_signals"] += 1
                        else:
                            stats["short_signals"] += 1

                        # Track regime
                        if signal_result.get("regime") == "TREND":
                            stats["trend_trades"] += 1
                        else:
                            stats["range_trades"] += 1

                        # Drawdown-adjusted position sizing
                        peak_capital = (
                            max(self.equity_curve)
                            if self.equity_curve
                            else self.config.initial_capital
                        )
                        current_dd = (
                            (capital - peak_capital) / peak_capital
                            if peak_capital > 0
                            else 0
                        )

                        dd_mult = 1.0
                        if current_dd < -0.20:
                            dd_mult = 0.25
                        elif current_dd < -0.15:
                            dd_mult = 0.5
                        elif current_dd < -0.10:
                            dd_mult = 0.75

                        entry_price = signal_result["price"]
                        if signal_result["signal"] == "LONG":
                            entry_price *= 1 + self.config.slippage_pct
                        else:
                            entry_price *= 1 - self.config.slippage_pct

                        position_size = (
                            capital
                            * self.config.position_size_pct
                            * signal_result["position_mult"]
                            * dd_mult
                        )

                        positions[symbol] = {
                            "side": signal_result["signal"],
                            "entry_price": entry_price,
                            "entry_time": current_time,
                            "size": position_size,
                            "leverage": self.config.leverage,
                            "stop_loss": signal_result["stop_loss"],
                            "atr": signal_result["atr"],
                            "stop_mult": signal_result["stop_mult"],
                            "highest": entry_price,
                            "lowest": entry_price,
                            "regime": signal_result.get("regime", "UNKNOWN"),
                        }

                        stats["trades_opened"] += 1

                        if stats["trades_opened"] <= 10 or i % 1000 == 0:
                            logger.info(
                                f"[{current_time}] {signal_result['signal']} {symbol} @ {entry_price:.2f} "
                                f"(Regime={signal_result.get('regime')}, FG={fear_greed}, {signal_result['reasons'][-1]})"
                            )

            # Mark-to-market
            portfolio_value = capital
            for symbol, pos in positions.items():
                if symbol in self.data_4h:
                    current_price = (
                        self.data_4h[symbol]
                        .loc[self.data_4h[symbol].index <= current_time]["close"]
                        .iloc[-1]
                    )

                    if pos["side"] == "LONG":
                        unrealized = (current_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]
                    else:
                        unrealized = (pos["entry_price"] - current_price) / pos[
                            "entry_price"
                        ]

                    unrealized *= pos["leverage"]
                    portfolio_value += pos["size"] * unrealized

            self.equity_curve.append(portfolio_value)

        # Close remaining positions
        for symbol, pos in list(positions.items()):
            if symbol in self.data_4h:
                exit_price = self.data_4h[symbol].iloc[-1]["close"]

                if pos["side"] == "LONG":
                    pnl_pct = (
                        (exit_price - pos["entry_price"])
                        / pos["entry_price"]
                        * pos["leverage"]
                    )
                else:
                    pnl_pct = (
                        (pos["entry_price"] - exit_price)
                        / pos["entry_price"]
                        * pos["leverage"]
                    )

                pnl_usd = (
                    pos["size"] * pnl_pct - pos["size"] * self.config.commission_pct * 2
                )
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
                        "exit_reason": "backtest_end",
                    }
                )

                stats["trades_closed"] += 1
                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        return self._calculate_results(stats)

    def _calculate_results(self, stats: Dict) -> Dict[str, Any]:
        """Calculate final results."""
        equity = np.array(self.equity_curve)

        if len(equity) < 2:
            return {"error": "Insufficient data"}

        total_return = (
            equity[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        n_candles = len(equity) - 1
        n_years = n_candles / (252 * 6)
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        returns = np.diff(equity) / equity[:-1]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)
            if np.std(returns) > 0
            else 0
        )

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)

        total_trades = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total_trades if total_trades > 0 else 0

        if self.trade_log:
            gross_profit = sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] > 0)
            gross_loss = abs(
                sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] < 0)
            )
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )
        else:
            profit_factor = 0

        # Analyze by exit reason
        exit_reasons = {}
        for t in self.trade_log:
            reason = t.get("exit_reason", "unknown")
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["pnl"] += t["pnl_usd"]

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "avg_trade_pnl": (
                np.mean([t["pnl_pct"] for t in self.trade_log]) if self.trade_log else 0
            ),
            "long_signals": stats["long_signals"],
            "short_signals": stats["short_signals"],
            "trend_trades": stats["trend_trades"],
            "range_trades": stats["range_trades"],
            "final_capital": equity[-1],
            "exit_reasons": exit_reasons,
        }


def run_period_analysis(config: BacktestConfig) -> Dict[str, Dict]:
    """Run analysis by period."""
    periods = [
        ("2020-01-01", "2020-12-31", "2020"),
        ("2021-01-01", "2021-12-31", "2021"),
        ("2022-01-01", "2022-12-31", "2022"),
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2025-01-01", None, "2025"),
    ]

    results = {}

    for start, end, label in periods:
        period_config = BacktestConfig(
            initial_capital=config.initial_capital,
            start_date=start,
            end_date=end,
            symbols=config.symbols,
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
            leverage=config.leverage,
            max_positions=config.max_positions,
            position_size_pct=config.position_size_pct,
        )

        backtester = HybridBacktester(period_config)
        if backtester.load_data():
            result = backtester.run()
            if "error" not in result:
                results[label] = result
                logger.info(
                    f"\n{label}: Return={result['total_return']*100:.1f}%, MDD={result['max_drawdown']*100:.1f}%, "
                    f"WinRate={result['win_rate']*100:.1f}%, Trades={result['total_trades']}"
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Hybrid Strategy Backtester (v7)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--period-analysis", action="store_true")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(","),
        leverage=args.leverage,
    )

    logger.info("=" * 60)
    logger.info("V7 HYBRID STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Period: {config.start_date} to {config.end_date or 'now'}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Leverage: {config.leverage}x")

    if args.period_analysis:
        logger.info("\n=== PERIOD-BY-PERIOD ANALYSIS ===")
        period_results = run_period_analysis(config)

        logger.info("\n" + "=" * 60)
        logger.info("PERIOD ANALYSIS SUMMARY")
        logger.info("=" * 60)
        for period, result in period_results.items():
            logger.info(
                f"{period}: Return={result['total_return']*100:+.1f}%, "
                f"MDD={result['max_drawdown']*100:.1f}%, "
                f"WinRate={result['win_rate']*100:.1f}%, "
                f"Trades={result['total_trades']}"
            )
        return 0

    backtester = HybridBacktester(config)

    if not backtester.load_data():
        logger.error("Failed to load data")
        return 1

    results = backtester.run()

    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"  Annualized Return: {results['annualized_return']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Total Trades: {results['total_trades']}")
    logger.info(f"  Wins/Losses: {results['wins']}/{results['losses']}")
    logger.info(f"  Avg Trade PnL: {results['avg_trade_pnl']*100:.2f}%")
    logger.info(f"  Trend Trades: {results['trend_trades']}")
    logger.info(f"  Range Trades: {results['range_trades']}")
    logger.info(f"  Final Capital: ${results['final_capital']:,.2f}")

    logger.info("\n  Exit Reasons:")
    for reason, data in results.get("exit_reasons", {}).items():
        logger.info(f"    {reason}: {data['count']} trades, ${data['pnl']:.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("vs TARGETS")
    logger.info("=" * 60)
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}% (target: 48-52%)")
    logger.info(
        f"  Annual Return: {results['annualized_return']*100:.1f}% (target: 25-45%)"
    )
    logger.info(f"  Max MDD: {results['max_drawdown']*100:.1f}% (target: < 25%)")

    # Target check
    targets_met = 0
    if 0.48 <= results["win_rate"] <= 0.52:
        targets_met += 1
    if results["annualized_return"] >= 0.25:
        targets_met += 1
    if results["max_drawdown"] >= -0.25:
        targets_met += 1

    logger.info(f"\n  Targets Met: {targets_met}/3")

    return 0


if __name__ == "__main__":
    sys.exit(main())
