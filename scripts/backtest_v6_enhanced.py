#!/usr/bin/env python3
"""
Binance Futures v6 Enhanced Backtester

Uses local data from E:/data/crypto_ohlcv including:
- OHLCV data (4H, 1D)
- Fear & Greed Index
- Funding Rate
- VIX (volatility)
- Open Interest
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    leverage: int = 3  # Reduced from 5
    max_positions: int = 2  # Reduced from 3
    position_size_pct: float = 0.10  # Reduced from 15%


@dataclass
class AlternativeData:
    """Alternative data for enhanced signals."""

    fear_greed: Optional[pd.DataFrame] = None
    funding_rate: Dict[str, pd.DataFrame] = field(default_factory=dict)
    vix: Optional[pd.DataFrame] = None
    open_interest: Dict[str, pd.DataFrame] = field(default_factory=dict)


class LocalDataLoader:
    """Load data from local CSV files."""

    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        """Load OHLCV data from local files."""
        # Map timeframe to folder
        tf_map = {
            "4h": "binance_futures_4h",
            "1d": "binance_futures_1d",
            "1h": "binance_futures_1h",
        }

        folder = tf_map.get(timeframe, "binance_futures_4h")

        # Try different file name formats
        file_paths = [
            self.data_root / folder / f"{symbol}.csv",
            self.data_root / folder / f"{symbol.replace('USDT', '')}.csv",
        ]

        for file_path in file_paths:
            if file_path.exists():
                df = pd.read_csv(file_path)

                # Handle different column names
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")

                df = df[["open", "high", "low", "close", "volume"]]
                df = df.sort_index()
                logger.info(
                    f"Loaded {symbol} {timeframe}: {len(df)} candles from {file_path.name}"
                )
                return df

        logger.warning(f"No data found for {symbol} {timeframe}")
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        """Load Fear & Greed Index."""
        file_path = self.data_root / "FEAR_GREED_INDEX.csv"
        if not file_path.exists():
            file_path = self.data_root / "sentiment" / "fear_greed_index.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp")
            df = df.rename(columns={"close": "value"})
            logger.info(f"Loaded Fear & Greed: {len(df)} records")
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

    def load_open_interest(self, symbol: str) -> pd.DataFrame:
        """Load open interest data."""
        file_path = self.data_root / "binance_open_interest" / f"{symbol}_oi.csv"

        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.set_index("datetime")
            return df

        return pd.DataFrame()


class EnhancedStrategy:
    """
    Enhanced v6 Strategy with Alternative Data.

    Key Changes from Original:
    1. Simplified MTF - use Fear/Greed as regime instead
    2. Funding rate as contrarian signal
    3. VIX as risk-off filter
    4. Relaxed quality filters
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

        # Thresholds (more selective)
        self.adx_threshold = 25  # Require stronger trend
        self.fear_greed_extreme_fear = 20  # More extreme fear (contrarian buy)
        self.fear_greed_extreme_greed = 80  # More extreme greed (contrarian caution)
        self.fear_greed_neutral_low = 40  # No trade zone
        self.fear_greed_neutral_high = 60  # No trade zone
        self.vix_risk_off = 35  # Higher threshold
        self.funding_rate_extreme = 0.0005  # More sensitive to funding

    def calculate_supertrend(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Calculate Supertrend indicator."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # ATR
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(self.supertrend_period).mean().values

        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + self.supertrend_factor * atr
        lower_band = hl2 - self.supertrend_factor * atr

        # Supertrend calculation
        supertrend = np.zeros(len(close))
        direction = np.ones(len(close))  # 1 = up, -1 = down

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

    def get_fear_greed_signal(self, fg_value: float) -> str:
        """Get signal from Fear & Greed Index."""
        if fg_value <= self.fear_greed_extreme_fear:
            return "EXTREME_FEAR"  # Contrarian bullish
        elif fg_value >= self.fear_greed_extreme_greed:
            return "EXTREME_GREED"  # Contrarian bearish
        return "NEUTRAL"

    def get_funding_signal(self, funding_rate: float) -> str:
        """Get signal from funding rate."""
        if funding_rate > self.funding_rate_extreme:
            return "HIGH_FUNDING"  # Longs paying shorts - potential short squeeze
        elif funding_rate < -self.funding_rate_extreme:
            return "LOW_FUNDING"  # Shorts paying longs - potential long squeeze
        return "NEUTRAL"

    def generate_signal(
        self,
        df_4h: pd.DataFrame,
        fear_greed: Optional[float] = None,
        funding_rate: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate trading signal with enhanced data."""

        if len(df_4h) < 100:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        close = df_4h["close"].values
        current_price = close[-1]

        # Core indicators
        supertrend, direction = self.calculate_supertrend(df_4h)
        ema_fast = self.calculate_ema(close, self.ema_fast)
        ema_slow = self.calculate_ema(close, self.ema_slow)
        atr = self.calculate_atr(df_4h, self.atr_period)
        adx = self.calculate_adx(df_4h, self.adx_period)

        # Current values
        st_direction = direction[-1]
        ema_bullish = ema_fast[-1] > ema_slow[-1]
        current_adx = adx[-1]
        current_atr = atr[-1]

        # Build signal score
        score = 0
        reasons = []

        # Supertrend signal (Primary - weight 3)
        if st_direction == 1:
            score += 3
            reasons.append("ST_LONG")
        else:
            score -= 3
            reasons.append("ST_SHORT")

        # EMA trend (weight 2)
        if ema_bullish:
            score += 2
            reasons.append("EMA_BULL")
        else:
            score -= 2
            reasons.append("EMA_BEAR")

        # Fear & Greed (Contrarian and filter - weight 2)
        fg_neutral_zone = False
        if fear_greed is not None:
            fg_signal = self.get_fear_greed_signal(fear_greed)
            if fg_signal == "EXTREME_FEAR":
                score += 3  # Strong buy signal in extreme fear
                reasons.append(f"FG_FEAR({fear_greed:.0f})")
            elif fg_signal == "EXTREME_GREED":
                score -= 3  # Strong sell signal in extreme greed
                reasons.append(f"FG_GREED({fear_greed:.0f})")
            elif (
                self.fear_greed_neutral_low
                <= fear_greed
                <= self.fear_greed_neutral_high
            ):
                fg_neutral_zone = True  # Neutral zone - reduce position
                reasons.append(f"FG_NEUTRAL({fear_greed:.0f})")

        # Funding Rate (Contrarian - weight 1)
        if funding_rate is not None:
            fr_signal = self.get_funding_signal(funding_rate)
            if fr_signal == "HIGH_FUNDING":
                score -= 1  # Too many longs, potential drop
                reasons.append(f"FR_HIGH({funding_rate:.4f})")
            elif fr_signal == "LOW_FUNDING":
                score += 1  # Too many shorts, potential squeeze
                reasons.append(f"FR_LOW({funding_rate:.4f})")

        # VIX filter (Risk-off)
        risk_off = False
        if vix is not None and vix > self.vix_risk_off:
            risk_off = True
            reasons.append(f"VIX_HIGH({vix:.1f})")

        # ADX filter (Trend strength) - relaxed
        trend_strong = current_adx > self.adx_threshold

        # Generate final signal - more selective
        signal = "HOLD"
        position_mult = 1.0

        # Higher thresholds for entry
        if score >= 5 and not risk_off and trend_strong:  # Strong bullish
            signal = "LONG"
            position_mult = 1.0 if not fg_neutral_zone else 0.5
        elif score <= -5 and not risk_off and trend_strong:  # Strong bearish
            signal = "SHORT"
            position_mult = 1.0 if not fg_neutral_zone else 0.5
        elif (
            score >= 4 and not risk_off and trend_strong and not fg_neutral_zone
        ):  # Moderate bullish
            signal = "LONG"
            position_mult = 0.7
        elif (
            score <= -4 and not risk_off and trend_strong and not fg_neutral_zone
        ):  # Moderate bearish
            signal = "SHORT"
            position_mult = 0.7

        # Calculate stop loss (wider to reduce whipsaw)
        if signal == "LONG":
            stop_loss = current_price - 2.0 * current_atr
        elif signal == "SHORT":
            stop_loss = current_price + 2.0 * current_atr
        else:
            stop_loss = None

        return {
            "signal": signal,
            "score": score,
            "reasons": reasons,
            "price": current_price,
            "stop_loss": stop_loss,
            "atr": current_atr,
            "adx": current_adx,
            "position_mult": position_mult,
            "risk_off": risk_off,
            "supertrend_dir": st_direction,
            "fear_greed": fear_greed,
            "funding_rate": funding_rate,
        }


class EnhancedBacktester:
    """Enhanced backtester using local data."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = LocalDataLoader()
        self.strategy = EnhancedStrategy(config)

        # Data storage
        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.data_1d: Dict[str, pd.DataFrame] = {}
        self.alt_data = AlternativeData()

        # Results
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def load_data(self) -> bool:
        """Load all data from local files."""
        logger.info("=" * 60)
        logger.info("LOADING LOCAL DATA")
        logger.info("=" * 60)

        # Load OHLCV
        for symbol in self.config.symbols:
            df_4h = self.loader.load_ohlcv(symbol, "4h")
            df_1d = self.loader.load_ohlcv(symbol, "1d")

            if len(df_4h) > 100:
                self.data_4h[symbol] = df_4h
            if len(df_1d) > 50:
                self.data_1d[symbol] = df_1d

        if not self.data_4h:
            logger.error("No OHLCV data loaded")
            return False

        # Load alternative data
        self.alt_data.fear_greed = self.loader.load_fear_greed()
        self.alt_data.vix = self.loader.load_vix()

        for symbol in self.config.symbols:
            fr = self.loader.load_funding_rate(symbol)
            if len(fr) > 0:
                self.alt_data.funding_rate[symbol] = fr

            oi = self.loader.load_open_interest(symbol)
            if len(oi) > 0:
                self.alt_data.open_interest[symbol] = oi

        logger.info(f"Loaded {len(self.data_4h)} symbols with OHLCV data")
        return True

    def get_fear_greed_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get Fear & Greed value at timestamp."""
        if self.alt_data.fear_greed is None or self.alt_data.fear_greed.empty:
            return None

        fg = self.alt_data.fear_greed
        date = timestamp.date()

        # Find closest date
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
        """Run the backtest."""
        logger.info("=" * 60)
        logger.info("RUNNING ENHANCED BACKTEST")
        logger.info("=" * 60)

        # Filter data by date range
        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = (
            pd.Timestamp(self.config.end_date)
            if self.config.end_date
            else pd.Timestamp.now()
        )

        # Use BTC as reference
        if "BTCUSDT" not in self.data_4h:
            logger.error("BTC data required")
            return {}

        btc_4h = self.data_4h["BTCUSDT"]
        btc_4h = btc_4h[(btc_4h.index >= start_dt) & (btc_4h.index <= end_dt)]
        timestamps = btc_4h.index.tolist()

        if len(timestamps) < 200:
            logger.error("Insufficient data for backtest")
            return {}

        logger.info(f"Backtest period: {timestamps[0]} to {timestamps[-1]}")
        logger.info(f"Total candles: {len(timestamps)}")

        # Initialize
        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        self.equity_curve = [capital]
        self.trade_log = []

        # Stats
        stats = {
            "total_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "trades_opened": 0,
            "trades_closed": 0,
            "wins": 0,
            "losses": 0,
        }

        for i in range(100, len(timestamps)):  # Start after warmup
            current_time = timestamps[i]

            # Get alternative data
            fear_greed = self.get_fear_greed_at(current_time)
            vix = self.get_vix_at(current_time)

            # Process each symbol
            for symbol in self.data_4h.keys():
                df_4h = self.data_4h[symbol]

                # Get data up to current time
                df_4h_current = df_4h[df_4h.index <= current_time].tail(200)

                if len(df_4h_current) < 100:
                    continue

                funding_rate = self.get_funding_rate_at(symbol, current_time)

                # Check exit conditions first
                if symbol in positions:
                    pos = positions[symbol]
                    current_price = df_4h_current["close"].iloc[-1]

                    # Update trailing stop and take profit
                    exit_signal = False
                    exit_reason = ""

                    if pos["side"] == "LONG":
                        pos["highest"] = max(pos["highest"], current_price)

                        # Trailing stop (wider - 2.5x ATR from highest)
                        trailing_stop = pos["highest"] - pos["atr"] * 2.5

                        # Take profit at 3x ATR gain
                        take_profit = pos["entry_price"] + pos["atr"] * 3.0
                        profit_pct = (current_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]

                        # Breakeven stop after 1.5x ATR profit
                        if profit_pct > 0.02:  # 2% profit
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

                        # Trailing stop (wider - 2.5x ATR from lowest)
                        trailing_stop = pos["lowest"] + pos["atr"] * 2.5

                        # Take profit at 3x ATR gain
                        take_profit = pos["entry_price"] - pos["atr"] * 3.0
                        profit_pct = (pos["entry_price"] - current_price) / pos[
                            "entry_price"
                        ]

                        # Breakeven stop after 2% profit
                        if profit_pct > 0.02:
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

                    stats["total_signals"] += 1

                    if signal_result["signal"] in ["LONG", "SHORT"]:
                        if signal_result["signal"] == "LONG":
                            stats["long_signals"] += 1
                        else:
                            stats["short_signals"] += 1

                        # Calculate drawdown-adjusted position size
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

                        # Reduce position size in drawdown
                        dd_mult = 1.0
                        if current_dd < -0.20:  # >20% drawdown
                            dd_mult = 0.25
                        elif current_dd < -0.15:  # >15% drawdown
                            dd_mult = 0.5
                        elif current_dd < -0.10:  # >10% drawdown
                            dd_mult = 0.75

                        # Open position
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
                            "highest": entry_price,
                            "lowest": entry_price,
                        }

                        stats["trades_opened"] += 1

                        if i % 500 == 0 or stats["trades_opened"] <= 5:
                            logger.info(
                                f"[{current_time}] {signal_result['signal']} {symbol} @ {entry_price:.2f} "
                                f"(score={signal_result['score']}, FG={fear_greed}, reasons={signal_result['reasons'][:3]})"
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
                        "reason": "End of backtest",
                    }
                )

                stats["trades_closed"] += 1
                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        # Calculate metrics
        return self._calculate_results(stats)

    def _calculate_results(self, stats: Dict) -> Dict[str, Any]:
        """Calculate final results."""
        equity = np.array(self.equity_curve)

        if len(equity) < 2:
            return {"error": "Insufficient data"}

        # Returns
        total_return = (
            equity[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Trading days
        n_candles = len(equity) - 1
        n_years = n_candles / (252 * 6)  # 6 4H candles per day
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        # Sharpe
        returns = np.diff(equity) / equity[:-1]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)
            if np.std(returns) > 0
            else 0
        )

        # Max Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = np.min(drawdown)

        # Win rate
        total_trades = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total_trades if total_trades > 0 else 0

        # Profit factor
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

        results = {
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
            "final_capital": equity[-1],
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Binance Futures v6 Backtester"
    )
    parser.add_argument(
        "--symbols", type=str, default="BTCUSDT,ETHUSDT", help="Comma-separated symbols"
    )
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date")
    parser.add_argument("--end", type=str, default=None, help="End date")
    parser.add_argument(
        "--capital", type=float, default=10000.0, help="Initial capital"
    )
    parser.add_argument("--leverage", type=int, default=5, help="Leverage")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(","),
        leverage=args.leverage,
    )

    logger.info("=" * 60)
    logger.info("ENHANCED V6 STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Period: {config.start_date} to {config.end_date or 'now'}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Leverage: {config.leverage}x")

    backtester = EnhancedBacktester(config)

    if not backtester.load_data():
        logger.error("Failed to load data")
        return 1

    results = backtester.run()

    logger.info("=" * 60)
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
    logger.info(f"  Final Capital: ${results['final_capital']:,.2f}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("vs TARGETS")
    logger.info("=" * 60)
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}% (target: 48-52%)")
    logger.info(
        f"  Annual Return: {results['annualized_return']*100:.1f}% (target: 25-45%)"
    )
    logger.info(f"  Max MDD: {results['max_drawdown']*100:.1f}% (target: < 25%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
