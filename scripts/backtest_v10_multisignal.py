#!/usr/bin/env python3
"""
Binance Futures v10 Multi-Signal Strategy

Integrates multiple data sources for confirmation-based trading:
1. Fear/Greed Index - Sentiment timing (contrarian)
2. Funding Rate - Market positioning (crowded trades)
3. On-chain metrics - Network health
4. RSI - Technical momentum
5. EMA200 - Regime detection

Key improvement: Require 3+ signals to align before entry.
This reduces false signals and improves Profit Factor.

Signal Scoring System:
- Each indicator contributes a score from -2 to +2
- LONG when total score >= +3
- SHORT when total score <= -3
- HOLD otherwise

Target: Profit Factor > 1.3 for real-world viability
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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
    initial_capital: float = 10000.0
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    leverage: int = 3
    max_positions: int = 2
    position_size_pct: float = 0.15  # Base position size
    min_score_long: int = 3   # Minimum score for LONG
    min_score_short: int = -3  # Maximum score for SHORT


class MultiSignalDataLoader:
    """Loads and manages multiple data sources."""

    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        """Load OHLCV data for symbol."""
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")

        for file_path in [
            self.data_root / folder / f"{symbol}.csv",
            self.data_root / folder / f"{symbol.replace('USDT', '')}.csv"
        ]:
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                df = df[["open", "high", "low", "close", "volume"]].sort_index()
                return df
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        """Load Fear & Greed Index."""
        # Try updated file first
        for filename in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            file_path = self.data_root / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                # Handle different column names
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                # Normalize column name
                if "close" in df.columns:
                    df = df.rename(columns={"close": "value"})
                return df[["value"]] if "value" in df.columns else df
        return pd.DataFrame()

    def load_funding_rate(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load Funding Rate data."""
        # Try full history first
        for filename in [f"{symbol}_funding_full.csv", f"{symbol}_funding.csv"]:
            file_path = self.data_root / "binance_funding_rate" / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
                return df[["fundingRate"]]
        return pd.DataFrame()

    def load_onchain_active_addresses(self) -> pd.DataFrame:
        """Load BTC active addresses."""
        file_path = self.data_root / "onchain" / "btc_active_addresses.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            return df
        return pd.DataFrame()

    def load_onchain_tx_count(self) -> pd.DataFrame:
        """Load BTC transaction count."""
        file_path = self.data_root / "onchain" / "btc_tx_count.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            return df
        return pd.DataFrame()

    def load_tvl(self) -> pd.DataFrame:
        """Load DeFi TVL history."""
        file_path = self.data_root / "defillama" / "total_tvl_history.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            # Only use data where TVL > 0
            df = df[df["tvl_usd"] > 0]
            return df
        return pd.DataFrame()

    def load_google_trends(self, keyword: str = "bitcoin") -> pd.DataFrame:
        """Load Google Trends data."""
        file_path = self.data_root / "sentiment" / f"google_trends_{keyword}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            return df
        return pd.DataFrame()

    def load_vix(self) -> pd.DataFrame:
        """Load VIX data."""
        file_path = self.data_root / "macro" / "VIX.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.set_index("datetime")
            return df
        return pd.DataFrame()


class MultiSignalStrategy:
    """Multi-signal confirmation strategy."""

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Technical indicator periods
        self.ema_slow = 200
        self.ema_fast = 50
        self.rsi_period = 14
        self.atr_period = 14

        # Regime detection thresholds
        self.bull_threshold = 1.02
        self.bear_threshold = 0.98

        # Fear/Greed thresholds
        self.fg_extreme_fear = 20
        self.fg_fear = 30
        self.fg_greed = 70
        self.fg_extreme_greed = 80

        # Funding rate thresholds (daily aggregated)
        self.funding_negative = -0.0003  # Negative = shorts paying longs
        self.funding_positive = 0.001    # High positive = longs paying shorts

        # RSI thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Stop loss / Take profit
        self.stop_mult = 2.0
        self.trail_mult = 1.5
        self.take_profit_mult = 3.0

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        ema = np.zeros(len(data))
        ema[0] = data[0]
        mult = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * mult + ema[i-1]
        return ema

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ATR."""
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(high - low,
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                 np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        return pd.Series(tr).rolling(period).mean().values

    def calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        return 100 - (100 / (1 + rs))

    def detect_regime(self, price: float, ema200: float) -> str:
        """Detect market regime."""
        if price > ema200 * self.bull_threshold:
            return "BULL"
        elif price < ema200 * self.bear_threshold:
            return "BEAR"
        return "RANGE"

    def score_fear_greed(self, fg_value: Optional[float], regime: str) -> Tuple[int, str]:
        """Score based on Fear/Greed Index (contrarian)."""
        if fg_value is None:
            return 0, "no_fg"

        # In Bull market: Fear = buy opportunity
        if regime == "BULL":
            if fg_value <= self.fg_extreme_fear:
                return 2, f"extreme_fear_{fg_value:.0f}"
            elif fg_value <= self.fg_fear:
                return 1, f"fear_{fg_value:.0f}"
            elif fg_value >= self.fg_extreme_greed:
                return -1, f"greed_exit_{fg_value:.0f}"
            return 0, f"neutral_fg_{fg_value:.0f}"

        # In Bear market: Greed = sell opportunity
        elif regime == "BEAR":
            if fg_value >= self.fg_extreme_greed:
                return -2, f"extreme_greed_{fg_value:.0f}"
            elif fg_value >= self.fg_greed:
                return -1, f"greed_{fg_value:.0f}"
            elif fg_value <= self.fg_extreme_fear:
                return 1, f"fear_exit_{fg_value:.0f}"
            return 0, f"neutral_fg_{fg_value:.0f}"

        # Range: Mean reversion both ways
        else:
            if fg_value <= self.fg_extreme_fear:
                return 1, f"range_fear_{fg_value:.0f}"
            elif fg_value >= self.fg_extreme_greed:
                return -1, f"range_greed_{fg_value:.0f}"
            return 0, f"range_neutral_{fg_value:.0f}"

    def score_funding_rate(self, funding: Optional[float], regime: str) -> Tuple[int, str]:
        """Score based on Funding Rate."""
        if funding is None:
            return 0, "no_funding"

        # Negative funding = shorts paying longs = bullish (crowded shorts)
        # High positive = longs paying shorts = bearish (crowded longs)

        if regime == "BULL":
            if funding < self.funding_negative:
                return 2, f"neg_funding_{funding:.4f}"  # Strong buy in uptrend
            elif funding > self.funding_positive:
                return -1, f"high_funding_{funding:.4f}"  # Warning
            return 0, f"neutral_funding_{funding:.4f}"

        elif regime == "BEAR":
            if funding > self.funding_positive:
                return -2, f"high_funding_{funding:.4f}"  # Strong sell
            elif funding < self.funding_negative:
                return 1, f"neg_funding_{funding:.4f}"  # Potential reversal
            return 0, f"neutral_funding_{funding:.4f}"

        else:  # RANGE
            if funding < self.funding_negative * 2:
                return 1, f"very_neg_funding_{funding:.4f}"
            elif funding > self.funding_positive * 2:
                return -1, f"very_high_funding_{funding:.4f}"
            return 0, f"range_funding_{funding:.4f}"

    def score_rsi(self, rsi: float, regime: str) -> Tuple[int, str]:
        """Score based on RSI."""
        if regime == "BULL":
            if rsi < self.rsi_oversold:
                return 2, f"oversold_{rsi:.0f}"
            elif rsi < 40:
                return 1, f"low_rsi_{rsi:.0f}"
            elif rsi > 80:
                return -1, f"very_overbought_{rsi:.0f}"
            return 0, f"neutral_rsi_{rsi:.0f}"

        elif regime == "BEAR":
            if rsi > self.rsi_overbought:
                return -2, f"overbought_{rsi:.0f}"
            elif rsi > 60:
                return -1, f"high_rsi_{rsi:.0f}"
            elif rsi < 20:
                return 1, f"very_oversold_{rsi:.0f}"
            return 0, f"neutral_rsi_{rsi:.0f}"

        else:  # RANGE - mean reversion
            if rsi < self.rsi_oversold:
                return 1, f"range_oversold_{rsi:.0f}"
            elif rsi > self.rsi_overbought:
                return -1, f"range_overbought_{rsi:.0f}"
            return 0, f"range_rsi_{rsi:.0f}"

    def score_trend_strength(self, price: float, ema50: float, ema200: float, regime: str) -> Tuple[int, str]:
        """Score based on trend alignment."""
        ema_bullish = ema50 > ema200
        price_above_ema50 = price > ema50

        if regime == "BULL":
            if ema_bullish and price_above_ema50:
                return 1, "strong_uptrend"
            elif not ema_bullish or not price_above_ema50:
                return 0, "weak_trend"
            return 0, "neutral_trend"

        elif regime == "BEAR":
            if not ema_bullish and not price_above_ema50:
                return -1, "strong_downtrend"
            elif ema_bullish or price_above_ema50:
                return 0, "weak_trend"
            return 0, "neutral_trend"

        return 0, "range_trend"

    def score_vix(self, vix: Optional[float]) -> Tuple[int, str]:
        """Score based on VIX (fear indicator)."""
        if vix is None:
            return 0, "no_vix"

        # Very high VIX = market panic = potential opportunity
        # Low VIX = complacency = potential top
        if vix > 35:
            return 1, f"high_vix_{vix:.0f}"  # Panic = contrarian buy
        elif vix > 25:
            return 0, f"elevated_vix_{vix:.0f}"
        elif vix < 15:
            return -1, f"low_vix_{vix:.0f}"  # Complacency warning
        return 0, f"normal_vix_{vix:.0f}"

    def generate_signal(
        self,
        df_4h: pd.DataFrame,
        fear_greed: Optional[float] = None,
        funding_rate: Optional[float] = None,
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate trading signal based on multi-signal scoring."""

        if len(df_4h) < 250:
            return {"signal": "HOLD", "reason": "Insufficient data", "score": 0}

        close = df_4h["close"].values
        current_price = close[-1]

        # Calculate indicators
        ema_fast = self.calculate_ema(close, self.ema_fast)
        ema_slow = self.calculate_ema(close, self.ema_slow)
        atr = self.calculate_atr(df_4h, self.atr_period)
        rsi = self.calculate_rsi(close, self.rsi_period)

        current_ema50 = ema_fast[-1]
        current_ema200 = ema_slow[-1]
        current_atr = atr[-1]
        current_rsi = rsi[-1]

        # Detect regime
        regime = self.detect_regime(current_price, current_ema200)

        # Calculate scores from each signal
        total_score = 0
        reasons = [f"Regime={regime}"]

        # 1. Fear/Greed Score
        fg_score, fg_reason = self.score_fear_greed(fear_greed, regime)
        total_score += fg_score
        if fg_score != 0:
            reasons.append(f"FG:{fg_reason}({fg_score:+d})")

        # 2. Funding Rate Score
        fr_score, fr_reason = self.score_funding_rate(funding_rate, regime)
        total_score += fr_score
        if fr_score != 0:
            reasons.append(f"FR:{fr_reason}({fr_score:+d})")

        # 3. RSI Score
        rsi_score, rsi_reason = self.score_rsi(current_rsi, regime)
        total_score += rsi_score
        if rsi_score != 0:
            reasons.append(f"RSI:{rsi_reason}({rsi_score:+d})")

        # 4. Trend Strength Score
        trend_score, trend_reason = self.score_trend_strength(
            current_price, current_ema50, current_ema200, regime
        )
        total_score += trend_score
        if trend_score != 0:
            reasons.append(f"Trend:{trend_reason}({trend_score:+d})")

        # 5. VIX Score (if available)
        vix_score, vix_reason = self.score_vix(vix)
        total_score += vix_score
        if vix_score != 0:
            reasons.append(f"VIX:{vix_reason}({vix_score:+d})")

        reasons.append(f"Total={total_score:+d}")

        # Determine signal based on total score
        signal = "HOLD"
        position_mult = 1.0

        if total_score >= self.config.min_score_long and regime != "BEAR":
            signal = "LONG"
            position_mult = min(1.0, total_score / 4)  # Scale by conviction
        elif total_score <= self.config.min_score_short and regime != "BULL":
            signal = "SHORT"
            position_mult = min(1.0, abs(total_score) / 4)

        # Calculate stops
        if signal == "LONG":
            stop_loss = current_price - self.stop_mult * current_atr
            take_profit = current_price + self.take_profit_mult * current_atr
        elif signal == "SHORT":
            stop_loss = current_price + self.stop_mult * current_atr
            take_profit = current_price - self.take_profit_mult * current_atr
        else:
            stop_loss = None
            take_profit = None

        return {
            "signal": signal,
            "score": total_score,
            "reasons": reasons,
            "price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": current_atr,
            "rsi": current_rsi,
            "position_mult": position_mult,
            "regime": regime,
            "fear_greed": fear_greed,
            "funding_rate": funding_rate,
        }


class MultiSignalBacktester:
    """Backtester for Multi-Signal Strategy."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = MultiSignalDataLoader()
        self.strategy = MultiSignalStrategy(config)

        # Data storage
        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.fear_greed: Optional[pd.DataFrame] = None
        self.funding_rates: Dict[str, pd.DataFrame] = {}
        self.vix: Optional[pd.DataFrame] = None
        self.active_addresses: Optional[pd.DataFrame] = None
        self.tx_count: Optional[pd.DataFrame] = None
        self.tvl: Optional[pd.DataFrame] = None
        self.google_trends: Optional[pd.DataFrame] = None

        # Results
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def load_data(self) -> bool:
        """Load all data sources."""
        logger.info("Loading data...")

        # OHLCV data
        for symbol in self.config.symbols:
            df = self.loader.load_ohlcv(symbol, "4h")
            if len(df) > 100:
                self.data_4h[symbol] = df
                logger.info(f"  OHLCV {symbol}: {len(df)} candles")

        if not self.data_4h:
            logger.error("No OHLCV data loaded")
            return False

        # Fear & Greed
        self.fear_greed = self.loader.load_fear_greed()
        if not self.fear_greed.empty:
            logger.info(f"  Fear/Greed: {len(self.fear_greed)} records")

        # Funding Rates
        for symbol in self.config.symbols:
            fr = self.loader.load_funding_rate(symbol)
            if not fr.empty:
                self.funding_rates[symbol] = fr
                logger.info(f"  Funding {symbol}: {len(fr)} records")

        # VIX
        self.vix = self.loader.load_vix()
        if not self.vix.empty:
            logger.info(f"  VIX: {len(self.vix)} records")

        # On-chain (for future use)
        self.active_addresses = self.loader.load_onchain_active_addresses()
        self.tx_count = self.loader.load_onchain_tx_count()

        # TVL
        self.tvl = self.loader.load_tvl()
        if not self.tvl.empty:
            logger.info(f"  TVL: {len(self.tvl)} records")

        # Google Trends
        self.google_trends = self.loader.load_google_trends("bitcoin")
        if not self.google_trends.empty:
            logger.info(f"  Google Trends: {len(self.google_trends)} records")

        return True

    def get_fear_greed_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get Fear/Greed value at timestamp."""
        if self.fear_greed is None or self.fear_greed.empty:
            return None
        date = timestamp.date()
        mask = self.fear_greed.index.date <= date
        if mask.any():
            idx = self.fear_greed.index[mask][-1]
            return self.fear_greed.loc[idx, "value"]
        return None

    def get_funding_rate_at(self, symbol: str, timestamp: pd.Timestamp) -> Optional[float]:
        """Get daily aggregated funding rate at timestamp."""
        if symbol not in self.funding_rates or self.funding_rates[symbol].empty:
            return None

        df = self.funding_rates[symbol]
        date = timestamp.date()

        # Get all funding rates for the day (aggregated)
        day_start = pd.Timestamp(date)
        day_end = day_start + pd.Timedelta(days=1)

        mask = (df.index >= day_start - pd.Timedelta(days=1)) & (df.index < day_end)
        if mask.any():
            # Sum of last 24h funding rates (3 x 8-hour periods = daily impact)
            return df.loc[mask, "fundingRate"].sum()
        return None

    def get_vix_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get VIX at timestamp."""
        if self.vix is None or self.vix.empty:
            return None
        date = timestamp.date()
        mask = self.vix.index.date <= date
        if mask.any():
            return self.vix.loc[self.vix.index[mask][-1], "close"]
        return None

    def run(self) -> Dict[str, Any]:
        """Run backtest."""
        logger.info("=" * 60)
        logger.info("RUNNING MULTI-SIGNAL STRATEGY BACKTEST (v10)")
        logger.info("=" * 60)

        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = pd.Timestamp(self.config.end_date) if self.config.end_date else pd.Timestamp.now()

        if "BTCUSDT" not in self.data_4h:
            return {"error": "BTC data required"}

        btc_4h = self.data_4h["BTCUSDT"]
        btc_4h = btc_4h[(btc_4h.index >= start_dt) & (btc_4h.index <= end_dt)]
        timestamps = btc_4h.index.tolist()

        if len(timestamps) < 300:
            return {"error": "Insufficient data"}

        logger.info(f"Period: {timestamps[0]} to {timestamps[-1]}")
        logger.info(f"Min score for LONG: {self.config.min_score_long}")
        logger.info(f"Min score for SHORT: {self.config.min_score_short}")

        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        self.equity_curve = [capital]
        self.trade_log = []

        stats = {
            "long": 0, "short": 0, "wins": 0, "losses": 0,
            "bull_trades": 0, "bear_trades": 0, "range_trades": 0,
            "high_conviction": 0, "low_conviction": 0
        }

        for i in range(250, len(timestamps)):
            current_time = timestamps[i]

            # Get signals data
            fear_greed = self.get_fear_greed_at(current_time)
            vix = self.get_vix_at(current_time)

            for symbol in self.data_4h.keys():
                df_4h = self.data_4h[symbol]
                df_current = df_4h[df_4h.index <= current_time].tail(300)

                if len(df_current) < 250:
                    continue

                current_price = df_current["close"].iloc[-1]
                funding_rate = self.get_funding_rate_at(symbol, current_time)

                # === EXIT CHECK ===
                if symbol in positions:
                    pos = positions[symbol]
                    exit_signal = False
                    exit_reason = ""

                    # Update trailing stop
                    if pos["side"] == "LONG":
                        pos["highest"] = max(pos["highest"], current_price)
                        trailing = pos["highest"] - pos["atr"] * self.strategy.trail_mult
                        profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]

                        # Move stop to breakeven after 2% profit
                        if profit_pct > 0.02:
                            trailing = max(trailing, pos["entry_price"])

                        if current_price <= pos["stop_loss"]:
                            exit_signal, exit_reason = True, "stop_loss"
                        elif current_price <= trailing:
                            exit_signal, exit_reason = True, "trailing_stop"
                        elif current_price >= pos["take_profit"]:
                            exit_signal, exit_reason = True, "take_profit"
                        elif fear_greed and fear_greed >= 75:
                            exit_signal, exit_reason = True, "greed_exit"

                    else:  # SHORT
                        pos["lowest"] = min(pos["lowest"], current_price)
                        trailing = pos["lowest"] + pos["atr"] * self.strategy.trail_mult
                        profit_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

                        if profit_pct > 0.02:
                            trailing = min(trailing, pos["entry_price"])

                        if current_price >= pos["stop_loss"]:
                            exit_signal, exit_reason = True, "stop_loss"
                        elif current_price >= trailing:
                            exit_signal, exit_reason = True, "trailing_stop"
                        elif current_price <= pos["take_profit"]:
                            exit_signal, exit_reason = True, "take_profit"
                        elif fear_greed and fear_greed <= 25:
                            exit_signal, exit_reason = True, "fear_exit"

                    if exit_signal:
                        exit_price = current_price * (1 - self.config.slippage_pct) if pos["side"] == "LONG" \
                            else current_price * (1 + self.config.slippage_pct)

                        if pos["side"] == "LONG":
                            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["leverage"]
                        else:
                            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * pos["leverage"]

                        pnl_usd = pos["size"] * pnl_pct - pos["size"] * self.config.commission_pct * 2
                        capital += pnl_usd

                        self.trade_log.append({
                            "symbol": symbol, "side": pos["side"],
                            "entry_time": pos["entry_time"], "entry_price": pos["entry_price"],
                            "exit_time": current_time, "exit_price": exit_price,
                            "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                            "exit_reason": exit_reason, "regime": pos.get("regime"),
                            "entry_score": pos.get("entry_score", 0),
                        })

                        if pnl_usd > 0:
                            stats["wins"] += 1
                        else:
                            stats["losses"] += 1

                        del positions[symbol]
                        continue

                # === ENTRY CHECK ===
                if symbol not in positions and len(positions) < self.config.max_positions:
                    signal = self.strategy.generate_signal(
                        df_current, fear_greed, funding_rate, vix
                    )

                    if signal["signal"] in ["LONG", "SHORT"]:
                        # Drawdown sizing
                        peak = max(self.equity_curve) if self.equity_curve else self.config.initial_capital
                        dd = (capital - peak) / peak if peak > 0 else 0
                        dd_mult = 0.25 if dd < -0.20 else 0.5 if dd < -0.15 else 0.75 if dd < -0.10 else 1.0

                        entry_price = signal["price"] * (1 + self.config.slippage_pct) if signal["signal"] == "LONG" \
                            else signal["price"] * (1 - self.config.slippage_pct)

                        size = capital * self.config.position_size_pct * signal["position_mult"] * dd_mult

                        positions[symbol] = {
                            "side": signal["signal"],
                            "entry_price": entry_price,
                            "entry_time": current_time,
                            "size": size,
                            "leverage": self.config.leverage,
                            "stop_loss": signal["stop_loss"],
                            "take_profit": signal["take_profit"],
                            "atr": signal["atr"],
                            "highest": entry_price,
                            "lowest": entry_price,
                            "regime": signal["regime"],
                            "entry_score": signal["score"],
                        }

                        # Track stats
                        if signal["signal"] == "LONG":
                            stats["long"] += 1
                        else:
                            stats["short"] += 1

                        regime = signal["regime"]
                        if regime == "BULL":
                            stats["bull_trades"] += 1
                        elif regime == "BEAR":
                            stats["bear_trades"] += 1
                        else:
                            stats["range_trades"] += 1

                        if abs(signal["score"]) >= 4:
                            stats["high_conviction"] += 1
                        else:
                            stats["low_conviction"] += 1

                        # Log first few trades
                        if stats["long"] + stats["short"] <= 20 or i % 2000 == 0:
                            logger.info(
                                f"[{current_time.strftime('%Y-%m-%d')}] {signal['signal']} {symbol} @ {entry_price:.2f} "
                                f"Score={signal['score']:+d} ({signal['reasons'][-1]})"
                            )

            # Mark-to-market
            portfolio_value = capital
            for symbol, pos in positions.items():
                if symbol in self.data_4h:
                    curr_df = self.data_4h[symbol]
                    curr_price = curr_df.loc[curr_df.index <= current_time]["close"].iloc[-1]

                    if pos["side"] == "LONG":
                        unrealized = (curr_price - pos["entry_price"]) / pos["entry_price"]
                    else:
                        unrealized = (pos["entry_price"] - curr_price) / pos["entry_price"]

                    portfolio_value += pos["size"] * unrealized * pos["leverage"]

            self.equity_curve.append(portfolio_value)

        # Close remaining positions
        for symbol, pos in list(positions.items()):
            if symbol in self.data_4h:
                exit_price = self.data_4h[symbol].iloc[-1]["close"]
                if pos["side"] == "LONG":
                    pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["leverage"]
                else:
                    pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"] * pos["leverage"]

                pnl_usd = pos["size"] * pnl_pct - pos["size"] * self.config.commission_pct * 2
                capital += pnl_usd

                self.trade_log.append({
                    "symbol": symbol, "side": pos["side"],
                    "entry_time": pos["entry_time"], "entry_price": pos["entry_price"],
                    "exit_time": timestamps[-1], "exit_price": exit_price,
                    "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                    "exit_reason": "backtest_end",
                })

                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        return self._calculate_results(stats)

    def _calculate_results(self, stats: Dict) -> Dict[str, Any]:
        """Calculate backtest results."""
        equity = np.array(self.equity_curve)

        if len(equity) < 2:
            return {"error": "Insufficient data"}

        total_return = (equity[-1] - self.config.initial_capital) / self.config.initial_capital
        n_years = (len(equity) - 1) / (252 * 6)  # 4h candles
        annualized = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        returns = np.diff(equity) / equity[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 6) if np.std(returns) > 0 else 0

        peak = np.maximum.accumulate(equity)
        max_dd = np.min((equity - peak) / peak)

        total_trades = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total_trades if total_trades > 0 else 0

        # Profit Factor
        if self.trade_log:
            gross_profit = sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] > 0)
            gross_loss = abs(sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] < 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            avg_win = gross_profit / stats["wins"] if stats["wins"] > 0 else 0
            avg_loss = gross_loss / stats["losses"] if stats["losses"] > 0 else 0
        else:
            pf = 0
            avg_win = 0
            avg_loss = 0

        # Exit reason analysis
        exit_reasons = {}
        for t in self.trade_log:
            r = t.get("exit_reason", "unknown")
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t["pnl_usd"]
            if t["pnl_usd"] > 0:
                exit_reasons[r]["wins"] += 1

        # High vs Low conviction analysis
        high_conv_trades = [t for t in self.trade_log if abs(t.get("entry_score", 0)) >= 4]
        low_conv_trades = [t for t in self.trade_log if abs(t.get("entry_score", 0)) < 4]

        high_conv_wr = sum(1 for t in high_conv_trades if t["pnl_usd"] > 0) / len(high_conv_trades) if high_conv_trades else 0
        low_conv_wr = sum(1 for t in low_conv_trades if t["pnl_usd"] > 0) / len(low_conv_trades) if low_conv_trades else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": pf,
            "total_trades": total_trades,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "long_trades": stats["long"],
            "short_trades": stats["short"],
            "bull_trades": stats["bull_trades"],
            "bear_trades": stats["bear_trades"],
            "range_trades": stats["range_trades"],
            "high_conviction_trades": stats["high_conviction"],
            "low_conviction_trades": stats["low_conviction"],
            "high_conviction_wr": high_conv_wr,
            "low_conviction_wr": low_conv_wr,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "final_capital": equity[-1],
            "exit_reasons": exit_reasons,
        }


def run_parameter_sweep(config: BacktestConfig) -> Dict[str, Dict]:
    """Run parameter sweep on score thresholds."""
    results = {}

    for min_score in [2, 3, 4]:
        cfg = BacktestConfig(
            initial_capital=config.initial_capital,
            start_date=config.start_date,
            end_date=config.end_date,
            symbols=config.symbols,
            commission_pct=config.commission_pct,
            slippage_pct=config.slippage_pct,
            leverage=config.leverage,
            max_positions=config.max_positions,
            position_size_pct=config.position_size_pct,
            min_score_long=min_score,
            min_score_short=-min_score,
        )

        bt = MultiSignalBacktester(cfg)
        if bt.load_data():
            res = bt.run()
            if "error" not in res:
                label = f"score_{min_score}"
                results[label] = res
                logger.info(f"\n{label}: Return={res['total_return']*100:+.1f}%, "
                           f"MDD={res['max_drawdown']*100:.1f}%, "
                           f"WR={res['win_rate']*100:.1f}%, "
                           f"PF={res['profit_factor']:.2f}, "
                           f"Trades={res['total_trades']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-Signal Strategy Backtester (v10)")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--leverage", type=int, default=3)
    parser.add_argument("--min-score", type=int, default=3)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")

    args = parser.parse_args()

    config = BacktestConfig(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols.split(","),
        leverage=args.leverage,
        min_score_long=args.min_score,
        min_score_short=-args.min_score,
    )

    logger.info("=" * 60)
    logger.info("V10 MULTI-SIGNAL STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Period: {config.start_date} to {config.end_date or 'now'}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Leverage: {config.leverage}x")
    logger.info(f"Min Score: {config.min_score_long}")

    if args.sweep:
        sweep_results = run_parameter_sweep(config)

        logger.info("\n" + "=" * 60)
        logger.info("PARAMETER SWEEP SUMMARY")
        logger.info("=" * 60)
        for label, res in sweep_results.items():
            logger.info(f"{label}: Return={res['total_return']*100:+.1f}%, "
                       f"PF={res['profit_factor']:.2f}, "
                       f"WR={res['win_rate']*100:.1f}%, "
                       f"Trades={res['total_trades']}")
        return 0

    bt = MultiSignalBacktester(config)
    if not bt.load_data():
        logger.error("Failed to load data")
        return 1

    results = bt.run()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total Return: {results['total_return']*100:.2f}%")
    logger.info(f"  Annualized: {results['annualized_return']*100:.2f}%")
    logger.info(f"  Sharpe: {results['sharpe_ratio']:.2f}")
    logger.info(f"  Max DD: {results['max_drawdown']*100:.2f}%")
    logger.info(f"  Win Rate: {results['win_rate']*100:.1f}%")
    logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"  Trades: {results['total_trades']} (L:{results['long_trades']}/S:{results['short_trades']})")
    logger.info(f"  By Regime: Bull={results['bull_trades']}, Bear={results['bear_trades']}, Range={results['range_trades']}")
    logger.info(f"  Final: ${results['final_capital']:,.2f}")

    logger.info("\n  Conviction Analysis:")
    logger.info(f"    High conviction (score>=4): {results['high_conviction_trades']} trades, WR={results['high_conviction_wr']*100:.1f}%")
    logger.info(f"    Low conviction (score<4): {results['low_conviction_trades']} trades, WR={results['low_conviction_wr']*100:.1f}%")

    logger.info(f"\n  Avg Win: ${results['avg_win']:.2f}, Avg Loss: ${results['avg_loss']:.2f}")

    logger.info("\n  Exit Reasons:")
    for reason, data in results.get("exit_reasons", {}).items():
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        logger.info(f"    {reason}: {data['count']} trades, ${data['pnl']:.2f}, WR={wr:.0f}%")

    logger.info("\n" + "=" * 60)
    logger.info("TARGET CHECK")
    logger.info("=" * 60)
    pf_status = "✓" if results['profit_factor'] >= 1.3 else "✗"
    wr_status = "✓" if 0.48 <= results['win_rate'] <= 0.52 else "✗"
    mdd_status = "✓" if results['max_drawdown'] >= -0.25 else "✗"

    logger.info(f"  [{pf_status}] Profit Factor: {results['profit_factor']:.2f} (target: >= 1.30)")
    logger.info(f"  [{wr_status}] Win Rate: {results['win_rate']*100:.1f}% (target: 48-52%)")
    logger.info(f"  [{mdd_status}] Max MDD: {results['max_drawdown']*100:.1f}% (target: < 25%)")

    if results['profit_factor'] >= 1.3:
        logger.info("\n  ** PROFIT FACTOR TARGET MET - Strategy may be viable for live trading **")
    else:
        logger.info(f"\n  Need PF improvement: {results['profit_factor']:.2f} -> 1.30 ({(1.3 - results['profit_factor'])/results['profit_factor']*100:.1f}% increase needed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
