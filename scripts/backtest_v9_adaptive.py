#!/usr/bin/env python3
"""
Binance Futures v9 Adaptive Strategy

Final strategy combining best elements:
1. Macro trend detection (EMA200) to determine market bias
2. Fear/Greed timing for optimal entry/exit
3. RSI confirmation for better timing
4. Adaptive position sizing based on conviction

Bull market (Price > EMA200):
- LONG only
- Enter on Fear (<30)
- Exit on Greed (>70) or take profit

Bear market (Price < EMA200):
- SHORT only
- Enter on Greed (>70)
- Exit on Fear (<30) or stop loss

Range market (Sideways near EMA200):
- Mean-reversion with RSI
- Smaller positions
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
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
    initial_capital: float = 10000.0
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    leverage: int = 3
    max_positions: int = 2
    position_size_pct: float = 0.12


class LocalDataLoader:
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
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
                df = df[["open", "high", "low", "close", "volume"]].sort_index()
                return df
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        file_path = self.data_root / "FEAR_GREED_INDEX.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp")
            df = df.rename(columns={"close": "value"})
            return df
        return pd.DataFrame()

    def load_vix(self) -> pd.DataFrame:
        file_path = self.data_root / "macro" / "VIX.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["datetime"])
            df = df.set_index("datetime")
            return df
        return pd.DataFrame()


class AdaptiveStrategy:
    """Adaptive strategy based on market regime."""

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Indicator periods
        self.ema_fast = 50
        self.ema_slow = 200
        self.atr_period = 14
        self.rsi_period = 14

        # Regime detection
        self.bull_threshold = 1.02  # Price > EMA200 * 1.02
        self.bear_threshold = 0.98  # Price < EMA200 * 0.98

        # Fear/Greed thresholds
        self.fg_fear = 30  # Fear zone
        self.fg_greed = 70  # Greed zone
        self.fg_extreme_fear = 20
        self.fg_extreme_greed = 80

        # RSI thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Stops
        self.stop_mult = 2.5
        self.trail_mult = 2.0
        self.take_profit_mult = 3.5

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros(len(data))
        ema[0] = data[0]
        mult = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1]
        return ema

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        high, low, close = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
            ),
        )
        tr[0] = high[0] - low[0]
        return pd.Series(tr).rolling(period).mean().values

    def calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
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

    def generate_signal(
        self,
        df_4h: pd.DataFrame,
        fear_greed: Optional[float],
        vix: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate trading signal based on regime and sentiment."""

        if len(df_4h) < 250:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        close = df_4h["close"].values
        current_price = close[-1]

        # Indicators
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

        # Must have sentiment data
        if fear_greed is None:
            return {"signal": "HOLD", "reason": "No sentiment data", "regime": regime}

        # VIX filter
        if vix is not None and vix > 40:
            return {"signal": "HOLD", "reason": "VIX too high", "regime": regime}

        signal = "HOLD"
        position_mult = 1.0
        reasons = [f"Regime={regime}", f"FG={fear_greed:.0f}", f"RSI={current_rsi:.0f}"]

        # === BULL MARKET: Long only ===
        if regime == "BULL":
            # Extreme fear in bull = strong buy
            if fear_greed <= self.fg_extreme_fear:
                signal = "LONG"
                position_mult = 1.0
                reasons.append("BULL_EXTREME_FEAR")

            # Fear + RSI oversold = buy dip
            elif fear_greed <= self.fg_fear and current_rsi < self.rsi_oversold + 10:
                signal = "LONG"
                position_mult = 0.8
                reasons.append("BULL_FEAR_DIP")

            # Moderate fear + RSI recovering
            elif fear_greed <= 40 and current_rsi < 40 and current_rsi > rsi[-2]:
                signal = "LONG"
                position_mult = 0.5
                reasons.append("BULL_FEAR_RECOVERY")

        # === BEAR MARKET: Short only ===
        elif regime == "BEAR":
            # Extreme greed in bear = strong sell
            if fear_greed >= self.fg_extreme_greed:
                signal = "SHORT"
                position_mult = 1.0
                reasons.append("BEAR_EXTREME_GREED")

            # Greed + RSI overbought = sell rally
            elif fear_greed >= self.fg_greed and current_rsi > self.rsi_overbought - 10:
                signal = "SHORT"
                position_mult = 0.8
                reasons.append("BEAR_GREED_RALLY")

            # Moderate greed + RSI declining
            elif fear_greed >= 60 and current_rsi > 60 and current_rsi < rsi[-2]:
                signal = "SHORT"
                position_mult = 0.5
                reasons.append("BEAR_GREED_DECLINE")

        # === RANGE MARKET: Mean reversion ===
        elif regime == "RANGE":
            # Strong oversold = buy
            if current_rsi < self.rsi_oversold and fear_greed <= self.fg_fear:
                signal = "LONG"
                position_mult = 0.5
                reasons.append("RANGE_OVERSOLD")

            # Strong overbought = sell
            elif current_rsi > self.rsi_overbought and fear_greed >= self.fg_greed:
                signal = "SHORT"
                position_mult = 0.5
                reasons.append("RANGE_OVERBOUGHT")

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
            "reasons": reasons,
            "price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": current_atr,
            "rsi": current_rsi,
            "position_mult": position_mult,
            "regime": regime,
            "fear_greed": fear_greed,
        }


class AdaptiveBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = LocalDataLoader()
        self.strategy = AdaptiveStrategy(config)
        self.data_4h: Dict[str, pd.DataFrame] = {}
        self.fear_greed: Optional[pd.DataFrame] = None
        self.vix: Optional[pd.DataFrame] = None
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []

    def load_data(self) -> bool:
        logger.info("Loading data...")
        for symbol in self.config.symbols:
            df = self.loader.load_ohlcv(symbol, "4h")
            if len(df) > 100:
                self.data_4h[symbol] = df
                logger.info(f"Loaded {symbol}: {len(df)} candles")

        if not self.data_4h:
            return False

        self.fear_greed = self.loader.load_fear_greed()
        self.vix = self.loader.load_vix()

        if not self.fear_greed.empty:
            logger.info(f"Loaded Fear & Greed: {len(self.fear_greed)} records")

        return True

    def get_fear_greed_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        if self.fear_greed is None or self.fear_greed.empty:
            return None
        date = timestamp.date()
        mask = self.fear_greed.index.date <= date
        if mask.any():
            idx = self.fear_greed.index[mask][-1]
            return self.fear_greed.loc[idx, "value"]
        return None

    def get_vix_at(self, timestamp: pd.Timestamp) -> Optional[float]:
        if self.vix is None or self.vix.empty:
            return None
        date = timestamp.date()
        mask = self.vix.index.date <= date
        if mask.any():
            return self.vix.loc[self.vix.index[mask][-1], "close"]
        return None

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("RUNNING ADAPTIVE STRATEGY BACKTEST")
        logger.info("=" * 60)

        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = (
            pd.Timestamp(self.config.end_date)
            if self.config.end_date
            else pd.Timestamp.now()
        )

        if "BTCUSDT" not in self.data_4h:
            return {"error": "BTC data required"}

        btc_4h = self.data_4h["BTCUSDT"]
        btc_4h = btc_4h[(btc_4h.index >= start_dt) & (btc_4h.index <= end_dt)]
        timestamps = btc_4h.index.tolist()

        if len(timestamps) < 300:
            return {"error": "Insufficient data"}

        logger.info(f"Period: {timestamps[0]} to {timestamps[-1]}")

        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        self.equity_curve = [capital]
        self.trade_log = []

        stats = {
            "long": 0,
            "short": 0,
            "wins": 0,
            "losses": 0,
            "bull_trades": 0,
            "bear_trades": 0,
            "range_trades": 0,
        }

        for i in range(250, len(timestamps)):
            current_time = timestamps[i]
            fear_greed = self.get_fear_greed_at(current_time)
            vix = self.get_vix_at(current_time)

            for symbol in self.data_4h.keys():
                df_4h = self.data_4h[symbol]
                df_current = df_4h[df_4h.index <= current_time].tail(300)

                if len(df_current) < 250:
                    continue

                current_price = df_current["close"].iloc[-1]

                # Exit check
                if symbol in positions:
                    pos = positions[symbol]
                    exit_signal = False
                    exit_reason = ""

                    # Update trailing stop
                    if pos["side"] == "LONG":
                        pos["highest"] = max(pos["highest"], current_price)
                        trailing = (
                            pos["highest"] - pos["atr"] * self.strategy.trail_mult
                        )
                        profit_pct = (current_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]

                        if profit_pct > 0.015:
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
                        profit_pct = (pos["entry_price"] - current_price) / pos[
                            "entry_price"
                        ]

                        if profit_pct > 0.015:
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
                        exit_price = (
                            current_price * (1 - self.config.slippage_pct)
                            if pos["side"] == "LONG"
                            else current_price * (1 + self.config.slippage_pct)
                        )

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
                            pos["size"] * pnl_pct
                            - pos["size"] * self.config.commission_pct * 2
                        )
                        capital += pnl_usd

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
                                "exit_reason": exit_reason,
                                "regime": pos.get("regime"),
                            }
                        )

                        if pnl_usd > 0:
                            stats["wins"] += 1
                        else:
                            stats["losses"] += 1

                        del positions[symbol]
                        continue

                # Entry check
                if (
                    symbol not in positions
                    and len(positions) < self.config.max_positions
                ):
                    signal = self.strategy.generate_signal(df_current, fear_greed, vix)

                    if signal["signal"] in ["LONG", "SHORT"]:
                        # Drawdown sizing
                        peak = (
                            max(self.equity_curve)
                            if self.equity_curve
                            else self.config.initial_capital
                        )
                        dd = (capital - peak) / peak if peak > 0 else 0
                        dd_mult = (
                            0.25
                            if dd < -0.20
                            else 0.5 if dd < -0.15 else 0.75 if dd < -0.10 else 1.0
                        )

                        entry_price = (
                            signal["price"] * (1 + self.config.slippage_pct)
                            if signal["signal"] == "LONG"
                            else signal["price"] * (1 - self.config.slippage_pct)
                        )

                        size = (
                            capital
                            * self.config.position_size_pct
                            * signal["position_mult"]
                            * dd_mult
                        )

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
                        }

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

                        if stats["long"] + stats["short"] <= 15 or i % 2000 == 0:
                            logger.info(
                                f"[{current_time}] {signal['signal']} {symbol} @ {entry_price:.2f} "
                                f"({signal['reasons'][-1]})"
                            )

            # Mark-to-market
            portfolio_value = capital
            for symbol, pos in positions.items():
                if symbol in self.data_4h:
                    curr_price = (
                        self.data_4h[symbol]
                        .loc[self.data_4h[symbol].index <= current_time]["close"]
                        .iloc[-1]
                    )

                    if pos["side"] == "LONG":
                        unrealized = (curr_price - pos["entry_price"]) / pos[
                            "entry_price"
                        ]
                    else:
                        unrealized = (pos["entry_price"] - curr_price) / pos[
                            "entry_price"
                        ]

                    portfolio_value += pos["size"] * unrealized * pos["leverage"]

            self.equity_curve.append(portfolio_value)

        # Close remaining
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

                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        return self._calculate_results(stats)

    def _calculate_results(self, stats: Dict) -> Dict[str, Any]:
        equity = np.array(self.equity_curve)

        if len(equity) < 2:
            return {"error": "Insufficient data"}

        total_return = (
            equity[-1] - self.config.initial_capital
        ) / self.config.initial_capital
        n_years = (len(equity) - 1) / (252 * 6)
        annualized = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1

        returns = np.diff(equity) / equity[:-1]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)
            if np.std(returns) > 0
            else 0
        )

        peak = np.maximum.accumulate(equity)
        max_dd = np.min((equity - peak) / peak)

        total_trades = stats["wins"] + stats["losses"]
        win_rate = stats["wins"] / total_trades if total_trades > 0 else 0

        if self.trade_log:
            gross_profit = sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] > 0)
            gross_loss = abs(
                sum(t["pnl_usd"] for t in self.trade_log if t["pnl_usd"] < 0)
            )
            pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        else:
            pf = 0

        exit_reasons = {}
        for t in self.trade_log:
            r = t.get("exit_reason", "unknown")
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t["pnl_usd"]
            if t["pnl_usd"] > 0:
                exit_reasons[r]["wins"] += 1

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
            "final_capital": equity[-1],
            "exit_reasons": exit_reasons,
        }


def run_period_analysis(config: BacktestConfig) -> Dict[str, Dict]:
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
        cfg = BacktestConfig(
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
        bt = AdaptiveBacktester(cfg)
        if bt.load_data():
            res = bt.run()
            if "error" not in res:
                results[label] = res
                logger.info(
                    f"\n{label}: Return={res['total_return']*100:+.1f}%, "
                    f"MDD={res['max_drawdown']*100:.1f}%, "
                    f"WR={res['win_rate']*100:.1f}%, "
                    f"Trades={res['total_trades']}"
                )
    return results


def main():
    parser = argparse.ArgumentParser(description="Adaptive Strategy Backtester (v9)")
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
    logger.info("V9 ADAPTIVE STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {config.symbols}")
    logger.info(f"Period: {config.start_date} to {config.end_date or 'now'}")
    logger.info(f"Capital: ${config.initial_capital:,.2f}")
    logger.info(f"Leverage: {config.leverage}x")

    if args.period_analysis:
        period_results = run_period_analysis(config)

        logger.info("\n" + "=" * 60)
        logger.info("PERIOD SUMMARY")
        logger.info("=" * 60)
        for period, res in period_results.items():
            logger.info(
                f"{period}: Return={res['total_return']*100:+.1f}%, "
                f"MDD={res['max_drawdown']*100:.1f}%, "
                f"WR={res['win_rate']*100:.1f}%, "
                f"Trades={res['total_trades']}, "
                f"L/S={res['long_trades']}/{res['short_trades']}"
            )
        return 0

    bt = AdaptiveBacktester(config)
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
    logger.info(
        f"  Trades: {results['total_trades']} (L:{results['long_trades']}/S:{results['short_trades']})"
    )
    logger.info(
        f"  By Regime: Bull={results['bull_trades']}, Bear={results['bear_trades']}, Range={results['range_trades']}"
    )
    logger.info(f"  Final: ${results['final_capital']:,.2f}")

    logger.info("\n  Exit Reasons:")
    for reason, data in results.get("exit_reasons", {}).items():
        wr = data["wins"] / data["count"] * 100 if data["count"] > 0 else 0
        logger.info(
            f"    {reason}: {data['count']} trades, ${data['pnl']:.2f}, WR={wr:.0f}%"
        )

    logger.info("\n" + "=" * 60)
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
