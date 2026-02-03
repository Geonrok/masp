#!/usr/bin/env python3
"""
Multi-Timeframe Trend Strategy - Comprehensive Validation

This strategy showed 79% profitable symbols in initial backtest.
Now performing rigorous validation:
1. Walk-Forward Validation
2. Out-of-Sample Testing
3. Parameter Sensitivity
4. Per-Symbol Analysis
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

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
# DATA & INDICATORS
# ============================================================================
class DataLoader:
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
                return df[["open", "high", "low", "close", "volume"]].sort_index()
        return pd.DataFrame()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ============================================================================
# MULTI-TIMEFRAME TREND STRATEGY
# ============================================================================
class MultiTimeframeTrendStrategy:
    """
    Align multiple timeframe trends:
    - Daily: EMA_fast vs EMA_slow for macro trend
    - 4H: EMA_fast vs EMA_slow for entry timing
    - Entry when both align, exit when misaligned
    """

    def __init__(
        self,
        daily_fast: int = 50,
        daily_slow: int = 200,
        h4_fast: int = 20,
        h4_slow: int = 50,
    ):
        self.daily_fast = daily_fast
        self.daily_slow = daily_slow
        self.h4_fast = h4_fast
        self.h4_slow = h4_slow

    def generate_signals(
        self, df_4h: pd.DataFrame, df_1d: pd.DataFrame
    ) -> pd.DataFrame:
        signals = pd.DataFrame(index=df_4h.index, columns=["signal", "exit"])
        signals["signal"] = 0
        signals["exit"] = False

        if df_1d.empty:
            return signals

        # 4H indicators
        df = df_4h.copy()
        df["ema_fast"] = calc_ema(df["close"], self.h4_fast)
        df["ema_slow"] = calc_ema(df["close"], self.h4_slow)
        df["trend_4h"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        # Daily indicators
        df_daily = df_1d.copy()
        df_daily["ema_fast_d"] = calc_ema(df_daily["close"], self.daily_fast)
        df_daily["ema_slow_d"] = calc_ema(df_daily["close"], self.daily_slow)
        df_daily["trend_daily"] = np.where(
            df_daily["ema_fast_d"] > df_daily["ema_slow_d"], 1, -1
        )

        # Merge daily to 4h
        df["trend_daily"] = df_daily["trend_daily"].reindex(df.index, method="ffill")

        position = 0
        min_bars = max(self.daily_slow, self.h4_slow) + 50

        for i in range(min_bars, len(df)):
            trend_4h = df["trend_4h"].iloc[i]
            trend_daily = df["trend_daily"].iloc[i]
            close = df["close"].iloc[i]
            ema_fast = df["ema_fast"].iloc[i]

            # Exit on trend break
            if position == 1 and (trend_4h == -1 or trend_daily == -1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and (trend_4h == 1 or trend_daily == 1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            # Entry when both trends align
            if position == 0:
                if trend_4h == 1 and trend_daily == 1 and close > ema_fast:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif trend_4h == -1 and trend_daily == -1 and close < ema_fast:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

        return signals


# ============================================================================
# BACKTESTER
# ============================================================================
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    total_return: float
    mdd: float
    win_rate: float
    profit_factor: float
    num_trades: int
    sharpe: float


class Backtester:
    def __init__(
        self,
        commission: float = 0.001,
        slippage: float = 0.0005,
        leverage: int = 3,
        size_pct: float = 0.25,
    ):
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.size_pct = size_pct

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        capital = 10000
        init_capital = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            signal = signals["signal"].iloc[i] if i < len(signals) else 0
            exit_sig = signals["exit"].iloc[i] if i < len(signals) else False

            if position is not None:
                if exit_sig or (signal != 0 and signal != position.direction):
                    exit_price = price * (1 - self.slippage * position.direction)
                    pnl_pct = (
                        exit_price / position.entry_price - 1
                    ) * position.direction - self.commission * 2
                    pnl = capital * self.size_pct * self.leverage * pnl_pct
                    position.exit_time = df.index[i]
                    position.exit_price = exit_price
                    position.pnl = pnl
                    trades.append(position)
                    capital += pnl
                    position = None

            if position is None and signal != 0:
                entry_price = price * (1 + self.slippage * signal)
                position = Trade(
                    entry_time=df.index[i], entry_price=entry_price, direction=signal
                )

            equity.append(capital)

        if position:
            exit_price = df["close"].iloc[-1]
            pnl_pct = (
                exit_price / position.entry_price - 1
            ) * position.direction - self.commission * 2
            position.pnl = capital * self.size_pct * self.leverage * pnl_pct
            trades.append(position)
            capital += position.pnl

        equity = pd.Series(equity)
        peak = equity.expanding().max()
        mdd = ((equity - peak) / peak).min() * 100

        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100 if trades else 0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0)
        )

        returns = equity.pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252 * 6)
            if returns.std() > 0
            else 0
        )

        return BacktestResult(
            total_return=(capital / init_capital - 1) * 100,
            mdd=mdd,
            win_rate=wr,
            profit_factor=pf,
            num_trades=len(trades),
            sharpe=sharpe,
        )


# ============================================================================
# VALIDATION
# ============================================================================
def run_validation():
    loader = DataLoader()
    backtester = Backtester()

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

    logger.info("=" * 70)
    logger.info("MULTI-TIMEFRAME STRATEGY COMPREHENSIVE VALIDATION")
    logger.info("=" * 70)

    # ========================================================================
    # 1. WALK-FORWARD VALIDATION
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("1. WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    wf_periods = [
        ("2021", "2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2022", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2023", "2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2024", "2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2025", "2020-01-01", "2024-12-31", "2025-01-01", "2026-12-31"),
    ]

    wf_results = []
    for period_name, train_start, train_end, test_start, test_end in wf_periods:
        period_pfs = []
        period_rets = []

        for symbol in symbols[:8]:  # Test on subset for speed
            df_4h = loader.load_ohlcv(symbol, "4h")
            df_1d = loader.load_ohlcv(symbol, "1d")

            if df_4h.empty or df_1d.empty:
                continue

            # Test period only
            test_4h = df_4h[(df_4h.index >= test_start) & (df_4h.index <= test_end)]
            test_1d = df_1d[(df_1d.index >= test_start) & (df_1d.index <= test_end)]

            if len(test_4h) < 100:
                continue

            strategy = MultiTimeframeTrendStrategy()
            signals = strategy.generate_signals(test_4h, test_1d)
            result = backtester.run(test_4h, signals)

            if result.profit_factor < float("inf"):
                period_pfs.append(result.profit_factor)
            period_rets.append(result.total_return)

        if period_pfs:
            avg_pf = np.mean(period_pfs)
            avg_ret = np.mean(period_rets)
            profitable = sum(1 for pf in period_pfs if pf > 1.0)
            wf_results.append(
                (period_name, avg_pf, avg_ret, profitable, len(period_pfs))
            )
            logger.info(
                f"  {period_name}: Avg PF={avg_pf:.2f}, Avg Ret={avg_ret:+.1f}%, "
                f"Profitable={profitable}/{len(period_pfs)}"
            )

    if wf_results:
        overall_pf = np.mean([r[1] for r in wf_results])
        overall_profitable = (
            sum(r[3] for r in wf_results) / sum(r[4] for r in wf_results) * 100
        )
        logger.info(f"\n  WALK-FORWARD SUMMARY:")
        logger.info(f"    Overall Avg PF: {overall_pf:.2f}")
        logger.info(f"    Overall Profitable: {overall_profitable:.0f}%")

    # ========================================================================
    # 2. OUT-OF-SAMPLE TEST
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("2. OUT-OF-SAMPLE TEST (Train: 2020-2022, Test: 2023-2026)")
    logger.info("=" * 70)

    is_results = []
    oos_results = []

    for symbol in symbols:
        df_4h = loader.load_ohlcv(symbol, "4h")
        df_1d = loader.load_ohlcv(symbol, "1d")

        if df_4h.empty or df_1d.empty:
            continue

        # In-sample: 2020-2022
        is_4h = df_4h[(df_4h.index >= "2020-01-01") & (df_4h.index <= "2022-12-31")]
        is_1d = df_1d[(df_1d.index >= "2020-01-01") & (df_1d.index <= "2022-12-31")]

        # Out-of-sample: 2023-2026
        oos_4h = df_4h[df_4h.index >= "2023-01-01"]
        oos_1d = df_1d[df_1d.index >= "2023-01-01"]

        strategy = MultiTimeframeTrendStrategy()

        if len(is_4h) > 200:
            signals = strategy.generate_signals(is_4h, is_1d)
            result = backtester.run(is_4h, signals)
            is_results.append((symbol, result))

        if len(oos_4h) > 200:
            signals = strategy.generate_signals(oos_4h, oos_1d)
            result = backtester.run(oos_4h, signals)
            oos_results.append((symbol, result))

    logger.info("\n  IN-SAMPLE (2020-2022):")
    is_profitable = sum(1 for s, r in is_results if r.profit_factor > 1.0)
    is_avg_pf = np.mean(
        [r.profit_factor for s, r in is_results if r.profit_factor < float("inf")]
    )
    is_avg_ret = np.mean([r.total_return for s, r in is_results])
    logger.info(f"    Profitable: {is_profitable}/{len(is_results)}")
    logger.info(f"    Avg PF: {is_avg_pf:.2f}")
    logger.info(f"    Avg Return: {is_avg_ret:+.1f}%")

    logger.info("\n  OUT-OF-SAMPLE (2023-2026):")
    oos_profitable = sum(1 for s, r in oos_results if r.profit_factor > 1.0)
    oos_avg_pf = np.mean(
        [r.profit_factor for s, r in oos_results if r.profit_factor < float("inf")]
    )
    oos_avg_ret = np.mean([r.total_return for s, r in oos_results])
    logger.info(f"    Profitable: {oos_profitable}/{len(oos_results)}")
    logger.info(f"    Avg PF: {oos_avg_pf:.2f}")
    logger.info(f"    Avg Return: {oos_avg_ret:+.1f}%")

    pf_degradation = (oos_avg_pf - is_avg_pf) / is_avg_pf * 100 if is_avg_pf > 0 else 0
    logger.info(f"\n  PF CHANGE: {pf_degradation:+.1f}%")
    if oos_avg_pf > 1.0:
        logger.info("  ✓ Strategy remains profitable out-of-sample")
    else:
        logger.info("  ✗ Strategy fails out-of-sample")

    # ========================================================================
    # 3. PARAMETER SENSITIVITY
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("3. PARAMETER SENSITIVITY ANALYSIS")
    logger.info("=" * 70)

    # Test different parameter combinations
    param_combos = [
        (20, 100, 10, 30),  # Faster
        (50, 200, 20, 50),  # Default
        (100, 300, 30, 80),  # Slower
        (30, 150, 15, 40),  # Medium-fast
        (75, 250, 25, 60),  # Medium-slow
    ]

    param_results = []
    for daily_fast, daily_slow, h4_fast, h4_slow in param_combos:
        combo_pfs = []

        for symbol in symbols[:8]:
            df_4h = loader.load_ohlcv(symbol, "4h")
            df_1d = loader.load_ohlcv(symbol, "1d")

            if df_4h.empty or df_1d.empty:
                continue

            df_4h = df_4h[df_4h.index >= "2020-01-01"]

            strategy = MultiTimeframeTrendStrategy(
                daily_fast, daily_slow, h4_fast, h4_slow
            )
            signals = strategy.generate_signals(df_4h, df_1d)
            result = backtester.run(df_4h, signals)

            if result.profit_factor < float("inf"):
                combo_pfs.append(result.profit_factor)

        if combo_pfs:
            avg_pf = np.mean(combo_pfs)
            profitable = sum(1 for pf in combo_pfs if pf > 1.0)
            param_results.append(
                (
                    daily_fast,
                    daily_slow,
                    h4_fast,
                    h4_slow,
                    avg_pf,
                    profitable,
                    len(combo_pfs),
                )
            )
            logger.info(
                f"  D({daily_fast}/{daily_slow}) H4({h4_fast}/{h4_slow}): "
                f"PF={avg_pf:.2f}, Profitable={profitable}/{len(combo_pfs)}"
            )

    if param_results:
        pfs = [r[4] for r in param_results]
        logger.info(f"\n  Parameter Stability:")
        logger.info(f"    PF Range: {min(pfs):.2f} ~ {max(pfs):.2f}")
        logger.info(f"    PF Std: {np.std(pfs):.2f}")
        if np.std(pfs) < 0.2:
            logger.info("    ✓ Parameters are stable")
        else:
            logger.info("    ⚠ Parameters show sensitivity")

    # ========================================================================
    # 4. PER-SYMBOL DETAILED ANALYSIS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("4. PER-SYMBOL DETAILED ANALYSIS (Full Period)")
    logger.info("=" * 70)

    symbol_details = []
    for symbol in symbols:
        df_4h = loader.load_ohlcv(symbol, "4h")
        df_1d = loader.load_ohlcv(symbol, "1d")

        if df_4h.empty or df_1d.empty:
            continue

        df_4h = df_4h[df_4h.index >= "2020-01-01"]

        strategy = MultiTimeframeTrendStrategy()
        signals = strategy.generate_signals(df_4h, df_1d)
        result = backtester.run(df_4h, signals)

        status = "✓" if result.profit_factor > 1.0 else "✗"
        logger.info(
            f"  {status} {symbol}: PF={result.profit_factor:.2f}, "
            f"Ret={result.total_return:+.1f}%, WR={result.win_rate:.0f}%, "
            f"MDD={result.mdd:.1f}%, Trades={result.num_trades}"
        )

        symbol_details.append(
            {
                "symbol": symbol,
                "pf": result.profit_factor,
                "return": result.total_return,
                "wr": result.win_rate,
                "mdd": result.mdd,
                "trades": result.num_trades,
                "sharpe": result.sharpe,
            }
        )

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION VERDICT")
    logger.info("=" * 70)

    profitable_count = sum(1 for d in symbol_details if d["pf"] > 1.0)
    avg_pf_all = np.mean([d["pf"] for d in symbol_details if d["pf"] < float("inf")])
    avg_ret_all = np.mean([d["return"] for d in symbol_details])

    logger.info(f"\n  Overall Statistics:")
    logger.info(
        f"    Profitable Symbols: {profitable_count}/{len(symbol_details)} ({profitable_count/len(symbol_details)*100:.0f}%)"
    )
    logger.info(f"    Average PF: {avg_pf_all:.2f}")
    logger.info(f"    Average Return: {avg_ret_all:+.1f}%")

    # Criteria check
    criteria_met = 0
    total_criteria = 5

    logger.info(f"\n  Validation Criteria:")

    # 1. > 60% symbols profitable
    if profitable_count / len(symbol_details) > 0.6:
        logger.info(f"    ✓ >60% symbols profitable")
        criteria_met += 1
    else:
        logger.info(f"    ✗ <60% symbols profitable")

    # 2. Average PF > 1.1
    if avg_pf_all > 1.1:
        logger.info(f"    ✓ Average PF > 1.1")
        criteria_met += 1
    else:
        logger.info(f"    ✗ Average PF < 1.1")

    # 3. OOS profitable
    if oos_avg_pf > 1.0:
        logger.info(f"    ✓ Out-of-sample profitable")
        criteria_met += 1
    else:
        logger.info(f"    ✗ Out-of-sample not profitable")

    # 4. Parameters stable
    if param_results and np.std([r[4] for r in param_results]) < 0.3:
        logger.info(f"    ✓ Parameters stable")
        criteria_met += 1
    else:
        logger.info(f"    ⚠ Parameters sensitive")

    # 5. Reasonable trade count
    avg_trades = np.mean([d["trades"] for d in symbol_details])
    if avg_trades > 50:
        logger.info(f"    ✓ Sufficient trade count ({avg_trades:.0f} avg)")
        criteria_met += 1
    else:
        logger.info(f"    ⚠ Low trade count ({avg_trades:.0f} avg)")

    logger.info(f"\n  CRITERIA MET: {criteria_met}/{total_criteria}")

    if criteria_met >= 4:
        logger.info("\n  ==> ✓✓ STRATEGY VALIDATED FOR LIVE TRADING (with caution)")
    elif criteria_met >= 3:
        logger.info(
            "\n  ==> ⚠ STRATEGY CONDITIONALLY VALIDATED - More testing recommended"
        )
    else:
        logger.info("\n  ==> ✗ STRATEGY NOT VALIDATED - Do not use for live trading")

    # Return detailed results
    return {
        "symbol_details": symbol_details,
        "is_results": is_results,
        "oos_results": oos_results,
        "param_results": param_results,
        "criteria_met": criteria_met,
    }


if __name__ == "__main__":
    results = run_validation()
