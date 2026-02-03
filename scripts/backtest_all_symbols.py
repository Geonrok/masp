#!/usr/bin/env python3
"""
Multi-Timeframe Strategy - ALL SYMBOLS Test
Tests on all available symbols in binance_futures_1d
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


class DataLoader:
    def __init__(self):
        self.data_root = DATA_ROOT

    def get_all_symbols(self) -> List[str]:
        """Get all symbols from 1d folder"""
        folder = self.data_root / "binance_futures_1d"
        symbols = []
        for f in folder.iterdir():
            if f.suffix == ".csv":
                symbol = f.stem
                # Add USDT if not present
                if not symbol.endswith("USDT"):
                    symbol = symbol + "USDT"
                symbols.append(symbol)
        return sorted(symbols)

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")

        # Try with USDT suffix
        file_path = self.data_root / folder / f"{symbol}.csv"
        if not file_path.exists():
            # Try without USDT suffix
            file_path = self.data_root / folder / f"{symbol.replace('USDT', '')}.csv"

        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")

            required_cols = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required_cols):
                return pd.DataFrame()

            return df[required_cols].sort_index()
        except Exception:
            return pd.DataFrame()


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
    symbol: str
    total_return: float
    mdd: float
    win_rate: float
    profit_factor: float
    num_trades: int
    sharpe: float
    data_start: str
    data_end: str


class MultiTFStrategy:
    def __init__(self, daily_fast=50, daily_slow=200, h4_fast=20, h4_slow=50):
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

        if df_1d.empty or len(df_4h) < 300:
            return signals

        df = df_4h.copy()
        df["ema_fast"] = calc_ema(df["close"], self.h4_fast)
        df["ema_slow"] = calc_ema(df["close"], self.h4_slow)
        df["trend_4h"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)

        df_daily = df_1d.copy()
        df_daily["ema_fast_d"] = calc_ema(df_daily["close"], self.daily_fast)
        df_daily["ema_slow_d"] = calc_ema(df_daily["close"], self.daily_slow)
        df_daily["trend_daily"] = np.where(
            df_daily["ema_fast_d"] > df_daily["ema_slow_d"], 1, -1
        )

        df["trend_daily"] = df_daily["trend_daily"].reindex(df.index, method="ffill")

        position = 0
        min_bars = max(self.daily_slow, self.h4_slow) + 50

        for i in range(min_bars, len(df)):
            trend_4h = df["trend_4h"].iloc[i]
            trend_daily = df["trend_daily"].iloc[i]

            if pd.isna(trend_daily):
                continue

            close = df["close"].iloc[i]
            ema_fast = df["ema_fast"].iloc[i]

            if position == 1 and (trend_4h == -1 or trend_daily == -1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0
            elif position == -1 and (trend_4h == 1 or trend_daily == 1):
                signals.iloc[i, signals.columns.get_loc("exit")] = True
                position = 0

            if position == 0:
                if trend_4h == 1 and trend_daily == 1 and close > ema_fast:
                    signals.iloc[i, signals.columns.get_loc("signal")] = 1
                    position = 1
                elif trend_4h == -1 and trend_daily == -1 and close < ema_fast:
                    signals.iloc[i, signals.columns.get_loc("signal")] = -1
                    position = -1

        return signals


class Backtester:
    def __init__(self, commission=0.001, slippage=0.0005, leverage=3, size_pct=0.25):
        self.commission = commission
        self.slippage = slippage
        self.leverage = leverage
        self.size_pct = size_pct

    def run(
        self, df: pd.DataFrame, signals: pd.DataFrame, symbol: str
    ) -> Optional[BacktestResult]:
        if len(df) < 300:
            return None

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

        if not trades:
            return None

        equity = pd.Series(equity)
        peak = equity.expanding().max()
        mdd = ((equity - peak) / peak).min() * 100

        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100

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
            symbol=symbol,
            total_return=(capital / init_capital - 1) * 100,
            mdd=mdd,
            win_rate=wr,
            profit_factor=pf if pf < float("inf") else 999,
            num_trades=len(trades),
            sharpe=sharpe,
            data_start=str(df.index[0].date()),
            data_end=str(df.index[-1].date()),
        )


def main():
    loader = DataLoader()
    backtester = Backtester()
    strategy = MultiTFStrategy()

    all_symbols = loader.get_all_symbols()
    logger.info(f"Total symbols found: {len(all_symbols)}")

    results = []
    skipped = 0
    errors = 0

    logger.info("=" * 70)
    logger.info("MULTI-TIMEFRAME STRATEGY - ALL SYMBOLS TEST")
    logger.info("=" * 70)

    for idx, symbol in enumerate(all_symbols):
        if (idx + 1) % 50 == 0:
            logger.info(f"Progress: {idx + 1}/{len(all_symbols)} symbols processed...")

        try:
            df_4h = loader.load_ohlcv(symbol, "4h")
            df_1d = loader.load_ohlcv(symbol, "1d")

            if df_4h.empty or df_1d.empty:
                skipped += 1
                continue

            # Filter by date - need at least from 2021
            df_4h = df_4h[df_4h.index >= "2021-01-01"]
            df_1d = df_1d[df_1d.index >= "2021-01-01"]

            if len(df_4h) < 500:  # Need ~3 months of 4h data minimum
                skipped += 1
                continue

            signals = strategy.generate_signals(df_4h, df_1d)
            result = backtester.run(df_4h, signals, symbol)

            if result and result.num_trades >= 10:  # Minimum 10 trades
                results.append(result)

        except Exception as e:
            errors += 1
            continue

    logger.info(
        f"\nProcessing complete: {len(results)} valid results, {skipped} skipped, {errors} errors"
    )

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    if not results:
        logger.info("No valid results to analyze")
        return

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)

    # Overall stats
    profitable = [r for r in results if r.profit_factor > 1.0]
    losing = [r for r in results if r.profit_factor <= 1.0]

    logger.info(f"\nTotal symbols tested: {len(results)}")
    logger.info(
        f"Profitable (PF > 1.0): {len(profitable)} ({len(profitable)/len(results)*100:.1f}%)"
    )
    logger.info(
        f"Losing (PF <= 1.0): {len(losing)} ({len(losing)/len(results)*100:.1f}%)"
    )

    # PF distribution
    pfs = [r.profit_factor for r in results if r.profit_factor < 999]
    logger.info(f"\nProfit Factor Distribution:")
    logger.info(f"  Mean: {np.mean(pfs):.2f}")
    logger.info(f"  Median: {np.median(pfs):.2f}")
    logger.info(f"  Std: {np.std(pfs):.2f}")
    logger.info(f"  Min: {min(pfs):.2f}")
    logger.info(f"  Max: {max(pfs):.2f}")

    # Return distribution
    rets = [r.total_return for r in results]
    logger.info(f"\nReturn Distribution:")
    logger.info(f"  Mean: {np.mean(rets):+.1f}%")
    logger.info(f"  Median: {np.median(rets):+.1f}%")
    logger.info(f"  Positive returns: {sum(1 for r in rets if r > 0)}/{len(rets)}")

    # Win rate distribution
    wrs = [r.win_rate for r in results]
    logger.info(f"\nWin Rate Distribution:")
    logger.info(f"  Mean: {np.mean(wrs):.1f}%")
    logger.info(f"  Median: {np.median(wrs):.1f}%")

    # MDD distribution
    mdds = [r.mdd for r in results]
    logger.info(f"\nMDD Distribution:")
    logger.info(f"  Mean: {np.mean(mdds):.1f}%")
    logger.info(f"  Worst: {min(mdds):.1f}%")

    # ========================================================================
    # TOP/BOTTOM PERFORMERS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 PERFORMERS (by Profit Factor)")
    logger.info("=" * 70)

    sorted_by_pf = sorted(results, key=lambda x: x.profit_factor, reverse=True)[:20]
    for r in sorted_by_pf:
        logger.info(
            f"  {r.symbol:12s} PF={r.profit_factor:5.2f} Ret={r.total_return:+8.1f}% "
            f"WR={r.win_rate:4.0f}% MDD={r.mdd:6.1f}% Trades={r.num_trades:3d}"
        )

    logger.info("\n" + "=" * 70)
    logger.info("BOTTOM 20 PERFORMERS (by Profit Factor)")
    logger.info("=" * 70)

    sorted_by_pf_asc = sorted(results, key=lambda x: x.profit_factor)[:20]
    for r in sorted_by_pf_asc:
        logger.info(
            f"  {r.symbol:12s} PF={r.profit_factor:5.2f} Ret={r.total_return:+8.1f}% "
            f"WR={r.win_rate:4.0f}% MDD={r.mdd:6.1f}% Trades={r.num_trades:3d}"
        )

    # ========================================================================
    # PF BUCKETS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PROFIT FACTOR DISTRIBUTION BUCKETS")
    logger.info("=" * 70)

    buckets = [
        ("PF >= 1.5 (Excellent)", [r for r in results if r.profit_factor >= 1.5]),
        ("PF 1.3-1.5 (Good)", [r for r in results if 1.3 <= r.profit_factor < 1.5]),
        (
            "PF 1.1-1.3 (Acceptable)",
            [r for r in results if 1.1 <= r.profit_factor < 1.3],
        ),
        ("PF 1.0-1.1 (Marginal)", [r for r in results if 1.0 <= r.profit_factor < 1.1]),
        ("PF 0.8-1.0 (Losing)", [r for r in results if 0.8 <= r.profit_factor < 1.0]),
        ("PF < 0.8 (Bad)", [r for r in results if r.profit_factor < 0.8]),
    ]

    for bucket_name, bucket_results in buckets:
        pct = len(bucket_results) / len(results) * 100
        logger.info(f"  {bucket_name}: {len(bucket_results)} symbols ({pct:.1f}%)")

    # ========================================================================
    # RECOMMENDED SYMBOLS
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDED SYMBOLS FOR TRADING")
    logger.info("Criteria: PF >= 1.3, MDD >= -60%, Trades >= 30")
    logger.info("=" * 70)

    recommended = [
        r
        for r in results
        if r.profit_factor >= 1.3 and r.mdd >= -60 and r.num_trades >= 30
    ]

    recommended = sorted(recommended, key=lambda x: x.profit_factor, reverse=True)

    logger.info(f"\nFound {len(recommended)} recommended symbols:\n")
    for r in recommended:
        logger.info(
            f"  {r.symbol:12s} PF={r.profit_factor:5.2f} Ret={r.total_return:+8.1f}% "
            f"WR={r.win_rate:4.0f}% MDD={r.mdd:6.1f}% Trades={r.num_trades:3d} "
            f"({r.data_start} ~ {r.data_end})"
        )

    # ========================================================================
    # SAVE RESULTS TO CSV
    # ========================================================================
    results_df = pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "profit_factor": r.profit_factor,
                "total_return": r.total_return,
                "win_rate": r.win_rate,
                "mdd": r.mdd,
                "num_trades": r.num_trades,
                "sharpe": r.sharpe,
                "data_start": r.data_start,
                "data_end": r.data_end,
            }
            for r in results
        ]
    )

    output_path = Path(
        "E:/투자/Multi-Asset Strategy Platform/data/backtests/multitf_all_symbols_results.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    # ========================================================================
    # FINAL VERDICT
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)

    profitable_pct = len(profitable) / len(results) * 100
    avg_pf = np.mean(pfs)

    logger.info(f"\n  Total symbols analyzed: {len(results)}")
    logger.info(f"  Profitable symbols: {len(profitable)} ({profitable_pct:.1f}%)")
    logger.info(f"  Average Profit Factor: {avg_pf:.2f}")
    logger.info(f"  Recommended symbols: {len(recommended)}")

    if profitable_pct >= 60 and avg_pf >= 1.1:
        logger.info("\n  ==> ✓ STRATEGY SHOWS POSITIVE EDGE ACROSS MARKET")
    elif profitable_pct >= 50:
        logger.info("\n  ==> ⚠ STRATEGY HAS MARGINAL EDGE - SELECT SYMBOLS CAREFULLY")
    else:
        logger.info("\n  ==> ✗ STRATEGY DOES NOT GENERALIZE WELL")

    return results


if __name__ == "__main__":
    results = main()
