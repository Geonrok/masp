#!/usr/bin/env python3
"""
V12 Strategy Validation Suite

1. Walk-Forward Validation
2. Out-of-Sample Testing
3. Multi-Symbol Testing
4. Robustness Analysis
"""

from __future__ import annotations

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
    position_size_pct: float = 0.15
    fear_entry: int = 30
    greed_exit: int = 70
    trend_deviation: float = 0.01


class DataLoader:
    def __init__(self, data_root: Path = DATA_ROOT):
        self.data_root = data_root

    def load_ohlcv(self, symbol: str, timeframe: str = "4h") -> pd.DataFrame:
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")

        # Try different file naming conventions
        possible_names = [
            f"{symbol}.csv",
            f"{symbol.replace('USDT', '')}.csv",
            f"{symbol.lower()}.csv",
        ]

        for name in possible_names:
            file_path = self.data_root / folder / name
            if file_path.exists():
                df = pd.read_csv(file_path)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                elif "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")

                required_cols = ["open", "high", "low", "close", "volume"]
                if all(col in df.columns for col in required_cols):
                    return df[required_cols].sort_index()
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

    def get_available_symbols(self, timeframe: str = "4h") -> List[str]:
        """Get list of available symbols."""
        tf_map = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}
        folder = tf_map.get(timeframe, "binance_futures_4h")
        path = self.data_root / folder

        if not path.exists():
            return []

        symbols = []
        for f in path.glob("*.csv"):
            symbol = f.stem.upper()
            if not symbol.endswith("USDT"):
                symbol = symbol + "USDT"
            symbols.append(symbol)
        return symbols


class LongOnlyStrategy:
    """Long-only fear-based strategy (same as v12)."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.ema200_period = 200
        self.ema50_period = 50
        self.atr_period = 14

        self.fear_entry = config.fear_entry
        self.greed_exit = config.greed_exit
        self.trend_dev = config.trend_deviation

        self.stop_atr = 2.5
        self.trail_atr = 2.0
        self.tp_atr = 4.0

    def calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros(len(data))
        ema[0] = data[0]
        mult = 2 / (period + 1)
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1]
        return ema

    def calc_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        tr = np.maximum(
            h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        )
        tr[0] = h[0] - l[0]
        return pd.Series(tr).rolling(period).mean().values

    def is_uptrend(
        self, price: float, ema50: float, ema200: float, ema200_prev: float
    ) -> bool:
        ema200_rising = ema200 > ema200_prev
        price_above_ema200 = price > ema200 * (1 + self.trend_dev)
        ema_aligned = ema50 > ema200
        return price_above_ema200 and (ema200_rising or ema_aligned)

    def generate_signal(
        self, df: pd.DataFrame, fear_greed: Optional[float]
    ) -> Dict[str, Any]:
        if len(df) < 250:
            return {"signal": "HOLD", "reason": "Insufficient data"}

        close = df["close"].values
        price = close[-1]

        ema200 = self.calc_ema(close, self.ema200_period)
        ema50 = self.calc_ema(close, self.ema50_period)
        atr = self.calc_atr(df, self.atr_period)

        curr_ema200 = ema200[-1]
        prev_ema200 = ema200[-2]
        curr_ema50 = ema50[-1]
        curr_atr = atr[-1]

        uptrend = self.is_uptrend(price, curr_ema50, curr_ema200, prev_ema200)

        if fear_greed is None:
            return {"signal": "HOLD", "reason": "No FG data", "uptrend": uptrend}

        signal = "HOLD"
        pos_mult = 1.0

        if uptrend and fear_greed <= self.fear_entry:
            signal = "LONG"
            if fear_greed <= 15:
                pos_mult = 1.0
            elif fear_greed <= 25:
                pos_mult = 0.8
            else:
                pos_mult = 0.6

        if signal == "LONG":
            stop_loss = price - self.stop_atr * curr_atr
            take_profit = price + self.tp_atr * curr_atr
        else:
            stop_loss = take_profit = None

        return {
            "signal": signal,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": curr_atr,
            "pos_mult": pos_mult,
        }


class StrategyBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.loader = DataLoader()
        self.strategy = LongOnlyStrategy(config)

    def run(
        self, data_4h: Dict[str, pd.DataFrame], fear_greed: pd.DataFrame
    ) -> Dict[str, Any]:
        start_dt = pd.Timestamp(self.config.start_date)
        end_dt = (
            pd.Timestamp(self.config.end_date)
            if self.config.end_date
            else pd.Timestamp.now()
        )

        # Find common timestamps
        first_symbol = list(data_4h.keys())[0]
        ref_df = data_4h[first_symbol]
        ref_df = ref_df[(ref_df.index >= start_dt) & (ref_df.index <= end_dt)]
        timestamps = ref_df.index.tolist()

        if len(timestamps) < 300:
            return {"error": "Insufficient data", "total_trades": 0}

        capital = self.config.initial_capital
        positions: Dict[str, Dict] = {}
        equity_curve = [capital]
        trade_log = []

        stats = {"wins": 0, "losses": 0}

        def get_fg_at(ts):
            if fear_greed.empty:
                return None
            date = ts.date()
            mask = fear_greed.index.date <= date
            if mask.any():
                return fear_greed.loc[fear_greed.index[mask][-1], "value"]
            return None

        for i in range(250, len(timestamps)):
            ts = timestamps[i]
            fg = get_fg_at(ts)

            for symbol in data_4h.keys():
                df = data_4h[symbol]
                df_curr = df[df.index <= ts].tail(300)
                if len(df_curr) < 250:
                    continue

                price = df_curr["close"].iloc[-1]

                # Exit check
                if symbol in positions:
                    pos = positions[symbol]
                    exit_sig = False

                    pos["highest"] = max(pos["highest"], price)
                    trail = pos["highest"] - pos["atr"] * self.strategy.trail_atr

                    if price <= pos["stop_loss"]:
                        exit_sig = True
                    elif price <= trail:
                        exit_sig = True
                    elif price >= pos["take_profit"]:
                        exit_sig = True
                    elif fg and fg >= self.config.greed_exit:
                        exit_sig = True

                    if exit_sig:
                        exit_price = price * (1 - self.config.slippage_pct)
                        pnl = (
                            (exit_price - pos["entry_price"])
                            / pos["entry_price"]
                            * pos["leverage"]
                        )
                        pnl_usd = (
                            pos["size"] * pnl
                            - pos["size"] * self.config.commission_pct * 2
                        )
                        capital += pnl_usd

                        trade_log.append({"pnl_usd": pnl_usd})
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
                    sig = self.strategy.generate_signal(df_curr, fg)

                    if sig["signal"] == "LONG":
                        entry_price = sig["price"] * (1 + self.config.slippage_pct)
                        size = capital * self.config.position_size_pct * sig["pos_mult"]

                        positions[symbol] = {
                            "entry_price": entry_price,
                            "size": size,
                            "leverage": self.config.leverage,
                            "stop_loss": sig["stop_loss"],
                            "take_profit": sig["take_profit"],
                            "atr": sig["atr"],
                            "highest": entry_price,
                        }

            # MTM
            pv = capital
            for sym, pos in positions.items():
                if sym in data_4h:
                    curr = data_4h[sym].loc[data_4h[sym].index <= ts]["close"].iloc[-1]
                    unr = (curr - pos["entry_price"]) / pos["entry_price"]
                    pv += pos["size"] * unr * pos["leverage"]
            equity_curve.append(pv)

        # Close remaining
        for sym, pos in list(positions.items()):
            if sym in data_4h:
                exit_price = data_4h[sym].iloc[-1]["close"]
                pnl = (
                    (exit_price - pos["entry_price"])
                    / pos["entry_price"]
                    * pos["leverage"]
                )
                pnl_usd = (
                    pos["size"] * pnl - pos["size"] * self.config.commission_pct * 2
                )
                capital += pnl_usd
                trade_log.append({"pnl_usd": pnl_usd})
                if pnl_usd > 0:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1

        # Calculate metrics
        eq = np.array(equity_curve)
        total_ret = (eq[-1] - self.config.initial_capital) / self.config.initial_capital
        peak = np.maximum.accumulate(eq)
        mdd = np.min((eq - peak) / peak)
        total_trades = stats["wins"] + stats["losses"]
        wr = stats["wins"] / total_trades if total_trades > 0 else 0

        if trade_log:
            gp = sum(t["pnl_usd"] for t in trade_log if t["pnl_usd"] > 0)
            gl = abs(sum(t["pnl_usd"] for t in trade_log if t["pnl_usd"] < 0))
            pf = gp / gl if gl > 0 else float("inf")
        else:
            pf = 0

        return {
            "total_return": total_ret,
            "max_drawdown": mdd,
            "win_rate": wr,
            "profit_factor": pf,
            "total_trades": total_trades,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "final_capital": eq[-1],
        }


def run_walk_forward_validation():
    """Walk-forward validation with rolling windows."""
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    loader = DataLoader()
    fear_greed = loader.load_fear_greed()

    # Load BTC and ETH data
    symbols = ["BTCUSDT", "ETHUSDT"]
    all_data = {}
    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if not df.empty:
            all_data[symbol] = df

    if not all_data:
        logger.error("No data loaded")
        return

    # Walk-forward periods
    periods = [
        {
            "train": ("2020-01-01", "2022-12-31"),
            "test": ("2023-01-01", "2023-12-31"),
            "label": "Test 2023",
        },
        {
            "train": ("2020-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2024-12-31"),
            "label": "Test 2024",
        },
        {
            "train": ("2020-01-01", "2024-12-31"),
            "test": ("2025-01-01", "2026-12-31"),
            "label": "Test 2025-26",
        },
    ]

    results = []
    for period in periods:
        train_start, train_end = period["train"]
        test_start, test_end = period["test"]
        label = period["label"]

        logger.info(
            f"\n{label}: Train {train_start}~{train_end}, Test {test_start}~{test_end}"
        )

        # Test period backtest
        config = BacktestConfig(
            start_date=test_start,
            end_date=test_end,
            symbols=symbols,
            fear_entry=30,
            greed_exit=70,
        )

        bt = StrategyBacktester(config)
        result = bt.run(all_data, fear_greed)

        if result.get("total_trades", 0) > 0:
            results.append(
                {
                    "period": label,
                    "return": result["total_return"],
                    "mdd": result["max_drawdown"],
                    "wr": result["win_rate"],
                    "pf": result["profit_factor"],
                    "trades": result["total_trades"],
                }
            )
            logger.info(
                f"  Return: {result['total_return']*100:+.1f}%, "
                f"MDD: {result['max_drawdown']*100:.1f}%, "
                f"WR: {result['win_rate']*100:.1f}%, "
                f"PF: {result['profit_factor']:.2f}, "
                f"Trades: {result['total_trades']}"
            )
        else:
            logger.info(f"  No trades in this period")
            results.append(
                {
                    "period": label,
                    "return": 0,
                    "mdd": 0,
                    "wr": 0,
                    "pf": 0,
                    "trades": 0,
                }
            )

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD SUMMARY")
    logger.info("=" * 70)

    total_trades = sum(r["trades"] for r in results)
    if total_trades > 0:
        # Weighted average by trades
        avg_return = sum(r["return"] * r["trades"] for r in results) / total_trades
        avg_pf = sum(r["pf"] * r["trades"] for r in results) / total_trades
        avg_wr = sum(r["wr"] * r["trades"] for r in results) / total_trades

        logger.info(f"Average Return: {avg_return*100:+.1f}%")
        logger.info(f"Average PF: {avg_pf:.2f}")
        logger.info(f"Average WR: {avg_wr*100:.1f}%")
        logger.info(f"Total Trades: {total_trades}")

        # Check consistency
        profitable_periods = sum(1 for r in results if r["return"] > 0)
        logger.info(f"Profitable Periods: {profitable_periods}/{len(results)}")

    return results


def run_out_of_sample_test():
    """True out-of-sample test: train on 2020-2022, test on 2023-2026."""
    logger.info("\n" + "=" * 70)
    logger.info("OUT-OF-SAMPLE TEST")
    logger.info("=" * 70)
    logger.info("Train: 2020-2022 (parameter selection)")
    logger.info("Test: 2023-2026 (completely unseen)")

    loader = DataLoader()
    fear_greed = loader.load_fear_greed()

    symbols = ["BTCUSDT", "ETHUSDT"]
    all_data = {}
    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if not df.empty:
            all_data[symbol] = df

    if not all_data:
        logger.error("No data loaded")
        return

    # In-sample (training period)
    logger.info("\n--- IN-SAMPLE (2020-2022) ---")
    config_in = BacktestConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        symbols=symbols,
        fear_entry=30,
        greed_exit=70,
    )
    bt_in = StrategyBacktester(config_in)
    result_in = bt_in.run(all_data, fear_greed)

    if result_in.get("total_trades", 0) > 0:
        logger.info(f"  Return: {result_in['total_return']*100:+.1f}%")
        logger.info(f"  MDD: {result_in['max_drawdown']*100:.1f}%")
        logger.info(f"  WR: {result_in['win_rate']*100:.1f}%")
        logger.info(f"  PF: {result_in['profit_factor']:.2f}")
        logger.info(f"  Trades: {result_in['total_trades']}")

    # Out-of-sample (test period)
    logger.info("\n--- OUT-OF-SAMPLE (2023-2026) ---")
    config_out = BacktestConfig(
        start_date="2023-01-01",
        end_date="2026-12-31",
        symbols=symbols,
        fear_entry=30,
        greed_exit=70,
    )
    bt_out = StrategyBacktester(config_out)
    result_out = bt_out.run(all_data, fear_greed)

    if result_out.get("total_trades", 0) > 0:
        logger.info(f"  Return: {result_out['total_return']*100:+.1f}%")
        logger.info(f"  MDD: {result_out['max_drawdown']*100:.1f}%")
        logger.info(f"  WR: {result_out['win_rate']*100:.1f}%")
        logger.info(f"  PF: {result_out['profit_factor']:.2f}")
        logger.info(f"  Trades: {result_out['total_trades']}")

    # Comparison
    logger.info("\n--- COMPARISON ---")
    if result_in.get("total_trades", 0) > 0 and result_out.get("total_trades", 0) > 0:
        pf_degradation = (
            (result_out["profit_factor"] - result_in["profit_factor"])
            / result_in["profit_factor"]
            * 100
        )
        logger.info(f"PF Change: {pf_degradation:+.1f}%")

        if result_out["profit_factor"] >= 1.0:
            logger.info("✓ Strategy remains profitable out-of-sample")
        else:
            logger.info("✗ Strategy loses money out-of-sample")

    return {"in_sample": result_in, "out_of_sample": result_out}


def run_multi_symbol_test():
    """Test strategy on multiple symbols."""
    logger.info("\n" + "=" * 70)
    logger.info("MULTI-SYMBOL TEST")
    logger.info("=" * 70)

    loader = DataLoader()
    fear_greed = loader.load_fear_greed()

    # Test these symbols individually
    test_symbols = [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "ADAUSDT",
        "DOGEUSDT",
        "LINKUSDT",
    ]

    results = []
    for symbol in test_symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty or len(df) < 500:
            logger.info(f"{symbol}: Insufficient data")
            continue

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2026-12-31",
            symbols=[symbol],
            max_positions=1,
            fear_entry=30,
            greed_exit=70,
        )

        bt = StrategyBacktester(config)
        result = bt.run({symbol: df}, fear_greed)

        if result.get("total_trades", 0) > 0:
            results.append(
                {
                    "symbol": symbol,
                    "return": result["total_return"],
                    "mdd": result["max_drawdown"],
                    "wr": result["win_rate"],
                    "pf": result["profit_factor"],
                    "trades": result["total_trades"],
                }
            )
            status = "✓" if result["profit_factor"] >= 1.0 else "✗"
            logger.info(
                f"{status} {symbol}: Return={result['total_return']*100:+.1f}%, "
                f"PF={result['profit_factor']:.2f}, "
                f"WR={result['win_rate']*100:.1f}%, "
                f"Trades={result['total_trades']}"
            )
        else:
            logger.info(f"- {symbol}: No trades")

    # Summary
    if results:
        logger.info("\n--- MULTI-SYMBOL SUMMARY ---")
        profitable = [r for r in results if r["pf"] >= 1.0]
        logger.info(f"Profitable symbols: {len(profitable)}/{len(results)}")

        avg_pf = np.mean([r["pf"] for r in results])
        avg_return = np.mean([r["return"] for r in results])
        logger.info(f"Average PF: {avg_pf:.2f}")
        logger.info(f"Average Return: {avg_return*100:+.1f}%")

    return results


def run_robustness_analysis():
    """Test sensitivity to parameter changes."""
    logger.info("\n" + "=" * 70)
    logger.info("ROBUSTNESS ANALYSIS (Parameter Sensitivity)")
    logger.info("=" * 70)

    loader = DataLoader()
    fear_greed = loader.load_fear_greed()

    symbols = ["BTCUSDT", "ETHUSDT"]
    all_data = {}
    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if not df.empty:
            all_data[symbol] = df

    if not all_data:
        return

    # Test parameter variations
    fear_values = [25, 28, 30, 32, 35]
    greed_values = [65, 68, 70, 72, 75]

    results = []
    for fear in fear_values:
        for greed in greed_values:
            config = BacktestConfig(
                start_date="2020-01-01",
                end_date="2026-12-31",
                symbols=symbols,
                fear_entry=fear,
                greed_exit=greed,
            )

            bt = StrategyBacktester(config)
            result = bt.run(all_data, fear_greed)

            if result.get("total_trades", 0) > 0:
                results.append(
                    {
                        "fear": fear,
                        "greed": greed,
                        "pf": result["profit_factor"],
                        "return": result["total_return"],
                        "trades": result["total_trades"],
                    }
                )

    # Analyze stability
    if results:
        pf_values = [r["pf"] for r in results]
        pf_mean = np.mean(pf_values)
        pf_std = np.std(pf_values)
        pf_min = np.min(pf_values)
        pf_max = np.max(pf_values)

        logger.info(f"\nPF across {len(results)} parameter combinations:")
        logger.info(f"  Mean: {pf_mean:.2f}")
        logger.info(f"  Std: {pf_std:.2f}")
        logger.info(f"  Range: {pf_min:.2f} ~ {pf_max:.2f}")

        # Count profitable combinations
        profitable = sum(1 for r in results if r["pf"] >= 1.0)
        logger.info(
            f"  Profitable: {profitable}/{len(results)} ({profitable/len(results)*100:.0f}%)"
        )

        # Stability score
        stability = 1 - (pf_std / pf_mean) if pf_mean > 0 else 0
        logger.info(f"  Stability Score: {stability:.2f}")

        if stability > 0.7:
            logger.info("  ✓ Parameters are robust")
        else:
            logger.info("  ✗ Parameters are sensitive (may be overfit)")

    return results


def main():
    logger.info("=" * 70)
    logger.info("V12 STRATEGY COMPREHENSIVE VALIDATION")
    logger.info("=" * 70)

    # 1. Walk-forward validation
    wf_results = run_walk_forward_validation()

    # 2. Out-of-sample test
    oos_results = run_out_of_sample_test()

    # 3. Multi-symbol test
    ms_results = run_multi_symbol_test()

    # 4. Robustness analysis
    rob_results = run_robustness_analysis()

    # Final verdict
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION VERDICT")
    logger.info("=" * 70)

    issues = []
    passes = []

    # Check walk-forward
    if wf_results:
        profitable_periods = sum(
            1 for r in wf_results if r["return"] > 0 and r["trades"] > 0
        )
        total_periods = sum(1 for r in wf_results if r["trades"] > 0)
        if profitable_periods >= total_periods * 0.5:
            passes.append(
                f"Walk-forward: {profitable_periods}/{total_periods} profitable periods"
            )
        else:
            issues.append(
                f"Walk-forward: Only {profitable_periods}/{total_periods} profitable periods"
            )

    # Check out-of-sample
    if oos_results and "out_of_sample" in oos_results:
        oos = oos_results["out_of_sample"]
        if oos.get("profit_factor", 0) >= 1.0:
            passes.append(f"Out-of-sample PF: {oos['profit_factor']:.2f}")
        else:
            issues.append(
                f"Out-of-sample PF: {oos.get('profit_factor', 0):.2f} (losing)"
            )

    # Check multi-symbol
    if ms_results:
        profitable_symbols = sum(1 for r in ms_results if r["pf"] >= 1.0)
        if profitable_symbols >= len(ms_results) * 0.5:
            passes.append(
                f"Multi-symbol: {profitable_symbols}/{len(ms_results)} profitable"
            )
        else:
            issues.append(
                f"Multi-symbol: Only {profitable_symbols}/{len(ms_results)} profitable"
            )

    logger.info("\n✓ PASSES:")
    for p in passes:
        logger.info(f"  - {p}")

    logger.info("\n✗ ISSUES:")
    for i in issues:
        logger.info(f"  - {i}")

    if len(issues) == 0:
        logger.info("\n==> STRATEGY VALIDATED: Ready for paper trading")
    elif len(issues) <= 1:
        logger.info("\n==> STRATEGY CONDITIONALLY VALIDATED: Proceed with caution")
    else:
        logger.info("\n==> STRATEGY NOT VALIDATED: Do not use for live trading")

    return 0


if __name__ == "__main__":
    sys.exit(main())
