#!/usr/bin/env python3
"""
Multi-Factor Strategy Walk-Forward Validation
- Rolling window: 12 months training, 3 months testing
- Tests parameter stability across time periods
- Measures out-of-sample performance degradation
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
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


def calc_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def calc_sma(s, p):
    return s.rolling(p).mean()


def calc_rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))


def calc_bollinger(s, p=20, std=2.0):
    sma = calc_sma(s, p)
    st = s.rolling(p).std()
    return sma + std * st, sma, sma - std * st


class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def get_all_symbols(self):
        folder = self.root / "binance_futures_4h"
        return sorted(
            [
                f.stem
                for f in folder.iterdir()
                if f.suffix == ".csv" and f.stem.endswith("USDT")
            ]
        )

    def load_ohlcv(self, symbol, tf="4h"):
        key = f"{symbol}_{tf}"
        if key in self._cache:
            return self._cache[key].copy()
        folder = {"4h": "binance_futures_4h", "1d": "binance_futures_1d"}.get(tf)
        for fn in [f"{symbol}.csv", f"{symbol.replace('USDT', '')}.csv"]:
            fp = self.root / folder / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if all(
                    c in df.columns for c in ["open", "high", "low", "close", "volume"]
                ):
                    df = df[["open", "high", "low", "close", "volume"]].sort_index()
                    self._cache[key] = df
                    return df.copy()
        return pd.DataFrame()

    def load_fear_greed(self):
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()

    def load_macro(self):
        dfs = []
        for fn, col in [("DXY.csv", "dxy"), ("SP500.csv", "sp500"), ("VIX.csv", "vix")]:
            fp = self.root / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                if "datetime" in df.columns:
                    df["datetime"] = pd.to_datetime(df["datetime"])
                    df = df.set_index("datetime")
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def load_tvl(self):
        fp = self.root / "defillama" / "total_tvl_history.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for col in ["datetime", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            if len(df.columns) > 0:
                return df.iloc[:, 0:1].rename(columns={df.columns[0]: "tvl"})
        return pd.DataFrame()


class MultiFactorScorer:
    def score_all(self, df, btc_df, fear_greed, macro, tvl):
        scores = pd.Series(0.0, index=df.index)

        # 1. Technical (weight: 1.5)
        ema20, ema50, ema200 = (
            calc_ema(df["close"], 20),
            calc_ema(df["close"], 50),
            calc_ema(df["close"], 200),
        )
        tech = np.where(
            (df["close"] > ema20) & (ema20 > ema50) & (ema50 > ema200),
            2,
            np.where(
                (df["close"] > ema50),
                1,
                np.where(
                    (df["close"] < ema20) & (ema20 < ema50) & (ema50 < ema200),
                    -2,
                    np.where((df["close"] < ema50), -1, 0),
                ),
            ),
        )

        rsi = calc_rsi(df["close"], 14)
        tech_rsi = np.where(
            rsi < 30,
            1.5,
            np.where(
                rsi < 40, 0.5, np.where(rsi > 70, -1.5, np.where(rsi > 60, -0.5, 0))
            ),
        )

        bb_upper, _, bb_lower = calc_bollinger(df["close"])
        bb_pos = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        tech_bb = np.where(bb_pos < 0.2, 1, np.where(bb_pos > 0.8, -1, 0))

        scores += (tech + tech_rsi + tech_bb) / 3 * 1.5

        # 2. Fear & Greed (weight: 1.2)
        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = np.where(
                fg < 20,
                2,
                np.where(fg < 35, 1, np.where(fg > 80, -2, np.where(fg > 65, -1, 0))),
            )
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

        # 3. BTC Correlation (weight: 1.0)
        if not btc_df.empty:
            btc_close = btc_df["close"].reindex(df.index, method="ffill")
            btc_ret = btc_close.pct_change(20)
            btc_ema50 = calc_ema(btc_close, 50)
            btc_ema200 = calc_ema(btc_close, 200)
            btc_uptrend = (btc_close > btc_ema50) & (btc_ema50 > btc_ema200)
            btc_downtrend = (btc_close < btc_ema50) & (btc_ema50 < btc_ema200)
            btc_score = np.where(
                btc_uptrend & (btc_ret > 0.1),
                2,
                np.where(
                    btc_uptrend & (btc_ret > 0.03),
                    1,
                    np.where(
                        btc_downtrend & (btc_ret < -0.1),
                        -2,
                        np.where(btc_downtrend & (btc_ret < -0.03), -1, 0),
                    ),
                ),
            )
            scores += pd.Series(btc_score, index=df.index).fillna(0) * 1.0

        # 4. Macro (weight: 1.0)
        if not macro.empty:
            macro_r = macro.reindex(df.index, method="ffill")
            macro_score = pd.Series(0.0, index=df.index)
            if "dxy" in macro_r.columns:
                dxy = macro_r["dxy"]
                dxy_ma = calc_sma(dxy, 50)
                macro_score += np.where(
                    dxy < dxy_ma * 0.98, 1, np.where(dxy > dxy_ma * 1.02, -1, 0)
                )
            if "vix" in macro_r.columns:
                vix = macro_r["vix"]
                macro_score += np.where(
                    vix > 30, 1, np.where(vix > 25, 0.5, np.where(vix < 15, -0.5, 0))
                )
            scores += macro_score.fillna(0) * 1.0

        # 5. TVL (weight: 0.6)
        if not tvl.empty:
            tv = tvl["tvl"].reindex(df.index, method="ffill")
            tvl_ma = tv.rolling(30).mean()
            tvl_growth = (tv - tvl_ma) / tvl_ma
            tvl_score = np.where(
                tvl_growth > 0.1,
                1,
                np.where(
                    tvl_growth > 0.03,
                    0.5,
                    np.where(
                        tvl_growth < -0.1, -1, np.where(tvl_growth < -0.03, -0.5, 0)
                    ),
                ),
            )
            scores += pd.Series(tvl_score, index=df.index).fillna(0) * 0.6

        return scores


@dataclass
class PeriodResult:
    period: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_pf: float
    test_pf: float
    train_ret: float
    test_ret: float
    train_wr: float
    test_wr: float
    train_trades: int
    test_trades: int
    degradation: float


class WalkForwardBacktester:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def run_period(self, df, scores, start_date, end_date):
        """Run backtest for a specific period"""
        mask = (df.index >= start_date) & (df.index < end_date)
        period_df = df[mask]
        period_scores = scores[mask]

        if len(period_df) < 50:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(1, len(period_df)):
            price = period_df["close"].iloc[i]
            score = period_scores.iloc[i] if i < len(period_scores) else 0
            if pd.isna(score):
                score = 0

            if position:
                should_exit = (position["dir"] == 1 and score < 0) or (
                    position["dir"] == -1 and score > 0
                )
                if should_exit:
                    pnl_pct = (price / position["entry"] - 1) * position["dir"] - 0.002
                    pnl = capital * 0.2 * 2 * min(position["mult"], 2) * pnl_pct
                    trades.append(pnl)
                    capital += pnl
                    position = None

            if not position and abs(score) >= self.threshold:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / self.threshold, 2.0)
                position = {"entry": price, "dir": direction, "mult": mult}

            equity.append(capital)

        if position:
            pnl_pct = (period_df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(capital * 0.2 * 2 * position["mult"] * pnl_pct)
            capital += trades[-1]

        if len(trades) < 3:
            return None

        wins = sum(1 for t in trades if t > 0)
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        return {
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "trades": len(trades),
        }


def generate_walk_forward_periods(start_date, end_date, train_months=12, test_months=3):
    """Generate walk-forward periods"""
    periods = []
    current = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current + pd.DateOffset(months=train_months + test_months) <= end:
        train_start = current
        train_end = current + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        periods.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )

        current = current + pd.DateOffset(months=test_months)

    return periods


def main():
    loader = DataLoader()
    scorer = MultiFactorScorer()

    # Load common data
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()
    tvl = loader.load_tvl()
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")

    logger.info("=" * 70)
    logger.info("MULTI-FACTOR STRATEGY - WALK-FORWARD VALIDATION")
    logger.info("=" * 70)
    logger.info("Train: 9 months, Test: 3 months, Rolling forward")
    logger.info("=" * 70)

    # Generate periods (2022-01 to 2025-01) - more symbols have data from 2022
    periods = generate_walk_forward_periods(
        "2022-01-01", "2025-01-01", train_months=9, test_months=3
    )
    logger.info(f"Generated {len(periods)} walk-forward periods\n")

    for i, p in enumerate(periods):
        logger.info(
            f"  Period {i+1}: Train {p['train_start'].strftime('%Y-%m')} ~ {p['train_end'].strftime('%Y-%m')} | Test {p['test_start'].strftime('%Y-%m')} ~ {p['test_end'].strftime('%Y-%m')}"
        )

    # Get symbols with enough data
    all_symbols = loader.get_all_symbols()
    valid_symbols = []

    for symbol in all_symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if (
            not df.empty
            and df.index.min() <= pd.Timestamp("2022-06-01")
            and len(df) >= 500
        ):
            valid_symbols.append(symbol)

    logger.info(f"\nSymbols with data from 2021: {len(valid_symbols)}")

    # Aggregate results by period
    period_results = {i: {"train": [], "test": []} for i in range(len(periods))}
    symbol_results = []

    backtester = WalkForwardBacktester(threshold=3.0)

    for symbol in valid_symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        scores = scorer.score_all(df, btc_df, fear_greed, macro, tvl)

        symbol_train_pfs = []
        symbol_test_pfs = []

        for i, p in enumerate(periods):
            train_result = backtester.run_period(
                df, scores, p["train_start"], p["train_end"]
            )
            test_result = backtester.run_period(
                df, scores, p["test_start"], p["test_end"]
            )

            if train_result and test_result:
                period_results[i]["train"].append(train_result["pf"])
                period_results[i]["test"].append(test_result["pf"])
                symbol_train_pfs.append(train_result["pf"])
                symbol_test_pfs.append(test_result["pf"])

        if symbol_train_pfs and symbol_test_pfs:
            avg_train = np.mean([p for p in symbol_train_pfs if p < 999])
            avg_test = np.mean([p for p in symbol_test_pfs if p < 999])
            if avg_train > 0:
                degradation = (avg_train - avg_test) / avg_train * 100
            else:
                degradation = 0
            symbol_results.append(
                {
                    "symbol": symbol,
                    "avg_train_pf": avg_train,
                    "avg_test_pf": avg_test,
                    "degradation": degradation,
                    "periods_tested": len(symbol_train_pfs),
                }
            )

    # Period-by-period analysis
    logger.info("\n" + "=" * 70)
    logger.info("PERIOD-BY-PERIOD ANALYSIS")
    logger.info("=" * 70)
    logger.info(
        f"{'Period':<10} {'Train Period':<25} {'Test Period':<25} {'Train PF':>10} {'Test PF':>10} {'Degrad':>10}"
    )
    logger.info("-" * 90)

    all_train_pfs = []
    all_test_pfs = []

    for i, p in enumerate(periods):
        train_pfs = [pf for pf in period_results[i]["train"] if pf < 999]
        test_pfs = [pf for pf in period_results[i]["test"] if pf < 999]

        if train_pfs and test_pfs:
            avg_train = np.mean(train_pfs)
            avg_test = np.mean(test_pfs)
            degradation = (
                (avg_train - avg_test) / avg_train * 100 if avg_train > 0 else 0
            )

            train_str = f"{p['train_start'].strftime('%Y-%m')} ~ {p['train_end'].strftime('%Y-%m')}"
            test_str = f"{p['test_start'].strftime('%Y-%m')} ~ {p['test_end'].strftime('%Y-%m')}"

            logger.info(
                f"Period {i+1:<3} {train_str:<25} {test_str:<25} {avg_train:>10.2f} {avg_test:>10.2f} {degradation:>9.1f}%"
            )

            all_train_pfs.extend(train_pfs)
            all_test_pfs.extend(test_pfs)

    # Overall summary
    logger.info("\n" + "=" * 70)
    logger.info("OVERALL WALK-FORWARD SUMMARY")
    logger.info("=" * 70)

    if all_train_pfs and all_test_pfs:
        avg_train_pf = np.mean(all_train_pfs)
        avg_test_pf = np.mean(all_test_pfs)
        overall_degradation = (avg_train_pf - avg_test_pf) / avg_train_pf * 100

        train_profitable = (
            sum(1 for pf in all_train_pfs if pf > 1.0) / len(all_train_pfs) * 100
        )
        test_profitable = (
            sum(1 for pf in all_test_pfs if pf > 1.0) / len(all_test_pfs) * 100
        )

        logger.info(f"Total symbol-period combinations: {len(all_train_pfs)}")
        logger.info(f"")
        logger.info(
            f"{'Metric':<25} {'In-Sample (Train)':>20} {'Out-of-Sample (Test)':>20}"
        )
        logger.info("-" * 65)
        logger.info(f"{'Average PF':<25} {avg_train_pf:>20.2f} {avg_test_pf:>20.2f}")
        logger.info(
            f"{'Profitable Rate':<25} {train_profitable:>19.1f}% {test_profitable:>19.1f}%"
        )
        logger.info(
            f"{'Median PF':<25} {np.median(all_train_pfs):>20.2f} {np.median(all_test_pfs):>20.2f}"
        )
        logger.info(f"")
        logger.info(f"Overall Degradation: {overall_degradation:.1f}%")

        # Stability assessment
        logger.info("\n" + "=" * 70)
        logger.info("STABILITY ASSESSMENT")
        logger.info("=" * 70)

        criteria = []

        # 1. OOS profitable rate > 45%
        oos_check = test_profitable > 45
        criteria.append(("OOS Profitable > 45%", oos_check, f"{test_profitable:.1f}%"))

        # 2. Degradation < 40%
        degrad_check = overall_degradation < 40
        criteria.append(
            ("Degradation < 40%", degrad_check, f"{overall_degradation:.1f}%")
        )

        # 3. OOS PF > 1.0
        oos_pf_check = avg_test_pf > 1.0
        criteria.append(("OOS Avg PF > 1.0", oos_pf_check, f"{avg_test_pf:.2f}"))

        # 4. Consistent across periods (std of test PF < mean)
        period_test_avgs = []
        for i in range(len(periods)):
            test_pfs = [pf for pf in period_results[i]["test"] if pf < 999]
            if test_pfs:
                period_test_avgs.append(np.mean(test_pfs))

        if period_test_avgs:
            consistency = (
                np.std(period_test_avgs) / np.mean(period_test_avgs)
                if np.mean(period_test_avgs) > 0
                else 999
            )
            consistency_check = consistency < 0.5
            criteria.append(
                (
                    "Period Consistency (CV < 0.5)",
                    consistency_check,
                    f"CV={consistency:.2f}",
                )
            )

        passed = sum(1 for _, check, _ in criteria if check)
        logger.info(f"")
        for name, check, value in criteria:
            status = "PASS" if check else "FAIL"
            logger.info(f"  [{status}] {name}: {value}")

        logger.info(f"\nPassed: {passed}/{len(criteria)} criteria")

        if passed >= 3:
            logger.info("\n>>> STRATEGY IS REASONABLY STABLE FOR LIVE TRADING <<<")
        else:
            logger.info("\n>>> STRATEGY NEEDS IMPROVEMENT BEFORE LIVE TRADING <<<")

    # Top stable symbols
    logger.info("\n" + "=" * 70)
    logger.info("TOP 30 STABLE SYMBOLS (Low Degradation, Test PF > 1.0)")
    logger.info("=" * 70)

    stable_symbols = [
        s for s in symbol_results if s["avg_test_pf"] > 1.0 and s["degradation"] < 30
    ]
    stable_symbols = sorted(stable_symbols, key=lambda x: -x["avg_test_pf"])[:30]

    logger.info(
        f"{'Symbol':<14} {'Train PF':>10} {'Test PF':>10} {'Degradation':>12} {'Periods':>8}"
    )
    logger.info("-" * 55)
    for s in stable_symbols:
        logger.info(
            f"{s['symbol']:<14} {s['avg_train_pf']:>10.2f} {s['avg_test_pf']:>10.2f} {s['degradation']:>11.1f}% {s['periods_tested']:>8}"
        )

    # Save results
    if symbol_results:
        results_df = pd.DataFrame(symbol_results)
        output_path = Path("E:/data/crypto_ohlcv") / "walkforward_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

    return symbol_results


if __name__ == "__main__":
    results = main()
