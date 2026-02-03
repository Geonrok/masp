#!/usr/bin/env python3
"""
Multi-Factor Strategy Out-of-Sample Validation
- Train: 2021-01 ~ 2023-12 (completely separate)
- Test: 2024-01 ~ 2025-01 (held-out, never seen)
- Measures true generalization ability
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

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
class BacktestResult:
    symbol: str
    pf: float
    ret: float
    wr: float
    mdd: float
    trades: int
    sharpe: float


class Backtester:
    def __init__(self, threshold=3.0):
        self.threshold = threshold

    def run(self, df, scores, start_date, end_date):
        mask = (df.index >= start_date) & (df.index < end_date)
        period_df = df[mask]
        period_scores = scores[mask]

        if len(period_df) < 100:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]
        daily_returns = []

        for i in range(1, len(period_df)):
            price = period_df["close"].iloc[i]
            score = period_scores.iloc[i] if i < len(period_scores) else 0
            if pd.isna(score):
                score = 0

            prev_capital = capital

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
            if prev_capital > 0:
                daily_returns.append((capital - prev_capital) / prev_capital)

        if position:
            pnl_pct = (period_df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(capital * 0.2 * 2 * position["mult"] * pnl_pct)
            capital += trades[-1]

        if len(trades) < 5:
            return None

        equity = pd.Series(equity)
        mdd = (
            (equity - equity.expanding().max()) / equity.expanding().max()
        ).min() * 100
        wins = sum(1 for t in trades if t > 0)
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        # Sharpe ratio (annualized, assuming 4h bars = 6 per day)
        if daily_returns:
            sharpe = (
                np.mean(daily_returns)
                / (np.std(daily_returns) + 1e-10)
                * np.sqrt(252 * 6)
            )
        else:
            sharpe = 0

        return BacktestResult(
            symbol="",
            pf=min(pf, 999),
            ret=(capital / init - 1) * 100,
            wr=wins / len(trades) * 100,
            mdd=mdd,
            trades=len(trades),
            sharpe=sharpe,
        )


def main():
    loader = DataLoader()
    scorer = MultiFactorScorer()

    # Load common data
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()
    tvl = loader.load_tvl()
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")

    logger.info("=" * 70)
    logger.info("MULTI-FACTOR STRATEGY - OUT-OF-SAMPLE VALIDATION")
    logger.info("=" * 70)
    logger.info("Train Period: 2022-01-01 ~ 2023-12-31 (In-Sample)")
    logger.info("Test Period:  2024-01-01 ~ 2025-01-26 (Out-of-Sample)")
    logger.info("=" * 70)

    TRAIN_START = "2022-01-01"
    TRAIN_END = "2024-01-01"
    TEST_START = "2024-01-01"
    TEST_END = "2025-01-26"

    # Get all symbols
    all_symbols = loader.get_all_symbols()

    train_results = []
    test_results = []
    comparison = []

    backtester = Backtester(threshold=3.0)

    for symbol in all_symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        # Need data spanning both periods (with some buffer for indicators)
        if df.index.min() > pd.Timestamp("2022-03-01") or df.index.max() < pd.Timestamp(
            "2024-06-01"
        ):
            continue

        scores = scorer.score_all(df, btc_df, fear_greed, macro, tvl)

        train_result = backtester.run(df, scores, TRAIN_START, TRAIN_END)
        test_result = backtester.run(df, scores, TEST_START, TEST_END)

        if train_result and test_result:
            train_result.symbol = symbol
            test_result.symbol = symbol
            train_results.append(train_result)
            test_results.append(test_result)

            if train_result.pf < 999:
                degradation = (
                    (train_result.pf - test_result.pf) / train_result.pf * 100
                    if train_result.pf > 0
                    else 0
                )
            else:
                degradation = 0

            comparison.append(
                {
                    "symbol": symbol,
                    "train_pf": train_result.pf,
                    "test_pf": test_result.pf,
                    "train_ret": train_result.ret,
                    "test_ret": test_result.ret,
                    "train_wr": train_result.wr,
                    "test_wr": test_result.wr,
                    "train_mdd": train_result.mdd,
                    "test_mdd": test_result.mdd,
                    "train_trades": train_result.trades,
                    "test_trades": test_result.trades,
                    "degradation": degradation,
                }
            )

    logger.info(f"\nSymbols with valid results: {len(comparison)}")

    if not comparison:
        logger.info("No valid results found.")
        return

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 70)

    train_pfs = [c["train_pf"] for c in comparison if c["train_pf"] < 999]
    test_pfs = [c["test_pf"] for c in comparison if c["test_pf"] < 999]
    train_rets = [c["train_ret"] for c in comparison]
    test_rets = [c["test_ret"] for c in comparison]

    train_profitable = sum(1 for c in comparison if c["train_pf"] > 1.0)
    test_profitable = sum(1 for c in comparison if c["test_pf"] > 1.0)

    logger.info(
        f"\n{'Metric':<25} {'In-Sample (Train)':>20} {'Out-of-Sample (Test)':>20}"
    )
    logger.info("-" * 65)
    logger.info(f"{'Symbols':<25} {len(comparison):>20} {len(comparison):>20}")
    logger.info(f"{'Profitable':<25} {train_profitable:>20} {test_profitable:>20}")
    logger.info(
        f"{'Profitable Rate':<25} {train_profitable/len(comparison)*100:>19.1f}% {test_profitable/len(comparison)*100:>19.1f}%"
    )
    logger.info(
        f"{'Average PF':<25} {np.mean(train_pfs):>20.2f} {np.mean(test_pfs):>20.2f}"
    )
    logger.info(
        f"{'Median PF':<25} {np.median(train_pfs):>20.2f} {np.median(test_pfs):>20.2f}"
    )
    logger.info(
        f"{'Average Return':<25} {np.mean(train_rets):>19.1f}% {np.mean(test_rets):>19.1f}%"
    )
    logger.info(
        f"{'Median Return':<25} {np.median(train_rets):>19.1f}% {np.median(test_rets):>19.1f}%"
    )

    # Calculate overall degradation
    if train_pfs and test_pfs:
        avg_train_pf = np.mean(train_pfs)
        avg_test_pf = np.mean(test_pfs)
        overall_degradation = (
            (avg_train_pf - avg_test_pf) / avg_train_pf * 100 if avg_train_pf > 0 else 0
        )
        logger.info(f"\nOverall PF Degradation: {overall_degradation:.1f}%")

    # Distribution analysis
    logger.info("\n" + "=" * 70)
    logger.info("PF DISTRIBUTION")
    logger.info("=" * 70)

    buckets = [
        (2.0, "PF >= 2.0"),
        (1.5, "PF >= 1.5"),
        (1.3, "PF >= 1.3"),
        (1.0, "PF >= 1.0"),
        (0.0, "PF < 1.0"),
    ]

    logger.info(f"\n{'Bucket':<15} {'Train':>15} {'Test':>15}")
    logger.info("-" * 45)
    for thresh, name in buckets:
        if thresh > 0:
            train_count = sum(1 for c in comparison if c["train_pf"] >= thresh)
            test_count = sum(1 for c in comparison if c["test_pf"] >= thresh)
        else:
            train_count = sum(1 for c in comparison if c["train_pf"] < 1.0)
            test_count = sum(1 for c in comparison if c["test_pf"] < 1.0)
        logger.info(
            f"{name:<15} {train_count:>10} ({train_count/len(comparison)*100:4.1f}%) {test_count:>10} ({test_count/len(comparison)*100:4.1f}%)"
        )

    # Stability assessment
    logger.info("\n" + "=" * 70)
    logger.info("OUT-OF-SAMPLE STABILITY ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    # 1. OOS profitable rate >= 50%
    oos_rate = test_profitable / len(comparison) * 100
    criteria.append(("OOS Profitable >= 50%", oos_rate >= 50, f"{oos_rate:.1f}%"))

    # 2. PF Degradation <= 30%
    criteria.append(
        (
            "PF Degradation <= 30%",
            overall_degradation <= 30,
            f"{overall_degradation:.1f}%",
        )
    )

    # 3. OOS Avg PF >= 1.0
    criteria.append(("OOS Avg PF >= 1.0", avg_test_pf >= 1.0, f"{avg_test_pf:.2f}"))

    # 4. OOS Median PF >= 1.0
    median_test_pf = np.median(test_pfs)
    criteria.append(
        ("OOS Median PF >= 1.0", median_test_pf >= 1.0, f"{median_test_pf:.2f}")
    )

    # 5. Average OOS Return > 0
    avg_test_ret = np.mean(test_rets)
    criteria.append(("OOS Avg Return > 0%", avg_test_ret > 0, f"{avg_test_ret:.1f}%"))

    logger.info("")
    passed = 0
    for name, check, value in criteria:
        status = "PASS" if check else "FAIL"
        if check:
            passed += 1
        logger.info(f"  [{status}] {name}: {value}")

    logger.info(f"\nPassed: {passed}/{len(criteria)} criteria")

    if passed >= 4:
        logger.info("\n>>> STRATEGY PASSES OUT-OF-SAMPLE VALIDATION <<<")
        verdict = "PASS"
    elif passed >= 3:
        logger.info("\n>>> STRATEGY MARGINALLY ACCEPTABLE - USE WITH CAUTION <<<")
        verdict = "MARGINAL"
    else:
        logger.info("\n>>> STRATEGY FAILS OUT-OF-SAMPLE VALIDATION <<<")
        verdict = "FAIL"

    # Best performers in OOS
    logger.info("\n" + "=" * 70)
    logger.info("TOP 30 OOS PERFORMERS")
    logger.info("=" * 70)

    sorted_by_oos = sorted(comparison, key=lambda x: -x["test_pf"])[:30]
    logger.info(
        f"\n{'Symbol':<14} {'Train PF':>10} {'Test PF':>10} {'Train Ret':>12} {'Test Ret':>12} {'Degrad':>10}"
    )
    logger.info("-" * 70)
    for c in sorted_by_oos:
        logger.info(
            f"{c['symbol']:<14} {c['train_pf']:>10.2f} {c['test_pf']:>10.2f} {c['train_ret']:>11.1f}% {c['test_ret']:>11.1f}% {c['degradation']:>9.1f}%"
        )

    # Consistent performers (good in both)
    logger.info("\n" + "=" * 70)
    logger.info("CONSISTENT PERFORMERS (Train PF > 1.0 AND Test PF > 1.0)")
    logger.info("=" * 70)

    consistent = [c for c in comparison if c["train_pf"] > 1.0 and c["test_pf"] > 1.0]
    consistent = sorted(consistent, key=lambda x: -(x["train_pf"] + x["test_pf"]))

    logger.info(
        f"\nFound {len(consistent)} consistent performers ({len(consistent)/len(comparison)*100:.1f}%)\n"
    )

    if consistent:
        logger.info(
            f"{'Symbol':<14} {'Train PF':>10} {'Test PF':>10} {'Train Ret':>12} {'Test Ret':>12}"
        )
        logger.info("-" * 60)
        for c in consistent[:30]:
            logger.info(
                f"{c['symbol']:<14} {c['train_pf']:>10.2f} {c['test_pf']:>10.2f} {c['train_ret']:>11.1f}% {c['test_ret']:>11.1f}%"
            )

    # Save results
    results_df = pd.DataFrame(comparison)
    output_path = Path("E:/data/crypto_ohlcv") / "oos_validation_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    return comparison, verdict


if __name__ == "__main__":
    results, verdict = main()
