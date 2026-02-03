#!/usr/bin/env python3
"""
Multi-Factor + ML Strategy Validation V2
- Fixed: Paired comparison (same samples for train/test)
- Walk-Forward with proper pairing
- Out-of-Sample with strict separation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

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

    def load_ohlcv(self, symbol: str, tf: str = "4h") -> pd.DataFrame:
        key = f"{symbol}_{tf}"
        if key in self._cache:
            return self._cache[key].copy()

        filepath = self.root / f"binance_futures_{tf}" / f"{symbol}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            for col in ["datetime", "timestamp", "date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col).sort_index()
                    break
            if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
                self._cache[key] = df
                return df.copy()
        return pd.DataFrame()

    def load_fear_greed(self) -> pd.DataFrame:
        for fn in ["FEAR_GREED_INDEX_updated.csv", "FEAR_GREED_INDEX.csv"]:
            fp = self.root / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for col in ["datetime", "timestamp"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col)
                        break
                if "value" in df.columns:
                    return df[["value"]].rename(columns={"value": "fear_greed"})
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "fear_greed"})
        return pd.DataFrame()

    def load_macro(self) -> pd.DataFrame:
        dfs = []
        for fn, col in [("DXY.csv", "dxy"), ("VIX.csv", "vix")]:
            fp = self.root / "macro" / fn
            if fp.exists():
                df = pd.read_csv(fp)
                for c in ["datetime", "timestamp"]:
                    if c in df.columns:
                        df[c] = pd.to_datetime(df[c])
                        df = df.set_index(c)
                        break
                if "close" in df.columns:
                    dfs.append(df[["close"]].rename(columns={"close": col}))
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()


class MultiFactorMLStrategy:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def calc_mf_scores(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        fear_greed: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)

        # Technical
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

        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = np.where(
                fg < 20,
                2,
                np.where(fg < 35, 1, np.where(fg > 80, -2, np.where(fg > 65, -1, 0))),
            )
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

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

        return scores

    def create_features(self, df: pd.DataFrame, mf_scores: pd.Series) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        features["mf_score"] = mf_scores
        for p in [5, 10, 20, 50]:
            features[f"ret_{p}"] = close.pct_change(p)
        for p in [10, 20]:
            features[f"vol_{p}"] = close.pct_change().rolling(p).std()
        features["rsi_14"] = calc_rsi(close, 14) / 100
        for p in [20, 50]:
            features[f"ema_pos_{p}"] = close / calc_ema(close, p) - 1
        features["vol_zscore"] = (volume - volume.rolling(50).mean()) / (
            volume.rolling(50).std() + 1e-10
        )

        return features.replace([np.inf, -np.inf], np.nan).fillna(0)

    def train(self, features: pd.DataFrame, target: pd.Series) -> bool:
        valid = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid]
        y = target[valid]
        if len(X) < 100:
            return False
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
        self.model.fit(X_scaled, y)
        return True

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            return pd.Series(0.0, index=features.index)
        features = features.fillna(0)
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)
        return pd.Series(predictions * confidence * 3, index=features.index)

    def backtest(
        self, df: pd.DataFrame, scores: pd.Series, threshold: float = 4.0
    ) -> Dict:
        if len(df) < 50:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []

        for i in range(1, len(df)):
            price = df["close"].iloc[i]
            score = scores.iloc[i] if i < len(scores) else 0
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

            if not position and abs(score) >= threshold:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / threshold, 2.0)
                position = {"entry": price, "dir": direction, "mult": mult}

        if position:
            pnl_pct = (df["close"].iloc[-1] / position["entry"] - 1) * position[
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


def run_walk_forward_paired(
    loader: DataLoader,
    symbols: List[str],
    btc_df: pd.DataFrame,
    fear_greed: pd.DataFrame,
    macro: pd.DataFrame,
) -> Dict:
    """Walk-Forward with PAIRED samples (same count for train/test)"""
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION (Paired Comparison)")
    logger.info("=" * 70)
    logger.info("Train: 9 months, Test: 3 months, Rolling forward")

    periods = []
    current = pd.Timestamp("2022-01-01")
    end = pd.Timestamp("2025-01-01")

    while current + pd.DateOffset(months=12) <= end:
        train_start = current
        train_end = current + pd.DateOffset(months=9)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=3)
        periods.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        current = current + pd.DateOffset(months=3)

    logger.info(f"Periods: {len(periods)}")

    # PAIRED results - only count when BOTH train and test succeed
    paired_results = []

    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty or df.index.min() > pd.Timestamp("2022-03-01"):
            continue

        strategy = MultiFactorMLStrategy()
        mf_scores = strategy.calc_mf_scores(df, btc_df, fear_greed, macro)

        for p in periods:
            # Train period
            train_mask = (df.index >= p["train_start"]) & (df.index < p["train_end"])
            train_df = df[train_mask]
            train_mf = mf_scores[train_mask]

            if len(train_df) < 300:
                continue

            train_features = strategy.create_features(train_df, train_mf)
            future_ret = train_df["close"].shift(-6) / train_df["close"] - 1
            train_target = pd.Series(
                np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0)),
                index=train_df.index,
            )

            if not strategy.train(train_features, train_target):
                continue

            train_ml = strategy.predict(train_features)
            train_combined = train_mf + train_ml * 0.8
            train_result = strategy.backtest(train_df, train_combined)

            # Test period
            test_mask = (df.index >= p["test_start"]) & (df.index < p["test_end"])
            test_df = df[test_mask]
            test_mf = mf_scores[test_mask]

            if len(test_df) < 50:
                continue

            test_features = strategy.create_features(test_df, test_mf)
            test_ml = strategy.predict(test_features)
            test_combined = test_mf + test_ml * 0.8
            test_result = strategy.backtest(test_df, test_combined)

            # ONLY add if BOTH succeed
            if train_result and test_result:
                paired_results.append(
                    {
                        "symbol": symbol,
                        "period": f"{p['train_start'].strftime('%Y-%m')}~{p['test_end'].strftime('%Y-%m')}",
                        "train_pf": train_result["pf"],
                        "test_pf": test_result["pf"],
                        "train_ret": train_result["ret"],
                        "test_ret": test_result["ret"],
                        "train_wr": train_result["wr"],
                        "test_wr": test_result["wr"],
                        "train_trades": train_result["trades"],
                        "test_trades": test_result["trades"],
                    }
                )

    if not paired_results:
        logger.info("No valid paired results")
        return {}

    df_paired = pd.DataFrame(paired_results)

    # Summary with SAME sample count
    n = len(df_paired)
    train_pfs = df_paired[df_paired["train_pf"] < 999]["train_pf"]
    test_pfs = df_paired[df_paired["test_pf"] < 999]["test_pf"]

    train_profitable = (df_paired["train_pf"] > 1.0).sum()
    test_profitable = (df_paired["test_pf"] > 1.0).sum()

    logger.info(f"\nPaired Samples: {n}")
    logger.info(f"\n{'Metric':<25} {'Train':>15} {'Test':>15}")
    logger.info("-" * 55)
    logger.info(f"{'Sample Count':<25} {n:>15} {n:>15}")
    logger.info(
        f"{'Profitable Count':<25} {train_profitable:>15} {test_profitable:>15}"
    )
    logger.info(
        f"{'Profitable Rate':<25} {train_profitable/n*100:>14.1f}% {test_profitable/n*100:>14.1f}%"
    )
    logger.info(f"{'Avg PF':<25} {train_pfs.mean():>15.2f} {test_pfs.mean():>15.2f}")
    logger.info(
        f"{'Median PF':<25} {train_pfs.median():>15.2f} {test_pfs.median():>15.2f}"
    )
    logger.info(
        f"{'Avg Return':<25} {df_paired['train_ret'].mean():>14.1f}% {df_paired['test_ret'].mean():>14.1f}%"
    )
    logger.info(
        f"{'Avg Win Rate':<25} {df_paired['train_wr'].mean():>14.1f}% {df_paired['test_wr'].mean():>14.1f}%"
    )

    degradation = (
        (train_pfs.mean() - test_pfs.mean()) / train_pfs.mean() * 100
        if train_pfs.mean() > 0
        else 0
    )
    logger.info(f"\nPF Degradation: {degradation:.1f}%")

    # Period breakdown
    logger.info("\nBy Period:")
    for period in df_paired["period"].unique():
        subset = df_paired[df_paired["period"] == period]
        train_prof = (subset["train_pf"] > 1.0).mean() * 100
        test_prof = (subset["test_pf"] > 1.0).mean() * 100
        logger.info(
            f"  {period}: Train {train_prof:.0f}% → Test {test_prof:.0f}% ({len(subset)} samples)"
        )

    return {
        "n_samples": n,
        "train_profitable_rate": train_profitable / n * 100,
        "test_profitable_rate": test_profitable / n * 100,
        "train_avg_pf": train_pfs.mean(),
        "test_avg_pf": test_pfs.mean(),
        "degradation": degradation,
        "df": df_paired,
    }


def run_oos_validation_paired(
    loader: DataLoader,
    symbols: List[str],
    btc_df: pd.DataFrame,
    fear_greed: pd.DataFrame,
    macro: pd.DataFrame,
) -> Dict:
    """Out-of-Sample with paired comparison"""
    logger.info("\n" + "=" * 70)
    logger.info("OUT-OF-SAMPLE VALIDATION (Paired Comparison)")
    logger.info("=" * 70)
    logger.info("Train: 2022-01 ~ 2023-12")
    logger.info("Test:  2024-01 ~ 2025-01")

    TRAIN_START = pd.Timestamp("2022-01-01")
    TRAIN_END = pd.Timestamp("2024-01-01")
    TEST_START = pd.Timestamp("2024-01-01")
    TEST_END = pd.Timestamp("2025-01-26")

    paired_results = []

    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue
        if df.index.min() > TRAIN_START or df.index.max() < TEST_START:
            continue

        strategy = MultiFactorMLStrategy()
        mf_scores = strategy.calc_mf_scores(df, btc_df, fear_greed, macro)

        # Train
        train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
        train_df = df[train_mask]
        train_mf = mf_scores[train_mask]

        if len(train_df) < 500:
            continue

        train_features = strategy.create_features(train_df, train_mf)
        future_ret = train_df["close"].shift(-6) / train_df["close"] - 1
        train_target = pd.Series(
            np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0)),
            index=train_df.index,
        )

        if not strategy.train(train_features, train_target):
            continue

        train_ml = strategy.predict(train_features)
        train_combined = train_mf + train_ml * 0.8
        train_result = strategy.backtest(train_df, train_combined)

        # Test
        test_mask = (df.index >= TEST_START) & (df.index < TEST_END)
        test_df = df[test_mask]
        test_mf = mf_scores[test_mask]

        if len(test_df) < 100:
            continue

        test_features = strategy.create_features(test_df, test_mf)
        test_ml = strategy.predict(test_features)
        test_combined = test_mf + test_ml * 0.8
        test_result = strategy.backtest(test_df, test_combined)

        # ONLY add if BOTH succeed
        if train_result and test_result:
            paired_results.append(
                {
                    "symbol": symbol,
                    "train_pf": train_result["pf"],
                    "test_pf": test_result["pf"],
                    "train_ret": train_result["ret"],
                    "test_ret": test_result["ret"],
                    "train_wr": train_result["wr"],
                    "test_wr": test_result["wr"],
                    "train_trades": train_result["trades"],
                    "test_trades": test_result["trades"],
                }
            )

    if not paired_results:
        logger.info("No valid results")
        return {}

    df_paired = pd.DataFrame(paired_results)
    n = len(df_paired)

    train_pfs = df_paired[df_paired["train_pf"] < 999]["train_pf"]
    test_pfs = df_paired[df_paired["test_pf"] < 999]["test_pf"]

    train_profitable = (df_paired["train_pf"] > 1.0).sum()
    test_profitable = (df_paired["test_pf"] > 1.0).sum()

    logger.info(f"\nPaired Samples: {n}")
    logger.info(f"\n{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15}")
    logger.info("-" * 55)
    logger.info(f"{'Sample Count':<25} {n:>15} {n:>15}")
    logger.info(
        f"{'Profitable Count':<25} {train_profitable:>15} {test_profitable:>15}"
    )
    logger.info(
        f"{'Profitable Rate':<25} {train_profitable/n*100:>14.1f}% {test_profitable/n*100:>14.1f}%"
    )
    logger.info(f"{'Avg PF':<25} {train_pfs.mean():>15.2f} {test_pfs.mean():>15.2f}")
    logger.info(
        f"{'Median PF':<25} {train_pfs.median():>15.2f} {test_pfs.median():>15.2f}"
    )
    logger.info(
        f"{'Avg Return':<25} {df_paired['train_ret'].mean():>14.1f}% {df_paired['test_ret'].mean():>14.1f}%"
    )
    logger.info(
        f"{'Avg Win Rate':<25} {df_paired['train_wr'].mean():>14.1f}% {df_paired['test_wr'].mean():>14.1f}%"
    )

    degradation = (
        (train_pfs.mean() - test_pfs.mean()) / train_pfs.mean() * 100
        if train_pfs.mean() > 0
        else 0
    )
    logger.info(f"\nPF Degradation: {degradation:.1f}%")

    # Consistent performers
    consistent = df_paired[(df_paired["train_pf"] > 1.0) & (df_paired["test_pf"] > 1.0)]
    logger.info(
        f"\nConsistent (PF > 1.0 both): {len(consistent)}/{n} ({len(consistent)/n*100:.1f}%)"
    )

    if len(consistent) > 0:
        logger.info("\nTop 15 Consistent:")
        top = consistent.nlargest(15, "test_pf")
        for _, r in top.iterrows():
            logger.info(
                f"  {r['symbol']:<14} Train PF={r['train_pf']:.2f} Test PF={r['test_pf']:.2f} "
                f"Test Ret={r['test_ret']:+.1f}%"
            )

    return {
        "n_samples": n,
        "train_profitable_rate": train_profitable / n * 100,
        "test_profitable_rate": test_profitable / n * 100,
        "train_avg_pf": train_pfs.mean(),
        "test_avg_pf": test_pfs.mean(),
        "degradation": degradation,
        "consistent_count": len(consistent),
        "df": df_paired,
    }


def main():
    loader = DataLoader()

    logger.info("=" * 70)
    logger.info("MULTI-FACTOR + ML STRATEGY VALIDATION V2")
    logger.info("=" * 70)
    logger.info("Fixed: Paired comparison (same N for train/test)")

    btc_df = loader.load_ohlcv("BTCUSDT", "4h")
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()

    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )

    logger.info(f"Total symbols: {len(symbols)}")

    # 1. Walk-Forward (Paired)
    wf_results = run_walk_forward_paired(loader, symbols, btc_df, fear_greed, macro)

    # 2. Out-of-Sample (Paired)
    oos_results = run_oos_validation_paired(loader, symbols, btc_df, fear_greed, macro)

    # Final Assessment
    logger.info("\n" + "=" * 70)
    logger.info("FINAL ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    if oos_results:
        criteria.append(
            (
                "OOS Profitable >= 50%",
                oos_results["test_profitable_rate"] >= 50,
                f"{oos_results['test_profitable_rate']:.1f}%",
            )
        )
        criteria.append(
            (
                "OOS Avg PF >= 1.0",
                oos_results["test_avg_pf"] >= 1.0,
                f"{oos_results['test_avg_pf']:.2f}",
            )
        )
        criteria.append(
            (
                "PF Degradation <= 50%",
                oos_results["degradation"] <= 50,
                f"{oos_results['degradation']:.1f}%",
            )
        )
        cons_rate = oos_results["consistent_count"] / oos_results["n_samples"] * 100
        criteria.append(("Consistent >= 25%", cons_rate >= 25, f"{cons_rate:.1f}%"))

    if wf_results:
        criteria.append(
            (
                "Walk-Forward Test >= 40%",
                wf_results["test_profitable_rate"] >= 40,
                f"{wf_results['test_profitable_rate']:.1f}%",
            )
        )

    logger.info("")
    passed = 0
    for name, check, value in criteria:
        status = "PASS" if check else "FAIL"
        if check:
            passed += 1
        logger.info(f"  [{status}] {name}: {value}")

    logger.info(f"\nPassed: {passed}/{len(criteria)}")

    if passed >= 4:
        logger.info("\n>>> STRATEGY VALIDATED <<<")
    elif passed >= 3:
        logger.info("\n>>> MARGINALLY ACCEPTABLE <<<")
    else:
        logger.info("\n>>> NEEDS IMPROVEMENT <<<")

    # Save
    if oos_results and "df" in oos_results:
        output_path = DATA_ROOT / "mf_ml_validation_v2_results.csv"
        oos_results["df"].to_csv(output_path, index=False)
        logger.info(f"\nSaved to: {output_path}")

    return {"wf": wf_results, "oos": oos_results}


if __name__ == "__main__":
    results = main()
