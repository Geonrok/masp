#!/usr/bin/env python3
"""
Multi-Factor + ML Strategy Comprehensive Validation
1. Walk-Forward Validation (rolling train/test)
2. Out-of-Sample Validation (strict time separation)
3. Overfitting Analysis (train vs test comparison)
4. Stability Metrics
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List

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


# ============================================================================
# Utility Functions
# ============================================================================


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
    """Combined Multi-Factor + ML Strategy with proper time separation"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def _calc_mf_scores(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        fear_greed: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.Series:
        """Calculate Multi-Factor scores"""
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

        # Fear & Greed
        if not fear_greed.empty:
            fg = fear_greed["fear_greed"].reindex(df.index, method="ffill")
            fg_score = np.where(
                fg < 20,
                2,
                np.where(fg < 35, 1, np.where(fg > 80, -2, np.where(fg > 65, -1, 0))),
            )
            scores += pd.Series(fg_score, index=df.index).fillna(0) * 1.2

        # BTC Correlation
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

        # Macro
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

    def _create_ml_features(
        self, df: pd.DataFrame, mf_scores: pd.Series
    ) -> pd.DataFrame:
        """Create ML features"""
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # Multi-Factor score as feature
        features["mf_score"] = mf_scores

        # Returns
        for p in [5, 10, 20, 50]:
            features[f"ret_{p}"] = close.pct_change(p)

        # Volatility
        for p in [10, 20]:
            features[f"vol_{p}"] = close.pct_change().rolling(p).std()

        # RSI
        features["rsi_14"] = calc_rsi(close, 14) / 100

        # EMA position
        for p in [20, 50]:
            features[f"ema_pos_{p}"] = close / calc_ema(close, p) - 1

        # Volume
        features["vol_zscore"] = (volume - volume.rolling(50).mean()) / (
            volume.rolling(50).std() + 1e-10
        )

        return features.replace([np.inf, -np.inf], np.nan).fillna(0)

    def train_model(self, features: pd.DataFrame, target: pd.Series):
        """Train ML model"""
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
        """Generate predictions"""
        if self.model is None:
            return pd.Series(0.0, index=features.index)

        features = features.fillna(0)
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)

        return pd.Series(predictions * confidence * 3, index=features.index)

    def backtest_period(
        self,
        df: pd.DataFrame,
        scores: pd.Series,
        start_date,
        end_date,
        threshold: float = 4.0,
    ) -> Dict:
        """Backtest on a specific period"""
        mask = (df.index >= start_date) & (df.index < end_date)
        period_df = df[mask]
        period_scores = scores[mask]

        if len(period_df) < 50:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []

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

            if not position and abs(score) >= threshold:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / threshold, 2.0)
                position = {"entry": price, "dir": direction, "mult": mult}

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


def run_walk_forward_validation(
    loader: DataLoader,
    symbols: List[str],
    btc_df: pd.DataFrame,
    fear_greed: pd.DataFrame,
    macro: pd.DataFrame,
) -> Dict:
    """Walk-Forward Validation: Train on past, test on future, roll forward"""
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION")
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

    logger.info(f"Generated {len(periods)} periods\n")

    all_train_results = []
    all_test_results = []

    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty or df.index.min() > pd.Timestamp("2022-03-01"):
            continue

        strategy = MultiFactorMLStrategy()
        mf_scores = strategy._calc_mf_scores(df, btc_df, fear_greed, macro)

        for p in periods:
            # Get training data
            train_mask = (df.index >= p["train_start"]) & (df.index < p["train_end"])
            train_df = df[train_mask]
            train_mf = mf_scores[train_mask]

            if len(train_df) < 300:
                continue

            # Create features and target for training
            train_features = strategy._create_ml_features(train_df, train_mf)
            future_ret = train_df["close"].shift(-6) / train_df["close"] - 1
            train_target = pd.Series(
                np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0)),
                index=train_df.index,
            )

            # Train model
            if not strategy.train_model(train_features, train_target):
                continue

            # Generate predictions for train period
            train_ml_scores = strategy.predict(train_features)
            train_combined = train_mf + train_ml_scores * 0.8

            # Train backtest
            train_result = strategy.backtest_period(
                train_df, train_combined, p["train_start"], p["train_end"]
            )
            if train_result:
                all_train_results.append(train_result)

            # Test period - USE SAME MODEL (no retraining!)
            test_mask = (df.index >= p["test_start"]) & (df.index < p["test_end"])
            test_df = df[test_mask]
            test_mf = mf_scores[test_mask]

            if len(test_df) < 50:
                continue

            test_features = strategy._create_ml_features(test_df, test_mf)
            test_ml_scores = strategy.predict(test_features)
            test_combined = test_mf + test_ml_scores * 0.8

            # Test backtest
            test_result = strategy.backtest_period(
                test_df, test_combined, p["test_start"], p["test_end"]
            )
            if test_result:
                all_test_results.append(test_result)

    # Summary
    if all_train_results and all_test_results:
        train_pfs = [r["pf"] for r in all_train_results if r["pf"] < 999]
        test_pfs = [r["pf"] for r in all_test_results if r["pf"] < 999]

        train_profitable = (
            sum(1 for r in all_train_results if r["pf"] > 1.0)
            / len(all_train_results)
            * 100
        )
        test_profitable = (
            sum(1 for r in all_test_results if r["pf"] > 1.0)
            / len(all_test_results)
            * 100
        )

        logger.info(f"\n{'Metric':<25} {'Train':>15} {'Test (OOS)':>15}")
        logger.info("-" * 55)
        logger.info(
            f"{'Sample Count':<25} {len(all_train_results):>15} {len(all_test_results):>15}"
        )
        logger.info(
            f"{'Profitable Rate':<25} {train_profitable:>14.1f}% {test_profitable:>14.1f}%"
        )
        logger.info(
            f"{'Avg PF':<25} {np.mean(train_pfs):>15.2f} {np.mean(test_pfs):>15.2f}"
        )
        logger.info(
            f"{'Median PF':<25} {np.median(train_pfs):>15.2f} {np.median(test_pfs):>15.2f}"
        )
        logger.info(
            f"{'Avg Return':<25} {np.mean([r['ret'] for r in all_train_results]):>14.1f}% "
            f"{np.mean([r['ret'] for r in all_test_results]):>14.1f}%"
        )

        degradation = (
            (np.mean(train_pfs) - np.mean(test_pfs)) / np.mean(train_pfs) * 100
            if np.mean(train_pfs) > 0
            else 0
        )

        return {
            "train_profitable": train_profitable,
            "test_profitable": test_profitable,
            "train_avg_pf": np.mean(train_pfs),
            "test_avg_pf": np.mean(test_pfs),
            "degradation": degradation,
        }

    return {}


def run_oos_validation(
    loader: DataLoader,
    symbols: List[str],
    btc_df: pd.DataFrame,
    fear_greed: pd.DataFrame,
    macro: pd.DataFrame,
) -> Dict:
    """Out-of-Sample Validation: Strict time separation"""
    logger.info("\n" + "=" * 70)
    logger.info("OUT-OF-SAMPLE VALIDATION")
    logger.info("=" * 70)
    logger.info("Train: 2022-01 ~ 2023-12 (In-Sample)")
    logger.info("Test:  2024-01 ~ 2025-01 (Out-of-Sample, Never Seen)")

    TRAIN_START = pd.Timestamp("2022-01-01")
    TRAIN_END = pd.Timestamp("2024-01-01")
    TEST_START = pd.Timestamp("2024-01-01")
    TEST_END = pd.Timestamp("2025-01-26")

    train_results = []
    test_results = []
    comparison = []

    for symbol in symbols:
        df = loader.load_ohlcv(symbol, "4h")
        if df.empty:
            continue

        if df.index.min() > TRAIN_START or df.index.max() < TEST_START:
            continue

        strategy = MultiFactorMLStrategy()
        mf_scores = strategy._calc_mf_scores(df, btc_df, fear_greed, macro)

        # TRAIN PERIOD
        train_mask = (df.index >= TRAIN_START) & (df.index < TRAIN_END)
        train_df = df[train_mask]
        train_mf = mf_scores[train_mask]

        if len(train_df) < 500:
            continue

        # Create features and train
        train_features = strategy._create_ml_features(train_df, train_mf)
        future_ret = train_df["close"].shift(-6) / train_df["close"] - 1
        train_target = pd.Series(
            np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0)),
            index=train_df.index,
        )

        if not strategy.train_model(train_features, train_target):
            continue

        # Train predictions and backtest
        train_ml_scores = strategy.predict(train_features)
        train_combined = train_mf + train_ml_scores * 0.8
        train_result = strategy.backtest_period(
            train_df, train_combined, TRAIN_START, TRAIN_END
        )

        # TEST PERIOD - Apply trained model to completely unseen data
        test_mask = (df.index >= TEST_START) & (df.index < TEST_END)
        test_df = df[test_mask]
        test_mf = mf_scores[test_mask]

        if len(test_df) < 100:
            continue

        test_features = strategy._create_ml_features(test_df, test_mf)
        test_ml_scores = strategy.predict(test_features)
        test_combined = test_mf + test_ml_scores * 0.8
        test_result = strategy.backtest_period(
            test_df, test_combined, TEST_START, TEST_END
        )

        if train_result and test_result:
            train_results.append(train_result)
            test_results.append(test_result)
            comparison.append(
                {
                    "symbol": symbol,
                    "train_pf": train_result["pf"],
                    "test_pf": test_result["pf"],
                    "train_ret": train_result["ret"],
                    "test_ret": test_result["ret"],
                    "train_wr": train_result["wr"],
                    "test_wr": test_result["wr"],
                }
            )

    if not comparison:
        logger.info("No valid results")
        return {}

    df_comp = pd.DataFrame(comparison)

    # Summary
    train_pfs = [r["train_pf"] for r in comparison if r["train_pf"] < 999]
    test_pfs = [r["test_pf"] for r in comparison if r["test_pf"] < 999]

    train_profitable = (df_comp["train_pf"] > 1.0).mean() * 100
    test_profitable = (df_comp["test_pf"] > 1.0).mean() * 100

    logger.info(f"\nSymbols tested: {len(comparison)}")
    logger.info(f"\n{'Metric':<25} {'In-Sample':>15} {'Out-of-Sample':>15}")
    logger.info("-" * 55)
    logger.info(
        f"{'Profitable Rate':<25} {train_profitable:>14.1f}% {test_profitable:>14.1f}%"
    )
    logger.info(
        f"{'Avg PF':<25} {np.mean(train_pfs):>15.2f} {np.mean(test_pfs):>15.2f}"
    )
    logger.info(
        f"{'Median PF':<25} {np.median(train_pfs):>15.2f} {np.median(test_pfs):>15.2f}"
    )
    logger.info(
        f"{'Avg Return':<25} {df_comp['train_ret'].mean():>14.1f}% {df_comp['test_ret'].mean():>14.1f}%"
    )
    logger.info(
        f"{'Avg Win Rate':<25} {df_comp['train_wr'].mean():>14.1f}% {df_comp['test_wr'].mean():>14.1f}%"
    )

    degradation = (
        (np.mean(train_pfs) - np.mean(test_pfs)) / np.mean(train_pfs) * 100
        if np.mean(train_pfs) > 0
        else 0
    )
    logger.info(f"\nPF Degradation: {degradation:.1f}%")

    # Consistent performers
    consistent = df_comp[(df_comp["train_pf"] > 1.0) & (df_comp["test_pf"] > 1.0)]
    logger.info(
        f"\nConsistent Performers (PF > 1.0 in both): {len(consistent)}/{len(df_comp)} ({len(consistent)/len(df_comp)*100:.1f}%)"
    )

    if len(consistent) > 0:
        logger.info("\nTop 15 Consistent Performers:")
        top = consistent.nlargest(15, "test_pf")
        for _, r in top.iterrows():
            logger.info(
                f"  {r['symbol']:<14} Train PF={r['train_pf']:.2f} Test PF={r['test_pf']:.2f} "
                f"Test Ret={r['test_ret']:+.1f}%"
            )

    return {
        "symbols_tested": len(comparison),
        "train_profitable": train_profitable,
        "test_profitable": test_profitable,
        "train_avg_pf": np.mean(train_pfs),
        "test_avg_pf": np.mean(test_pfs),
        "degradation": degradation,
        "consistent_count": len(consistent),
        "comparison": df_comp,
    }


def main():
    loader = DataLoader()

    logger.info("=" * 70)
    logger.info("MULTI-FACTOR + ML STRATEGY - COMPREHENSIVE VALIDATION")
    logger.info("=" * 70)

    # Load common data
    btc_df = loader.load_ohlcv("BTCUSDT", "4h")
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()

    # Get symbols
    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )

    logger.info(f"Total symbols: {len(symbols)}")

    # 1. Walk-Forward Validation
    wf_results = run_walk_forward_validation(loader, symbols, btc_df, fear_greed, macro)

    # 2. Out-of-Sample Validation
    oos_results = run_oos_validation(loader, symbols, btc_df, fear_greed, macro)

    # 3. Final Assessment
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VALIDATION ASSESSMENT")
    logger.info("=" * 70)

    criteria = []

    # Criterion 1: OOS Profitable Rate >= 50%
    if oos_results:
        oos_rate = oos_results["test_profitable"]
        criteria.append(("OOS Profitable >= 50%", oos_rate >= 50, f"{oos_rate:.1f}%"))

    # Criterion 2: OOS Avg PF >= 1.0
    if oos_results:
        oos_pf = oos_results["test_avg_pf"]
        criteria.append(("OOS Avg PF >= 1.0", oos_pf >= 1.0, f"{oos_pf:.2f}"))

    # Criterion 3: PF Degradation <= 50%
    if oos_results:
        degrad = oos_results["degradation"]
        criteria.append(("PF Degradation <= 50%", degrad <= 50, f"{degrad:.1f}%"))

    # Criterion 4: Walk-Forward Test Profitable >= 45%
    if wf_results:
        wf_rate = wf_results["test_profitable"]
        criteria.append(("Walk-Forward Test >= 45%", wf_rate >= 45, f"{wf_rate:.1f}%"))

    # Criterion 5: Consistent performers >= 20%
    if oos_results:
        cons_rate = (
            oos_results["consistent_count"] / oos_results["symbols_tested"] * 100
        )
        criteria.append(
            ("Consistent Symbols >= 20%", cons_rate >= 20, f"{cons_rate:.1f}%")
        )

    logger.info("")
    passed = 0
    for name, check, value in criteria:
        status = "PASS" if check else "FAIL"
        if check:
            passed += 1
        logger.info(f"  [{status}] {name}: {value}")

    logger.info(f"\nPassed: {passed}/{len(criteria)} criteria")

    if passed >= 4:
        logger.info("\n>>> STRATEGY VALIDATED - READY FOR PAPER TRADING <<<")
        verdict = "PASS"
    elif passed >= 3:
        logger.info("\n>>> STRATEGY MARGINALLY ACCEPTABLE - USE WITH CAUTION <<<")
        verdict = "MARGINAL"
    else:
        logger.info("\n>>> STRATEGY FAILED VALIDATION - NEEDS IMPROVEMENT <<<")
        verdict = "FAIL"

    # Save results
    if oos_results and "comparison" in oos_results:
        output_path = DATA_ROOT / "mf_ml_validation_results.csv"
        oos_results["comparison"].to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

    return {
        "walk_forward": wf_results,
        "out_of_sample": oos_results,
        "verdict": verdict,
        "passed_criteria": passed,
        "total_criteria": len(criteria),
    }


if __name__ == "__main__":
    results = main()
