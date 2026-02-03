#!/usr/bin/env python3
"""
Combined Strategy Backtester
1. Multi-Factor + On-chain (simulated)
2. Multi-Factor + Korean Exchange Premium (Upbit/Bithumb)
3. Multi-Factor + ML Prediction

Tests synergies between different signal sources
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict

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


# ============================================================================
# Data Loader
# ============================================================================


class DataLoader:
    def __init__(self):
        self.root = DATA_ROOT
        self._cache = {}

    def load_ohlcv(
        self, symbol: str, source: str = "binance_futures", tf: str = "4h"
    ) -> pd.DataFrame:
        key = f"{source}_{tf}_{symbol}"
        if key in self._cache:
            return self._cache[key].copy()

        folder_map = {
            ("binance_futures", "4h"): "binance_futures_4h",
            ("binance_futures", "1d"): "binance_futures_1d",
            ("binance_spot", "4h"): "binance_spot_4h",
            ("binance_spot", "1d"): "binance_spot_1d",
            ("upbit", "4h"): "bithumb_4h",  # Using bithumb as proxy
            ("bithumb", "4h"): "bithumb_4h",
            ("bithumb", "1d"): "bithumb_1d",
        }

        folder = folder_map.get((source, tf))
        if not folder:
            return pd.DataFrame()

        # Try different filename formats
        for fn in [
            f"{symbol}.csv",
            f"{symbol.replace('USDT', '')}.csv",
            f"{symbol.replace('USDT', '')}_KRW.csv",
            f"{symbol.replace('USDT', 'KRW')}.csv",
        ]:
            filepath = self.root / folder / fn
            if filepath.exists():
                df = pd.read_csv(filepath)
                for col in ["datetime", "timestamp", "date"]:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.set_index(col).sort_index()
                        break
                if all(
                    c in df.columns for c in ["open", "high", "low", "close", "volume"]
                ):
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
        for fn, col in [("DXY.csv", "dxy"), ("SP500.csv", "sp500"), ("VIX.csv", "vix")]:
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

    def load_funding(self, symbol: str) -> pd.DataFrame:
        fp = self.root / "binance_funding_rate" / f"{symbol}.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for col in ["datetime", "timestamp"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df = df.set_index(col)
                    break
            if "funding_rate" in df.columns:
                return df[["funding_rate"]]
        return pd.DataFrame()


# ============================================================================
# Signal Generators
# ============================================================================


class MultiFactorSignal:
    """Original Multi-Factor scoring"""

    def generate(
        self,
        df: pd.DataFrame,
        btc_df: pd.DataFrame,
        fear_greed: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> pd.Series:
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

        return scores


class OnChainSignal:
    """Simulated on-chain signals from volume/price patterns"""

    def generate(self, df: pd.DataFrame) -> pd.Series:
        scores = pd.Series(0.0, index=df.index)
        volume = df["volume"]
        close = df["close"]

        # 1. Whale Activity (large volume spikes)
        vol_zscore = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()
        price_change = close.pct_change().abs()
        whale_activity = vol_zscore * (1 + price_change * 10)

        price_below_ma = close < close.rolling(20).mean()
        whale_buy = (whale_activity > 2) & price_below_ma
        whale_sell = (whale_activity > 2) & ~price_below_ma

        scores += np.where(whale_buy, 2, np.where(whale_sell, -1, 0)) * 1.5

        # 2. Exchange Flow (price/volume divergence)
        price_chg = close.pct_change()
        vol_chg = volume.pct_change()
        flow = -price_chg * 10 + vol_chg
        flow_ema = flow.ewm(span=5).mean()

        scores += (
            np.where(
                flow_ema < -0.5,
                2,
                np.where(
                    flow_ema < -0.2,
                    1,
                    np.where(flow_ema > 0.5, -2, np.where(flow_ema > 0.2, -1, 0)),
                ),
            )
            * 2.0
        )

        # 3. Accumulation/Distribution
        price_ma = close.rolling(20).mean()
        vol_ma = volume.rolling(20).mean()
        accumulation = ((close > price_ma) & (volume < vol_ma)).astype(int) - (
            (close < price_ma) & (volume > vol_ma)
        ).astype(int)
        accumulation = accumulation.rolling(10).sum()

        scores += (
            np.where(
                accumulation > 5,
                1.5,
                np.where(
                    accumulation > 2,
                    0.5,
                    np.where(
                        accumulation < -5, -1.5, np.where(accumulation < -2, -0.5, 0)
                    ),
                ),
            )
            * 1.0
        )

        return scores.fillna(0)


class KoreanPremiumSignal:
    """Korean exchange premium indicator (Kimchi Premium)"""

    def generate(
        self,
        binance_df: pd.DataFrame,
        korean_df: pd.DataFrame,
        usd_krw_rate: float = 1350,
    ) -> pd.Series:
        """
        Calculate Kimchi Premium
        Premium > 0: Korean buying pressure (bullish)
        Premium < 0: Korean selling pressure (bearish)
        """
        if korean_df.empty:
            return pd.Series(0.0, index=binance_df.index)

        # Align data
        common_idx = binance_df.index.intersection(korean_df.index)
        if len(common_idx) < 100:
            return pd.Series(0.0, index=binance_df.index)

        binance_price = binance_df.loc[common_idx, "close"]
        korean_price = korean_df.loc[common_idx, "close"] / usd_krw_rate

        # Calculate premium
        premium = (korean_price / binance_price - 1) * 100
        premium.rolling(20).mean()

        scores = pd.Series(0.0, index=binance_df.index)

        # High premium = excessive bullishness (contrarian sell)
        # Low/negative premium = excessive bearishness (contrarian buy)
        premium_score = np.where(
            premium < -2,
            2,  # Negative premium = buy signal
            np.where(
                premium < 0,
                1,
                np.where(
                    premium > 5,
                    -2,  # High premium = sell signal
                    np.where(premium > 2, -1, 0),
                ),
            ),
        )

        scores.loc[common_idx] = premium_score * 1.5
        return scores.fillna(0)


class MLSignal:
    """Machine Learning based signal"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        volume = df["volume"]

        # Returns
        for p in [5, 10, 20, 50]:
            features[f"ret_{p}"] = close.pct_change(p)
            features[f"vol_ret_{p}"] = volume.pct_change(p)

        # Volatility
        for p in [10, 20]:
            features[f"volatility_{p}"] = close.pct_change().rolling(p).std()

        # RSI
        features["rsi_14"] = calc_rsi(close, 14)

        # EMA position
        for p in [20, 50]:
            features[f"ema_pos_{p}"] = close / calc_ema(close, p) - 1

        # Volume profile
        features["vol_zscore"] = (volume - volume.rolling(50).mean()) / volume.rolling(
            50
        ).std()

        return features.replace([np.inf, -np.inf], np.nan).fillna(0)

    def train_and_predict(
        self, df: pd.DataFrame, train_ratio: float = 0.7
    ) -> pd.Series:
        """Train model and generate predictions"""
        features = self._create_features(df)

        # Target: Future 6-bar return direction
        future_ret = df["close"].shift(-6) / df["close"] - 1
        target = np.where(future_ret > 0.02, 1, np.where(future_ret < -0.02, -1, 0))
        target = pd.Series(target, index=df.index)

        # Remove NaN
        valid = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid]
        target = target[valid]

        if len(features) < 500:
            return pd.Series(0.0, index=df.index)

        train_size = int(len(features) * train_ratio)
        X_train = features.iloc[:train_size]
        y_train = target.iloc[:train_size]

        self.feature_names = features.columns.tolist()

        # Scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model = GradientBoostingClassifier(
            n_estimators=50, max_depth=4, random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Predict on all data
        X_scaled = self.scaler.transform(features)
        predictions = self.model.predict(X_scaled)
        proba = self.model.predict_proba(X_scaled)
        confidence = np.max(proba, axis=1)

        result = pd.Series(0.0, index=df.index)
        result.loc[features.index] = (
            predictions * confidence * 3
        )  # Scale to match other signals
        return result


# ============================================================================
# Combined Strategy Backtester
# ============================================================================


class CombinedBacktester:
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold

    def backtest(self, df: pd.DataFrame, scores: pd.Series, symbol: str) -> Dict:
        if len(df) < 200:
            return None

        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        for i in range(50, len(df)):
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

            if not position and abs(score) >= self.threshold:
                direction = 1 if score > 0 else -1
                mult = min(abs(score) / self.threshold, 2.0)
                position = {"entry": price, "dir": direction, "mult": mult}

            equity.append(capital)

        if position:
            pnl_pct = (df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(capital * 0.2 * 2 * position["mult"] * pnl_pct)
            capital += trades[-1]

        if len(trades) < 5:
            return None

        equity_s = pd.Series(equity)
        mdd = (
            (equity_s - equity_s.expanding().max()) / equity_s.expanding().max()
        ).min() * 100
        wins = sum(1 for t in trades if t > 0)
        gp = sum(t for t in trades if t > 0)
        gl = abs(sum(t for t in trades if t < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        return {
            "symbol": symbol,
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
        }


def main():
    loader = DataLoader()

    logger.info("=" * 70)
    logger.info("COMBINED STRATEGY BACKTESTER")
    logger.info("=" * 70)
    logger.info(
        "Testing: Multi-Factor + On-chain, Multi-Factor + Korean Premium, Multi-Factor + ML"
    )
    logger.info("=" * 70)

    # Load common data
    fear_greed = loader.load_fear_greed()
    macro = loader.load_macro()
    btc_df = loader.load_ohlcv("BTCUSDT", "binance_futures", "4h")

    # Get symbols
    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    all_symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )

    # Initialize signal generators
    mf_signal = MultiFactorSignal()
    oc_signal = OnChainSignal()
    kr_signal = KoreanPremiumSignal()
    ml_signal = MLSignal()

    backtester = CombinedBacktester(threshold=4.0)

    strategies = {
        "MultiFactor_Only": [],
        "MF_OnChain": [],
        "MF_KoreanPremium": [],
        "MF_ML": [],
        "MF_OnChain_ML": [],
    }

    logger.info(f"\nTesting {len(all_symbols)} symbols...\n")

    tested = 0
    for symbol in all_symbols:
        df = loader.load_ohlcv(symbol, "binance_futures", "4h")
        if df.empty:
            continue

        df = df[df.index >= "2022-01-01"]
        if len(df) < 500:
            continue

        tested += 1

        # Generate individual signals
        mf_scores = mf_signal.generate(df, btc_df, fear_greed, macro)
        oc_scores = oc_signal.generate(df)

        # Korean premium (try to load Korean data)
        kr_df = loader.load_ohlcv(symbol, "bithumb", "4h")
        kr_scores = (
            kr_signal.generate(df, kr_df)
            if not kr_df.empty
            else pd.Series(0.0, index=df.index)
        )

        # ML signal
        ml_scores = ml_signal.train_and_predict(df)

        # Combined scores
        scores_dict = {
            "MultiFactor_Only": mf_scores,
            "MF_OnChain": mf_scores + oc_scores * 0.7,
            "MF_KoreanPremium": mf_scores + kr_scores * 0.5,
            "MF_ML": mf_scores + ml_scores * 0.8,
            "MF_OnChain_ML": mf_scores + oc_scores * 0.5 + ml_scores * 0.5,
        }

        # Backtest each combination
        for name, scores in scores_dict.items():
            result = backtester.backtest(df, scores, symbol)
            if result and result["trades"] >= 10:
                strategies[name].append(result)

        if tested % 50 == 0:
            logger.info(f"  Progress: {tested} symbols tested")

    # Results summary
    logger.info(f"\nTested: {tested} symbols")
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 70)

    summary_data = []

    for name, results in strategies.items():
        if not results:
            continue

        df_r = pd.DataFrame(results)
        profitable = (df_r["pf"] > 1.0).sum()
        avg_pf = df_r[df_r["pf"] < 999]["pf"].mean()
        avg_ret = df_r["ret"].mean()
        avg_wr = df_r["wr"].mean()
        avg_mdd = df_r["mdd"].mean()

        summary_data.append(
            {
                "strategy": name,
                "symbols": len(results),
                "profitable": profitable,
                "profitable_pct": profitable / len(results) * 100,
                "avg_pf": avg_pf,
                "avg_ret": avg_ret,
                "avg_wr": avg_wr,
                "avg_mdd": avg_mdd,
            }
        )

        logger.info(f"\n{name}:")
        logger.info(f"  Symbols: {len(results)}")
        logger.info(
            f"  Profitable: {profitable}/{len(results)} ({profitable/len(results)*100:.1f}%)"
        )
        logger.info(f"  Avg PF: {avg_pf:.2f}")
        logger.info(f"  Avg Return: {avg_ret:+.1f}%")
        logger.info(f"  Avg Win Rate: {avg_wr:.1f}%")
        logger.info(f"  Avg MDD: {avg_mdd:.1f}%")

    # Comparison table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 70)
    logger.info(
        f"\n{'Strategy':<20} {'Symbols':>8} {'Profitable':>12} {'Avg PF':>10} {'Avg Ret':>10} {'Avg MDD':>10}"
    )
    logger.info("-" * 75)

    for s in sorted(summary_data, key=lambda x: -x["avg_pf"]):
        logger.info(
            f"{s['strategy']:<20} {s['symbols']:>8} {s['profitable_pct']:>11.1f}% {s['avg_pf']:>10.2f} "
            f"{s['avg_ret']:>9.1f}% {s['avg_mdd']:>9.1f}%"
        )

    # Best performers by strategy
    logger.info("\n" + "=" * 70)
    logger.info("TOP 10 BY EACH STRATEGY")
    logger.info("=" * 70)

    for name, results in strategies.items():
        if not results:
            continue

        df_r = pd.DataFrame(results)
        top10 = df_r.nlargest(10, "pf")

        logger.info(f"\n{name}:")
        for _, r in top10.iterrows():
            logger.info(
                f"  {r['symbol']:<14} PF={r['pf']:5.2f} Ret={r['ret']:+7.1f}% WR={r['wr']:.0f}%"
            )

    # Save all results
    all_results = []
    for name, results in strategies.items():
        for r in results:
            r["strategy"] = name
            all_results.append(r)

    if all_results:
        df_all = pd.DataFrame(all_results)
        output_path = DATA_ROOT / "combined_strategy_results.csv"
        df_all.to_csv(output_path, index=False)
        logger.info(f"\nResults saved to: {output_path}")

    return strategies


if __name__ == "__main__":
    results = main()
