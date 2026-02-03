#!/usr/bin/env python3
"""
AI/ML Strategy Predictor
- Random Forest for regime classification
- XGBoost for return prediction
- Symbol-specific optimal strategy selection

Features:
- Technical indicators
- Volume patterns
- Volatility regime
- Market correlation
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_ROOT = Path("E:/data/crypto_ohlcv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create ML features from OHLCV data"""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features"""
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. Price-based features
        for period in [5, 10, 20, 50, 100]:
            features[f"return_{period}"] = close.pct_change(period)
            features[f"sma_{period}"] = close.rolling(period).mean() / close - 1
            features[f"ema_{period}"] = close.ewm(span=period).mean() / close - 1

        # 2. Volatility features
        for period in [10, 20, 50]:
            features[f"volatility_{period}"] = close.pct_change().rolling(period).std()
            features[f"atr_{period}"] = self._calc_atr(high, low, close, period) / close

        # 3. Volume features
        for period in [5, 10, 20]:
            features[f"volume_sma_{period}"] = (
                volume / volume.rolling(period).mean() - 1
            )
            features[f"volume_std_{period}"] = (
                volume.rolling(period).std() / volume.rolling(50).mean()
            )

        # 4. Momentum features
        features["rsi_14"] = self._calc_rsi(close, 14)
        features["rsi_7"] = self._calc_rsi(close, 7)

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        features["macd"] = (ema12 - ema26) / close
        features["macd_signal"] = features["macd"].ewm(span=9).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        # 5. Bollinger Bands
        for period in [20]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f"bb_position_{period}"] = (close - sma) / (2 * std)
            features[f"bb_width_{period}"] = (4 * std) / sma

        # 6. Price position
        for period in [20, 50, 100]:
            features[f"high_position_{period}"] = (
                close - low.rolling(period).min()
            ) / (high.rolling(period).max() - low.rolling(period).min() + 1e-10)

        # 7. Trend features
        features["trend_strength"] = abs(features["return_20"]) / (
            features["volatility_20"] + 1e-10
        )

        # 8. Time features
        features["hour"] = df.index.hour
        features["dayofweek"] = df.index.dayofweek

        return features.replace([np.inf, -np.inf], np.nan).fillna(0)

    def _calc_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _calc_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int
    ) -> pd.Series:
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        return tr.rolling(period).mean()


class MLStrategy:
    """Machine Learning based trading strategy"""

    def __init__(self, prediction_horizon: int = 6):  # 6 bars = 1 day for 4h
        self.prediction_horizon = prediction_horizon
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        features = self.feature_engineer.create_features(df)

        # Target: Future return classification
        future_return = df["close"].shift(-self.prediction_horizon) / df["close"] - 1

        # Classify: -1 (down >2%), 0 (sideways), 1 (up >2%)
        target = pd.Series(0, index=df.index)
        target[future_return > 0.02] = 1  # Bullish
        target[future_return < -0.02] = -1  # Bearish

        return features, target

    def train(self, df: pd.DataFrame, train_ratio: float = 0.7) -> Dict:
        """Train the model"""
        features, target = self.prepare_data(df)

        # Remove NaN rows
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]

        # Time series split
        train_size = int(len(features) * train_ratio)
        X_train = features.iloc[:train_size]
        y_train = target.iloc[:train_size]
        X_test = features.iloc[train_size:]
        y_test = target.iloc[train_size:]

        self.feature_names = features.columns.tolist()

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Feature importance
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_importance": importance.head(10),
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model not trained")

        features = self.feature_engineer.create_features(df)
        features = features[self.feature_names]

        # Handle NaN
        features = features.fillna(0)

        scaled = self.scaler.transform(features)
        predictions = self.model.predict(scaled)

        # Get prediction probabilities for confidence
        proba = self.model.predict_proba(scaled)
        confidence = np.max(proba, axis=1)

        return pd.Series(predictions * confidence, index=df.index)

    def backtest(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Run backtest with trained model"""
        if len(df) < 500:
            return None

        # Train on first 70%, test on last 30%
        train_metrics = self.train(df, train_ratio=0.7)

        predictions = self.predict(df)

        # Trading simulation
        capital = 10000
        init = capital
        position = None
        trades = []
        equity = [capital]

        test_start = int(len(df) * 0.7)

        for i in range(test_start, len(df)):
            price = df["close"].iloc[i]
            signal = predictions.iloc[i]

            # Exit
            if position:
                bars_held = i - position["entry_idx"]
                exit_signal = (
                    (position["dir"] == 1 and signal < -0.3)
                    or (position["dir"] == -1 and signal > 0.3)
                    or bars_held >= self.prediction_horizon * 2
                )

                if exit_signal:
                    pnl_pct = (price / position["entry"] - 1) * position["dir"] - 0.002
                    pnl = capital * 0.2 * pnl_pct
                    trades.append(
                        {
                            "pnl": pnl,
                            "direction": position["dir"],
                            "bars_held": bars_held,
                        }
                    )
                    capital += pnl
                    position = None

            # Entry
            if not position and abs(signal) >= 0.5:
                direction = 1 if signal > 0 else -1
                position = {"entry": price, "dir": direction, "entry_idx": i}

            equity.append(capital)

        if position:
            pnl_pct = (df["close"].iloc[-1] / position["entry"] - 1) * position[
                "dir"
            ] - 0.002
            trades.append(
                {
                    "pnl": capital * 0.2 * pnl_pct,
                    "direction": position["dir"],
                    "bars_held": 0,
                }
            )
            capital += trades[-1]["pnl"]

        if len(trades) < 5:
            return None

        pnls = [t["pnl"] for t in trades]
        equity_s = pd.Series(equity)
        mdd = (
            (equity_s - equity_s.expanding().max()) / equity_s.expanding().max()
        ).min() * 100

        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = gp / gl if gl > 0 else (999 if gp > 0 else 0)

        return {
            "symbol": symbol,
            "pf": min(pf, 999),
            "ret": (capital / init - 1) * 100,
            "wr": wins / len(trades) * 100,
            "mdd": mdd,
            "trades": len(trades),
            "train_acc": train_metrics["train_accuracy"],
            "test_acc": train_metrics["test_accuracy"],
            "top_features": train_metrics["feature_importance"]
            .head(3)["feature"]
            .tolist(),
        }


class StrategySelector:
    """Select optimal strategy per symbol using ML"""

    def __init__(self):
        self.strategies = ["trend_follow", "mean_reversion", "momentum", "volatility"]

    def classify_symbol(self, df: pd.DataFrame) -> Dict:
        """Classify symbol characteristics"""
        close = df["close"]
        returns = close.pct_change()

        # Characteristics
        volatility = returns.std() * np.sqrt(6 * 365)  # Annualized
        trend_strength = abs(close.iloc[-1] / close.iloc[0] - 1)
        mean_reversion = self._calc_mean_reversion_score(returns)
        momentum_persistence = returns.autocorr(lag=1)

        # Determine best strategy
        scores = {
            "trend_follow": trend_strength * 10 + momentum_persistence * 5,
            "mean_reversion": mean_reversion * 10 - trend_strength * 5,
            "momentum": momentum_persistence * 10 if momentum_persistence > 0 else 0,
            "volatility": volatility * 2 - abs(momentum_persistence) * 5,
        }

        best_strategy = max(scores, key=scores.get)

        return {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "mean_reversion": mean_reversion,
            "momentum_persistence": momentum_persistence,
            "recommended_strategy": best_strategy,
            "strategy_scores": scores,
        }

    def _calc_mean_reversion_score(self, returns: pd.Series) -> float:
        """Calculate mean reversion tendency"""
        # Negative autocorrelation = mean reverting
        autocorr = returns.autocorr(lag=1)
        return -autocorr if not pd.isna(autocorr) else 0


def main():
    logger.info("=" * 70)
    logger.info("AI/ML STRATEGY PREDICTOR")
    logger.info("=" * 70)
    logger.info("Using Gradient Boosting for direction prediction")
    logger.info("Features: Technical, Volume, Volatility, Momentum")
    logger.info("=" * 70)

    # Load data
    ohlcv_dir = DATA_ROOT / "binance_futures_4h"
    symbols = sorted(
        [f.stem for f in ohlcv_dir.glob("*.csv") if f.stem.endswith("USDT")]
    )[
        :50
    ]  # Top 50

    logger.info(f"Testing {len(symbols)} symbols...\n")

    results = []
    selector = StrategySelector()

    for symbol in symbols:
        filepath = ohlcv_dir / f"{symbol}.csv"
        df = pd.read_csv(filepath)
        for col in ["datetime", "timestamp", "date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break
        df = df[df.index >= "2022-01-01"]

        if len(df) < 1000:
            continue

        try:
            # ML Strategy
            strategy = MLStrategy(prediction_horizon=6)
            result = strategy.backtest(df, symbol)

            if result:
                # Add strategy classification
                classification = selector.classify_symbol(df)
                result["recommended_strategy"] = classification["recommended_strategy"]
                result["volatility"] = classification["volatility"]
                results.append(result)

                logger.info(
                    f"{symbol:<14} PF={result['pf']:.2f} Acc={result['test_acc']:.1%} "
                    f"Rec={result['recommended_strategy']}"
                )

        except Exception as e:
            logger.warning(f"{symbol}: {e}")

    if not results:
        logger.info("No valid results")
        return

    # Summary
    df_results = pd.DataFrame(results)
    profitable = df_results[df_results["pf"] > 1.0]

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(
        f"Profitable: {len(profitable)}/{len(results)} ({len(profitable)/len(results)*100:.1f}%)"
    )
    logger.info(f"Avg PF: {df_results[df_results['pf'] < 999]['pf'].mean():.2f}")
    logger.info(f"Avg Test Accuracy: {df_results['test_acc'].mean():.1%}")
    logger.info(f"Avg Return: {df_results['ret'].mean():+.1f}%")

    # Best by strategy type
    logger.info("\n" + "=" * 70)
    logger.info("PERFORMANCE BY RECOMMENDED STRATEGY")
    logger.info("=" * 70)

    for strategy in df_results["recommended_strategy"].unique():
        subset = df_results[df_results["recommended_strategy"] == strategy]
        if len(subset) > 0:
            avg_pf = subset[subset["pf"] < 999]["pf"].mean()
            profitable_rate = (subset["pf"] > 1.0).mean() * 100
            logger.info(
                f"{strategy:<15}: {len(subset)} symbols, Avg PF={avg_pf:.2f}, Profitable={profitable_rate:.1f}%"
            )

    # Top performers
    logger.info("\n" + "=" * 70)
    logger.info("TOP 15 ML STRATEGY PERFORMERS")
    logger.info("=" * 70)

    top = df_results.nlargest(15, "pf")
    for _, r in top.iterrows():
        logger.info(
            f"  {r['symbol']:<14} PF={r['pf']:5.2f} Ret={r['ret']:+7.1f}% Acc={r['test_acc']:.1%} "
            f"Strategy={r['recommended_strategy']}"
        )

    # Save
    output_path = DATA_ROOT / "ml_strategy_results.csv"
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    results = main()
