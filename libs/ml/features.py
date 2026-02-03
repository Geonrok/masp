"""
Feature Engineering

Technical and statistical feature extraction for ML models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Price features
    include_returns: bool = True
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])

    # Volatility features
    include_volatility: bool = True
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 60])

    # Momentum features
    include_momentum: bool = True
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # Moving average features
    include_ma: bool = True
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])

    # Volume features
    include_volume: bool = True

    # Technical indicators
    include_rsi: bool = True
    include_macd: bool = True
    include_bollinger: bool = True
    include_atr: bool = True

    # Statistical features
    include_skewness: bool = True
    include_kurtosis: bool = True

    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5])

    # Normalization
    normalize: bool = True
    normalization_method: str = "zscore"  # zscore, minmax, robust


class FeatureEngineering:
    """
    Feature engineering pipeline for trading ML models.

    Extracts technical and statistical features from OHLCV data.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineering.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self._normalization_params: Dict[str, Dict[str, float]] = {}

        logger.info("[FeatureEngineering] Initialized with config")

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit normalizer and transform features.

        Args:
            df: OHLCV DataFrame
            target_col: Target column name (excluded from normalization)

        Returns:
            DataFrame with features
        """
        features = self.transform(df)

        if self.config.normalize:
            self._fit_normalization(features, target_col)
            features = self._apply_normalization(features, target_col)

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=df.index)

        # Ensure lowercase column names
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Price features
        if self.config.include_returns:
            features = self._add_return_features(df, features)

        # Volatility features
        if self.config.include_volatility:
            features = self._add_volatility_features(df, features)

        # Momentum features
        if self.config.include_momentum:
            features = self._add_momentum_features(df, features)

        # Moving average features
        if self.config.include_ma:
            features = self._add_ma_features(df, features)

        # Volume features
        if self.config.include_volume and "volume" in df.columns:
            features = self._add_volume_features(df, features)

        # Technical indicators
        if self.config.include_rsi:
            features = self._add_rsi(df, features)

        if self.config.include_macd:
            features = self._add_macd(df, features)

        if self.config.include_bollinger:
            features = self._add_bollinger(df, features)

        if self.config.include_atr and all(
            c in df.columns for c in ["high", "low", "close"]
        ):
            features = self._add_atr(df, features)

        # Statistical features
        if self.config.include_skewness:
            features = self._add_skewness(df, features)

        if self.config.include_kurtosis:
            features = self._add_kurtosis(df, features)

        # Lag features
        if self.config.lag_periods:
            features = self._add_lag_features(features)

        # Apply normalization if already fitted
        if self.config.normalize and self._normalization_params:
            features = self._apply_normalization(features)

        self.feature_names = list(features.columns)

        logger.debug(
            f"[FeatureEngineering] Generated {len(self.feature_names)} features"
        )

        return features

    def _add_return_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add return features."""
        close = df["close"]

        for period in self.config.return_periods:
            features[f"return_{period}d"] = close.pct_change(period)
            features[f"log_return_{period}d"] = np.log(close / close.shift(period))

        return features

    def _add_volatility_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add volatility features."""
        returns = df["close"].pct_change()

        for period in self.config.volatility_periods:
            features[f"volatility_{period}d"] = returns.rolling(period).std()
            features[f"volatility_ratio_{period}d"] = (
                returns.rolling(period).std() / returns.rolling(period * 2).std()
            )

        # Realized volatility
        if all(c in df.columns for c in ["high", "low"]):
            # Parkinson volatility
            features["parkinson_vol"] = (
                np.sqrt((1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2))
                .rolling(20)
                .mean()
            )

        return features

    def _add_momentum_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add momentum features."""
        close = df["close"]

        for period in self.config.momentum_periods:
            # Simple momentum
            features[f"momentum_{period}d"] = close / close.shift(period) - 1

            # Rate of change
            features[f"roc_{period}d"] = (
                (close - close.shift(period)) / close.shift(period) * 100
            )

        # Relative strength
        for short, long in [(5, 20), (10, 60)]:
            features[f"rs_{short}_{long}"] = (
                close.rolling(short).mean() / close.rolling(long).mean()
            )

        return features

    def _add_ma_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add moving average features."""
        close = df["close"]

        for period in self.config.ma_periods:
            ma = close.rolling(period).mean()
            features[f"ma_{period}d"] = ma
            features[f"close_to_ma_{period}d"] = close / ma - 1

            # EMA
            ema = close.ewm(span=period).mean()
            features[f"ema_{period}d"] = ema
            features[f"close_to_ema_{period}d"] = close / ema - 1

        # MA crossovers
        for short, long in [(5, 20), (10, 50), (50, 200)]:
            if short in self.config.ma_periods and long in self.config.ma_periods:
                ma_short = close.rolling(short).mean()
                ma_long = close.rolling(long).mean()
                features[f"ma_cross_{short}_{long}"] = (ma_short - ma_long) / ma_long

        return features

    def _add_volume_features(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add volume features."""
        volume = df["volume"]
        close = df["close"]

        # Volume change
        features["volume_change"] = volume.pct_change()

        # Relative volume
        for period in [5, 10, 20]:
            features[f"volume_ratio_{period}d"] = volume / volume.rolling(period).mean()

        # OBV (On-Balance Volume)
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv"] = obv
        features["obv_change"] = obv.pct_change(5)

        # Volume-weighted price
        features["vwap_ratio"] = close / (
            (close * volume).rolling(20).sum() / volume.rolling(20).sum()
        )

        return features

    def _add_rsi(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        periods: List[int] = [7, 14, 21],
    ) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)."""
        close = df["close"]
        delta = close.diff()

        for period in periods:
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            features[f"rsi_{period}d"] = 100 - (100 / (1 + rs))

        return features

    def _add_macd(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        close = df["close"]

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()

        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = macd - signal
        features["macd_hist_change"] = (macd - signal).diff()

        return features

    def _add_bollinger(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        close = df["close"]

        ma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = ma + 2 * std
        lower = ma - 2 * std

        features["bb_position"] = (close - lower) / (upper - lower)
        features["bb_width"] = (upper - lower) / ma

        return features

    def _add_atr(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """Add ATR (Average True Range)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        features["atr"] = atr
        features["atr_ratio"] = atr / close

        return features

    def _add_skewness(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:
        """Add return skewness."""
        returns = df["close"].pct_change()
        features["skewness"] = returns.rolling(period).skew()
        return features

    def _add_kurtosis(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:
        """Add return kurtosis."""
        returns = df["close"].pct_change()
        features["kurtosis"] = returns.rolling(period).kurt()
        return features

    def _add_lag_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for key indicators."""
        lag_cols = [
            c
            for c in features.columns
            if any(
                x in c for x in ["return_1d", "rsi_14d", "macd_hist", "volume_ratio"]
            )
        ]

        for col in lag_cols[:5]:  # Limit to 5 columns to avoid explosion
            for lag in self.config.lag_periods:
                features[f"{col}_lag{lag}"] = features[col].shift(lag)

        return features

    def _fit_normalization(
        self,
        features: pd.DataFrame,
        exclude_col: Optional[str] = None,
    ) -> None:
        """Fit normalization parameters."""
        self._normalization_params = {}

        for col in features.columns:
            if exclude_col and col == exclude_col:
                continue

            data = features[col].dropna()

            if self.config.normalization_method == "zscore":
                self._normalization_params[col] = {
                    "mean": float(data.mean()),
                    "std": float(data.std()) or 1.0,
                }
            elif self.config.normalization_method == "minmax":
                self._normalization_params[col] = {
                    "min": float(data.min()),
                    "max": float(data.max()) or 1.0,
                }
            elif self.config.normalization_method == "robust":
                self._normalization_params[col] = {
                    "median": float(data.median()),
                    "iqr": float(data.quantile(0.75) - data.quantile(0.25)) or 1.0,
                }

    def _apply_normalization(
        self,
        features: pd.DataFrame,
        exclude_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply normalization to features."""
        result = features.copy()

        for col, params in self._normalization_params.items():
            if col not in result.columns:
                continue
            if exclude_col and col == exclude_col:
                continue

            if self.config.normalization_method == "zscore":
                result[col] = (result[col] - params["mean"]) / params["std"]
            elif self.config.normalization_method == "minmax":
                result[col] = (result[col] - params["min"]) / (
                    params["max"] - params["min"]
                )
            elif self.config.normalization_method == "robust":
                result[col] = (result[col] - params["median"]) / params["iqr"]

        return result

    def get_feature_importance(
        self,
        model: Any,
        top_n: int = 20,
    ) -> Dict[str, float]:
        """
        Get feature importance from fitted model.

        Args:
            model: Fitted model with feature_importances_ attribute
            top_n: Number of top features to return

        Returns:
            Dictionary of feature name to importance
        """
        if not hasattr(model, "feature_importances_"):
            return {}

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        return {
            self.feature_names[i]: float(importances[i])
            for i in indices
            if i < len(self.feature_names)
        }


def create_technical_features(
    df: pd.DataFrame,
    include_all: bool = True,
) -> pd.DataFrame:
    """
    Quick technical feature extraction.

    Args:
        df: OHLCV DataFrame
        include_all: Include all feature types

    Returns:
        DataFrame with features
    """
    config = FeatureConfig(
        normalize=False,
        include_returns=include_all,
        include_volatility=include_all,
        include_momentum=include_all,
        include_ma=include_all,
        include_volume=include_all,
        include_rsi=include_all,
        include_macd=include_all,
        include_bollinger=include_all,
        include_atr=include_all,
        include_skewness=include_all,
        include_kurtosis=include_all,
    )
    fe = FeatureEngineering(config)
    return fe.transform(df)


def create_statistical_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Create statistical features for returns.

    Args:
        df: OHLCV DataFrame
        lookback: Rolling window for statistics

    Returns:
        DataFrame with statistical features
    """
    features = pd.DataFrame(index=df.index)
    returns = df["close"].pct_change()

    features["mean"] = returns.rolling(lookback).mean()
    features["std"] = returns.rolling(lookback).std()
    features["skew"] = returns.rolling(lookback).skew()
    features["kurt"] = returns.rolling(lookback).kurt()
    features["min"] = returns.rolling(lookback).min()
    features["max"] = returns.rolling(lookback).max()
    features["median"] = returns.rolling(lookback).median()
    features["q25"] = returns.rolling(lookback).quantile(0.25)
    features["q75"] = returns.rolling(lookback).quantile(0.75)

    return features
