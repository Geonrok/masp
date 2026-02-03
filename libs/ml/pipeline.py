"""
ML Pipeline

End-to-end ML pipeline for trading strategy integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from libs.ml.features import FeatureConfig, FeatureEngineering
from libs.ml.models import ModelInterface, SklearnModel

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""

    # Feature configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Target configuration
    target_type: str = "direction"  # direction, return, volatility
    target_horizon: int = 1  # Prediction horizon in periods
    target_threshold: float = 0.0  # Threshold for classification

    # Training configuration
    train_test_split: float = 0.8
    validation_split: float = 0.1
    walk_forward: bool = False
    walk_forward_windows: int = 5

    # Model configuration
    model_type: str = "random_forest"  # random_forest, gradient_boost, linear
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Prediction settings
    prediction_cache_size: int = 1000
    min_samples: int = 100


@dataclass
class PredictionResult:
    """Result of model prediction."""

    timestamp: datetime
    symbol: str
    prediction: float
    probability: Optional[float] = None
    confidence: Optional[float] = None
    features: Optional[Dict[str, float]] = None
    model_id: Optional[str] = None

    def to_signal(self) -> int:
        """Convert prediction to trading signal."""
        if self.prediction > 0:
            return 1  # Long
        elif self.prediction < 0:
            return -1  # Short
        return 0  # Neutral


class MLPipeline:
    """
    End-to-end ML pipeline for trading.

    Handles feature engineering, model training,
    and real-time predictions.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model: Optional[ModelInterface] = None,
    ):
        """
        Initialize ML pipeline.

        Args:
            config: Pipeline configuration
            model: Pre-trained model (optional)
        """
        self.config = config or PipelineConfig()
        self.feature_engineer = FeatureEngineering(self.config.feature_config)
        self.model = model
        self._prediction_cache: Dict[str, PredictionResult] = {}
        self._is_fitted = False

        logger.info("[MLPipeline] Initialized")

    def fit(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
    ) -> "MLPipeline":
        """
        Fit the ML pipeline.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            Self
        """
        logger.info(f"[MLPipeline] Fitting on {len(df)} samples for {symbol}")

        # Generate features
        features = self.feature_engineer.fit_transform(df)

        # Generate target
        target = self._generate_target(df)

        # Align features and target
        features, target = self._align_data(features, target)

        # Split data
        X_train, X_test, y_train, y_test = self._split_data(features, target)

        # Create model if not provided
        if self.model is None:
            self.model = self._create_model()

        # Fit model
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        # Evaluate
        train_score = self._evaluate(X_train, y_train)
        test_score = self._evaluate(X_test, y_test)

        # Update metadata
        if self.model.metadata:
            self.model.metadata.metrics = {
                "train_accuracy": train_score.get("accuracy", 0),
                "test_accuracy": test_score.get("accuracy", 0),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
            self.model.metadata.training_period = f"{df.index[0]} to {df.index[-1]}"

        logger.info(
            f"[MLPipeline] Fit complete - "
            f"Train acc: {train_score.get('accuracy', 0):.4f}, "
            f"Test acc: {test_score.get('accuracy', 0):.4f}"
        )

        return self

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
        use_cache: bool = True,
    ) -> PredictionResult:
        """
        Make prediction on latest data.

        Args:
            df: OHLCV DataFrame (should include recent history)
            symbol: Symbol name
            use_cache: Whether to use prediction cache

        Returns:
            PredictionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")

        # Check cache
        cache_key = f"{symbol}_{len(df)}_{df.index[-1]}"
        if use_cache and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        # Generate features
        features = self.feature_engineer.transform(df)

        # Get latest row (dropna to avoid NaN features)
        latest_features = features.iloc[[-1]].dropna(axis=1)

        if len(latest_features.columns) < 5:
            logger.warning("[MLPipeline] Insufficient features for prediction")
            return PredictionResult(
                timestamp=datetime.now(),
                symbol=symbol,
                prediction=0.0,
            )

        # Make prediction
        prediction = self.model.predict(latest_features)[0]

        # Get probability if available
        probability = None
        try:
            proba = self.model.predict_proba(latest_features)
            probability = float(np.max(proba[0]))
        except (NotImplementedError, AttributeError):
            pass

        # Calculate confidence
        confidence = self._calculate_confidence(latest_features, prediction)

        result = PredictionResult(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=float(prediction),
            probability=probability,
            confidence=confidence,
            features={
                col: float(latest_features[col].iloc[0])
                for col in latest_features.columns[:10]
            },
            model_id=self.model.model_id if self.model else None,
        )

        # Update cache
        if use_cache:
            self._prediction_cache[cache_key] = result
            # Limit cache size
            if len(self._prediction_cache) > self.config.prediction_cache_size:
                oldest = list(self._prediction_cache.keys())[0]
                del self._prediction_cache[oldest]

        return result

    def predict_batch(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
    ) -> List[PredictionResult]:
        """
        Make predictions for all rows in DataFrame.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            List of PredictionResult
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted")

        # Generate features
        features = self.feature_engineer.transform(df)

        # Drop rows with NaN
        valid_idx = features.dropna().index
        features_clean = features.loc[valid_idx]

        if len(features_clean) == 0:
            return []

        # Make predictions
        predictions = self.model.predict(features_clean)

        # Get probabilities if available
        probabilities = None
        try:
            probas = self.model.predict_proba(features_clean)
            probabilities = np.max(probas, axis=1)
        except (NotImplementedError, AttributeError):
            pass

        # Create results
        results = []
        for i, idx in enumerate(valid_idx):
            result = PredictionResult(
                timestamp=df.index[df.index.get_loc(idx)],
                symbol=symbol,
                prediction=float(predictions[i]),
                probability=(
                    float(probabilities[i]) if probabilities is not None else None
                ),
            )
            results.append(result)

        return results

    def walk_forward_fit(
        self,
        df: pd.DataFrame,
        symbol: str = "unknown",
    ) -> List[Dict[str, Any]]:
        """
        Walk-forward fitting with multiple windows.

        Args:
            df: OHLCV DataFrame
            symbol: Symbol name

        Returns:
            List of evaluation results per window
        """
        logger.info(
            f"[MLPipeline] Walk-forward fit with {self.config.walk_forward_windows} windows"
        )

        n = len(df)
        window_size = n // (self.config.walk_forward_windows + 1)

        results = []

        for i in range(self.config.walk_forward_windows):
            # Training data: all data up to this point
            train_end = window_size * (i + 1)
            test_end = min(train_end + window_size, n)

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            if len(train_df) < self.config.min_samples:
                continue

            # Fit on training data
            self.fit(train_df, symbol)

            # Evaluate on test data
            test_features = self.feature_engineer.transform(test_df)
            test_target = self._generate_target(test_df)
            test_features, test_target = self._align_data(test_features, test_target)

            if len(test_features) > 0:
                metrics = self._evaluate(test_features, test_target)
                metrics["window"] = i + 1
                metrics["train_size"] = len(train_df)
                metrics["test_size"] = len(test_df)
                results.append(metrics)

                logger.info(
                    f"[MLPipeline] Window {i+1}: "
                    f"accuracy={metrics.get('accuracy', 0):.4f}"
                )

        return results

    def _generate_target(self, df: pd.DataFrame) -> pd.Series:
        """Generate target variable."""
        close = df["close"]

        if self.config.target_type == "direction":
            # Binary classification: up/down
            future_return = close.shift(-self.config.target_horizon) / close - 1
            target = (future_return > self.config.target_threshold).astype(int)

        elif self.config.target_type == "return":
            # Regression: future return
            target = close.shift(-self.config.target_horizon) / close - 1

        elif self.config.target_type == "volatility":
            # Regression: future volatility
            returns = close.pct_change()
            target = (
                returns.rolling(self.config.target_horizon)
                .std()
                .shift(-self.config.target_horizon)
            )

        else:
            raise ValueError(f"Unknown target type: {self.config.target_type}")

        target.name = "target"
        return target

    def _align_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, dropping NaN."""
        # Combine and drop NaN
        combined = pd.concat([features, target], axis=1).dropna()

        if len(combined) == 0:
            return pd.DataFrame(), pd.Series()

        return combined[features.columns], combined["target"]

    def _split_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        n = len(features)
        train_size = int(n * self.config.train_test_split)

        X_train = features.iloc[:train_size]
        X_test = features.iloc[train_size:]
        y_train = target.iloc[:train_size]
        y_test = target.iloc[train_size:]

        return X_train, X_test, y_train, y_test

    def _create_model(self) -> ModelInterface:
        """Create model based on configuration."""
        model_type = self.config.model_type
        params = self.config.model_params

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            estimator = RandomForestClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 10),
                min_samples_split=params.get("min_samples_split", 5),
                random_state=42,
            )

        elif model_type == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier

            estimator = GradientBoostingClassifier(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.1),
                random_state=42,
            )

        elif model_type == "linear":
            from sklearn.linear_model import LogisticRegression

            estimator = LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 1000),
                random_state=42,
            )

        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier

                estimator = XGBClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 5),
                    learning_rate=params.get("learning_rate", 0.1),
                    random_state=42,
                )
            except ImportError:
                logger.warning("XGBoost not available, using RandomForest")
                from sklearn.ensemble import RandomForestClassifier

                estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier

                estimator = LGBMClassifier(
                    n_estimators=params.get("n_estimators", 100),
                    max_depth=params.get("max_depth", 5),
                    learning_rate=params.get("learning_rate", 0.1),
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                logger.warning("LightGBM not available, using RandomForest")
                from sklearn.ensemble import RandomForestClassifier

                estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return SklearnModel(estimator)

    def _evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on data."""
        predictions = self.model.predict(X)

        if self.config.target_type == "direction":
            # Classification metrics
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            return {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, zero_division=0),
                "recall": recall_score(y, predictions, zero_division=0),
                "f1": f1_score(y, predictions, zero_division=0),
            }

        else:
            # Regression metrics
            from sklearn.metrics import (
                mean_absolute_error,
                mean_squared_error,
                r2_score,
            )

            return {
                "mse": mean_squared_error(y, predictions),
                "mae": mean_absolute_error(y, predictions),
                "r2": r2_score(y, predictions),
            }

    def _calculate_confidence(
        self,
        features: pd.DataFrame,
        prediction: float,
    ) -> float:
        """Calculate prediction confidence."""
        # Simple confidence based on feature completeness
        non_nan_ratio = features.notna().sum().sum() / features.size

        # Could add more sophisticated confidence measures
        return float(non_nan_ratio)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from model."""
        return self.feature_engineer.get_feature_importance(
            self.model._model if hasattr(self.model, "_model") else self.model,
            top_n,
        )

    def save(self, path: str) -> None:
        """Save pipeline to disk."""
        if self.model:
            self.model.save(path)

    def load(self, path: str, model_id: str) -> "MLPipeline":
        """Load pipeline from disk."""
        if self.model is None:
            self.model = self._create_model()
        self.model.model_id = model_id
        self.model.load(path)
        self._is_fitted = True
        return self
