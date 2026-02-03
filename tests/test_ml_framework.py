"""
Tests for ML integration framework.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from libs.ml.features import (
    FeatureConfig,
    FeatureEngineering,
    create_statistical_features,
    create_technical_features,
)
from libs.ml.models import (
    EnsembleModel,
    ModelMetadata,
    ModelRegistry,
    SklearnModel,
)
from libs.ml.pipeline import (
    MLPipeline,
    PipelineConfig,
    PredictionResult,
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 500

    # Generate realistic price series
    returns = np.random.normal(0.0005, 0.02, n)
    close = 100 * np.exp(np.cumsum(returns))

    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")

    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.uniform(-0.01, 0.01, n)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.015, n))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.015, n))),
            "close": close,
            "volume": np.random.randint(1000000, 5000000, n),
        },
        index=dates,
    )

    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


class TestFeatureEngineering:
    """Tests for FeatureEngineering."""

    def test_default_features(self, sample_ohlcv):
        """Test default feature extraction."""
        fe = FeatureEngineering()
        features = fe.transform(sample_ohlcv)

        assert len(features) == len(sample_ohlcv)
        assert len(features.columns) > 20  # Should have many features

    def test_return_features(self, sample_ohlcv):
        """Test return features."""
        config = FeatureConfig(
            include_returns=True,
            include_volatility=False,
            include_momentum=False,
            include_ma=False,
            include_volume=False,
            include_rsi=False,
            include_macd=False,
            include_bollinger=False,
            include_atr=False,
            include_skewness=False,
            include_kurtosis=False,
            lag_periods=[],
            normalize=False,
        )
        fe = FeatureEngineering(config)
        features = fe.transform(sample_ohlcv)

        assert "return_1d" in features.columns
        assert "return_5d" in features.columns
        assert "log_return_1d" in features.columns

    def test_volatility_features(self, sample_ohlcv):
        """Test volatility features."""
        config = FeatureConfig(
            include_returns=False,
            include_volatility=True,
            include_momentum=False,
            include_ma=False,
            include_volume=False,
            include_rsi=False,
            include_macd=False,
            include_bollinger=False,
            include_atr=False,
            include_skewness=False,
            include_kurtosis=False,
            lag_periods=[],
            normalize=False,
        )
        fe = FeatureEngineering(config)
        features = fe.transform(sample_ohlcv)

        assert "volatility_10d" in features.columns
        assert "volatility_20d" in features.columns

    def test_technical_indicators(self, sample_ohlcv):
        """Test technical indicator features."""
        config = FeatureConfig(
            include_returns=False,
            include_volatility=False,
            include_momentum=False,
            include_ma=False,
            include_volume=False,
            include_rsi=True,
            include_macd=True,
            include_bollinger=True,
            include_atr=True,
            include_skewness=False,
            include_kurtosis=False,
            lag_periods=[],
            normalize=False,
        )
        fe = FeatureEngineering(config)
        features = fe.transform(sample_ohlcv)

        assert "rsi_14d" in features.columns
        assert "macd" in features.columns
        assert "bb_position" in features.columns
        assert "atr" in features.columns

    def test_normalization(self, sample_ohlcv):
        """Test feature normalization."""
        config = FeatureConfig(normalize=True, normalization_method="zscore")
        fe = FeatureEngineering(config)
        features = fe.fit_transform(sample_ohlcv)

        # Check that features are roughly standardized
        for col in features.columns[:5]:
            data = features[col].dropna()
            if len(data) > 0:
                assert abs(data.mean()) < 1  # Should be close to 0
                assert 0.5 < data.std() < 2  # Should be close to 1

    def test_convenience_functions(self, sample_ohlcv):
        """Test convenience feature functions."""
        tech_features = create_technical_features(sample_ohlcv)
        assert len(tech_features.columns) > 20

        stat_features = create_statistical_features(sample_ohlcv)
        assert "mean" in stat_features.columns
        assert "skew" in stat_features.columns


class TestSklearnModel:
    """Tests for SklearnModel."""

    def test_fit_predict(self):
        """Test basic fit and predict."""
        from sklearn.ensemble import RandomForestClassifier

        model = SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42))

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model.fit(X, y)

        assert model.is_fitted
        assert model.metadata is not None

        predictions = model.predict(X)
        assert len(predictions) == 100

    def test_predict_proba(self):
        """Test probability prediction."""
        from sklearn.ensemble import RandomForestClassifier

        model = SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42))

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_feature_importances(self):
        """Test feature importance extraction."""
        from sklearn.ensemble import RandomForestClassifier

        model = SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42))

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model.fit(X, y)

        assert len(model.feature_importances_) == 5


class TestEnsembleModel:
    """Tests for EnsembleModel."""

    def test_soft_voting(self):
        """Test soft voting ensemble."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = [
            SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42)),
            SklearnModel(LogisticRegression(random_state=42)),
        ]

        ensemble = EnsembleModel(models, voting="soft")

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        ensemble.fit(X, y)

        assert ensemble.is_fitted
        predictions = ensemble.predict(X)
        assert len(predictions) == 100

    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        from sklearn.ensemble import RandomForestClassifier

        models = [
            SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42)),
            SklearnModel(RandomForestClassifier(n_estimators=20, random_state=43)),
        ]

        ensemble = EnsembleModel(models, weights=[0.7, 0.3])

        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        ensemble.fit(X, y)
        predictions = ensemble.predict(X)

        assert len(predictions) == 100


class TestModelMetadata:
    """Tests for ModelMetadata."""

    def test_to_dict(self):
        """Test serialization to dict."""
        meta = ModelMetadata(
            model_id="test_123",
            model_type="random_forest",
            version="1.0.0",
            created_at=datetime.now(),
            features=["f1", "f2"],
            target="target",
            metrics={"accuracy": 0.85},
        )

        d = meta.to_dict()
        assert d["model_id"] == "test_123"
        assert d["metrics"]["accuracy"] == 0.85

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "model_id": "test_456",
            "model_type": "linear",
            "version": "2.0.0",
            "created_at": "2024-01-01T10:00:00",
            "features": ["a", "b"],
            "target": "y",
            "metrics": {},
            "parameters": {},
            "training_samples": 100,
            "training_period": None,
            "description": "",
        }

        meta = ModelMetadata.from_dict(d)
        assert meta.model_id == "test_456"
        assert meta.model_type == "linear"


class TestMLPipeline:
    """Tests for MLPipeline."""

    def test_fit(self, sample_ohlcv):
        """Test pipeline fitting."""
        config = PipelineConfig(
            model_type="random_forest",
            model_params={"n_estimators": 10},
        )
        pipeline = MLPipeline(config)
        pipeline.fit(sample_ohlcv, "TEST")

        assert pipeline._is_fitted
        assert pipeline.model is not None

    def test_predict(self, sample_ohlcv):
        """Test pipeline prediction."""
        config = PipelineConfig(
            model_type="random_forest",
            model_params={"n_estimators": 10},
        )
        pipeline = MLPipeline(config)
        pipeline.fit(sample_ohlcv, "TEST")

        result = pipeline.predict(sample_ohlcv, "TEST")

        assert isinstance(result, PredictionResult)
        assert result.symbol == "TEST"
        assert result.prediction is not None

    def test_predict_batch(self, sample_ohlcv):
        """Test batch prediction."""
        config = PipelineConfig(
            model_type="random_forest",
            model_params={"n_estimators": 10},
        )
        pipeline = MLPipeline(config)
        pipeline.fit(sample_ohlcv, "TEST")

        results = pipeline.predict_batch(sample_ohlcv, "TEST")

        assert len(results) > 0
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_walk_forward(self, sample_ohlcv):
        """Test walk-forward fitting."""
        # Use minimal feature config to avoid losing too many samples to NaN
        feature_config = FeatureConfig(
            include_returns=True,
            include_volatility=True,
            include_momentum=False,
            include_ma=False,  # Disable long lookback MAs
            include_volume=False,
            include_rsi=False,  # Disable RSI (14-day lookback)
            include_macd=False,  # Disable MACD (26-day lookback)
            include_bollinger=False,
            include_atr=False,
            include_skewness=False,
            include_kurtosis=False,
            lag_periods=[],
            normalize=False,
        )
        config = PipelineConfig(
            model_type="random_forest",
            model_params={"n_estimators": 10},
            walk_forward=True,
            walk_forward_windows=2,  # Reduced from 3 to allow more data per window
            min_samples=50,  # Reduced from 100
            feature_config=feature_config,
        )
        pipeline = MLPipeline(config)
        results = pipeline.walk_forward_fit(sample_ohlcv, "TEST")

        assert len(results) > 0
        assert all("accuracy" in r for r in results)

    def test_different_model_types(self, sample_ohlcv):
        """Test different model types."""
        for model_type in ["random_forest", "gradient_boost", "linear"]:
            config = PipelineConfig(model_type=model_type)
            pipeline = MLPipeline(config)
            pipeline.fit(sample_ohlcv, "TEST")

            assert pipeline._is_fitted

    def test_prediction_cache(self, sample_ohlcv):
        """Test prediction caching."""
        config = PipelineConfig(
            model_type="random_forest",
            model_params={"n_estimators": 10},
        )
        pipeline = MLPipeline(config)
        pipeline.fit(sample_ohlcv, "TEST")

        # First prediction
        result1 = pipeline.predict(sample_ohlcv, "TEST", use_cache=True)

        # Second prediction should use cache
        result2 = pipeline.predict(sample_ohlcv, "TEST", use_cache=True)

        assert result1.prediction == result2.prediction


class TestPredictionResult:
    """Tests for PredictionResult."""

    def test_to_signal_positive(self):
        """Test conversion to positive signal."""
        result = PredictionResult(
            timestamp=datetime.now(),
            symbol="TEST",
            prediction=0.5,
        )
        assert result.to_signal() == 1

    def test_to_signal_negative(self):
        """Test conversion to negative signal."""
        result = PredictionResult(
            timestamp=datetime.now(),
            symbol="TEST",
            prediction=-0.3,
        )
        assert result.to_signal() == -1

    def test_to_signal_neutral(self):
        """Test conversion to neutral signal."""
        result = PredictionResult(
            timestamp=datetime.now(),
            symbol="TEST",
            prediction=0.0,
        )
        assert result.to_signal() == 0


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_get(self, tmp_path):
        """Test model registration and retrieval."""
        from sklearn.ensemble import RandomForestClassifier

        registry = ModelRegistry(tmp_path)

        model = SklearnModel(RandomForestClassifier(n_estimators=10, random_state=42))
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model.fit(X, y)

        model_id = registry.register(model, "test_model", tags=["v1", "production"])

        # Retrieve
        info = registry.get_model("test_model")
        assert info is not None
        assert info["model_id"] == model_id

    def test_list_models(self, tmp_path):
        """Test listing models."""
        from sklearn.ensemble import RandomForestClassifier

        registry = ModelRegistry(tmp_path)

        # Register multiple models
        for i in range(3):
            model = SklearnModel(
                RandomForestClassifier(n_estimators=10, random_state=i)
            )
            X = np.random.randn(100, 5)
            y = (X[:, 0] > 0).astype(int)
            model.fit(X, y)
            registry.register(model, f"model_{i}")

        models = registry.list_models()
        assert len(models) == 3
