"""
Machine Learning Integration Module

Framework for integrating ML models into trading strategies:
- Feature engineering pipeline
- Model interface (sklearn, XGBoost, LightGBM, PyTorch)
- Online learning support
- Model persistence and versioning
- Prediction caching
"""

from libs.ml.features import (
    FeatureConfig,
    FeatureEngineering,
    create_statistical_features,
    create_technical_features,
)
from libs.ml.models import (
    EnsembleModel,
    ModelInterface,
    ModelMetadata,
    ModelRegistry,
    SklearnModel,
)
from libs.ml.pipeline import (
    MLPipeline,
    PipelineConfig,
    PredictionResult,
)

__all__ = [
    # Features
    "FeatureEngineering",
    "FeatureConfig",
    "create_technical_features",
    "create_statistical_features",
    # Models
    "ModelInterface",
    "SklearnModel",
    "EnsembleModel",
    "ModelRegistry",
    "ModelMetadata",
    # Pipeline
    "MLPipeline",
    "PipelineConfig",
    "PredictionResult",
]
