"""
ML Model Interfaces

Unified interface for various ML model types with
persistence and versioning support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""

    model_id: str
    model_type: str
    version: str
    created_at: datetime
    features: List[str]
    target: str
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    training_samples: int = 0
    training_period: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "features": self.features,
            "target": self.target,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "training_samples": self.training_samples,
            "training_period": self.training_period,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


class ModelInterface(ABC):
    """
    Abstract base class for ML models.

    Provides a unified interface for training, prediction,
    and model management.
    """

    model_type: str = "base"

    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize model interface.

        Args:
            model_id: Unique model identifier
        """
        self.model_id = model_id or self._generate_model_id()
        self.metadata: Optional[ModelMetadata] = None
        self._model: Any = None
        self._is_fitted = False

        logger.info(f"[{self.model_type}] Initialized model: {self.model_id}")

    def _generate_model_id(self) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
        return f"{self.model_type}_{timestamp}_{hash_part}"

    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "ModelInterface":
        """
        Fit the model.

        Args:
            X: Feature matrix
            y: Target values
            **kwargs: Additional arguments

        Returns:
            Self
        """
        pass

    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        pass

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Predict class probabilities (for classifiers).

        Args:
            X: Feature matrix

        Returns:
            Class probabilities
        """
        raise NotImplementedError("predict_proba not implemented for this model")

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / f"{self.model_id}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self._model, f)

        # Save metadata
        if self.metadata:
            meta_path = path / f"{self.model_id}_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"[{self.model_type}] Saved model to {path}")

    def load(self, path: Union[str, Path]) -> "ModelInterface":
        """
        Load model from disk.

        Args:
            path: Directory path containing model

        Returns:
            Self
        """
        path = Path(path)

        # Load model
        model_path = path / f"{self.model_id}.pkl"
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)
        self._is_fitted = True

        # Load metadata
        meta_path = path / f"{self.model_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = ModelMetadata.from_dict(json.load(f))

        logger.info(f"[{self.model_type}] Loaded model from {path}")
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted


class SklearnModel(ModelInterface):
    """
    Wrapper for scikit-learn compatible models.

    Supports any model with fit/predict interface.
    """

    model_type = "sklearn"

    def __init__(
        self,
        estimator: Any,
        model_id: Optional[str] = None,
    ):
        """
        Initialize sklearn model wrapper.

        Args:
            estimator: Sklearn-compatible estimator
            model_id: Unique model identifier
        """
        super().__init__(model_id)
        self._model = estimator
        self._estimator_class = type(estimator).__name__

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "SklearnModel":
        """Fit the sklearn model."""
        # Convert to numpy if DataFrame
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        # Fit model
        self._model.fit(X_arr, y_arr, **kwargs)
        self._is_fitted = True

        # Create metadata
        features = list(X.columns) if isinstance(X, pd.DataFrame) else [
            f"feature_{i}" for i in range(X_arr.shape[1])
        ]
        target = y.name if isinstance(y, pd.Series) else "target"

        self.metadata = ModelMetadata(
            model_id=self.model_id,
            model_type=self._estimator_class,
            version="1.0.0",
            created_at=datetime.now(),
            features=features,
            target=target,
            parameters=self.get_params(),
            training_samples=len(X_arr),
        )

        logger.info(
            f"[{self.model_type}] Fitted {self._estimator_class} "
            f"on {len(X_arr)} samples"
        )

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict(X_arr)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(
                f"{self._estimator_class} does not support predict_proba"
            )

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self._model.predict_proba(X_arr)

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if hasattr(self._model, "get_params"):
            return self._model.get_params()
        return {}

    @property
    def feature_importances_(self) -> np.ndarray:
        """Get feature importances if available."""
        if hasattr(self._model, "feature_importances_"):
            return self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            return np.abs(self._model.coef_).flatten()
        return np.array([])


class EnsembleModel(ModelInterface):
    """
    Ensemble of multiple models.

    Combines predictions from multiple models using
    voting or averaging.
    """

    model_type = "ensemble"

    def __init__(
        self,
        models: List[ModelInterface],
        weights: Optional[List[float]] = None,
        voting: str = "soft",  # soft, hard
        model_id: Optional[str] = None,
    ):
        """
        Initialize ensemble model.

        Args:
            models: List of models to ensemble
            weights: Weights for each model (None = equal)
            voting: Voting method (soft=average probas, hard=majority)
            model_id: Unique model identifier
        """
        super().__init__(model_id)
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.voting = voting

        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs,
    ) -> "EnsembleModel":
        """Fit all models in ensemble."""
        for i, model in enumerate(self.models):
            logger.info(f"[{self.model_type}] Fitting model {i+1}/{len(self.models)}")
            model.fit(X, y, **kwargs)

        self._is_fitted = True

        # Create metadata
        features = list(X.columns) if isinstance(X, pd.DataFrame) else [
            f"feature_{i}" for i in range(X.shape[1])
        ]
        target = y.name if isinstance(y, pd.Series) else "target"

        self.metadata = ModelMetadata(
            model_id=self.model_id,
            model_type="ensemble",
            version="1.0.0",
            created_at=datetime.now(),
            features=features,
            target=target,
            parameters={
                "n_models": len(self.models),
                "weights": self.weights,
                "voting": self.voting,
                "model_types": [m.model_type for m in self.models],
            },
            training_samples=len(X),
        )

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make ensemble predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        if self.voting == "soft":
            # Average predictions
            predictions = np.zeros(len(X))
            for model, weight in zip(self.models, self.weights):
                pred = model.predict(X)
                predictions += weight * pred
            return predictions

        else:  # hard voting
            # Majority vote for classification
            predictions = np.array([m.predict(X) for m in self.models])
            # Weighted majority
            result = np.zeros(len(X))
            for i in range(len(X)):
                votes = {}
                for j, model in enumerate(self.models):
                    pred = predictions[j, i]
                    votes[pred] = votes.get(pred, 0) + self.weights[j]
                result[i] = max(votes.keys(), key=lambda k: votes[k])
            return result

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict averaged class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        probas = None
        for model, weight in zip(self.models, self.weights):
            try:
                proba = model.predict_proba(X)
                if probas is None:
                    probas = weight * proba
                else:
                    probas += weight * proba
            except NotImplementedError:
                continue

        if probas is None:
            raise NotImplementedError("No models support predict_proba")

        return probas

    def get_params(self) -> Dict[str, Any]:
        """Get ensemble parameters."""
        return {
            "n_models": len(self.models),
            "weights": self.weights,
            "voting": self.voting,
            "model_params": [m.get_params() for m in self.models],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model in enumerate(self.models):
            model.save(path / f"model_{i}")

        # Save ensemble config
        config = {
            "model_id": self.model_id,
            "weights": self.weights,
            "voting": self.voting,
            "model_ids": [m.model_id for m in self.models],
        }
        with open(path / "ensemble_config.json", "w") as f:
            json.dump(config, f)

        # Save metadata
        if self.metadata:
            with open(path / f"{self.model_id}_meta.json", "w") as f:
                json.dump(self.metadata.to_dict(), f)


class ModelRegistry:
    """
    Registry for managing multiple models.

    Provides versioning and model selection.
    """

    def __init__(self, storage_path: Union[str, Path]):
        """
        Initialize model registry.

        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.storage_path / "registry.json"
        if registry_file.exists():
            with open(registry_file, "r") as f:
                self._registry = json.load(f)

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.storage_path / "registry.json"
        with open(registry_file, "w") as f:
            json.dump(self._registry, f, indent=2)

    def register(
        self,
        model: ModelInterface,
        name: str,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a model in the registry.

        Args:
            model: Model to register
            name: Model name
            tags: Optional tags for categorization

        Returns:
            Model ID
        """
        # Save model
        model_path = self.storage_path / name / model.model_id
        model.save(model_path)

        # Update registry
        if name not in self._registry:
            self._registry[name] = {"versions": []}

        version_info = {
            "model_id": model.model_id,
            "version": len(self._registry[name]["versions"]) + 1,
            "created_at": datetime.now().isoformat(),
            "tags": tags or [],
            "path": str(model_path),
        }

        if model.metadata:
            version_info["metrics"] = model.metadata.metrics

        self._registry[name]["versions"].append(version_info)
        self._registry[name]["latest"] = model.model_id

        self._save_registry()

        logger.info(f"[ModelRegistry] Registered {name} v{version_info['version']}")
        return model.model_id

    def get_model(
        self,
        name: str,
        version: Optional[int] = None,
        model_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get model info from registry.

        Args:
            name: Model name
            version: Specific version (None = latest)
            model_id: Specific model ID

        Returns:
            Model info dictionary
        """
        if name not in self._registry:
            return None

        versions = self._registry[name]["versions"]

        if model_id:
            for v in versions:
                if v["model_id"] == model_id:
                    return v
            return None

        if version:
            for v in versions:
                if v["version"] == version:
                    return v
            return None

        # Return latest
        return versions[-1] if versions else None

    def list_models(self, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List registered models.

        Args:
            name: Filter by model name

        Returns:
            List of model info
        """
        if name:
            return self._registry.get(name, {}).get("versions", [])

        result = []
        for model_name, data in self._registry.items():
            for version in data["versions"]:
                result.append({"name": model_name, **version})
        return result

    def delete_model(self, name: str, version: Optional[int] = None) -> bool:
        """
        Delete a model from registry.

        Args:
            name: Model name
            version: Specific version (None = all versions)

        Returns:
            True if deleted
        """
        if name not in self._registry:
            return False

        if version:
            versions = self._registry[name]["versions"]
            self._registry[name]["versions"] = [
                v for v in versions if v["version"] != version
            ]
        else:
            del self._registry[name]

        self._save_registry()
        return True
