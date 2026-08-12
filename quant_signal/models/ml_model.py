"""
Machine Learning Model Lifecycle Manager for QuantSignal AI.

Handles Scikit-learn model loading, saving via Joblib, feature validation, and scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import pandas as pd

from quant_signal.exceptions import ModelNotFoundError, ModelPredictionError
from quant_signal.logger import get_logger, log_execution_time
from quant_signal.models.base import IModelEngine

logger = get_logger(__name__)


class SklearnModelEngine(IModelEngine):
    """Scikit-Learn ML Model Manager implementing IModelEngine interface."""

    def __init__(self, model_name: str = "quant_signal_rf_v1.joblib") -> None:
        """Initialize ML model manager.

        Args:
            model_name: Filename of saved model.
        """
        self.model_name: str = model_name
        self._model: Any | None = None
        self._is_loaded: bool = False

    @log_execution_time()
    def load_model(self, model_path: str) -> bool:
        """Loads serialized Scikit-learn model using Joblib.

        Args:
            model_path: Filepath to target joblib artifact.

        Returns:
            bool: True if model loaded successfully.

        Raises:
            ModelNotFoundError: If path does not exist on disk.
        """
        path = Path(model_path)
        if not path.exists():
            logger.error(f"Model file not found at: {model_path}")
            raise ModelNotFoundError(f"Target model file '{model_path}' does not exist.")

        try:
            logger.info(f"Loading ML model from '{model_path}' using Joblib...")
            self._model = joblib.load(path)
            self._is_loaded = True
            logger.info(f"Successfully loaded model '{path.name}'")
            return True
        except Exception as err:
            logger.error(f"Failed to load model file '{model_path}': {err}")
            raise ModelPredictionError(f"Joblib load error for model '{model_path}': {err}") from err

    @log_execution_time()
    def save_model(self, model_path: str) -> bool:
        """Serializes current model instance to disk using Joblib.

        Args:
            model_path: Output filepath destination.

        Returns:
            bool: True if save operation succeeded.
        """
        if self._model is None:
            logger.warning("Attempted to save uninitialized or empty model.")
            return False

        try:
            out_path = Path(model_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, out_path)
            logger.info(f"Model saved successfully to '{model_path}'")
            return True
        except Exception as err:
            logger.error(f"Failed to save model to '{model_path}': {err}")
            return False

    @log_execution_time()
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Runs inference/scoring on input features DataFrame.

        Args:
            features: Input indicator and price feature matrix.

        Returns:
            pd.Series: Model classification predictions.

        Raises:
            ModelPredictionError: If prediction fails or model is unloaded.
        """
        if not self._is_loaded or self._model is None:
            logger.warning("Predict called on uninitialized model. Returning empty series.")
            return pd.Series(dtype=float)

        try:
            logger.info(f"Executing model prediction across {len(features)} feature rows...")
            predictions = self._model.predict(features)
            return pd.Series(predictions, index=features.index)
        except Exception as err:
            logger.error(f"Inference error during model prediction: {err}")
            raise ModelPredictionError(f"Model prediction failed: {err}") from err

    @log_execution_time()
    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        """Runs probability scoring on input features DataFrame.

        Args:
            features: Input indicator and price feature matrix.

        Returns:
            pd.Series: Bullish probability score per row.
        """
        if not self._is_loaded or self._model is None:
            logger.warning("Predict_proba called on uninitialized model. Returning 0.5 default series.")
            return pd.Series([0.5] * len(features), index=features.index if hasattr(features, "index") else None)

        try:
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(features)
                bull_probs = [float(p[1]) if len(p) > 1 else float(p[0]) for p in probs]
                return pd.Series(bull_probs, index=features.index)
            else:
                preds = self._model.predict(features)
                return pd.Series(preds, index=features.index).astype(float)
        except Exception as err:
            logger.error(f"Probability inference error during model prediction: {err}")
            return pd.Series([0.5] * len(features), index=features.index if hasattr(features, "index") else None)

