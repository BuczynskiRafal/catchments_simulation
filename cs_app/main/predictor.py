"""
This module contains functions to predict runoff using a machine learning model.

The model weights are loaded from the 'swmm_model' directory.
The predict_runoff function takes a SWMM model object as input and returns an array of
predicted runoff values for each subcatchment.

This is a pure NumPy implementation - no TensorFlow/Keras required.
"""

import logging
import os
import threading
import time

import numpy as np
import swmmio

logger = logging.getLogger(__name__)


class SimpleMLPModel:
    """
    A simple MLP model for inference using pure NumPy.

    Architecture:
        Input (8) -> Normalization -> Dense(8, ReLU) -> Dense(8, ReLU) -> Dense(1, ReLU)
    """

    def __init__(self, weights_path: str):
        """
        Load model weights from .npz file.

        Parameters
        ----------
        weights_path : str
            Path to the .npz file containing model weights.
        """
        try:
            weights = np.load(weights_path)
        except FileNotFoundError:
            logger.error(f"Cannot load model weights: {weights_path}")
            raise

        # Normalization parameters
        self.norm_mean = weights["norm_mean"]
        self.norm_variance = weights["norm_variance"]

        # Dense layer weights
        self.dense_0_kernel = weights["dense_0_kernel"]
        self.dense_0_bias = weights["dense_0_bias"]
        self.dense_1_kernel = weights["dense_1_kernel"]
        self.dense_1_bias = weights["dense_1_bias"]
        self.dense_2_kernel = weights["dense_2_kernel"]
        self.dense_2_bias = weights["dense_2_bias"]

        logger.debug("Model weights loaded successfully")

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization layer."""
        return (x - self.norm_mean) / np.sqrt(self.norm_variance + 1e-7)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU activation."""
        return np.maximum(0, x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (batch_size, 8).

        Returns
        -------
        np.ndarray
            Predictions of shape (batch_size, 1).
        """
        # Ensure float32
        x = np.asarray(x, dtype=np.float32)

        # Normalization
        x = self._normalize(x)

        # Dense layer 0
        x = x @ self.dense_0_kernel + self.dense_0_bias
        x = self._relu(x)

        # Dense layer 1
        x = x @ self.dense_1_kernel + self.dense_1_bias
        x = self._relu(x)

        # Dense layer 2 (output)
        x = x @ self.dense_2_kernel + self.dense_2_bias
        x = self._relu(x)

        return x


_model: SimpleMLPModel | None = None
_model_lock = threading.Lock()


def _default_weights_path() -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, "..", "swmm_model", "weights.npz")


def _get_weights_path() -> str:
    """Resolve model weights path from env/settings with a safe default."""
    env_path = os.environ.get("CATCHMENT_SIMULATION_WEIGHTS_PATH")
    if env_path:
        return env_path

    try:
        from django.conf import settings

        configured_path = getattr(settings, "SWMM_MODEL_WEIGHTS_PATH", None)
        if configured_path:
            return str(configured_path)
    except Exception:
        # Predictor can also run outside Django context.
        pass

    return _default_weights_path()


def _get_model() -> SimpleMLPModel:
    """Get or create the model instance (lazy loading)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SimpleMLPModel(_get_weights_path())
    return _model


def preload_model() -> bool:
    """Eagerly load model weights into process memory.

    Returns True when model was not loaded before this call.
    """
    loaded_before = _model is not None
    started_at = time.perf_counter()
    _get_model()
    if not loaded_before:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info("ANN predictor model loaded in %.1f ms", elapsed_ms)
        return True
    return False


def predict_runoff(swmmio_model: swmmio.Model) -> np.ndarray:
    """
    Predict runoff using a machine learning model.

    Parameters
    ----------
    swmmio_model : swmmio.Model
        A SWMM model object with subcatchment data.

    Returns
    -------
    np.ndarray
        An array of predicted runoff values for each subcatchment.

    Example
    -------
    >>> swmm_model = swmmio.Model("example.inp")
    >>> predict_runoff(swmm_model)
    array([0.1, 0.2, 0.3, 0.4, 0.5])
    """
    model = _get_model()

    data = swmmio_model.subcatchments.dataframe[
        [
            "PercImperv",
            "Width",
            "PercSlope",
            "N-Imperv",
            "N-Perv",
            "S-Imperv",
            "S-Perv",
            "PctZero",
        ]
    ].values

    return model.predict(data).flatten()
