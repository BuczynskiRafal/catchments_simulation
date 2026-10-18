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
        return self._relu(x)


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
        from django.core.exceptions import ImproperlyConfigured
    except ImportError:
        # Predictor can run outside Django environments.
        return _default_weights_path()

    try:
        configured_path = getattr(settings, "SWMM_MODEL_WEIGHTS_PATH", None)
    except ImproperlyConfigured:
        return _default_weights_path()

    if configured_path:
        return str(configured_path)

    return _default_weights_path()


def _get_or_create_model() -> tuple[SimpleMLPModel, bool]:
    """Return model instance and whether it was created in this call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SimpleMLPModel(_get_weights_path())
                return _model, True
    return _model, False


def _get_model() -> SimpleMLPModel:
    """Get or create the model instance (lazy loading)."""
    model, _ = _get_or_create_model()
    return model


def preload_model() -> bool:
    """Eagerly load model weights into process memory.

    Returns True when model was not loaded before this call.
    """
    started_at = time.perf_counter()
    _, loaded_now = _get_or_create_model()
    if loaded_now:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info("ANN predictor model loaded in %.1f ms", elapsed_ms)
        return True
    return False


REQUIRED_SUBCATCHMENT_COLUMNS: tuple[str, ...] = (
    "PercImperv",
    "Width",
    "PercSlope",
    "N-Imperv",
    "N-Perv",
    "S-Imperv",
    "S-Perv",
    "PctZero",
)
_FT_TO_M = 0.3048
_IN_TO_MM = 25.4


def _is_us_flow_units(swmmio_model: object) -> bool:
    """Return True when swmmio_model specifies US flow units (CFS, GPM, MGD)."""
    from main.units import _extract_options_flow_units, _unit_system

    inp = getattr(swmmio_model, "inp", None)
    options = getattr(inp, "options", None)
    if options is None:
        return False
    return _unit_system(_extract_options_flow_units(options)) == "US"


def predict_runoff(swmmio_model: swmmio.Model) -> np.ndarray:
    """
    Predict subcatchment runoff volume using the pre-trained MLP model.

    The network weights were trained on the reference SI model (``example.inp``,
    ``Area = 5 ha``) with ``Width`` in meters and ``S-Imperv`` / ``S-Perv`` in
    millimeters, and its output is always expressed in ``10^6 L`` (``10^3 m³``).
    When ``swmmio_model`` uses US flow units (``CFS``, ``GPM``, ``MGD``), ``Width``
    (ft -> m) and ``S-Imperv`` / ``S-Perv`` (in -> mm) are converted to SI prior
    to normalization.

    Parameters
    ----------
    swmmio_model : swmmio.Model
        A SWMM model object with ``subcatchments.dataframe`` containing all 8
        required ``[SUBCATCHMENTS]`` and ``[SUBAREAS]`` columns.

    Returns
    -------
    np.ndarray
        1-D array of predicted runoff volumes in ``10^6 L`` (``10^3 m³``) for each
        subcatchment.

    Raises
    ------
    KeyError
        If any required ``[SUBCATCHMENTS]`` or ``[SUBAREAS]`` column is missing.
    """
    model = _get_model()
    dataframe = swmmio_model.subcatchments.dataframe
    missing = [col for col in REQUIRED_SUBCATCHMENT_COLUMNS if col not in dataframe.columns]
    if missing:
        raise KeyError(
            f"SWMM model is missing required subcatchment/subarea columns: {', '.join(missing)}"
        )

    data = dataframe[list(REQUIRED_SUBCATCHMENT_COLUMNS)].to_numpy(dtype=np.float32, copy=True)
    if _is_us_flow_units(swmmio_model):
        # Column 1 is Width (ft -> m); columns 5 and 6 are S-Imperv and S-Perv (in -> mm).
        data[:, 1] *= _FT_TO_M
        data[:, 5] *= _IN_TO_MM
        data[:, 6] *= _IN_TO_MM

    return model.predict(data).flatten()
