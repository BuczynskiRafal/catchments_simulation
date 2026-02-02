"""Property-based tests for MLP predictor stability using Hypothesis.

Fast tests (pure NumPy, no SWMM). Requires cs_app/swmm_model/weights.npz.
"""

import os

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from main.predictor import SimpleMLPModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_weights_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "swmm_model", "weights.npz"
)


@pytest.fixture(scope="module")
def mlp_model():
    """Load the MLP model once for all tests in this module."""
    return SimpleMLPModel(_weights_path)


# ---------------------------------------------------------------------------
# Strategies — physically valid feature ranges
# ---------------------------------------------------------------------------

valid_features = st.fixed_dictionaries(
    {
        "PercImperv": st.floats(0, 100, allow_nan=False, allow_infinity=False),
        "Width": st.floats(1, 10000, allow_nan=False, allow_infinity=False),
        "PercSlope": st.floats(0.01, 100, allow_nan=False, allow_infinity=False),
        "N-Imperv": st.floats(0.001, 1.0, allow_nan=False, allow_infinity=False),
        "N-Perv": st.floats(0.001, 1.0, allow_nan=False, allow_infinity=False),
        "S-Imperv": st.floats(0, 100, allow_nan=False, allow_infinity=False),
        "S-Perv": st.floats(0, 100, allow_nan=False, allow_infinity=False),
        "PctZero": st.floats(0, 100, allow_nan=False, allow_infinity=False),
    }
)


def _features_to_array(features: dict[str, float]) -> np.ndarray:
    """Convert feature dict to numpy array in the correct column order."""
    keys = [
        "PercImperv",
        "Width",
        "PercSlope",
        "N-Imperv",
        "N-Perv",
        "S-Imperv",
        "S-Perv",
        "PctZero",
    ]
    return np.array([[features[k] for k in keys]], dtype=np.float32)


# ===========================================================================
# MLP property tests
# ===========================================================================


class TestPredictorProperties:
    """Property-based tests for the SimpleMLPModel predictor."""

    @given(features=valid_features)
    @settings(max_examples=50)
    def test_predictions_always_non_negative(
        self, features: dict[str, float], mlp_model: SimpleMLPModel
    ) -> None:
        x = _features_to_array(features)
        output = mlp_model.predict(x)
        assert (output >= 0).all(), f"Negative prediction: {output}"

    @given(features=valid_features)
    @settings(max_examples=50)
    def test_predictions_deterministic(
        self, features: dict[str, float], mlp_model: SimpleMLPModel
    ) -> None:
        x = _features_to_array(features)
        result1 = mlp_model.predict(x)
        result2 = mlp_model.predict(x)
        np.testing.assert_array_equal(result1, result2)

    @given(features=valid_features)
    @settings(max_examples=50)
    def test_predictions_no_nan_inf(
        self, features: dict[str, float], mlp_model: SimpleMLPModel
    ) -> None:
        x = _features_to_array(features)
        output = mlp_model.predict(x)
        assert np.all(np.isfinite(output)), f"Non-finite prediction: {output}"

    @given(features=valid_features)
    @settings(max_examples=50)
    def test_output_shape_correct(
        self, features: dict[str, float], mlp_model: SimpleMLPModel
    ) -> None:
        x = _features_to_array(features)
        output = mlp_model.predict(x)
        assert output.shape == (1, 1), f"Unexpected shape: {output.shape}"

    @given(
        batch=st.lists(valid_features, min_size=2, max_size=5),
    )
    @settings(max_examples=20)
    def test_batch_output_shape(
        self, batch: list[dict[str, float]], mlp_model: SimpleMLPModel
    ) -> None:
        keys = [
            "PercImperv",
            "Width",
            "PercSlope",
            "N-Imperv",
            "N-Perv",
            "S-Imperv",
            "S-Perv",
            "PctZero",
        ]
        x = np.array([[f[k] for k in keys] for f in batch], dtype=np.float32)
        output = mlp_model.predict(x)
        assert output.shape == (len(batch), 1)
