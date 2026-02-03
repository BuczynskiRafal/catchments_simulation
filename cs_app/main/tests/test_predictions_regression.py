"""Regression tests for MLP runoff predictions.

Ensures that model predictions don't change unexpectedly between versions.
Reference data stored in tests/fixtures/regression_mlp.json.
"""

import json
import os

import numpy as np
import pytest

from main.predictor import SimpleMLPModel

FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests", "fixtures"
)
REFERENCE_FILE = os.path.join(FIXTURES_DIR, "regression_mlp.json")
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "swmm_model", "weights.npz"
)
TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def reference_data():
    with open(REFERENCE_FILE) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mlp_model():
    return SimpleMLPModel(WEIGHTS_PATH)


class TestMLPRegression:
    """Regression tests for MLP predictions against reference values."""

    def test_individual_predictions_match_reference(self, mlp_model, reference_data):
        for case in reference_data["cases"]:
            x = np.array([case["input"]], dtype=np.float32)
            prediction = mlp_model.predict(x).flatten()[0]
            assert prediction == pytest.approx(
                case["expected"], abs=TOLERANCE
            ), f"Failed for {case['label']}: got {prediction}, expected {case['expected']}"

    def test_batch_predictions_match_reference(self, mlp_model, reference_data):
        inputs = np.array([c["input"] for c in reference_data["cases"]], dtype=np.float32)
        expected = np.array([c["expected"] for c in reference_data["cases"]])
        predictions = mlp_model.predict(inputs).flatten()
        np.testing.assert_allclose(predictions, expected, atol=TOLERANCE)

    def test_predictions_reproducible_across_calls(self, mlp_model, reference_data):
        inputs = np.array([c["input"] for c in reference_data["cases"]], dtype=np.float32)
        result1 = mlp_model.predict(inputs)
        result2 = mlp_model.predict(inputs)
        np.testing.assert_array_equal(result1, result2)
