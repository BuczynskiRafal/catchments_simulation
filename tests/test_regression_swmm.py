"""Regression tests for SWMM simulation results.

Ensures that simulation outputs don't change unexpectedly between versions.
Reference data generated from known-good runs and stored in
tests/fixtures/regression_swmm.json.
"""

import json
import os

import pytest

from catchment_simulation import FeaturesSimulation

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REFERENCE_FILE = os.path.join(FIXTURES_DIR, "regression_swmm.json")
TOLERANCE = 1e-5


@pytest.fixture(scope="module")
def reference_data() -> dict:
    with open(REFERENCE_FILE) as f:
        return json.load(f)


@pytest.fixture
def simulation_instance():
    with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
        yield sim


class TestRegressionArea:
    """Regression tests for simulate_area."""

    def test_area_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["area"]
        df = simulation_instance.simulate_area(**ref["params"])
        for actual, expected in zip(df["runoff"], ref["expected"]["runoff"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_area_peak_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["area"]
        df = simulation_instance.simulate_area(**ref["params"])
        for actual, expected in zip(df["peak_runoff_rate"], ref["expected"]["peak_runoff_rate"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_area_infiltration_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["area"]
        df = simulation_instance.simulate_area(**ref["params"])
        for actual, expected in zip(df["infiltration"], ref["expected"]["infiltration"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)


class TestRegressionPercentImpervious:
    """Regression tests for simulate_percent_impervious."""

    def test_imperv_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["percent_impervious"]
        df = simulation_instance.simulate_percent_impervious(**ref["params"])
        for actual, expected in zip(df["runoff"], ref["expected"]["runoff"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_imperv_peak_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["percent_impervious"]
        df = simulation_instance.simulate_percent_impervious(**ref["params"])
        for actual, expected in zip(df["peak_runoff_rate"], ref["expected"]["peak_runoff_rate"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_imperv_infiltration_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["percent_impervious"]
        df = simulation_instance.simulate_percent_impervious(**ref["params"])
        for actual, expected in zip(df["infiltration"], ref["expected"]["infiltration"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)


class TestRegressionManningN:
    """Regression tests for simulate_n_imperv."""

    def test_n_imperv_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["n_imperv"]
        df = simulation_instance.simulate_n_imperv()
        for actual, expected in zip(df["runoff"], ref["expected"]["runoff"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_n_imperv_peak_runoff_matches_reference(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["n_imperv"]
        df = simulation_instance.simulate_n_imperv()
        for actual, expected in zip(df["peak_runoff_rate"], ref["expected"]["peak_runoff_rate"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)

    def test_n_imperv_infiltration_constant(
        self, simulation_instance: FeaturesSimulation, reference_data: dict
    ) -> None:
        ref = reference_data["n_imperv"]
        df = simulation_instance.simulate_n_imperv()
        for actual, expected in zip(df["infiltration"], ref["expected"]["infiltration"]):
            assert actual == pytest.approx(expected, abs=TOLERANCE)
