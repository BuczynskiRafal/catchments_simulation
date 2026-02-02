"""Property-based tests for SWMM simulation invariants using Hypothesis.

Slow tests (runs SWMM engine). Uses small parameter ranges to control execution time.
Marked with @pytest.mark.slow for selective CI execution.
"""

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from catchment_simulation import FeaturesSimulation

# ---------------------------------------------------------------------------
# Strategies — limited ranges for fast SWMM runs (~10 steps max)
# ---------------------------------------------------------------------------

small_range = st.fixed_dictionaries(
    {
        "start": st.just(1.0),
        "stop": st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        "step": st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    }
)


@pytest.fixture
def simulation_instance():
    """Create a FeaturesSimulation instance with cleanup."""
    with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
        yield sim


# ===========================================================================
# Simulation invariant tests
# ===========================================================================


@pytest.mark.slow
class TestSimulationProperties:
    """Property-based tests for SWMM simulation invariants."""

    @given(params=small_range)
    @settings(max_examples=3, deadline=None)
    def test_simulate_area_returns_valid_dataframe(self, params: dict[str, float]) -> None:
        with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
            df = sim.simulate_area(**params)
            expected_cols = set(FeaturesSimulation.RESULT_KEYS) | {"Area"}
            assert set(df.columns) == expected_cols
            for key in FeaturesSimulation.RESULT_KEYS:
                assert (df[key] >= 0).all(), f"Column {key} has negative values"

    @given(params=small_range)
    @settings(max_examples=3, deadline=None)
    def test_simulate_area_row_count_matches_range(self, params: dict[str, float]) -> None:
        with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
            df = sim.simulate_area(**params)
            expected_count = len(np.arange(params["start"], params["stop"] + 0.001, params["step"]))
            assert len(df) == expected_count

    @given(params=small_range)
    @settings(max_examples=3, deadline=None)
    def test_simulate_percent_impervious_non_negative(self, params: dict[str, float]) -> None:
        with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
            df = sim.simulate_percent_impervious(**params)
            for key in FeaturesSimulation.RESULT_KEYS:
                assert (df[key] >= 0).all(), f"Column {key} has negative values"

    @given(params=small_range)
    @settings(max_examples=3, deadline=None)
    def test_simulate_width_non_negative(self, params: dict[str, float]) -> None:
        with FeaturesSimulation("S1", "tests/fixtures/example.inp") as sim:
            df = sim.simulate_width(**params)
            for key in FeaturesSimulation.RESULT_KEYS:
                assert (df[key] >= 0).all(), f"Column {key} has negative values"
