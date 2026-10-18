"""Property-based tests for SWMM simulation invariants using Hypothesis.

Slow tests (they run the SWMM engine), kept to a few small examples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from catchment_simulation import FeaturesSimulation, build_range

if TYPE_CHECKING:
    from pathlib import Path

small_range = st.fixed_dictionaries(
    {
        "start": st.just(1.0),
        "stop": st.floats(min_value=2.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        "step": st.floats(min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    }
)

# The library never writes to the model file, so one copy can serve every example.
swmm_settings = settings(
    max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)


@pytest.mark.slow
class TestSimulationProperties:
    @given(params=small_range)
    @swmm_settings
    def test_simulate_area_rows_match_range(
        self, model_path: Path, params: dict[str, float]
    ) -> None:
        frame = FeaturesSimulation("S1", model_path).simulate_area(**params)
        assert frame["Area"].tolist() == build_range(**params)
        assert (frame[list(FeaturesSimulation.RESULT_KEYS)] >= 0).all().all()

    @pytest.mark.parametrize("method", ["simulate_percent_impervious", "simulate_width"])
    @given(params=small_range)
    @swmm_settings
    def test_results_non_negative(
        self, model_path: Path, method: str, params: dict[str, float]
    ) -> None:
        frame = getattr(FeaturesSimulation("S1", model_path), method)(**params)
        assert (frame[list(FeaturesSimulation.RESULT_KEYS)] >= 0).all().all()
