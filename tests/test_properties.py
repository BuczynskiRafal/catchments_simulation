"""Property-based tests for Pydantic schemas using Hypothesis.

Fast tests (no I/O, no SWMM). Validates that schema constraints are
consistently enforced across random inputs.
"""

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import ValidationError

from catchment_simulation.schemas import (
    SimulationMethodParams,
    SimulationParams,
    SubcatchmentParams,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

valid_start = st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False)
valid_step = st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False)


@st.composite
def valid_simulation_params(draw: st.DrawFn) -> dict[str, float]:
    start = draw(valid_start)
    stop = draw(
        st.floats(min_value=start, max_value=start + 1000, allow_nan=False, allow_infinity=False)
    )
    step = draw(valid_step)
    return {"start": start, "stop": stop, "step": step}


@st.composite
def valid_subcatchment_params(draw: st.DrawFn) -> dict[str, float]:
    return {
        "area": draw(
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False)
        ),
        "slope": draw(
            st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False)
        ),
        "imperviousness": draw(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
        ),
        "width": draw(
            st.floats(min_value=0.01, max_value=10000, allow_nan=False, allow_infinity=False)
        ),
        "n_imperv": draw(
            st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        "n_perv": draw(
            st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        "s_imperv": draw(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
        ),
        "s_perv": draw(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
        ),
        "pct_zero": draw(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False)
        ),
    }


VALID_METHOD_NAMES = [
    "simulate_percent_slope",
    "simulate_area",
    "simulate_width",
    "simulate_percent_impervious",
    "simulate_percent_zero_imperv",
]


# ===========================================================================
# A. SimulationParams — range validation
# ===========================================================================


class TestSimulationParamsProperties:
    """Property-based tests for SimulationParams schema."""

    @given(data=valid_simulation_params())
    @settings(max_examples=100)
    def test_valid_params_always_accepted(self, data: dict[str, float]) -> None:
        params = SimulationParams(**data)
        assert params.start >= 0
        assert params.step > 0
        assert params.start <= params.stop

    @given(
        start=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
        stop=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        step=valid_step,
    )
    @settings(max_examples=100)
    def test_negative_start_always_rejected(self, start: float, stop: float, step: float) -> None:
        with pytest.raises(ValidationError):
            SimulationParams(start=start, stop=stop, step=step)

    @given(
        start=valid_start,
        stop=st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        step=st.floats(max_value=0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_non_positive_step_always_rejected(
        self, start: float, stop: float, step: float
    ) -> None:
        with pytest.raises(ValidationError):
            SimulationParams(start=start, stop=stop, step=step)

    @given(
        start=st.floats(min_value=1, max_value=2000, allow_nan=False, allow_infinity=False),
        step=valid_step,
    )
    @settings(max_examples=100)
    def test_start_gt_stop_always_rejected(self, start: float, step: float) -> None:
        stop = start - 0.001  # guarantee start > stop
        with pytest.raises(ValidationError):
            SimulationParams(start=start, stop=stop, step=step)

    @given(data=valid_simulation_params())
    @settings(max_examples=100)
    def test_roundtrip_preserves_values(self, data: dict[str, float]) -> None:
        model = SimulationParams(**data)
        dumped = model.model_dump()
        restored = SimulationParams(**dumped)
        assert model.start == restored.start
        assert model.stop == restored.stop
        assert model.step == restored.step


# ===========================================================================
# B. SubcatchmentParams — physical range validation
# ===========================================================================


class TestSubcatchmentParamsProperties:
    """Property-based tests for SubcatchmentParams schema."""

    @given(data=valid_subcatchment_params())
    @settings(max_examples=100)
    def test_valid_subcatchment_params_accepted(self, data: dict[str, float]) -> None:
        params = SubcatchmentParams(**data)
        assert params.area > 0
        assert 0 < params.slope <= 100
        assert 0 <= params.imperviousness <= 100

    @given(
        area=st.floats(max_value=0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_negative_area_rejected(self, area: float) -> None:
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=area,
                slope=5.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    @given(
        imperviousness=st.one_of(
            st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False),
            st.floats(min_value=100.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_imperviousness_out_of_range_rejected(self, imperviousness: float) -> None:
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=5.0,
                imperviousness=imperviousness,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    @given(
        slope=st.one_of(
            st.floats(max_value=0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=100.001, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_slope_out_of_range_rejected(self, slope: float) -> None:
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=slope,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )


# ===========================================================================
# C. SimulationMethodParams — web form validation
# ===========================================================================


class TestSimulationMethodParamsProperties:
    """Property-based tests for SimulationMethodParams schema."""

    @given(
        method_name=st.sampled_from(VALID_METHOD_NAMES),
        start=st.integers(min_value=1, max_value=50),
        step=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100)
    def test_valid_method_params_accepted(self, method_name: str, start: int, step: int) -> None:
        stop = min(start + step * 5, 100)
        params = SimulationMethodParams(
            method_name=method_name,
            start=start,
            stop=stop,
            step=step,
            catchment_name="S1",
        )
        assert params.start <= params.stop

    @given(
        method_name=st.text(min_size=1, max_size=50).filter(lambda s: s not in VALID_METHOD_NAMES),
    )
    @settings(max_examples=100)
    def test_invalid_method_name_rejected(self, method_name: str) -> None:
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name=method_name,
                start=1,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    @given(
        start=st.integers(min_value=2, max_value=100),
    )
    @settings(max_examples=100)
    def test_start_gt_stop_rejected(self, start: int) -> None:
        stop = max(1, start - 1)  # guarantee start > stop
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=start,
                stop=stop,
                step=1,
                catchment_name="S1",
            )
