"""Tests for Pydantic schemas in the catchment_simulation package."""

import pytest
from pydantic import ValidationError

from catchment_simulation.schemas import (
    SimulationMethodParams,
    SimulationParams,
    SubcatchmentParams,
)


class TestSimulationParams:
    """Tests for SimulationParams schema."""

    def test_valid_params(self):
        params = SimulationParams(start=0, stop=100, step=10)
        assert params.start == 0
        assert params.stop == 100
        assert params.step == 10

    def test_start_equals_stop(self):
        params = SimulationParams(start=5, stop=5, step=1)
        assert params.start == params.stop

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            SimulationParams(start=-1, stop=10, step=1)

    def test_zero_step_rejected(self):
        with pytest.raises(ValidationError):
            SimulationParams(start=0, stop=10, step=0)

    def test_negative_step_rejected(self):
        with pytest.raises(ValidationError):
            SimulationParams(start=0, stop=10, step=-5)

    def test_start_greater_than_stop_rejected(self):
        with pytest.raises(ValidationError, match="start .* must be <= stop"):
            SimulationParams(start=100, stop=10, step=1)

    def test_float_values_accepted(self):
        params = SimulationParams(start=0.5, stop=10.5, step=0.1)
        assert params.start == pytest.approx(0.5)
        assert params.step == pytest.approx(0.1)


class TestSubcatchmentParams:
    """Tests for SubcatchmentParams schema."""

    def test_valid_params(self):
        params = SubcatchmentParams(
            area=10.0,
            slope=5.0,
            imperviousness=50.0,
            width=100.0,
            n_imperv=0.012,
            n_perv=0.15,
            s_imperv=1.27,
            s_perv=5.08,
            pct_zero=25.0,
        )
        assert params.area == 10.0
        assert params.slope == 5.0

    def test_zero_area_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=0,
                slope=5.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_negative_area_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=-1.0,
                slope=5.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_slope_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=101.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_zero_slope_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=0.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_imperviousness_boundaries(self):
        # 0% is valid
        params = SubcatchmentParams(
            area=10.0,
            slope=5.0,
            imperviousness=0.0,
            width=100.0,
            n_imperv=0.012,
            n_perv=0.15,
            s_imperv=1.27,
            s_perv=5.08,
            pct_zero=25.0,
        )
        assert params.imperviousness == 0.0

        # 100% is valid
        params = SubcatchmentParams(
            area=10.0,
            slope=5.0,
            imperviousness=100.0,
            width=100.0,
            n_imperv=0.012,
            n_perv=0.15,
            s_imperv=1.27,
            s_perv=5.08,
            pct_zero=25.0,
        )
        assert params.imperviousness == 100.0

    def test_imperviousness_over_100_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=5.0,
                imperviousness=101.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_negative_manning_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=5.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=-0.01,
                n_perv=0.15,
                s_imperv=1.27,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_negative_depression_storage_rejected(self):
        with pytest.raises(ValidationError):
            SubcatchmentParams(
                area=10.0,
                slope=5.0,
                imperviousness=50.0,
                width=100.0,
                n_imperv=0.012,
                n_perv=0.15,
                s_imperv=-1.0,
                s_perv=5.08,
                pct_zero=25.0,
            )

    def test_pct_zero_boundaries(self):
        # 0% valid
        params = SubcatchmentParams(
            area=10.0,
            slope=5.0,
            imperviousness=50.0,
            width=100.0,
            n_imperv=0.012,
            n_perv=0.15,
            s_imperv=0.0,
            s_perv=0.0,
            pct_zero=0.0,
        )
        assert params.pct_zero == 0.0

        # 100% valid
        params = SubcatchmentParams(
            area=10.0,
            slope=5.0,
            imperviousness=50.0,
            width=100.0,
            n_imperv=0.012,
            n_perv=0.15,
            s_imperv=0.0,
            s_perv=0.0,
            pct_zero=100.0,
        )
        assert params.pct_zero == 100.0


class TestSimulationMethodParams:
    """Tests for SimulationMethodParams schema (web form parameters)."""

    def test_valid_params(self):
        params = SimulationMethodParams(
            method_name="simulate_area",
            start=1,
            stop=10,
            step=1,
            catchment_name="S1",
        )
        assert params.method_name == "simulate_area"
        assert params.catchment_name == "S1"

    def test_all_valid_method_names(self):
        range_methods = [
            "simulate_percent_slope",
            "simulate_area",
            "simulate_width",
            "simulate_percent_impervious",
            "simulate_percent_zero_imperv",
            "simulate_curb_length",
        ]
        for method in range_methods:
            params = SimulationMethodParams(
                method_name=method, start=1, stop=10, step=1, catchment_name="S1"
            )
            assert params.method_name == method

        predefined_methods = [
            "simulate_n_imperv",
            "simulate_n_perv",
            "simulate_s_imperv",
            "simulate_s_perv",
        ]
        for method in predefined_methods:
            params = SimulationMethodParams(method_name=method, catchment_name="S1")
            assert params.method_name == method

    def test_invalid_method_name_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="invalid_method",
                start=1,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    def test_start_below_min_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=-1,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    def test_stop_below_min_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=0,
                stop=-1,
                step=1,
                catchment_name="S1",
            )

    def test_start_greater_than_stop_rejected(self):
        with pytest.raises(ValidationError, match="start .* must be <= stop"):
            SimulationMethodParams(
                method_name="simulate_area",
                start=50,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    def test_empty_catchment_name_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=1,
                stop=10,
                step=1,
                catchment_name="",
            )

    def test_catchment_name_too_long_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=1,
                stop=10,
                step=1,
                catchment_name="x" * 101,
            )

    def test_predefined_method_without_range_params(self):
        """Test that predefined methods work without start/stop/step."""
        params = SimulationMethodParams(method_name="simulate_n_imperv", catchment_name="S1")
        assert params.start is None
        assert params.stop is None
        assert params.step is None

    def test_predefined_method_rejects_start_stop_step(self):
        """Test that predefined methods reject start/stop/step parameters."""
        with pytest.raises(ValidationError, match="predefined literature values"):
            SimulationMethodParams(
                method_name="simulate_n_imperv",
                start=1,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    def test_range_method_requires_start_stop_step(self):
        """Test that range-based methods require start/stop/step."""
        with pytest.raises(ValidationError, match="requires start, stop, and step"):
            SimulationMethodParams(method_name="simulate_area", catchment_name="S1")

    def test_range_method_partial_params_rejected(self):
        """Test that providing only some range params is rejected."""
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=1,
                catchment_name="S1",
            )

    def test_predefined_methods_constant(self):
        """Test that PREDEFINED_METHODS contains expected methods."""
        assert SimulationMethodParams.PREDEFINED_METHODS == frozenset(
            {
                "simulate_n_imperv",
                "simulate_n_perv",
                "simulate_s_imperv",
                "simulate_s_perv",
            }
        )
