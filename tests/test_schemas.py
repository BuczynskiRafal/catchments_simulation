"""Tests for Pydantic schemas in the catchment_simulation package.

Imports the schemas module directly (via importlib) to avoid pulling in the
heavy pyswmm / swmmio C-extensions through ``__init__.py``.
"""

import importlib.util
import os

import pytest
from pydantic import ValidationError

# Load schemas.py directly so that importing the package __init__
# (which eagerly loads pyswmm) is not required.
_schemas_path = os.path.join(
    os.path.dirname(__file__), os.pardir, "src", "catchment_simulation", "schemas.py"
)
_spec = importlib.util.spec_from_file_location("catchment_simulation.schemas", _schemas_path)
_schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schemas)

SimulationParams = _schemas.SimulationParams
SubcatchmentParams = _schemas.SubcatchmentParams
SimulationMethodParams = _schemas.SimulationMethodParams


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
        methods = [
            "simulate_percent_slope",
            "simulate_area",
            "simulate_width",
            "simulate_percent_impervious",
            "simulate_percent_zero_imperv",
        ]
        for method in methods:
            params = SimulationMethodParams(
                method_name=method, start=1, stop=10, step=1, catchment_name="S1"
            )
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
                start=0,
                stop=10,
                step=1,
                catchment_name="S1",
            )

    def test_stop_above_max_rejected(self):
        with pytest.raises(ValidationError):
            SimulationMethodParams(
                method_name="simulate_area",
                start=1,
                stop=101,
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
