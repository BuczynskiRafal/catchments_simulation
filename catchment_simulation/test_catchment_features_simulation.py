import glob
import os

import pandas as pd
import pytest
import swmmio

from .catchment_features_simulation import FeaturesSimulation


@pytest.fixture
def simulation_instance():
    """
    Returns a FeaturesSimulation instance with cleanup.
    """
    subcatchment_id = "S1"
    raw_file = "catchment_simulation/example.inp"
    instance = FeaturesSimulation(subcatchment_id, raw_file)
    yield instance
    instance._cleanup_temp_files()


def test_init(simulation_instance):
    """
    Test the initialization of the simulation instance.
    """
    assert simulation_instance.subcatchment_id == "S1"
    assert simulation_instance.raw_file == "catchment_simulation/example.inp"
    assert simulation_instance.file.endswith("_copy.inp")
    assert isinstance(simulation_instance.model, swmmio.Model)


def test_copy_file(simulation_instance):
    """
    Test the copy_file method.
    """
    new_path = simulation_instance.copy_file(suffix="test")
    assert os.path.exists(new_path)
    assert new_path.endswith("_test.inp")
    os.remove(new_path)


def test_get_section(simulation_instance):
    """
    Test the get_section method.
    """
    section_df = simulation_instance.get_section(section="subcatchments")
    assert isinstance(section_df, pd.DataFrame)
    assert not section_df.empty

    with pytest.raises(AttributeError):
        simulation_instance.get_section(section="non_existing_section")


def test_calculate(simulation_instance):
    """
    Test the calculate method.
    """
    subcatchment_stats = simulation_instance.calculate()
    assert isinstance(subcatchment_stats, dict)
    assert "runoff" in subcatchment_stats
    assert "peak_runoff_rate" in subcatchment_stats
    assert "infiltration" in subcatchment_stats
    assert "evaporation" in subcatchment_stats


def test_simulate_subcatchment(simulation_instance):
    """
    Test the simulate_subcatchment method.
    """
    feature = "Area"
    df = simulation_instance.simulate_subcatchment(feature=feature, start=1, stop=10, step=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", feature]
    )


def test_simulate_area(simulation_instance):
    """
    Test the simulate_area method.
    """
    df = simulation_instance.simulate_area(start=1, stop=10, step=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Area"]
    )


def test_simulate_percent_impervious(simulation_instance):
    """
    Test the simulate_percent_impervious method.
    """
    df = simulation_instance.simulate_percent_impervious(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "PercImperv"]
    )


def test_simulate_percent_slope(simulation_instance):
    """
    Test the simulate_percent_slope method.
    """
    df = simulation_instance.simulate_percent_slope(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "PercSlope"]
    )


def test_simulate_width(simulation_instance):
    """
    Test the simulate_width method.
    """
    df = simulation_instance.simulate_width(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Width"]
    )


def test_simulate_curb_length(simulation_instance):
    """
    Test the simulate_curb_length method.
    """
    df = simulation_instance.simulate_curb_length(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "CurbLength"]
    )


def test_simulate_n_imperv(simulation_instance):
    """
    Test the simulate_n_imperv method.
    """
    df = simulation_instance.simulate_n_imperv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "N-Imperv"]
    )


def test_simulate_n_perv(simulation_instance):
    """
    Test the simulate_n_perv method.
    """
    df = simulation_instance.simulate_n_perv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "N-Perv"]
    )


def test_simulate_destore_imperv(simulation_instance):
    """
    Test the simulate_destore_imperv method.
    """
    df = simulation_instance.simulate_destore(param="Imperv")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Destore-Imperv"]
    )


def test_simulate_destore_perv(simulation_instance):
    """
    Test the simulate_destore_perv method.
    """
    df = simulation_instance.simulate_destore(param="Perv")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Destore-Perv"]
    )


def test_simulate_s_imperv(simulation_instance):
    """
    Test the simulate_s_imperv method.
    """
    df = simulation_instance.simulate_s_imperv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Destore-Imperv"]
    )


def test_simulate_s_perv(simulation_instance):
    """
    Test the simulate_s_perv method.
    """
    df = simulation_instance.simulate_s_perv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Destore-Perv"]
    )


def test_simulate_percent_zero_imperv(simulation_instance):
    """
    Test the simulate_percent_zero_imperv method.
    """
    df = simulation_instance.simulate_percent_zero_imperv(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == set(
        ["runoff", "peak_runoff_rate", "infiltration", "evaporation", "Zero-Imperv"]
    )


class TestContextManager:
    """Tests for context manager functionality."""

    def test_context_manager_cleanup(self):
        """Test that context manager cleans up temp files."""
        subcatchment_id = "S1"
        raw_file = "catchment_simulation/example.inp"

        with FeaturesSimulation(subcatchment_id, raw_file) as sim:
            temp_files = sim._temp_files.copy()
            assert len(temp_files) > 0
            for f in temp_files:
                assert os.path.exists(f)

        for f in temp_files:
            assert not os.path.exists(f)


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_negative_start_raises_error(self, simulation_instance):
        """Test that negative start value raises ValueError."""
        with pytest.raises(ValueError, match="start must be >= 0"):
            simulation_instance.simulate_subcatchment("Area", start=-1, stop=10, step=1)

    def test_zero_step_raises_error(self, simulation_instance):
        """Test that zero step value raises ValueError."""
        with pytest.raises(ValueError, match="step must be > 0"):
            simulation_instance.simulate_subcatchment("Area", start=0, stop=10, step=0)

    def test_negative_step_raises_error(self, simulation_instance):
        """Test that negative step value raises ValueError."""
        with pytest.raises(ValueError, match="step must be > 0"):
            simulation_instance.simulate_subcatchment("Area", start=0, stop=10, step=-1)

    def test_start_greater_than_stop_raises_error(self, simulation_instance):
        """Test that start > stop raises ValueError."""
        with pytest.raises(ValueError, match="start .* must be <= stop"):
            simulation_instance.simulate_subcatchment("Area", start=100, stop=10, step=1)


class TestClassConstants:
    """Tests for class constants."""

    def test_result_keys(self):
        """Test that RESULT_KEYS contains expected keys."""
        assert FeaturesSimulation.RESULT_KEYS == (
            "runoff",
            "peak_runoff_rate",
            "infiltration",
            "evaporation",
        )

    def test_manning_n_values_sorted(self):
        """Test that MANNING_N_VALUES is sorted."""
        values = FeaturesSimulation.MANNING_N_VALUES
        assert list(values) == sorted(values)
        assert len(values) == 15

    def test_depression_storage_values(self):
        """Test that DEPRESSION_STORAGE_VALUES are in mm."""
        values = FeaturesSimulation.DEPRESSION_STORAGE_VALUES
        assert len(values) == 4
        assert values[0] == pytest.approx(0.05 * 25.4)

    def test_create_result_dict(self):
        """Test that _create_result_dict returns correct structure."""
        result = FeaturesSimulation._create_result_dict()
        assert set(result.keys()) == set(FeaturesSimulation.RESULT_KEYS)
        for key in result:
            assert result[key] == []


class TestCleanup:
    """Tests for file cleanup logic."""

    def test_fixture_safety_net_cleanup(self):
        """Test the safety net cleanup for files not tracked by instance."""

        extra_file = "catchment_simulation/example_untracked.inp"
        with open(extra_file, "w") as f:
            f.write("test")

        def safety_cleanup():
            for f in glob.glob("catchment_simulation/example_*.inp"):
                if os.path.exists(f):
                    os.remove(f)

        assert os.path.exists(extra_file)
        safety_cleanup()
        assert not os.path.exists(extra_file)


class TestDefaultParameters:
    """Tests for default parameter values."""

    def test_simulate_area_defaults(self, simulation_instance):
        """Test simulate_area with default parameters."""
        df = simulation_instance.simulate_area()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert "Area" in df.columns

    def test_simulate_percent_impervious_defaults(self, simulation_instance):
        """Test simulate_percent_impervious with default parameters."""
        df = simulation_instance.simulate_percent_impervious()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "PercImperv" in df.columns

    def test_simulate_percent_slope_defaults(self, simulation_instance):
        """Test simulate_percent_slope with default parameters."""
        df = simulation_instance.simulate_percent_slope()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "PercSlope" in df.columns

    def test_simulate_width_defaults(self, simulation_instance):
        """Test simulate_width with default parameters."""
        df = simulation_instance.simulate_width()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "Width" in df.columns

    def test_simulate_curb_length_defaults(self, simulation_instance):
        """Test simulate_curb_length with default parameters."""
        df = simulation_instance.simulate_curb_length()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "CurbLength" in df.columns

    def test_simulate_percent_zero_imperv_defaults(self, simulation_instance):
        """Test simulate_percent_zero_imperv with default parameters."""
        df = simulation_instance.simulate_percent_zero_imperv()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 11
        assert "Zero-Imperv" in df.columns

    def test_copy_file_default_suffix(self, simulation_instance):
        """Test copy_file with default suffix."""
        new_path = simulation_instance.copy_file()
        assert new_path.endswith("_copy.inp")
        assert os.path.exists(new_path)

    def test_get_section_default_subcatchments(self, simulation_instance):
        """Test get_section with default section (subcatchments)."""
        section_df = simulation_instance.get_section()
        assert isinstance(section_df, pd.DataFrame)
        assert not section_df.empty


class TestBoundaryConditions:
    """Tests for boundary conditions."""

    def test_simulate_single_step(self, simulation_instance):
        """Test simulation with start == stop (single iteration)."""
        df = simulation_instance.simulate_subcatchment("Area", start=5, stop=5, step=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["Area"].iloc[0] == pytest.approx(5.0)

    def test_simulate_very_small_step(self, simulation_instance):
        """Test simulation with very small step."""
        df = simulation_instance.simulate_subcatchment("Area", start=1, stop=1.01, step=0.005)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_simulate_float_precision(self, simulation_instance):
        """Test that float precision is maintained in range values."""
        df = simulation_instance.simulate_subcatchment("Area", start=0.1, stop=0.3, step=0.1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_copy_file_multiple_copies(self, simulation_instance):
        """Test creating multiple copies with different suffixes."""
        path1 = simulation_instance.copy_file(suffix="first")
        path2 = simulation_instance.copy_file(suffix="second")
        assert path1 != path2
        assert os.path.exists(path1)
        assert os.path.exists(path2)
        assert path1.endswith("_first.inp")
        assert path2.endswith("_second.inp")

    def test_simulate_with_zero_start(self, simulation_instance):
        """Test simulation with start=0."""
        df = simulation_instance.simulate_subcatchment("Width", start=0, stop=10, step=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df["Width"].iloc[0] == pytest.approx(0.0)


class TestDataFrameValidation:
    """Tests for DataFrame output validation."""

    def test_simulate_returns_correct_columns(self, simulation_instance):
        """Test that simulation returns DataFrame with correct columns."""
        df = simulation_instance.simulate_area(start=1, stop=3, step=1)
        expected_columns = {"runoff", "peak_runoff_rate", "infiltration", "evaporation", "Area"}
        assert set(df.columns) == expected_columns

    def test_simulate_returns_correct_dtypes(self, simulation_instance):
        """Test that DataFrame columns have correct data types."""
        df = simulation_instance.simulate_area(start=1, stop=3, step=1)
        for col in ["runoff", "peak_runoff_rate", "infiltration", "evaporation"]:
            assert df[col].dtype in [float, "float64"]

    def test_simulate_feature_column_matches_range(self, simulation_instance):
        """Test that feature column contains values from the specified range."""
        import numpy as np

        df = simulation_instance.simulate_subcatchment("Area", start=1, stop=5, step=2)
        expected = np.arange(1, 5 + 0.001, 2)
        assert len(df["Area"]) == len(expected)
        for actual, exp in zip(df["Area"], expected):
            assert actual == pytest.approx(exp)

    def test_result_values_non_negative(self, simulation_instance):
        """Test that simulation results are non-negative."""
        df = simulation_instance.simulate_area(start=1, stop=3, step=1)
        for col in ["runoff", "peak_runoff_rate", "infiltration", "evaporation"]:
            assert (df[col] >= 0).all(), f"Column {col} contains negative values"

    def test_manning_n_result_has_correct_length(self, simulation_instance):
        """Test that Manning's n simulation returns correct number of rows."""
        df = simulation_instance.simulate_n_imperv()
        assert len(df) == len(FeaturesSimulation.MANNING_N_VALUES)

    def test_destore_result_has_correct_length(self, simulation_instance):
        """Test that depression storage simulation returns correct number of rows."""
        df = simulation_instance.simulate_s_imperv()
        assert len(df) == len(FeaturesSimulation.DEPRESSION_STORAGE_VALUES)


class TestErrorHandling:
    """Tests for error handling."""

    def test_init_invalid_file_path(self):
        """Test that invalid file path raises an error."""
        with pytest.raises(FileNotFoundError):
            FeaturesSimulation("S1", "non_existent_file.inp")

    def test_get_section_invalid_section(self, simulation_instance):
        """Test that invalid section name raises AttributeError."""
        with pytest.raises(AttributeError):
            simulation_instance.get_section(section="invalid_section_name")

    def test_simulate_subcatchment_invalid_feature(self, simulation_instance):
        """Test simulation with invalid feature name raises an error."""
        with pytest.raises((KeyError, Exception)):
            simulation_instance.simulate_subcatchment("InvalidFeature", start=0, stop=10, step=1)


class TestStateManagement:
    """Tests for state management across simulations."""

    def test_consecutive_simulations(self, simulation_instance):
        """Test running consecutive simulations on the same instance."""
        df1 = simulation_instance.simulate_area(start=1, stop=3, step=1)
        df2 = simulation_instance.simulate_percent_impervious(start=0, stop=20, step=10)
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert "Area" in df1.columns
        assert "PercImperv" in df2.columns

    def test_mixed_simulation_types(self, simulation_instance):
        """Test mixing different simulation types."""
        df1 = simulation_instance.simulate_n_imperv()
        df2 = simulation_instance.simulate_s_perv()
        df3 = simulation_instance.simulate_width(start=10, stop=20, step=5)
        assert len(df1) == len(FeaturesSimulation.MANNING_N_VALUES)
        assert len(df2) == len(FeaturesSimulation.DEPRESSION_STORAGE_VALUES)
        assert len(df3) == 3

    def test_temp_files_tracking(self, simulation_instance):
        """Test that temp files are properly tracked."""
        initial_count = len(simulation_instance._temp_files)
        simulation_instance.copy_file(suffix="track_test")
        assert len(simulation_instance._temp_files) == initial_count + 1


class TestContextManagerExceptions:
    """Tests for context manager behavior with exceptions."""

    def test_context_manager_cleanup_on_exception(self):
        """Test that temp files are cleaned up even when exception occurs."""
        temp_files = []
        try:
            with FeaturesSimulation("S1", "catchment_simulation/example.inp") as sim:
                temp_files = sim._temp_files.copy()
                assert len(temp_files) > 0
                raise ValueError("Test exception")
        except ValueError:
            pass

        for f in temp_files:
            assert not os.path.exists(f), f"File {f} was not cleaned up after exception"

    def test_cleanup_after_multiple_simulations(self):
        """Test cleanup after running multiple simulations."""
        with FeaturesSimulation("S1", "catchment_simulation/example.inp") as sim:
            sim.simulate_area(start=1, stop=2, step=1)
            sim.simulate_width(start=10, stop=20, step=10)
            temp_files = sim._temp_files.copy()

        for f in temp_files:
            assert not os.path.exists(f)


class TestNumericalCorrectness:
    """Tests for numerical correctness."""

    def test_manning_n_values_unique(self):
        """Test that MANNING_N_VALUES contains unique values."""
        values = FeaturesSimulation.MANNING_N_VALUES
        assert len(values) == len(set(values))

    def test_depression_storage_conversion_correctness(self):
        """Test depression storage conversion from inches to mm."""
        values = FeaturesSimulation.DEPRESSION_STORAGE_VALUES
        expected_inches = [0.05, 0.1, 0.2, 0.3]
        for actual, inches in zip(values, expected_inches):
            assert actual == pytest.approx(inches * 25.4)

    def test_simulate_row_count_matches_range(self, simulation_instance):
        """Test that number of rows matches expected range count."""
        import numpy as np

        start, stop, step = 0, 50, 10
        df = simulation_instance.simulate_percent_impervious(start=start, stop=stop, step=step)
        expected_count = len(np.arange(start, stop + 0.001, step))
        assert len(df) == expected_count

    def test_manning_n_column_values_match_constant(self, simulation_instance):
        """Test that N-Imperv column contains MANNING_N_VALUES."""
        df = simulation_instance.simulate_n_imperv()
        for actual, expected in zip(df["N-Imperv"], FeaturesSimulation.MANNING_N_VALUES):
            assert actual == pytest.approx(expected)

    def test_destore_column_values_match_constant(self, simulation_instance):
        """Test that Destore column contains DEPRESSION_STORAGE_VALUES."""
        df = simulation_instance.simulate_s_imperv()
        for actual, expected in zip(
            df["Destore-Imperv"], FeaturesSimulation.DEPRESSION_STORAGE_VALUES
        ):
            assert actual == pytest.approx(expected)
