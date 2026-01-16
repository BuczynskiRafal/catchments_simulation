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
