import pytest
import os
import swmmio
import pandas as pd
from .catchment_features_simulation import FeaturesSimulation


@pytest.fixture
def simulation_instance():
    """
    Returns a swmmio.Simulation instance.
    """
    subcatchment_id = "S1"
    raw_file = "catchment_simulation/example.inp"
    return FeaturesSimulation(subcatchment_id, raw_file)


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
    df = simulation_instance.simulate_subcatchment(
        feature=feature, start=1, stop=10, step=1
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        feature,
    }


def test_simulate_area(simulation_instance):
    """
    Test the simulate_area method.
    """
    df = simulation_instance.simulate_area(start=1, stop=10, step=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Area",
    }


def test_simulate_percent_impervious(simulation_instance):
    """
    Test the simulate_percent_impervious method.
    """
    df = simulation_instance.simulate_percent_impervious(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "PercImperv",
    }


def test_simulate_percent_slope(simulation_instance):
    """
    Test the simulate_percent_slope method.
    """
    df = simulation_instance.simulate_percent_slope(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "PercSlope",
    }


def test_simulate_width(simulation_instance):
    """
    Test the simulate_width method.
    """
    df = simulation_instance.simulate_width(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Width",
    }


def test_simulate_curb_length(simulation_instance):
    """
    Test the simulate_curb_length method.
    """
    df = simulation_instance.simulate_curb_length(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "CurbLength",
    }


def test_simulate_n_imperv(simulation_instance):
    """
    Test the simulate_n_imperv method.
    """
    df = simulation_instance.simulate_n_imperv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "N-Imperv",
    }


def test_simulate_n_perv(simulation_instance):
    """
    Test the simulate_n_perv method.
    """
    df = simulation_instance.simulate_n_perv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "N-Perv",
    }


def test_simulate_destore_imperv(simulation_instance):
    """
    Test the simulate_destore_imperv method.
    """
    df = simulation_instance.simulate_destore(param="Imperv")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Destore-Imperv",
    }


def test_simulate_destore_perv(simulation_instance):
    """
    Test the simulate_destore_perv method.
    """
    df = simulation_instance.simulate_destore(param="Perv")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Destore-Perv",
    }


def test_simulate_s_imperv(simulation_instance):
    """
    Test the simulate_s_imperv method.
    """
    df = simulation_instance.simulate_s_imperv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Destore-Imperv",
    }


def test_simulate_s_perv(simulation_instance):
    """
    Test the simulate_s_perv method.
    """
    df = simulation_instance.simulate_s_perv()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Destore-Perv",
    }


def test_simulate_percent_zero_imperv(simulation_instance):
    """
    Test the simulate_percent_zero_imperv method.
    """
    df = simulation_instance.simulate_percent_zero_imperv(start=0, stop=100, step=10)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "Zero-Imperv",
    }
