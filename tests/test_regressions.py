"""Regression tests for the bugs fixed in 0.1.0, named after the audit findings (P-K1..P-S6).

Each test failed on 0.0.7 and pins the corrected behaviour.
"""

from __future__ import annotations

import os
import pickle
import stat
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from catchment_simulation import (
    FeaturesSimulation,
    InvalidParameterError,
    SimulationError,
    SubcatchmentNotFoundError,
    UnknownFeatureError,
)
from catchment_simulation import catchment_features_simulation as cfs_module


def write_variant(model_path: Path, target: Path, *replacements: tuple[str, str]) -> Path:
    """Write a copy of *model_path* to *target* with plain-text replacements applied."""
    text = model_path.read_text()
    for old, new in replacements:
        assert old in text, f"{old!r} not in the model"
        text = text.replace(old, new)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def move_rain_to_file(model_path: Path, target: Path, data_file: str) -> Path:
    """Copy the model to *target* with the ``test_series`` rainfall moved to an external file.

    The data file is written next to *target* and referenced by a relative path,
    the way SWMM GUI users keep rainfall records.
    """
    lines = model_path.read_text().splitlines()
    series = [line for line in lines if line.startswith("test_series")]
    kept = [line for line in lines if not line.startswith("test_series")]
    header = kept.index("[TIMESERIES]")
    kept.insert(header + 1, f'test_series FILE "{data_file}"')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(kept) + "\n")
    (target.parent / data_file).write_text(
        "".join(" ".join(line.split()[1:]) + "\n" for line in series)
    )
    return target


def files_under(directory: Path) -> set[str]:
    return {path.relative_to(directory).as_posix() for path in directory.rglob("*")}


# --- P-K1: state leaked between sweeps -----------------------------------------------------


def test_pk1_sweep_after_sweep_of_same_section_matches_fresh_instance(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", model_path)
    sim.simulate_area(1, 3, 1)
    width_after_area = sim.simulate_width(10, 20, 10)
    sim.simulate_percent_zero_imperv(0, 50, 50)
    n_imperv_after_pct_zero = sim.simulate_n_imperv()

    assert_frame_equal(
        width_after_area, FeaturesSimulation("S1", model_path).simulate_width(10, 20, 10)
    )
    assert_frame_equal(
        n_imperv_after_pct_zero, FeaturesSimulation("S1", model_path).simulate_n_imperv()
    )


def test_pk1_calculate_after_sweep_runs_the_original_model(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", model_path)
    before = sim.calculate()
    sim.simulate_area(1, 2, 1)
    assert sim.calculate() == before
    assert_frame_equal(
        sim.calculate_timeseries(), FeaturesSimulation("S1", model_path).calculate_timeseries()
    )


# --- P-K2: working file in the CWD under a fixed name --------------------------------------


def test_pk2_model_with_relative_external_rain_file_runs(model_path: Path, tmp_path: Path) -> None:
    external = move_rain_to_file(model_path, tmp_path / "external" / "example.inp", "rain.dat")
    inline_result = FeaturesSimulation("S1", model_path).calculate()

    result = FeaturesSimulation("S1", external).calculate()

    assert result == pytest.approx(inline_result)
    assert files_under(external.parent) == {"example.inp", "rain.dat"}


def test_pk2_instances_on_same_named_models_do_not_collide(
    model_path: Path, tmp_path: Path
) -> None:
    first = write_variant(model_path, tmp_path / "a" / "example.inp")
    second = write_variant(
        model_path,
        tmp_path / "b" / "example.inp",
        ("J1               5        25", "J1               2        25"),
    )
    expected_first = FeaturesSimulation("S1", first).calculate()
    expected_second = FeaturesSimulation("S1", second).calculate()
    assert expected_first["runoff"] != pytest.approx(expected_second["runoff"])

    sim_first = FeaturesSimulation("S1", first)
    sim_second = FeaturesSimulation("S1", second)

    assert sim_first.calculate() == pytest.approx(expected_first)
    assert sim_second.calculate() == pytest.approx(expected_second)


def test_pk2_nothing_is_left_behind_without_context_manager(
    model_path: Path, tmp_path: Path
) -> None:
    before = files_under(tmp_path)
    sim = FeaturesSimulation("S1", model_path)
    sim.calculate()
    sim.calculate_timeseries()
    sim.simulate_area(1, 2, 1)
    sim.simulate_subcatchment_timeseries("Area", 1, 2, 1)
    assert files_under(tmp_path) == before


def test_pk2_nothing_is_left_behind_with_context_manager(model_path: Path, tmp_path: Path) -> None:
    before = files_under(tmp_path)
    with FeaturesSimulation("S1", model_path) as sim:
        sim.simulate_n_perv()
        assert files_under(tmp_path) == before
    assert files_under(tmp_path) == before


def test_pk2_read_only_model_directory_falls_back_to_temp_dir(
    model_path: Path, request: pytest.FixtureRequest
) -> None:
    expected = FeaturesSimulation("S1", model_path).simulate_area(1, 2, 1)
    model_dir = model_path.parent
    model_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    request.addfinalizer(lambda: model_dir.chmod(stat.S_IRWXU))
    if os.access(model_dir, os.W_OK):
        pytest.skip("the model directory stays writable (root or Windows)")

    result = FeaturesSimulation("S1", model_path).simulate_area(1, 2, 1)

    assert_frame_equal(result, expected)
    assert files_under(model_dir) == {"example.inp"}


# --- P-K3: misspelled names gave silent, constant results ----------------------------------


@pytest.mark.parametrize("method", ["simulate_destore", "simulate_manning_n"])
def test_pk3_misspelled_surface_raises(model_path: Path, method: str) -> None:
    sim = FeaturesSimulation("S1", model_path)
    with pytest.raises(InvalidParameterError, match="Imperv.*Perv"):
        getattr(sim, method)("imperv")


@pytest.mark.parametrize("feature", ["Nope", "Outlet", "S-imperv"])
def test_pk3_unknown_feature_raises(model_path: Path, feature: str) -> None:
    sim = FeaturesSimulation("S1", model_path)
    with pytest.raises(UnknownFeatureError) as excinfo:
        sim.simulate(feature, [1])  # type: ignore[arg-type]
    assert excinfo.value.feature == feature
    assert "S-Imperv" in excinfo.value.allowed
    with pytest.raises(UnknownFeatureError):
        sim.simulate_subcatchment(feature, 1, 2, 1)  # type: ignore[arg-type]


# --- P-K4: the range overshot stop and drifted ---------------------------------------------


def test_pk4_range_never_overshoots_stop(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", model_path)
    assert sim.simulate_area(1, 2.9, 1)["Area"].tolist() == [1.0, 2.0]
    assert sim.simulate_percent_impervious(60, 100, 50)["PercImperv"].tolist() == [60.0]


def test_pk4_timeseries_keys_are_clean_floats(model_path: Path) -> None:
    results = FeaturesSimulation("S1", model_path).simulate_subcatchment_timeseries(
        feature="Area", start=0.1, stop=0.3, step=0.1
    )
    assert list(results) == [0.1, 0.2, 0.3]
    assert isinstance(results[0.3], pd.DataFrame)


# --- P-W4: physical bounds were never checked ----------------------------------------------


def test_pw4_out_of_bounds_values_raise_before_any_run(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sim = FeaturesSimulation("S1", model_path)

    def no_run(*args: object, **kwargs: object) -> None:
        raise AssertionError("SWMM must not run for invalid input")

    monkeypatch.setattr(cfs_module, "Simulation", no_run)
    with pytest.raises(InvalidParameterError, match="Width"):
        sim.simulate_width(0, 0, 1)
    with pytest.raises(InvalidParameterError, match="PercImperv.*110"):
        sim.simulate("PercImperv", [50, 110])
    with pytest.raises(InvalidParameterError, match="PercSlope"):
        sim.simulate_percent_slope(0, 10, 10)


# --- P-S1, P-S6: path types and SWMM failures ----------------------------------------------


def test_ps1_pathlike_model_path_is_accepted(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", model_path)
    assert sim.raw_file == str(model_path)
    assert sim.calculate()["runoff"] > 0


def test_ps6_swmm_failure_raises_simulation_error_with_report_errors(
    model_path: Path, tmp_path: Path
) -> None:
    broken = move_rain_to_file(model_path, tmp_path / "broken" / "example.inp", "rain.dat")
    (broken.parent / "rain.dat").unlink()
    sim = FeaturesSimulation("S1", broken)

    with pytest.raises(SimulationError) as excinfo:
        sim.simulate_area(1, 2, 1)

    error = excinfo.value
    assert isinstance(error, RuntimeError)
    assert error.__cause__ is not None
    assert any("ERROR 361" in line for line in error.report_errors)
    assert "ERROR 361" in str(error)
    assert files_under(broken.parent) == {"example.inp"}
    # The engine was released: the next run in this process works.
    assert FeaturesSimulation("S1", model_path).calculate()["runoff"] > 0


def test_ps6_refused_open_leaves_the_running_simulation_alone(model_path: Path) -> None:
    # pyswmm >= 2.0 refuses a second simulation; pyswmm 1.x cannot detect one.
    errors = pytest.importorskip("pyswmm.errors")
    if not hasattr(errors, "MultiSimulationError"):
        pytest.skip("pyswmm < 2.0 does not refuse a second simulation")
    from pyswmm import Simulation, Subcatchments

    expected = FeaturesSimulation("S1", model_path).calculate()["runoff"]
    other = model_path.with_name("other.inp")
    other.write_bytes(model_path.read_bytes())
    try:
        with Simulation(str(other)) as running:
            with pytest.raises(SimulationError, match="Multi-Simulation"):
                FeaturesSimulation("S1", model_path).calculate()
            # The refused open did not close the engine under the running simulation.
            for _ in running:
                pass
            assert Subcatchments(running)["S1"].statistics["runoff"] == expected
    finally:
        for suffix in (".rpt", ".out"):
            other.with_suffix(suffix).unlink(missing_ok=True)


def test_ps6_unknown_subcatchment_lists_available_ids(model_path: Path) -> None:
    with pytest.raises(SubcatchmentNotFoundError, match="Available subcatchments") as excinfo:
        FeaturesSimulation("NOPE", model_path)
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.subcatchment_id == "NOPE"
    assert excinfo.value.available == ("S1",)


@pytest.mark.parametrize(
    "error",
    [
        UnknownFeatureError("Nope", ("Area", "Width")),
        SubcatchmentNotFoundError("NOPE", ("S1",)),
        SimulationError("SWMM failed", ("ERROR 361: x",)),
    ],
)
def test_ps6_exceptions_survive_pickling(error: Exception) -> None:
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert vars(restored) == vars(error)


# --- Units: depression storage in inches for US models -------------------------------------


def test_us_unit_model_sweeps_depression_storage_in_inches(
    model_path: Path, tmp_path: Path
) -> None:
    us_model = write_variant(
        model_path, tmp_path / "us" / "example.inp", ("FLOW_UNITS           CMS", "FLOW_UNITS CFS")
    )
    sim = FeaturesSimulation("S1", us_model)
    assert sim.flow_units == "CFS"
    assert sim.simulate_s_imperv()["S-Imperv"].tolist() == [0.05, 0.1, 0.2, 0.3]
    assert FeaturesSimulation("S1", model_path).simulate_s_perv()["S-Perv"].tolist() == list(
        FeaturesSimulation.DEPRESSION_STORAGE_VALUES
    )


# --- P-S5: implementation details are deprecated, still working ----------------------------


def test_ps5_deprecated_api_warns_and_still_works(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", model_path)
    with pytest.warns(DeprecationWarning, match="copy_file"):
        copy = Path(sim.copy_file(suffix="mine"))
    assert copy == model_path.with_name("example_mine.inp")
    assert copy.read_bytes() == model_path.read_bytes()
    with pytest.warns(DeprecationWarning, match="get_section"):
        assert list(sim.get_section("subareas").index) == ["S1"]
    with pytest.warns(DeprecationWarning, match="model"):
        assert list(sim.model.inp.subcatchments.index) == ["S1"]

    sim.close()
    sim.close()
    assert not copy.exists()
    assert files_under(model_path.parent) == {"example.inp"}
