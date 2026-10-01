"""Tests for the public contract of FeaturesSimulation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from catchment_simulation import (
    Feature,
    FeaturesSimulation,
    InvalidParameterError,
    SubcatchmentNotFoundError,
    UnknownFeatureError,
    build_range,
)
from catchment_simulation.features import FEATURES

if TYPE_CHECKING:
    from pathlib import Path

RESULT_COLUMNS = list(FeaturesSimulation.RESULT_KEYS)


@pytest.fixture
def sim(model_path: Path) -> FeaturesSimulation:
    return FeaturesSimulation("S1", model_path)


# --- construction --------------------------------------------------------------------------


def test_init_keeps_arguments_and_reads_flow_units(model_path: Path) -> None:
    sim = FeaturesSimulation("S1", str(model_path))
    assert sim.subcatchment_id == "S1"
    assert sim.raw_file == str(model_path)
    assert sim.flow_units == "CMS"


def test_init_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FeaturesSimulation("S1", tmp_path / "missing.inp")


@pytest.mark.parametrize("subcatchment_id", ["NONEXISTENT", ""])
def test_init_unknown_subcatchment_raises(model_path: Path, subcatchment_id: str) -> None:
    with pytest.raises(SubcatchmentNotFoundError, match=r"Available subcatchments: \['S1'\]"):
        FeaturesSimulation(subcatchment_id, model_path)


def test_flow_units_default_to_cfs_when_unset(model_path: Path) -> None:
    text = model_path.read_text().replace("FLOW_UNITS           CMS\n", "")
    model_path.write_text(text)
    assert FeaturesSimulation("S1", model_path).flow_units == "CFS"


def test_context_manager_and_close(model_path: Path) -> None:
    with FeaturesSimulation("S1", model_path) as sim:
        assert isinstance(sim, FeaturesSimulation)
    sim.close()
    assert sim.calculate()["runoff"] > 0


# --- single runs ---------------------------------------------------------------------------


def test_calculate_returns_all_statistics(sim: FeaturesSimulation) -> None:
    stats = sim.calculate()
    assert set(stats) == set(FeaturesSimulation.RESULT_KEYS)
    assert all(value >= 0 for value in stats.values())
    assert stats["runoff"] > 0
    assert stats["precipitation"] == pytest.approx(97.0)


def test_sweeping_to_the_model_value_reproduces_calculate(sim: FeaturesSimulation) -> None:
    row = sim.simulate("Area", [5]).iloc[0]
    stats = sim.calculate()
    assert row[RESULT_COLUMNS].tolist() == [stats[key] for key in RESULT_COLUMNS]


class TestCalculateTimeseries:
    @pytest.fixture
    def timeseries(self, sim: FeaturesSimulation) -> pd.DataFrame:
        return sim.calculate_timeseries()

    def test_shape(self, timeseries: pd.DataFrame) -> None:
        assert isinstance(timeseries.index, pd.DatetimeIndex)
        assert timeseries.index.name == "datetime"
        assert list(timeseries.columns) == list(FeaturesSimulation.TIMESERIES_KEYS)
        assert not timeseries.empty
        assert timeseries.index.is_monotonic_increasing

    def test_values(self, timeseries: pd.DataFrame) -> None:
        assert (timeseries >= 0).all().all()
        assert (timeseries["runoff"] > 0).any()

    def test_timestamps_within_simulation_period(self, timeseries: pd.DataFrame) -> None:
        assert timeseries.index.min() > datetime(2022, 6, 17, 0, 0)
        assert timeseries.index.max() <= datetime(2022, 6, 17, 12, 0)


# --- sweeps --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "feature", "defaults"),
    [
        ("simulate_area", "Area", (1, 10, 1)),
        ("simulate_percent_impervious", "PercImperv", (0, 100, 10)),
        ("simulate_percent_slope", "PercSlope", (10, 100, 10)),
        ("simulate_width", "Width", (10, 100, 10)),
        ("simulate_curb_length", "CurbLength", (0, 100, 10)),
        ("simulate_percent_zero_imperv", "PctZero", (0, 100, 10)),
    ],
)
def test_range_methods_default_sweep(
    sim: FeaturesSimulation, method: str, feature: str, defaults: tuple[float, float, float]
) -> None:
    frame = getattr(sim, method)()

    assert list(frame.columns) == [feature, *RESULT_COLUMNS]
    assert frame[feature].tolist() == build_range(*defaults)
    assert all(frame[column].dtype == np.float64 for column in frame.columns)
    assert (frame[RESULT_COLUMNS] >= 0).all().all()


@pytest.mark.parametrize(
    ("method", "feature", "values"),
    [
        ("simulate_n_imperv", "N-Imperv", FeaturesSimulation.MANNING_N_VALUES),
        ("simulate_n_perv", "N-Perv", FeaturesSimulation.MANNING_N_VALUES),
        ("simulate_s_imperv", "S-Imperv", FeaturesSimulation.DEPRESSION_STORAGE_VALUES),
        ("simulate_s_perv", "S-Perv", FeaturesSimulation.DEPRESSION_STORAGE_VALUES),
    ],
)
def test_literature_value_methods(
    sim: FeaturesSimulation, method: str, feature: str, values: tuple[float, ...]
) -> None:
    frame = getattr(sim, method)()
    assert list(frame.columns) == [feature, *RESULT_COLUMNS]
    assert frame[feature].tolist() == list(values)


def test_surface_methods_match_their_shortcuts(sim: FeaturesSimulation) -> None:
    pd.testing.assert_frame_equal(sim.simulate_manning_n("Perv"), sim.simulate_n_perv())
    pd.testing.assert_frame_equal(sim.simulate_destore("Imperv"), sim.simulate_s_imperv())


# One low and one high physically valid value per feature.
SWEEP_VALUES: dict[Feature, tuple[float, float]] = {
    "Area": (2, 8),
    "PercImperv": (10, 90),
    "Width": (20, 400),
    "PercSlope": (0.5, 30),
    "CurbLength": (0, 500),
    "N-Imperv": (0.011, 0.4),
    "N-Perv": (0.05, 0.8),
    "S-Imperv": (0, 7.62),
    "S-Perv": (0, 7.62),
    "PctZero": (0, 100),
}


@pytest.mark.parametrize("feature", list(FEATURES))
def test_every_feature_reaches_swmm(sim: FeaturesSimulation, feature: Feature) -> None:
    frame = sim.simulate(feature, SWEEP_VALUES[feature])
    assert frame[feature].tolist() == list(SWEEP_VALUES[feature])
    if feature == "CurbLength":
        # Curb length only drives pollutant build-up, so runoff must not change.
        assert frame["runoff"].iloc[0] == frame["runoff"].iloc[1]
    else:
        assert frame["runoff"].iloc[0] != pytest.approx(frame["runoff"].iloc[1])


def test_simulate_subcatchment_accepts_subarea_features(sim: FeaturesSimulation) -> None:
    frame = sim.simulate_subcatchment("N-Perv", 0.1, 0.3, 0.1)
    assert frame["N-Perv"].tolist() == [0.1, 0.2, 0.3]


def test_simulate_accepts_any_iterable_in_given_order(sim: FeaturesSimulation) -> None:
    from_generator = sim.simulate("Width", (w for w in (300.0, 30.0)))
    from_array = sim.simulate("Width", np.array([300, 30]))
    assert from_generator["Width"].tolist() == [300.0, 30.0]
    pd.testing.assert_frame_equal(from_generator, from_array)
    assert from_generator["runoff"].iloc[0] > from_generator["runoff"].iloc[1]


@pytest.mark.parametrize("values", [[], [float("nan")], [-1.0]])
def test_simulate_rejects_bad_values(sim: FeaturesSimulation, values: list[float]) -> None:
    with pytest.raises(InvalidParameterError, match="Area"):
        sim.simulate("Area", values)


@pytest.mark.parametrize(
    ("start", "stop", "step", "match"),
    [
        (-1, 10, 1, "start"),
        (0, 10, 0, "step"),
        (0, 10, -1, "step"),
        (100, 10, 1, "start .* must be <= stop"),
        (1, 5000, 1, "max_runs"),
    ],
)
def test_range_methods_reject_invalid_ranges(
    sim: FeaturesSimulation, start: float, stop: float, step: float, match: str
) -> None:
    with pytest.raises(InvalidParameterError, match=match):
        sim.simulate_area(start, stop, step)
    with pytest.raises(InvalidParameterError, match=match):
        sim.simulate_subcatchment_timeseries("Area", start, stop, step)


def test_unknown_feature_in_timeseries_sweep(sim: FeaturesSimulation) -> None:
    with pytest.raises(UnknownFeatureError):
        sim.simulate_timeseries("Outlet", [1])  # type: ignore[arg-type]


class TestTimeseriesSweeps:
    def test_simulate_subcatchment_timeseries(self, sim: FeaturesSimulation) -> None:
        results = sim.simulate_subcatchment_timeseries(feature="Area", start=1, stop=3, step=1)

        assert list(results) == [1.0, 2.0, 3.0]
        assert len({len(frame) for frame in results.values()}) == 1
        for frame in results.values():
            assert isinstance(frame.index, pd.DatetimeIndex)
            assert frame.index.name == "datetime"
            assert list(frame.columns) == list(FeaturesSimulation.TIMESERIES_KEYS)
        assert results[3.0]["runoff"].sum() > results[1.0]["runoff"].sum()

    def test_single_value(self, sim: FeaturesSimulation) -> None:
        results = sim.simulate_subcatchment_timeseries("Area", start=5, stop=5, step=1)
        assert list(results) == [5.0]
        pd.testing.assert_frame_equal(results[5.0], sim.calculate_timeseries())

    def test_simulate_timeseries_with_values(self, sim: FeaturesSimulation) -> None:
        results = sim.simulate_timeseries("PctZero", [100, 0])
        assert list(results) == [100.0, 0.0]


# --- class constants -----------------------------------------------------------------------


def test_result_and_timeseries_keys() -> None:
    assert FeaturesSimulation.RESULT_KEYS == (
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "precipitation",
        "runon",
    )
    assert FeaturesSimulation.TIMESERIES_KEYS == (
        "rainfall",
        "runoff",
        "infiltration_loss",
        "evaporation_loss",
        "runon",
    )
    assert FeaturesSimulation.SWMM_SIDE_EXTENSIONS == (".inp", ".rpt", ".out")


def test_manning_n_values() -> None:
    labelled = FeaturesSimulation.MANNING_N
    values = FeaturesSimulation.MANNING_N_VALUES
    assert len(labelled) == 15
    assert values == tuple(labelled.values())
    assert list(values) == sorted(set(values))
    assert labelled["smooth asphalt"] == 0.011
    assert labelled["woods, dense underbrush"] == 0.8


def test_depression_storage_values() -> None:
    inches = FeaturesSimulation.DEPRESSION_STORAGE_VALUES_IN
    assert inches == (0.05, 0.1, 0.2, 0.3)
    millimetres = FeaturesSimulation.DEPRESSION_STORAGE_VALUES
    assert millimetres == (1.27, 2.54, 5.08, 7.62)
    assert millimetres == pytest.approx([value * 25.4 for value in inches])
