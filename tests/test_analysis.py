"""Tests for the hydrograph helpers in catchment_simulation.analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from catchment_simulation import FeaturesSimulation, runoff_volume, time_to_peak

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def timeseries(model_path: Path) -> pd.DataFrame:
    return FeaturesSimulation("S1", model_path).calculate_timeseries()


def hourly(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=len(values), freq="1h", name="datetime")
    return pd.DataFrame({"runoff": values}, index=index)


# --- time_to_peak --------------------------------------------------------------------------


def test_time_to_peak_on_simulated_hydrograph(timeseries: pd.DataFrame) -> None:
    result = time_to_peak(timeseries)
    assert isinstance(result, pd.Timedelta)
    assert pd.Timedelta(0) <= result <= timeseries.index[-1] - timeseries.index[0]
    assert isinstance(time_to_peak(timeseries, column="rainfall"), pd.Timedelta)


def test_time_to_peak_numerical_correctness() -> None:
    assert time_to_peak(hourly([0.0, 1.0, 3.0, 0.5])) == pd.Timedelta(hours=2)
    # NaN is skipped and the first of equal peaks counts.
    assert time_to_peak(hourly([np.nan, 1.0, 3.0, 3.0, np.nan])) == pd.Timedelta(hours=2)


def test_time_to_peak_from_explicit_start() -> None:
    frame = hourly([0.0, 1.0, 3.0, 0.5])
    start = frame.index[0] - pd.Timedelta(minutes=5)
    assert time_to_peak(frame, start=start) == pd.Timedelta(hours=2, minutes=5)
    assert time_to_peak(frame, start=frame.index[0].to_pydatetime()) == pd.Timedelta(hours=2)


def test_time_to_peak_rejects_start_after_first_timestamp() -> None:
    frame = hourly([0.0, 3.0])
    with pytest.raises(ValueError, match="later than the first timestamp"):
        time_to_peak(frame, start=frame.index[1])


@pytest.mark.parametrize(
    ("frame", "column", "match"),
    [
        (pd.DataFrame(columns=["runoff"]), "runoff", "empty"),
        (hourly([1.0]), "nonexistent", "not found"),
        (hourly([0.0, 0.0, 0.0]), "runoff", "zero"),
        (hourly([np.nan, np.nan]), "runoff", "NaN"),
        (pd.DataFrame({"runoff": [1.0, 3.0]}), "runoff", "DatetimeIndex"),
    ],
)
def test_time_to_peak_rejects_degenerate_input(
    frame: pd.DataFrame, column: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        time_to_peak(frame, column=column)


def test_time_to_peak_rejects_unsorted_index() -> None:
    # Reversed, the first timestamp is the latest: the peak would lie before it.
    with pytest.raises(ValueError, match="sorted"):
        time_to_peak(hourly([1.0, 3.0, 2.0]).iloc[::-1])


# --- runoff_volume -------------------------------------------------------------------------


def test_runoff_volume_on_simulated_hydrograph(timeseries: pd.DataFrame) -> None:
    total = runoff_volume(timeseries)
    assert isinstance(total, float)
    assert total > 0
    middle = timeseries.index[len(timeseries) // 2]
    quarter, three_quarters = timeseries.index[[len(timeseries) // 4, 3 * len(timeseries) // 4]]
    assert runoff_volume(timeseries, end=middle) <= total
    assert runoff_volume(timeseries, start=quarter, end=three_quarters) <= total
    assert runoff_volume(timeseries, column="infiltration_loss") >= 0


def test_runoff_volume_numerical_correctness() -> None:
    index = pd.date_range("2022-01-01", periods=3, freq="50s", name="datetime")
    frame = pd.DataFrame({"runoff": [2.0, 2.0, 2.0]}, index=index)
    assert runoff_volume(frame) == pytest.approx(200.0)


def test_runoff_volume_rejects_non_datetime_index() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        runoff_volume(pd.DataFrame({"runoff": [1.0, 2.0]}))


def test_runoff_volume_rejects_unsorted_index() -> None:
    with pytest.raises(ValueError, match="sorted"):
        runoff_volume(hourly([1.0, 2.0, 3.0]).iloc[::-1])


@pytest.mark.parametrize(
    ("frame", "column", "match"),
    [
        (hourly([1.0]), "runoff", "2 data points"),
        (hourly([1.0, 2.0]), "nonexistent", "not found"),
    ],
)
def test_runoff_volume_rejects_degenerate_input(
    frame: pd.DataFrame, column: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        runoff_volume(frame, column=column)
