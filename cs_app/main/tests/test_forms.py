"""The sweep forms accept exactly the ranges catchment_simulation will run.

Bounds come from ``catchment_simulation.features.FEATURES`` and the run count from
``build_range``, so a range the form accepts never fails in the package.
"""

from typing import get_args

import pytest
from django.urls import reverse

from catchment_simulation.features import FEATURES, MAX_RUNS, build_range
from catchment_simulation.schemas import SimulationMethodParams
from main.forms import (
    MAX_SWEEP_RUNS,
    METHOD_FEATURES,
    STOP_BEFORE_START_MESSAGE,
    TOO_MANY_RUNS_MESSAGE,
    SimulationForm,
    TimeseriesForm,
)

CATCHMENTS = [("S1", "S1")]


def _simulation_form(option: str, start, stop, step) -> SimulationForm:
    data = {"option": option, "start": start, "stop": stop, "step": step, "catchment_name": "S1"}
    return SimulationForm(data=data, catchment_choices=CATCHMENTS)


def _timeseries_form(feature: str, start, stop, step) -> TimeseriesForm:
    data = {
        "mode": "sweep",
        "feature": feature,
        "start": start,
        "stop": stop,
        "step": step,
        "catchment_name": "S1",
    }
    return TimeseriesForm(data=data, catchment_choices=CATCHMENTS)


def test_method_features_cover_every_simulation_method():
    methods = get_args(SimulationMethodParams.model_fields["method_name"].annotation)
    assert set(METHOD_FEATURES) == set(methods)
    assert set(METHOD_FEATURES.values()) == set(FEATURES)
    offered = {value for _, group in SimulationForm.OPTIONS for value, _ in group}
    assert offered == set(METHOD_FEATURES)
    assert SimulationForm.PREDEFINED_METHODS == SimulationMethodParams.PREDEFINED_METHODS


def test_timeseries_features_are_package_features():
    assert {value for value, _ in TimeseriesForm.FEATURE_CHOICES} <= set(FEATURES)


def test_run_limit_stays_within_the_package_limit():
    assert MAX_SWEEP_RUNS <= MAX_RUNS
    assert SimulationForm.MAX_SWEEP_RUNS == TimeseriesForm.MAX_SWEEP_RUNS == MAX_SWEEP_RUNS


SLOPE_RANGE = "Slope (%) must be greater than 0 and at most 100."
PERCENT_RANGE = "{} (%) must be between 0 and 100."


@pytest.mark.parametrize(
    ("option", "start", "stop", "step", "invalid_field", "message"),
    [
        ("simulate_percent_slope", 0, 10, 5, "start", SLOPE_RANGE),
        ("simulate_percent_slope", 10, 110, 10, "stop", SLOPE_RANGE),
        ("simulate_area", 0, 5, 1, "start", "Area (ha) must be greater than 0."),
        ("simulate_width", 0, 100, 10, "start", "Width (m) must be greater than 0."),
        ("simulate_percent_impervious", 50, 150, 50, "stop", PERCENT_RANGE.format("Impervious")),
        ("simulate_percent_zero_imperv", 0, 200, 100, "stop", PERCENT_RANGE.format("Zero-Imperv")),
    ],
)
def test_simulation_form_rejects_values_outside_the_physical_range(
    option, start, stop, step, invalid_field, message
):
    form = _simulation_form(option, start, stop, step)

    assert not form.is_valid()
    assert form.errors == {invalid_field: [message]}


@pytest.mark.parametrize(
    ("option", "start", "stop", "step"),
    [
        ("simulate_percent_slope", 1, 100, 1),
        ("simulate_area", 1, 10, 1),
        ("simulate_width", 10, 100, 10),
        ("simulate_percent_impervious", 0, 100, 10),
        ("simulate_percent_zero_imperv", 0, 100, 10),
        ("simulate_curb_length", 0, 100, 10),
        # The package never runs past the last value on the grid (0 and 100), so the stop may be.
        ("simulate_percent_impervious", 0, 150, 100),
    ],
)
def test_simulation_form_accepts_physical_ranges(option, start, stop, step):
    form = _simulation_form(option, start, stop, step)

    assert form.is_valid(), form.errors
    spec = FEATURES[METHOD_FEATURES[option]]
    assert all(spec.contains(value) for value in build_range(start, stop, step))


def test_simulation_form_counts_runs_like_the_package():
    assert _simulation_form("simulate_curb_length", 1, 100, 1).is_valid()

    form = _simulation_form("simulate_curb_length", 1, 101, 1)

    assert not form.is_valid()
    assert form.errors == {"step": [TOO_MANY_RUNS_MESSAGE]}


def test_simulation_form_reports_a_reversed_range_once():
    """A reversed range gets its own message, not a run-count or bounds error on top."""
    form = _simulation_form("simulate_percent_slope", 200, 10, 1)

    assert form.errors == {"stop": [STOP_BEFORE_START_MESSAGE]}


@pytest.mark.parametrize(
    ("feature", "start", "stop", "step", "invalid_field"),
    [
        ("PercSlope", 0, 100, 10, "start"),
        ("Area", 0, 5, 1, "start"),
        ("Width", 0, 50, 10, "start"),
        ("PercImperv", 50, 150, 50, "stop"),
    ],
)
def test_timeseries_form_rejects_values_outside_the_physical_range(
    feature, start, stop, step, invalid_field
):
    form = _timeseries_form(feature, start, stop, step)

    assert not form.is_valid()
    assert list(form.errors) == [invalid_field]


def test_timeseries_form_counts_float_runs_like_the_package():
    assert len(build_range(0, 9.9, 0.1)) == 100
    assert _timeseries_form("PercImperv", 0, 9.9, 0.1).is_valid()

    form = _timeseries_form("PercImperv", 0, 10, 0.1)

    assert not form.is_valid()
    assert form.errors == {"step": [TOO_MANY_RUNS_MESSAGE]}


def test_form_defaults_are_runnable():
    """The values a fresh form shows must pass its own validation."""
    simulation = SimulationForm(catchment_choices=CATCHMENTS)
    first_option = SimulationForm.OPTIONS[0][1][0][0]
    simulation_data = {
        "option": first_option,
        **{field: simulation.fields[field].initial for field in ("start", "stop", "step")},
        "catchment_name": "S1",
    }
    assert SimulationForm(data=simulation_data, catchment_choices=CATCHMENTS).is_valid()

    timeseries = TimeseriesForm(catchment_choices=CATCHMENTS)
    timeseries_data = {
        "mode": "sweep",
        **{
            field: timeseries.fields[field].initial
            for field in ("feature", "start", "stop", "step")
        },
        "catchment_name": "S1",
    }
    form = TimeseriesForm(data=timeseries_data, catchment_choices=CATCHMENTS)
    assert form.is_valid(), form.errors


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation", "main:timeseries"])
def test_pages_hand_the_run_limit_to_the_live_hint(client, user, url_name):
    client.force_login(user)

    response = client.get(reverse(url_name))

    assert f'data-max-runs="{MAX_SWEEP_RUNS}"'.encode() in response.content
