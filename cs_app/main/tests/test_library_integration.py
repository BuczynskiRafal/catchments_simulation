"""The run views on the real catchment_simulation package and its typed errors.

Stubs elsewhere keep the view tests fast; these run SWMM through the views, so a
change in the package's result layout (column names, order) fails here instead of
in production. Every model is a copy in the user's upload directory, which the
autouse ``isolated_media_root`` fixture puts in ``tmp_path``: the views run only
models from there, and SWMM writes its report and output next to the model.
"""

import json
import os
from io import BytesIO
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
from django.core.cache import cache
from django.urls import reverse

from catchment_simulation import (
    FeaturesSimulation,
    InvalidParameterError,
    SimulationError,
    SubcatchmentNotFoundError,
    UnknownFeatureError,
    example_model_path,
)
from catchment_simulation.features import FEATURES
from catchment_simulation.schemas import SimulationMethodParams, SimulationParams
from main.views import (
    NO_RUNOFF_SUMMARY_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
    SIMULATION_OUTPUT_COLUMNS,
    SUBCATCHMENT_NOT_FOUND_MESSAGE,
    TS_RESULT_TOKEN_SESSION_KEY,
    _result_cache_key,
    _user_upload_dir,
    get_feature_name,
)

AJAX = {"HTTP_X_REQUESTED_WITH": "XMLHttpRequest"}
SAMPLE_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "data", "example.inp")
# The example model's only subcatchment uses rain gage "1"; an unknown gage fails SWMM's input check.
BROKEN_GAGE_LINE = ("S1               1                J1", "S1               RGX              J1")


def _user_model_file(user, name: str) -> Path:
    path = Path(_user_upload_dir(user.id)) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def example_model(user):
    path = _user_model_file(user, "example.inp")
    path.write_bytes(example_model_path().read_bytes())
    return path


@pytest.fixture
def broken_model(user):
    path = _user_model_file(user, "broken.inp")
    text = example_model_path().read_text()
    assert BROKEN_GAGE_LINE[0] in text
    path.write_text(text.replace(*BROKEN_GAGE_LINE))
    return path


def _login_with_model(client, user, model_path, subcatchments=("S1",)):
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = str(model_path)
    session["_subcatchment_ids_file"] = str(model_path)
    session["_subcatchment_ids"] = list(subcatchments)
    session.save()
    return client


def _cached_payload(client, user, scope, session_key):
    return json.loads(cache.get(_result_cache_key(scope, user.id, client.session[session_key])))


# ---------------------------------------------------------------------------
# Contracts between the app and the package
# ---------------------------------------------------------------------------


def test_sample_model_is_the_package_example_model():
    """The home page snippets run example_model_path(); its charts come from the site's copy."""
    with open(SAMPLE_MODEL, "rb") as sample:
        assert sample.read() == example_model_path().read_bytes()


def test_simulation_outputs_are_package_results():
    keys = FeaturesSimulation.RESULT_KEYS
    assert set(SIMULATION_OUTPUT_COLUMNS) <= set(keys)
    assert list(SIMULATION_OUTPUT_COLUMNS) == [k for k in keys if k in SIMULATION_OUTPUT_COLUMNS]


def test_every_simulation_method_names_a_package_feature():
    for method in get_args(SimulationMethodParams.model_fields["method_name"].annotation):
        assert get_feature_name(method) in FEATURES
        assert callable(getattr(FeaturesSimulation, method))


# ---------------------------------------------------------------------------
# Real SWMM runs through the views
# ---------------------------------------------------------------------------

STORAGE_MM = list(FeaturesSimulation.DEPRESSION_STORAGE_VALUES)
MANNING_N = list(FeaturesSimulation.MANNING_N_VALUES)

REAL_RUNS = [
    # (method, range, swept values, x-axis label)
    ("simulate_percent_zero_imperv", (0, 100, 50), [0, 50, 100], "Zero-Impervious Area [%]"),
    ("simulate_s_imperv", None, STORAGE_MM, "Depression Storage (Impervious) [mm]"),
    ("simulate_s_perv", None, STORAGE_MM, "Depression Storage (Pervious) [mm]"),
    ("simulate_percent_slope", (10, 20, 10), [10, 20], "Percent Slope [%]"),
    ("simulate_area", (1, 2, 1), [1, 2], "Area [ha]"),
    ("simulate_width", (10, 20, 10), [10, 20], "Width [m]"),
    ("simulate_percent_impervious", (0, 100, 50), [0, 50, 100], "Impervious Area [%]"),
    ("simulate_curb_length", (0, 10, 10), [0, 10], "Curb Length [m]"),
    ("simulate_n_imperv", None, MANNING_N, "Manning n (Impervious) [-]"),
    ("simulate_n_perv", None, MANNING_N, "Manning n (Pervious) [-]"),
]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("method", "sweep", "values", "x_label"), REAL_RUNS, ids=[run[0] for run in REAL_RUNS]
)
def test_simulation_view_runs_the_real_package(
    client, user, example_model, method, sweep, values, x_label
):
    _login_with_model(client, user, example_model)
    data = {"option": method, "catchment_name": "S1"}
    if sweep:
        data.update(zip(("start", "stop", "step"), map(str, sweep), strict=True))

    response = client.post(reverse("main:simulation"), data=data)

    assert response.status_code == 302, response.content.decode()[:500]
    payload = _cached_payload(client, user, "sim", SIM_RESULT_TOKEN_SESSION_KEY)
    feature = get_feature_name(method)
    columns = [feature, *SIMULATION_OUTPUT_COLUMNS]
    assert payload["results_columns"] == columns
    assert [row[0] for row in payload["results_data"]] == pytest.approx(values)
    assert all(len(row) == len(columns) for row in payload["results_data"])
    chart = payload["chart_config"]
    assert (chart["x"], chart["y"], chart["xLabel"]) == (feature, columns[1:], x_label)
    assert list(chart["yLabels"]) == columns[1:]
    assert [record[feature] for record in chart["data"]] == pytest.approx(values)
    # Nothing but the model is left next to it: the package cleans up after every call.
    assert os.listdir(example_model.parent) == [example_model.name]


@pytest.mark.django_db
def test_simulation_download_has_the_columns_of_the_table(client, user, example_model):
    _login_with_model(client, user, example_model)
    client.post(
        reverse("main:simulation"),
        data={
            "option": "simulate_percent_zero_imperv",
            "start": "0",
            "stop": "100",
            "step": "50",
            "catchment_name": "S1",
        },
    )
    token = client.session[SIM_RESULT_TOKEN_SESSION_KEY]

    response = client.post(reverse("main:download_simulation_results"), data={"token": token})

    sheet = pd.read_excel(BytesIO(b"".join(response.streaming_content)), sheet_name="results")
    assert list(sheet.columns) == ["PctZero", *SIMULATION_OUTPUT_COLUMNS]
    assert sheet["PctZero"].tolist() == [0, 50, 100]


@pytest.mark.django_db
def test_timeseries_sweep_runs_the_real_package(client, user, example_model):
    _login_with_model(client, user, example_model)

    response = client.post(
        reverse("main:timeseries"),
        data={
            "mode": "sweep",
            "feature": "PercImperv",
            "start": "0",
            "stop": "50",
            "step": "25",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 302
    payload = _cached_payload(client, user, "ts", TS_RESULT_TOKEN_SESSION_KEY)
    assert list(payload["data"]) == ["0", "25", "50"]
    assert [run["value"] for run in payload["sweep_summary"]] == ["0", "25", "50"]
    first_row = payload["data"]["0"][0]
    assert set(first_row) == {"datetime", *FeaturesSimulation.TIMESERIES_KEYS}
    assert os.listdir(example_model.parent) == [example_model.name]


@pytest.mark.django_db
def test_unknown_subcatchment_is_reported_on_the_catchment_field(client, user, example_model):
    """The package raises SubcatchmentNotFoundError; the page names the field, not a crash."""
    _login_with_model(client, user, example_model, subcatchments=("S1", "S9"))
    data = {"mode": "single", "catchment_name": "S9"}

    response = client.post(reverse("main:timeseries"), data=data, **AJAX)

    assert response.status_code == 400
    assert response.json() == {
        "message": SUBCATCHMENT_NOT_FOUND_MESSAGE,
        "field_errors": {"catchment_name": [SUBCATCHMENT_NOT_FOUND_MESSAGE]},
    }

    page = client.post(reverse("main:timeseries"), data=data).content.decode()

    assert 'id="id_catchment_name"' in page
    assert "is-invalid" in page
    assert page.count(SUBCATCHMENT_NOT_FOUND_MESSAGE) == 2, "flash message and field error"


@pytest.mark.django_db
def test_model_swmm_cannot_run_shows_the_report_errors(client, user, broken_model):
    _login_with_model(client, user, broken_model)

    response = client.post(
        reverse("main:simulation"),
        data={
            "option": "simulate_percent_impervious",
            "start": "0",
            "stop": "50",
            "step": "50",
            "catchment_name": "S1",
        },
        **AJAX,
    )

    assert response.status_code == 400
    message = response.json()["message"]
    assert message.startswith("SWMM could not run the model: ERROR 209: undefined object RGX")
    assert str(broken_model.parent) not in message
    assert os.listdir(broken_model.parent) == [broken_model.name]


@pytest.mark.django_db
def test_failed_comparison_does_not_block_the_next_one(client, user):
    """
    A model SWMM cannot open must not leave the engine open: the report's errors reach
    the user, and the next comparison in the same process runs.
    """
    user_dir = _user_upload_dir(user.id)
    os.makedirs(user_dir)
    text = example_model_path().read_text()
    broken = os.path.join(user_dir, "broken.inp")
    with open(broken, "w") as model_file:
        model_file.write(text.replace(*BROKEN_GAGE_LINE))
    good = os.path.join(user_dir, "example.inp")
    with open(good, "w") as model_file:
        model_file.write(text)

    _login_with_model(client, user, broken)
    failed = client.post(reverse("main:calculations"), **AJAX)

    assert failed.status_code == 400
    assert "ERROR 209: undefined object RGX" in failed.json()["message"]

    _login_with_model(client, user, good)
    succeeded = client.post(reverse("main:calculations"), **AJAX)

    assert succeeded.status_code == 200
    assert "calculations-chart-data" in succeeded.content.decode()
    assert sorted(os.listdir(user_dir)) == ["broken.inp", "example.inp"]


@pytest.mark.django_db
def test_comparison_of_a_model_without_runoff_is_a_user_error(client, user):
    """
    With IGNORE_RAINFALL the model runs, but SWMM computes no runoff and its report has
    no Subcatchment Runoff Summary: that is the model's problem (400), not a crash.
    """
    user_dir = _user_upload_dir(user.id)
    os.makedirs(user_dir)
    text = example_model_path().read_text()
    ponding = "ALLOW_PONDING        NO\n"
    assert ponding in text
    model = os.path.join(user_dir, "no_rain.inp")
    with open(model, "w") as model_file:
        model_file.write(text.replace(ponding, ponding + "IGNORE_RAINFALL      YES\n"))

    _login_with_model(client, user, model)
    response = client.post(reverse("main:calculations"), **AJAX)

    assert response.status_code == 400
    assert response.json()["message"] == NO_RUNOFF_SUMMARY_MESSAGE
    assert os.listdir(user_dir) == ["no_rain.inp"]


@pytest.mark.django_db
def test_comparison_of_a_model_without_subareas_is_a_user_error(client, user):
    """A model lacking [SUBAREAS] columns returns HTTP 400 instead of an unhandled KeyError (500)."""
    user_dir = _user_upload_dir(user.id)
    os.makedirs(user_dir)
    text = example_model_path().read_text()
    subareas_header = "[SUBAREAS]"
    infiltration_header = "[INFILTRATION]"
    assert subareas_header in text
    assert infiltration_header in text
    stripped = text[: text.index(subareas_header)] + text[text.index(infiltration_header) :]
    model = os.path.join(user_dir, "no_subareas.inp")
    with open(model, "w") as model_file:
        model_file.write(stripped)

    _login_with_model(client, user, model)
    response = client.post(reverse("main:calculations"), **AJAX)

    assert response.status_code == 400
    assert (
        response.json()["message"]
        == "The model is missing required [SUBCATCHMENTS] or [SUBAREAS] columns."
    )
    assert os.listdir(user_dir) == ["no_subareas.inp"]


@pytest.mark.django_db
def test_comparison_converts_us_units_for_ann_predictions(client, user):
    """
    US models convert Width (ft -> m) and S-Imperv/S-Perv (in -> mm) before MLP normalization,
    and scale the ANN output (10^3 m³) to ft³ rather than treating it as 10^6 US gal.
    """
    from main.views import _M3_TO_FT3, CALC_RESULT_TOKEN_SESSION_KEY

    user_dir = _user_upload_dir(user.id)
    os.makedirs(user_dir)
    si_text = example_model_path().read_text()
    si_path = os.path.join(user_dir, "si.inp")
    with open(si_path, "w") as f:
        f.write(si_text)

    # Build a US-unit model with physically identical Width (100 m in ft) and S-Imperv/S-Perv (0.05 mm in inches).
    width_ft = f"{100.0 / 0.3048:.6f}"
    storage_in = f"{0.05 / 25.4:.6f}"
    subcatch_line = (
        "S1               1                J1               5        25       100      10       0"
    )
    subareas_line = "S1               0.01       1          0.05       0.05       25         OUTLET"
    assert subcatch_line in si_text
    assert subareas_line in si_text
    us_text = (
        si_text.replace("FLOW_UNITS           CMS", "FLOW_UNITS           CFS")
        .replace(
            subcatch_line,
            f"S1               1                J1               5        25       {width_ft} 10       0",
        )
        .replace(
            subareas_line,
            f"S1               0.01       1          {storage_in}   {storage_in}   25         OUTLET",
        )
    )
    us_path = os.path.join(user_dir, "us.inp")
    with open(us_path, "w") as f:
        f.write(us_text)

    _login_with_model(client, user, si_path)
    si_resp = client.post(reverse("main:calculations"), **AJAX)
    assert si_resp.status_code == 200
    si_payload = _cached_payload(client, user, "calc", CALC_RESULT_TOKEN_SESSION_KEY)

    _login_with_model(client, user, us_path)
    us_resp = client.post(reverse("main:calculations"), **AJAX)
    assert us_resp.status_code == 200
    us_payload = _cached_payload(client, user, "calc", CALC_RESULT_TOKEN_SESSION_KEY)

    assert si_payload["unit"] == "m³"
    assert us_payload["unit"] == "ft³"
    si_ann_m3 = si_payload["rows"][0]["ANN_Runoff"]
    us_ann_ft3 = us_payload["rows"][0]["ANN_Runoff"]
    assert us_ann_ft3 == pytest.approx(si_ann_m3 * _M3_TO_FT3, rel=1e-4)


# ---------------------------------------------------------------------------
# Typed errors of the package, raised by a stand-in
# ---------------------------------------------------------------------------


def _range_params_error() -> Exception:
    try:
        SimulationParams(start=5, stop=1, step=1)
    except Exception as exc:  # pydantic.ValidationError
        return exc
    raise AssertionError("SimulationParams accepted a reversed range")


def _raising_simulation(error: Exception) -> type:
    class RaisingFeaturesSimulation:
        TIMESERIES_KEYS = FeaturesSimulation.TIMESERIES_KEYS

        def __init__(self, subcatchment_id, raw_file):
            pass

        def simulate_percent_slope(self, **_range):
            raise error

        def simulate_subcatchment_timeseries(self, **_range):
            raise error

        def calculate_timeseries(self):
            raise error

    return RaisingFeaturesSimulation


SIMULATION_POST = {
    "option": "simulate_percent_slope",
    "start": "10",
    "stop": "20",
    "step": "10",
    "catchment_name": "S1",
}
TIMESERIES_POST = {
    "mode": "sweep",
    "feature": "PercImperv",
    "start": "0",
    "stop": "20",
    "step": "10",
    "catchment_name": "S1",
}
OUT_OF_RANGE = "PercSlope value 0 is outside the physical range (0, 100]."
REPORT_LINE = "ERROR 317: cannot open rainfall data file /srv/app/media/uploaded_files/7/rain.dat."
TYPED_ERRORS = [
    # (error, page, message, field errors)
    (
        SubcatchmentNotFoundError("S1", ["S2"]),
        "main:simulation",
        SUBCATCHMENT_NOT_FOUND_MESSAGE,
        {"catchment_name": [SUBCATCHMENT_NOT_FOUND_MESSAGE]},
    ),
    (InvalidParameterError(OUT_OF_RANGE), "main:simulation", OUT_OF_RANGE, {}),
    (
        UnknownFeatureError("Nope", ["Area", "Width"]),
        "main:timeseries",
        "Unknown feature 'Nope'. Allowed features: Area, Width",
        {"feature": ["Unknown feature 'Nope'. Allowed features: Area, Width"]},
    ),
    # The simulation form has no feature field, so the error is not placed on one.
    (
        UnknownFeatureError("Nope", ["Area"]),
        "main:simulation",
        "Unknown feature 'Nope'. Allowed features: Area",
        {},
    ),
    (
        SimulationError("SWMM failed to run example.inp", [REPORT_LINE]),
        "main:timeseries",
        "SWMM could not run the model: ERROR 317: cannot open rainfall data file rain.dat.",
        {},
    ),
    (
        SimulationError("SWMM failed to run example.inp: no report"),
        "main:simulation",
        "SWMM could not run the model. Open the model in SWMM to find the problem.",
        {},
    ),
    (
        _range_params_error(),
        "main:simulation",
        "Input file error. Please validate your model and selected parameters.",
        {},
    ),
]


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("error", "url_name", "message", "field_errors"),
    TYPED_ERRORS,
    ids=[f"{type(case[0]).__name__}-{case[1]}" for case in TYPED_ERRORS],
)
def test_package_errors_are_recognised_by_type(
    client, user, example_model, monkeypatch, error, url_name, message, field_errors
):
    _login_with_model(client, user, example_model)
    monkeypatch.setattr("main.views.FeaturesSimulation", _raising_simulation(error))
    data = SIMULATION_POST if url_name == "main:simulation" else TIMESERIES_POST

    response = client.post(reverse(url_name), data=data, **AJAX)

    assert response.status_code == 400
    assert response.json() == {"message": message, "field_errors": field_errors}


@pytest.mark.django_db
def test_other_errors_keep_the_generic_message(client, user, example_model, monkeypatch):
    _login_with_model(client, user, example_model)
    monkeypatch.setattr(
        "main.views.FeaturesSimulation", _raising_simulation(RuntimeError("subcatchment not found"))
    )

    response = client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    assert response.status_code == 500
    assert response.json()["message"] == "An error occurred while running the simulation."


@pytest.mark.django_db
def test_report_errors_are_capped(client, user, example_model, monkeypatch):
    errors = [f"ERROR 2{index:02d}: bad line {index}" for index in range(8)]
    _login_with_model(client, user, example_model)
    monkeypatch.setattr(
        "main.views.FeaturesSimulation", _raising_simulation(SimulationError("failed", errors))
    )

    response = client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    message = response.json()["message"]
    assert "ERROR 204: bad line 4" in message
    assert "ERROR 205" not in message
    assert message.endswith("(and 3 more errors)")


@pytest.mark.django_db
def test_engine_busy_error_returns_503(client, user, example_model, monkeypatch):
    """Concurrent SWMM runs raising MultiSimulationError (direct or wrapped) return HTTP 503 (WA1)."""
    from main.views import ENGINE_BUSY_MESSAGE

    class FakeBusyError(Exception):
        pass

    monkeypatch.setattr("main.views._ENGINE_BUSY_ERRORS", (FakeBusyError,))
    _login_with_model(client, user, example_model)

    wrapped = SimulationError("SWMM failed")
    wrapped.__cause__ = FakeBusyError("Multi-Simulation Error.")
    with monkeypatch.context() as mp:
        mp.setattr("main.views.FeaturesSimulation", _raising_simulation(wrapped))
        sim_resp = client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)
        assert sim_resp.status_code == 503
        assert sim_resp.json() == {"message": ENGINE_BUSY_MESSAGE, "field_errors": {}}

    with monkeypatch.context() as mp:

        def busy_sim(_path):
            raise FakeBusyError("Multi-Simulation Error.")

        mp.setattr("main.views.Simulation", busy_sim)
        calc_resp = client.post(reverse("main:calculations"), **AJAX)
        assert calc_resp.status_code == 503
        assert calc_resp.json() == {"message": ENGINE_BUSY_MESSAGE, "field_errors": {}}
