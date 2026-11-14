"""What the sweep pages tell the user about the model: values, suggestions, balance, links."""

import json
import math
import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from django.conf import settings
from django.core.cache import cache
from django.urls import reverse
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from catchment_simulation import FeaturesSimulation
from main.forms import SimulationForm, TimeseriesForm
from main.guide import PARAMETER_GUIDE
from main.insights import (
    SUGGESTED_RANGE_FEATURES,
    feature_units,
    lab_query,
    read_subcatchment_values,
    suggest_range,
    water_balance,
)
from main.views import (
    NO_MODEL_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
    SUBCATCHMENT_NOT_FOUND_MESSAGE,
    _cached_subcatchment_values,
    _result_cache_key,
    _user_upload_dir,
)

SAMPLE = Path(settings.BASE_DIR) / "data" / "example.inp"
SAMPLE_VALUES = {
    "Area": 5.0,
    "PercImperv": 25.0,
    "Width": 100.0,
    "PercSlope": 10.0,
    "CurbLength": 0.0,
    "N-Imperv": 0.01,
    "N-Perv": 1.0,
    "S-Imperv": 0.05,
    "S-Perv": 0.05,
    "PctZero": 25.0,
}
SIMULATION_METHODS = {
    "PercSlope": "simulate_percent_slope",
    "Area": "simulate_area",
    "Width": "simulate_width",
    "PercImperv": "simulate_percent_impervious",
    "PctZero": "simulate_percent_zero_imperv",
    "CurbLength": "simulate_curb_length",
}
VALUES_URL = reverse("main:subcatchment_values")


@pytest.fixture(autouse=True)
def _fresh_values_cache():
    _cached_subcatchment_values.cache_clear()
    yield
    _cached_subcatchment_values.cache_clear()


def _model_copy(user, content: bytes, name: str = "example.inp") -> str:
    path = Path(_user_upload_dir(user.id)) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _us_model_bytes() -> bytes:
    return re.sub(rb"FLOW_UNITS\s+CMS", b"FLOW_UNITS           CFS", SAMPLE.read_bytes())


def _load(client, user, path: str) -> None:
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = path
    session.save()


# ── suggest_range ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("PercSlope", 10, (5, 15, 1)),
        ("Width", 100, (50, 150, 10)),
        ("Area", 5, (2.5, 7.5, 0.5)),
        ("PercImperv", 25, (0, 100, 10)),
        ("PctZero", 80, (0, 100, 10)),
        ("Area", None, (1, 10, 1)),
        ("Width", 0, (10, 100, 10)),
        ("PercSlope", -3, (1, 10, 1)),
        ("CurbLength", float("nan"), (0, 100, 10)),
        # Capped at the feature's upper bound, on the step grid.
        ("PercSlope", 100, (50, 100, 10)),
        # An exclusive lower bound moves a start of 0 up by one step.
        ("PercSlope", 0.05, (0.1, 1.1, 0.1)),
        # Far above the forms' maximum of 10000: the top ten steps they allow.
        ("Width", 50000, (1000, 10000, 1000)),
    ],
)
def test_suggest_range_examples(feature, value, expected):
    assert suggest_range(feature, value) == pytest.approx(expected)


def _form_accepts(feature: str, start: float, stop: float, step: float) -> bool:
    data = {"start": start, "stop": stop, "step": step, "catchment_name": "S1"}
    simulation = SimulationForm(
        {**data, "option": SIMULATION_METHODS[feature]}, catchment_choices=[("S1", "S1")]
    )
    accepted = simulation.is_valid()
    if feature in dict(TimeseriesForm.FEATURE_CHOICES):
        timeseries = TimeseriesForm(
            {**data, "mode": "sweep", "feature": feature}, catchment_choices=[("S1", "S1")]
        )
        accepted = accepted and timeseries.is_valid()
    return accepted


@hypothesis_settings(max_examples=300, deadline=None)
@given(
    feature=st.sampled_from(SUGGESTED_RANGE_FEATURES),
    value=st.one_of(
        st.none(),
        st.floats(min_value=-10, max_value=1e7),
        st.sampled_from([math.nan, math.inf]),
    ),
)
def test_every_suggestion_passes_the_forms_validation(feature, value):
    start, stop, step = suggest_range(feature, value)
    assert _form_accepts(feature, start, stop, step)


@pytest.mark.parametrize("value", [0.3, 1, 3, 7, 12.5, 33, 99, 250, 999, 4321])
def test_suggestion_contains_the_models_value_when_it_can(value):
    start, stop, _ = suggest_range("Width", value)
    assert start <= value <= stop


def test_the_forms_reject_ranges_a_suggestion_must_avoid():
    assert not _form_accepts("PercSlope", 0, 10, 1)  # a slope of 0 is not physical
    assert not _form_accepts("Width", 1, 500, 1)  # more than MAX_SWEEP_RUNS runs
    assert not _form_accepts("Width", 10, 5, 1)
    assert _form_accepts("CurbLength", 0, 100, 10)


# ── model values and units ──────────────────────────────────────────────────


def test_read_subcatchment_values_of_the_sample():
    assert read_subcatchment_values(str(SAMPLE), "S1") == SAMPLE_VALUES
    assert read_subcatchment_values(str(SAMPLE), "missing") is None


def test_read_subcatchment_values_without_a_subareas_row(tmp_path):
    content = re.sub(rb"\nS1\s+0\.01[^\n]*", b"\n", SAMPLE.read_bytes())
    path = tmp_path / "no_subareas.inp"
    path.write_bytes(content)

    values = read_subcatchment_values(str(path), "S1")

    assert values["Area"] == 5.0
    assert values["N-Imperv"] is None
    assert values["PctZero"] is None


def test_feature_units_follow_the_unit_system():
    assert feature_units("CMS")["Area"] == "ha"
    assert feature_units("CMS")["S-Perv"] == "mm"
    assert feature_units("CFS") == {
        "Area": "acre",
        "PercImperv": "%",
        "Width": "ft",
        "PercSlope": "%",
        "CurbLength": "ft",
        "N-Imperv": "",
        "N-Perv": "",
        "S-Imperv": "in",
        "S-Perv": "in",
        "PctZero": "%",
    }


# ── lab link ─────────────────────────────────────────────────────────────────


def test_lab_query_in_si_units_with_the_swept_parameter():
    query = parse_qs(lab_query(SAMPLE_VALUES, "CMS", vary="PercSlope"))
    assert query == {
        "area": ["5"],
        "imperviousness": ["25"],
        "width": ["100"],
        "slope": ["10"],
        "nImperv": ["0.01"],
        "nPerv": ["1"],
        "storageImperv": ["0.05"],
        "storagePerv": ["0.05"],
        "zeroImperv": ["25"],
        "vary": ["PercSlope"],
    }
    # Every key is a parameter of the interactive model.
    lab_keys = {guide.lab_key for guide in PARAMETER_GUIDE.values()} | {"vary"}
    assert set(query) <= lab_keys


def test_lab_query_converts_us_units_to_si():
    query = parse_qs(lab_query({"Area": 1.0, "Width": 100.0, "S-Perv": 0.1}, "CFS"))
    assert query == {"area": ["0.4046856422"], "width": ["30.48"], "storagePerv": ["2.54"]}


def test_lab_query_skips_missing_values_and_parameters_without_a_lab_key():
    query = parse_qs(
        lab_query({"Area": None, "Width": math.nan, "PercSlope": 4}, "LPS", "CurbLength")
    )
    assert query == {"slope": ["4"]}


def test_lab_query_is_none_for_an_unknown_unit_system():
    assert lab_query(SAMPLE_VALUES, "XYZ") is None


# ── water balance ─────────────────────────────────────────────────────────────


def _rows(feature, values, **overrides):
    base = {
        "runoff": 3000.0,
        "peak_runoff_rate": 0.2,
        "infiltration": 400.0,
        "evaporation": 10.0,
        "precipitation": 97.0,
        "runon": 0.0,
    }
    return [{feature: value, **base, **overrides} for value in values]


def test_water_balance_volumes_add_up_to_the_rain():
    balance = water_balance(_rows("Width", [50, 100]), "Width", 5.0, "CMS")

    assert balance["unit"] == "m³"
    row = balance["rows"][0]
    assert row["x"] == 50.0
    assert row["rainfall"] == pytest.approx(97 * 5 * 10)
    assert row["storage"] == pytest.approx(4850 - 3000 - 400 - 10)


def test_water_balance_uses_the_swept_area_and_us_units():
    balance = water_balance(_rows("Area", [1.0, 2.0], runoff=1.0), "Area", 5.0, "CFS")

    assert balance["unit"] == "ft³"
    assert [row["rainfall"] for row in balance["rows"]] == pytest.approx(
        [97 * 1 * 3630, 97 * 2 * 3630]
    )


def test_water_balance_never_reports_negative_storage():
    balance = water_balance(_rows("Width", [1], runoff=1e9), "Width", 5.0, "CMS")
    assert balance["rows"][0]["storage"] == 0.0


@pytest.mark.parametrize(
    ("rows", "area", "flow_units"),
    [
        (_rows("Width", [1]), 5.0, "XYZ"),
        (_rows("Width", [1]), None, "CMS"),
        (_rows("Width", [1], runoff=None), 5.0, "CMS"),
        (_rows("Width", [1], infiltration=math.inf), 5.0, "CMS"),
        ([], 5.0, "CMS"),
    ],
)
def test_water_balance_is_none_when_it_cannot_be_trusted(rows, area, flow_units):
    assert water_balance(rows, "Width", area, flow_units) is None


def test_water_balance_of_the_sample_model_closes(example_model):
    """On a real SWMM run the parts add up to the rain that fell on the subcatchment."""
    frame = FeaturesSimulation("S1", example_model).simulate_percent_slope(1, 10, 9)
    balance = water_balance(frame.to_dict("records"), "PercSlope", 5.0, "CMS")

    for row in balance["rows"]:
        parts = row["runoff"] + row["infiltration"] + row["evaporation"] + row["storage"]
        assert parts == pytest.approx(row["rainfall"] + row["runon"])
        assert 0 < row["storage"] < row["rainfall"]


@pytest.fixture
def example_model(tmp_path):
    path = tmp_path / "example.inp"
    shutil.copyfile(SAMPLE, path)
    return str(path)


# ── values endpoint ───────────────────────────────────────────────────────────


@pytest.mark.django_db
def test_values_endpoint_returns_values_units_ranges_and_lab_link(client, user, user_model):
    _load(client, user, user_model)

    response = client.get(VALUES_URL, {"catchment": "S1"}, HTTP_X_REQUESTED_WITH="XMLHttpRequest")

    assert response.status_code == 200
    data = response.json()
    assert data["catchment"] == "S1"
    assert data["values"] == SAMPLE_VALUES
    assert data["units"]["Area"] == "ha"
    assert data["units"]["Width"] == "m"
    assert data["suggested_ranges"]["PercSlope"] == [5, 15, 1]
    assert data["suggested_ranges"]["Area"] == [2.5, 7.5, 0.5]
    assert data["suggested_ranges"]["Width"] == [50, 150, 10]
    assert set(data["suggested_ranges"]) == set(SUGGESTED_RANGE_FEATURES)
    lab = urlsplit(data["lab_url"])
    assert lab.path == reverse("main:how_it_works")
    assert lab.fragment == "lab"
    assert parse_qs(lab.query)["area"] == ["5"]


@pytest.mark.django_db
def test_values_endpoint_for_a_us_model(client, user):
    _load(client, user, _model_copy(user, _us_model_bytes(), "us_model.inp"))

    data = client.get(VALUES_URL, {"catchment": "S1"}).json()

    assert data["units"]["Area"] == "acre"
    assert data["units"]["S-Imperv"] == "in"
    # The model's numbers are kept; the lab link converts them to SI.
    assert data["values"]["Width"] == 100.0
    assert parse_qs(urlsplit(data["lab_url"]).query)["width"] == ["30.48"]


@pytest.mark.django_db
def test_values_endpoint_does_not_store_values_in_the_session(client, user, user_model):
    _load(client, user, user_model)
    before = set(client.session.keys())

    client.get(VALUES_URL, {"catchment": "S1"})

    assert set(client.session.keys()) - before <= {"_flow_units", "_flow_units_file"}
    assert "S1" not in json.dumps(dict(client.session))


@pytest.mark.django_db
def test_values_endpoint_parses_an_unchanged_model_once(client, user, user_model, monkeypatch):
    _load(client, user, user_model)
    client.post(reverse("main:upload_sample"), HTTP_X_REQUESTED_WITH="XMLHttpRequest")
    calls = []
    real = read_subcatchment_values
    monkeypatch.setattr(
        "main.views.read_subcatchment_values",
        lambda path, catchment: calls.append(catchment) or real(path, catchment),
    )

    for _ in range(3):
        assert client.get(VALUES_URL, {"catchment": "S1"}).status_code == 200

    assert calls == ["S1"]


@pytest.mark.django_db
def test_values_endpoint_reads_a_model_written_over_another_again(client, user):
    path = _model_copy(user, SAMPLE.read_bytes())
    _load(client, user, path)
    assert client.get(VALUES_URL, {"catchment": "S1"}).json()["values"]["Width"] == 100.0

    wider = re.sub(rb"(S1\s+1\s+J1\s+5\s+25\s+)100(\s)", rb"\g<1>250\g<2>", SAMPLE.read_bytes())
    assert wider != SAMPLE.read_bytes()
    _model_copy(user, wider)

    assert client.get(VALUES_URL, {"catchment": "S1"}).json()["values"]["Width"] == 250.0


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("params", "status", "message"),
    [
        ({}, 400, "Choose a subcatchment."),
        ({"catchment": "nope"}, 404, SUBCATCHMENT_NOT_FOUND_MESSAGE),
    ],
)
def test_values_endpoint_errors(client, user, user_model, params, status, message):
    _load(client, user, user_model)

    response = client.get(VALUES_URL, params)

    assert response.status_code == status
    assert response.json() == {"message": message}


@pytest.mark.django_db
def test_values_endpoint_without_a_model(client, user):
    client.force_login(user)

    response = client.get(VALUES_URL, {"catchment": "S1"})

    assert response.status_code == 400
    assert response.json() == {"message": NO_MODEL_MESSAGE}


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ValueError("could not convert string to float: 'x'"), "Input file contains non-numeric"),
        (RuntimeError("parser crashed"), "could not be read from the model"),
    ],
)
def test_values_endpoint_with_an_unreadable_model(
    client, user, user_model, monkeypatch, error, message
):
    _load(client, user, user_model)

    def fail(path, catchment):
        raise error

    monkeypatch.setattr("main.views.read_subcatchment_values", fail)

    response = client.get(VALUES_URL, {"catchment": "S1"})

    assert response.status_code == 400
    assert message in response.json()["message"]


@pytest.mark.django_db
def test_values_endpoint_requires_login(client):
    response = client.get(VALUES_URL, {"catchment": "S1"}, HTTP_X_REQUESTED_WITH="XMLHttpRequest")
    assert response.status_code == 401


@pytest.mark.django_db
def test_values_endpoint_is_get_only(client, user, user_model):
    _load(client, user, user_model)
    assert client.post(VALUES_URL, {"catchment": "S1"}).status_code == 405


# ── is_sample ─────────────────────────────────────────────────────────────────


@pytest.mark.django_db
def test_upload_responses_say_whether_the_model_is_the_sample(client, user):
    client.force_login(user)
    ajax = {"HTTP_X_REQUESTED_WITH": "XMLHttpRequest"}

    sample = client.post(reverse("main:upload_sample"), **ajax).json()
    assert sample["is_sample"] is True
    assert client.get(reverse("main:upload_status"), **ajax).json()["is_sample"] is True

    from django.core.files.uploadedfile import SimpleUploadedFile

    same = SimpleUploadedFile("copy.inp", SAMPLE.read_bytes(), content_type="text/plain")
    assert client.post(reverse("main:upload"), {"file": same}, **ajax).json()["is_sample"] is True

    other = SimpleUploadedFile("other.inp", _us_model_bytes(), content_type="text/plain")
    assert client.post(reverse("main:upload"), {"file": other}, **ajax).json()["is_sample"] is False
    assert client.get(reverse("main:upload_status"), **ajax).json()["is_sample"] is False


# ── simulation results ──────────────────────────────────────────────────────


def _realistic_sweep(feature, values):
    import pandas as pd

    rows = _rows(feature, values)
    for index, row in enumerate(rows):
        row["runoff"] = 3000.0 + 10 * index
    return pd.DataFrame(rows)[[feature, *FeaturesSimulation.RESULT_KEYS]]


@pytest.fixture
def stub_simulation(monkeypatch):
    class StubSimulation:
        def __init__(self, subcatchment_id, raw_file):
            pass

        def simulate_percent_slope(self, start, stop, step):
            return _realistic_sweep("PercSlope", [start, start + step, stop])

        def simulate_curb_length(self, start, stop, step):
            return _realistic_sweep("CurbLength", [start, stop])

    monkeypatch.setattr("main.views.FeaturesSimulation", StubSimulation)


def _run(client, option="simulate_percent_slope", start="5", stop="15", step="5"):
    return client.post(
        reverse("main:simulation"),
        {"option": option, "start": start, "stop": stop, "step": step, "catchment_name": "S1"},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )


def _cached_chart_config(client, user) -> dict:
    token = client.session[SIM_RESULT_TOKEN_SESSION_KEY]
    return json.loads(cache.get(_result_cache_key("sim", user.id, token)))["chart_config"]


@pytest.mark.django_db
def test_simulation_result_has_baseline_balance_and_lab_link(
    client, user, user_model, stub_simulation
):
    _load(client, user, user_model)

    response = _run(client)

    assert response.status_code == 200
    config = _cached_chart_config(client, user)
    assert config["baseline"] == 10.0
    assert config["balance"]["unit"] == "m³"
    assert [row["x"] for row in config["balance"]["rows"]] == [5.0, 10.0, 15.0]
    assert config["balance"]["rows"][0]["rainfall"] == pytest.approx(4850.0)
    lab = urlsplit(config["labUrl"])
    assert parse_qs(lab.query)["vary"] == ["PercSlope"]
    assert lab.fragment == "lab"
    # The Excel columns stay as they were.
    payload = json.loads(
        cache.get(_result_cache_key("sim", user.id, client.session[SIM_RESULT_TOKEN_SESSION_KEY]))
    )
    assert payload["results_columns"] == [
        "PercSlope",
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
    ]

    html = response.content.decode()
    assert "Where the rain goes" in html
    assert 'id="simulation-lab-link"' in html
    assert f'href="{reverse("main:how_it_works")}#parameter-PercSlope"' in html
    assert "How Slope enters the model" in html
    assert 'data-parameter-name="Slope"' in html


@pytest.mark.django_db
def test_simulation_result_without_a_lab_parameter_keeps_the_link_without_vary(
    client, user, user_model, stub_simulation
):
    _load(client, user, user_model)

    _run(client, "simulate_curb_length", "0", "10", "10")

    config = _cached_chart_config(client, user)
    assert config["baseline"] == 0.0
    assert "vary" not in parse_qs(urlsplit(config["labUrl"]).query)


@pytest.mark.django_db
def test_simulation_result_without_model_values(client, user, user_model, stub_simulation):
    """A model swmmio cannot read for the values still runs; the extras are just missing."""
    _load(client, user, user_model)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("main.views.read_subcatchment_values", lambda *args: 1 / 0)
        response = _run(client)

    assert response.status_code == 200
    config = _cached_chart_config(client, user)
    assert config["baseline"] is None
    assert config["balance"] is None
    assert config["labUrl"] is None
    assert "Where the rain goes" not in response.content.decode()


@pytest.mark.django_db
def test_results_cached_before_the_new_keys_still_render(client, user, user_model):
    """An older payload has no baseline, balance or labUrl: the page renders without them."""
    _load(client, user, user_model)
    token = "a" * 32
    payload = {
        "chart_config": {
            "data": [{"PercSlope": 1.0, "runoff": 1.0}],
            "x": "PercSlope",
            "y": ["runoff"],
            "title": "t",
            "xLabel": "Percent Slope [%]",
            "yLabels": {"runoff": "Total Runoff Volume [m³]"},
        },
        "results_columns": ["PercSlope", "runoff"],
        "results_data": [[1.0, 1.0]],
        "feature_name": "PercSlope",
        "output_file_name": "old.xlsx",
    }
    cache.set(_result_cache_key("sim", user.id, token), json.dumps(payload))
    session = client.session
    session[SIM_RESULT_TOKEN_SESSION_KEY] = token
    session.save()

    response = client.get(reverse("main:simulation"))

    html = response.content.decode()
    assert response.status_code == 200
    assert "Simulation results" in html
    assert "Where the rain goes" not in html
    assert "simulation-lab-link" not in html
    # The guide entry comes from the code, not the cache.
    assert "How Slope enters the model" in html


# ── page context ──────────────────────────────────────────────────────────────


def _json_script(html: str, element_id: str):
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>', html, re.S
    )
    assert match, element_id
    return json.loads(match.group(1))


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url_name", "form_id"),
    [("main:simulation", "simulation-form"), ("main:timeseries", "timeseries-form")],
)
def test_sweep_pages_hand_the_guide_and_urls_to_their_scripts(client, user, url_name, form_id):
    client.force_login(user)

    html = client.get(reverse(url_name)).content.decode()

    guide = _json_script(html, "sweep-guide")
    assert set(guide) == set(PARAMETER_GUIDE)
    assert guide["PercSlope"]["anchor"] == "parameter-PercSlope"
    form = re.search(rf'<form[^>]*id="{form_id}"[^>]*>', html, re.S).group(0)
    assert f'data-values-url="{VALUES_URL}"' in form
    assert f'data-how-it-works-url="{reverse("main:how_it_works")}"' in form
    assert "data-cs-run-form" in form
    assert "data-restored" not in form
    # Without a model, step 1 is the active one and step 2 waits.
    assert 'data-cs-step="model" data-state="active"' in html
    assert 'data-cs-step="run" data-state="todo"' in html
    if url_name == "main:simulation":
        assert _json_script(html, "method-features")["simulate_area"] == "Area"


@pytest.mark.django_db
def test_restored_form_is_marked_and_steps_show_the_loaded_model(client, user, user_model):
    _load(client, user, user_model)
    session = client.session
    session["sim_form_state"] = {"option": "simulate_area", "start": 1, "stop": 3, "step": 1}
    session.save()

    html = client.get(reverse("main:simulation")).content.decode()

    assert 'data-restored="true"' in html
    assert 'data-cs-step="model" data-state="done"' in html
    assert 'data-cs-step="run" data-state="active"' in html
    assert 'data-check="model" data-state="done"' in html


@pytest.mark.django_db
def test_calculations_panel_tracks_the_model_and_describes_its_run(client, user, user_model):
    _load(client, user, user_model)

    html = client.get(reverse("main:calculations")).content.decode()

    assert 'data-cs-step="model" data-state="done"' in html
    assert 'data-cs-run-status="Running SWMM and the neural network"' in html
