"""What the sweep pages tell the user about the model: values, suggestions, links."""

import json
import math
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from django.conf import settings
from django.urls import reverse
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from main.forms import SimulationForm, TimeseriesForm
from main.guide import PARAMETER_GUIDE
from main.insights import (
    SUGGESTED_RANGE_FEATURES,
    feature_units,
    lab_query,
    read_subcatchment_values,
    suggest_range,
)
from main.views import (
    NO_MODEL_MESSAGE,
    SUBCATCHMENT_NOT_FOUND_MESSAGE,
    _cached_subcatchment_values,
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
def test_calculations_panel_tracks_the_model(client, user, user_model):
    _load(client, user, user_model)

    html = client.get(reverse("main:calculations")).content.decode()

    assert 'data-cs-step="model" data-state="done"' in html
