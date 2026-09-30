"""Tests for the async (fetch) contract of the run views and the contact/profile fixes.

Run views (simulation, timeseries, calculations) answer requests sent with
``X-Requested-With: XMLHttpRequest`` with the results fragment on success and a
``{"message", "field_errors"}`` JSON body on failure, without flash messages.
"""

import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from django.conf import settings
from django.core import mail
from django.core.cache import cache
from django.urls import reverse

from main.models import UserProfile
from main.views import (
    CALC_RESULT_TOKEN_SESSION_KEY,
    FORM_INVALID_MESSAGE,
    NO_MODEL_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
    TS_RESULT_TOKEN_SESSION_KEY,
    _result_cache_key,
    _user_upload_dir,
)

AJAX = {"HTTP_X_REQUESTED_WITH": "XMLHttpRequest"}
SIMULATION_PARTIAL = "main/partials/_simulation_results.html"
TIMESERIES_PARTIAL = "main/partials/_timeseries_results.html"
CALCULATIONS_PARTIAL = "main/partials/_calculations_results.html"
SIMULATION_POST = {
    "option": "simulate_percent_slope",
    "start": "1",
    "stop": "5",
    "step": "2",
    "catchment_name": "S1",
}
TIMESERIES_SINGLE_POST = {"mode": "single", "catchment_name": "S1"}
TIMESERIES_SWEEP_POST = {
    "mode": "sweep",
    "feature": "PercSlope",
    "start": "0",
    "stop": "20",
    "step": "5",
    "catchment_name": "S1",
}
NON_NUMERIC_MESSAGE = "Input file contains non-numeric values where numbers are required."


def _timeseries_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=3, freq="h", name="datetime")
    return pd.DataFrame(
        {
            "rainfall": [0.2, 0.1, 0.0],
            "runoff": [0.5, 1.2, 0.3],
            "infiltration_loss": [0.1, 0.1, 0.1],
            "evaporation_loss": [0.0, 0.0, 0.0],
            "runon": [0.0, 0.0, 0.0],
        },
        index=idx,
    )


class DummyFeaturesSimulation:
    """Fast stand-in for FeaturesSimulation covering the methods the views call."""

    TIMESERIES_KEYS = ("rainfall", "runoff", "infiltration_loss", "evaporation_loss", "runon")

    def __init__(self, subcatchment_id, raw_file):
        self.subcatchment_id = subcatchment_id
        self.raw_file = raw_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def simulate_percent_slope(self, start, stop, step):
        return pd.DataFrame({"PercSlope": [start, stop], "runoff": [1.0, 2.0]})

    def calculate_timeseries(self):
        return _timeseries_frame()

    def simulate_subcatchment_timeseries(self, feature, start, stop, step):
        values = np.arange(start, stop + step / 2, step)
        return {float(value): _timeseries_frame() for value in values}


class InvalidInputFeaturesSimulation(DummyFeaturesSimulation):
    def __enter__(self):
        raise ValueError("Expected numeric value in slope sheet.")


class CrashingFeaturesSimulation(DummyFeaturesSimulation):
    def __enter__(self):
        raise RuntimeError("boom")


@pytest.fixture
def run_client(client, user):
    """Logged-in client whose session points at the bundled example model."""
    client.force_login(user)
    uploaded_file_path = os.path.join(settings.BASE_DIR, "data", "example.inp")
    session = client.session
    session["uploaded_file_path"] = uploaded_file_path
    session["_subcatchment_ids_file"] = uploaded_file_path
    session["_subcatchment_ids"] = ["S1"]
    session.save()
    return client


@pytest.fixture
def fake_simulation(monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)


@pytest.fixture
def fake_calculations(monkeypatch):
    """Replace the SWMM run and ANN prediction with a one-subcatchment SI (CMS) result."""

    class FakeSimulation:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __iter__(self):
            return iter(())

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeModel:
        def __init__(self, _path):
            dataframe = pd.DataFrame(index=["S1"], data={"TotalRunoffMG": [12.34]})
            self.subcatchments = SimpleNamespace(dataframe=dataframe)
            options = pd.DataFrame(index=["FLOW_UNITS"], data={"Value": ["CMS"]})
            self.inp = SimpleNamespace(options=options)

    monkeypatch.setattr("main.views.Simulation", FakeSimulation)
    monkeypatch.setattr("main.views.swmmio.Model", FakeModel)
    monkeypatch.setattr("main.views.predict_runoff", lambda _model: np.array([4.56]))
    monkeypatch.setattr("main.views._cleanup_swmm_side_files", lambda _path: None)


@pytest.fixture
def calc_model_path(user):
    """A stub model file in the user's upload directory (the SWMM run itself is faked)."""
    user_dir = _user_upload_dir(user.id)
    os.makedirs(user_dir, exist_ok=True)
    path = os.path.join(user_dir, "test.inp")
    with open(path, "w", encoding="utf-8") as model_file:
        model_file.write("[TITLE]\n[OPTIONS]\n")
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def calc_client(client, user, calc_model_path):
    """Logged-in client whose session points at ``calc_model_path``."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = calc_model_path
    session.save()
    return client


def _template_names(response) -> list[str]:
    return [template.name for template in response.templates]


def _assert_fragment(response, partial: str, root_id: str) -> None:
    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/html")
    assert partial in _template_names(response)
    content = response.content.decode()
    assert content.lstrip().startswith(f'<div id="{root_id}">')
    assert "<html" not in content
    assert "<form" not in content


def _assert_json_error(response, status: int, message: str) -> dict:
    assert response.status_code == status
    assert response["Content-Type"] == "application/json"
    data = response.json()
    assert data["message"] == message
    assert isinstance(data["field_errors"], dict)
    return data


def _assert_no_flash_on_next_page(client, url: str, message: str) -> None:
    response = client.get(url)
    assert response.status_code == 200
    assert b"alert-danger" not in response.content
    assert message.encode() not in response.content


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_simulation_ajax_success_returns_results_fragment(run_client, user, fake_simulation):
    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    _assert_fragment(response, SIMULATION_PARTIAL, "simulation-results")
    assert "main/simulation.html" not in _template_names(response)
    content = response.content.decode()
    assert 'id="chart-config"' in content
    assert 'id="simulation-results-heading"' in content
    assert 'tabindex="-1"' in content
    assert "Download Results" in content
    assert "cs-table" in content

    session = run_client.session
    token = session[SIM_RESULT_TOKEN_SESSION_KEY]
    assert f'value="{token}"' in content
    assert session["sim_form_state"]["catchment_name"] == "S1"


@pytest.mark.django_db
def test_simulation_download_uses_page_form_outside_run_form(run_client, fake_simulation):
    """Download targets its own form, so it never submits (or spins) the run form (audit S2)."""
    run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    content = run_client.get(reverse("main:simulation")).content.decode()

    download_form = re.search(r'<form id="simulation-download-form"[^>]*>', content)
    assert download_form is not None
    assert f'action="{reverse("main:download_simulation_results")}"' in download_form.group(0)
    run_form = re.search(r'<form[^>]*id="simulation-form".*?</form>', content, re.S).group(0)
    assert "download-simulation-results-button" not in run_form
    token = run_client.session[SIM_RESULT_TOKEN_SESSION_KEY]
    assert re.search(rf'name="token" value="{token}" form="simulation-download-form"', content)
    assert re.search(
        r'id="download-simulation-results-button"\s+form="simulation-download-form"', content
    )


@pytest.mark.django_db
def test_simulation_full_page_renders_same_results_partial(run_client, fake_simulation):
    run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    response = run_client.get(reverse("main:simulation"))

    assert response.status_code == 200
    assert SIMULATION_PARTIAL in _template_names(response)
    assert b'id="chart-config"' in response.content


@pytest.mark.django_db
def test_simulation_page_without_results_renders_empty_results_root(run_client):
    response = run_client.get(reverse("main:simulation"))

    assert SIMULATION_PARTIAL in _template_names(response)
    assert b'id="simulation-results"' in response.content
    assert b'id="simulation-results-heading"' not in response.content
    assert b"No results yet" in response.content


@pytest.mark.django_db
def test_simulation_non_ajax_post_still_redirects(run_client, fake_simulation):
    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST)

    assert response.status_code == 302
    assert response.url == reverse("main:simulation")


@pytest.mark.django_db
def test_simulation_ajax_invalid_form_returns_field_errors(run_client, fake_simulation):
    data = {**SIMULATION_POST, "start": "10", "stop": "5", "catchment_name": ""}

    response = run_client.post(reverse("main:simulation"), data=data, **AJAX)

    body = _assert_json_error(response, 400, FORM_INVALID_MESSAGE)
    assert body["field_errors"]["stop"] == ["Stop must be greater than or equal to start."]
    assert body["field_errors"]["catchment_name"] == ["This field is required."]
    assert SIM_RESULT_TOKEN_SESSION_KEY not in run_client.session
    _assert_no_flash_on_next_page(run_client, reverse("main:simulation"), "Stop must be")


@pytest.mark.django_db
def test_simulation_ajax_input_error_returns_json_without_flash(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", InvalidInputFeaturesSimulation)

    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    body = _assert_json_error(response, 400, NON_NUMERIC_MESSAGE)
    assert body["field_errors"] == {}
    _assert_no_flash_on_next_page(run_client, reverse("main:simulation"), NON_NUMERIC_MESSAGE)


@pytest.mark.django_db
def test_simulation_ajax_oversized_result_returns_413(run_client, fake_simulation, monkeypatch):
    monkeypatch.setattr("main.views.MAX_RESULT_CACHE_BYTES", 10)

    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    _assert_json_error(
        response,
        413,
        "Result set is too large to keep for download. Narrow the simulation range.",
    )
    assert SIM_RESULT_TOKEN_SESSION_KEY not in run_client.session


@pytest.mark.django_db
def test_simulation_ajax_unexpected_error_returns_500_without_flash(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", CrashingFeaturesSimulation)
    message = "An error occurred while running the simulation."

    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST, **AJAX)

    _assert_json_error(response, 500, message)
    _assert_no_flash_on_next_page(run_client, reverse("main:simulation"), message)


@pytest.mark.django_db
def test_simulation_non_ajax_input_error_still_flashes_message(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", InvalidInputFeaturesSimulation)

    response = run_client.post(reverse("main:simulation"), data=SIMULATION_POST)

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert NON_NUMERIC_MESSAGE.encode() in response.content


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation", "main:timeseries"])
def test_run_view_anonymous_ajax_post_returns_401_json(client, url_name):
    response = client.post(reverse(url_name), data=SIMULATION_POST, **AJAX)

    body = _assert_json_error(response, 401, "Authentication required.")
    assert body["login_url"] == settings.LOGIN_URL


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation", "main:timeseries"])
def test_run_view_anonymous_get_redirects_to_login(client, url_name):
    url = reverse(url_name)

    response = client.get(url)

    assert response.status_code == 302
    assert response.url == f"{settings.LOGIN_URL}?next={url}"


@pytest.mark.django_db
def test_run_view_login_redirect_keeps_the_query_string(client):
    response = client.get(reverse("main:simulation"), {"option": "simulate_area"})

    assert response.status_code == 302
    assert response.url == f"{settings.LOGIN_URL}?next=/simulation%3Foption%3Dsimulate_area"


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url_name", "data"),
    [("main:simulation", SIMULATION_POST), ("main:timeseries", TIMESERIES_SINGLE_POST)],
)
def test_run_view_without_model_returns_400_json(client, user, url_name, data):
    """There is no fallback model: a run needs one loaded in the session."""
    client.force_login(user)

    response = client.post(reverse(url_name), data=data, **AJAX)

    _assert_json_error(response, 400, NO_MODEL_MESSAGE)
    _assert_no_flash_on_next_page(client, reverse(url_name), NO_MODEL_MESSAGE)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url_name", "data"),
    [("main:simulation", SIMULATION_POST), ("main:timeseries", TIMESERIES_SINGLE_POST)],
)
def test_run_view_without_model_flashes_message(client, user, url_name, data):
    client.force_login(user)

    response = client.post(reverse(url_name), data=data)

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert NO_MODEL_MESSAGE.encode() in response.content


# ---------------------------------------------------------------------------
# Timeseries
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_timeseries_ajax_single_returns_results_fragment(run_client, fake_simulation):
    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX)

    _assert_fragment(response, TIMESERIES_PARTIAL, "timeseries-results")
    content = response.content.decode()
    assert 'id="ts-chart-config"' in content
    assert 'id="timeseries-results-heading"' in content
    assert "Time to peak" in content
    assert "Runoff volume" in content
    assert 'id="download-timeseries-png-button"' in content
    token = run_client.session[TS_RESULT_TOKEN_SESSION_KEY]
    assert f'value="{token}"' in content


@pytest.mark.django_db
def test_timeseries_single_metrics_are_rendered_by_the_server(run_client, fake_simulation):
    """Peak, time to peak and volume (with units) are in the fragment, not left to JS."""
    content = run_client.post(
        reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX
    ).content.decode()

    metric_values = re.findall(r'<p class="cs-metric__value">(.*?)</p>', content, re.S)
    # Hourly runoff 0.5, 1.2, 0.3 CMS: peak in the second hour; trapezoidal volume 5760 m³.
    assert metric_values == [
        '1.200<span class="cs-metric__unit">CMS</span>',
        "1 h",
        '5,760.00<span class="cs-metric__unit">m³</span>',
    ]
    assert "At 2025-01-01 01:00" in content


@pytest.mark.django_db
def test_timeseries_ajax_sweep_returns_results_fragment(run_client, fake_simulation):
    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SWEEP_POST, **AJAX)

    _assert_fragment(response, TIMESERIES_PARTIAL, "timeseries-results")
    content = response.content.decode()
    assert '"mode": "sweep"' in content
    assert "Time to peak</p>" not in content, "a sweep has a summary table, not metric tiles"
    assert 'data-ts-table="sweep"' in content
    assert run_client.session["ts_form_state"]["feature"] == "PercSlope"


@pytest.mark.django_db
def test_timeseries_sweep_summary_and_series_are_rendered_by_the_server(
    run_client, fake_simulation
):
    content = run_client.post(
        reverse("main:timeseries"), data=TIMESERIES_SWEEP_POST, **AJAX
    ).content.decode()

    table = re.search(
        r'<table class="table cs-table" data-ts-table="sweep">.*?</table>', content, re.S
    )
    rows = re.findall(r'<tr>\s*<th scope="row"[^>]*>(.*?)</th>', table.group(0), re.S)
    # Swept values are labelled without float noise ("5", not "5.0").
    assert rows == ["0", "5", "10", "15", "20"]
    assert "Peak runoff [CMS]" in table.group(0)
    assert "Runoff volume [m³]" in table.group(0)
    # The swept parameter reads as on the simulation page, with its unit, not as the SWMM key.
    assert '<th scope="col" class="cs-num">Percent Slope [%]</th>' in table.group(0)
    assert "Timeseries sweep: Percent Slope for S1" in content
    assert "PercSlope" not in content
    assert 'data-sort-value="3600.0">1 h</td>' in table.group(0)
    options = re.findall(r'<option value="(\w+)"', content)
    assert options == ["runoff", "infiltration_loss", "evaporation_loss", "runon"]


@pytest.mark.django_db
def test_timeseries_payload_keeps_one_copy_of_the_rows(run_client, fake_simulation):
    """Rows are cached once (for the downloads) and merged into the chart config on render."""
    run_client.post(reverse("main:timeseries"), data=TIMESERIES_SWEEP_POST, **AJAX)
    token = run_client.session[TS_RESULT_TOKEN_SESSION_KEY]
    user_id = int(run_client.session["_auth_user_id"])
    payload = json.loads(cache.get(_result_cache_key("ts", user_id, token)))

    assert "data" not in payload["chart_config"]
    assert list(payload["data"]) == ["0", "5", "10", "15", "20"]
    page = run_client.get(reverse("main:timeseries"))
    assert list(page.context["ts_chart_config"]["data"]) == ["0", "5", "10", "15", "20"]


@pytest.mark.django_db
def test_timeseries_exports_submit_the_page_export_form(run_client, fake_simulation):
    """The fragment is swapped outside the run form, so its exports target a page-level form."""
    fragment = run_client.post(
        reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX
    ).content.decode()
    page = run_client.get(reverse("main:timeseries")).content.decode()

    token = run_client.session[TS_RESULT_TOKEN_SESSION_KEY]
    assert f'value="{token}" form="timeseries-export-form"' in fragment
    assert fragment.count('form="timeseries-export-form"') == 3
    assert f'formaction="{reverse("main:download_timeseries_csv")}"' in fragment
    export_form = re.search(r'<form id="timeseries-export-form"[^>]*>(.*?)</form>', page, re.S)
    assert export_form
    assert f'action="{reverse("main:download_timeseries_results")}"' in export_form.group(0)
    assert 'name="csrfmiddlewaretoken"' in export_form.group(1)


@pytest.mark.django_db
def test_timeseries_full_page_renders_same_results_partial(run_client, fake_simulation):
    run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX)

    response = run_client.get(reverse("main:timeseries"))

    assert TIMESERIES_PARTIAL in _template_names(response)
    assert b'id="ts-chart-config"' in response.content


@pytest.mark.django_db
def test_timeseries_non_ajax_post_still_redirects(run_client, fake_simulation):
    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST)

    assert response.status_code == 302
    assert response.url == reverse("main:timeseries")


@pytest.mark.django_db
def test_timeseries_ajax_invalid_form_returns_field_errors(run_client, fake_simulation):
    data = {"mode": "sweep", "catchment_name": "S1"}

    response = run_client.post(reverse("main:timeseries"), data=data, **AJAX)

    body = _assert_json_error(response, 400, FORM_INVALID_MESSAGE)
    assert set(body["field_errors"]) == {"feature", "start", "stop", "step"}
    assert body["field_errors"]["feature"] == ["Required for parameter sweep mode."]
    _assert_no_flash_on_next_page(run_client, reverse("main:timeseries"), "Required for")


@pytest.mark.django_db
def test_timeseries_page_without_results_renders_empty_results_root(run_client):
    response = run_client.get(reverse("main:timeseries"))

    assert TIMESERIES_PARTIAL in _template_names(response)
    assert b'id="timeseries-results"' in response.content
    assert b'id="timeseries-results-heading"' not in response.content
    assert b"No results yet" in response.content


@pytest.mark.django_db
def test_timeseries_ajax_oversized_result_returns_413(run_client, fake_simulation, monkeypatch):
    monkeypatch.setattr("main.views.MAX_RESULT_CACHE_BYTES", 10)

    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SWEEP_POST, **AJAX)

    _assert_json_error(
        response,
        413,
        "Result set is too large to keep for download. Narrow the timeseries range.",
    )
    assert TS_RESULT_TOKEN_SESSION_KEY not in run_client.session


@pytest.mark.django_db
def test_timeseries_ajax_input_error_returns_json_without_flash(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", InvalidInputFeaturesSimulation)

    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX)

    _assert_json_error(response, 400, NON_NUMERIC_MESSAGE)
    assert "ts_form_state" not in run_client.session
    _assert_no_flash_on_next_page(run_client, reverse("main:timeseries"), NON_NUMERIC_MESSAGE)


@pytest.mark.django_db
def test_timeseries_non_ajax_input_error_still_flashes_message(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", InvalidInputFeaturesSimulation)

    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST)

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert NON_NUMERIC_MESSAGE.encode() in response.content


@pytest.mark.django_db
def test_timeseries_ajax_unexpected_error_returns_500(run_client, monkeypatch):
    monkeypatch.setattr("main.views.FeaturesSimulation", CrashingFeaturesSimulation)

    response = run_client.post(reverse("main:timeseries"), data=TIMESERIES_SINGLE_POST, **AJAX)

    _assert_json_error(response, 500, "An error occurred while running the analysis.")


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_calculations_ajax_success_returns_results_fragment(calc_client, fake_calculations):
    response = calc_client.post(reverse("main:calculations"), **AJAX)

    _assert_fragment(response, CALCULATIONS_PARTIAL, "calculations-results")
    content = response.content.decode()
    assert 'id="calculations-results-heading"' in content
    assert "Results comparison" in content
    assert "S1" in content
    # The report's 10^6 L become m³: 12.34 -> 12,340 and the ANN's 4.56 -> 4,560.
    assert "12340.00" in content
    assert "4560.00" in content
    assert "SWMM Runoff [m³]" in content
    assert 'data-calc-unit="m³"' in content
    assert 'id="calculations-table"' in content
    assert 'id="calculations-chart-data"' in content
    assert "cs-empty-state" not in content


@pytest.mark.django_db
def test_calculations_non_ajax_success_redirects_to_results_page(calc_client, fake_calculations):
    """Post/Redirect/Get: refreshing the results page does not run the comparison again."""
    response = calc_client.post(reverse("main:calculations"), follow=True)

    assert response.redirect_chain == [(reverse("main:calculations"), 302)]
    names = _template_names(response)
    assert "main/calculations.html" in names
    assert CALCULATIONS_PARTIAL in names
    assert b"12340.00" in response.content


@pytest.mark.django_db
def test_calculations_page_shows_last_async_comparison(calc_client, fake_calculations):
    calc_client.post(reverse("main:calculations"), **AJAX)

    response = calc_client.get(reverse("main:calculations"))

    assert b'id="calculations-table"' in response.content
    assert b"12340.00" in response.content


@pytest.mark.django_db
def test_calculations_page_drops_an_expired_comparison(calc_client, fake_calculations):
    calc_client.post(reverse("main:calculations"), **AJAX)
    cache.clear()

    response = calc_client.get(reverse("main:calculations"))

    assert b"No comparison yet" in response.content
    assert CALC_RESULT_TOKEN_SESSION_KEY not in calc_client.session


@pytest.mark.django_db
def test_calculations_get_renders_empty_state(client):
    response = client.get(reverse("main:calculations"))

    assert response.status_code == 200
    assert CALCULATIONS_PARTIAL in _template_names(response)
    assert b'id="calculations-results-heading"' not in response.content
    assert b'<h2 class="cs-empty-state__title">No comparison yet</h2>' in response.content
    assert b'id="calculations-table"' not in response.content
    assert b'id="calculations-chart-data"' not in response.content


@pytest.mark.django_db
def test_calculations_ajax_without_upload_returns_400_without_flash(client, user):
    client.force_login(user)
    message = "Please upload a file first."

    response = client.post(reverse("main:calculations"), **AJAX)

    _assert_json_error(response, 400, message)
    _assert_no_flash_on_next_page(client, reverse("main:calculations"), message)


@pytest.mark.django_db
def test_calculations_with_deleted_model_returns_400(calc_client, calc_model_path):
    os.remove(calc_model_path)
    message = "Input file is missing. Please upload the model again."

    ajax_response = calc_client.post(reverse("main:calculations"), **AJAX)
    page_response = calc_client.post(reverse("main:calculations"))

    _assert_json_error(ajax_response, 400, message)
    assert b"alert-danger" in page_response.content
    assert message.encode() in page_response.content


@pytest.mark.django_db
def test_calculations_model_without_subcatchments_returns_400(calc_client, calc_model_path):
    """A real SWMM run of the example model with its subcatchment sections removed."""
    with open(os.path.join(settings.BASE_DIR, "data", "example.inp"), encoding="utf-8") as src:
        sections = re.split(r"(?m)^(?=\[)", src.read())
    dropped = ("[SUBCATCHMENTS]", "[SUBAREAS]", "[INFILTRATION]", "[Polygons]")
    with open(calc_model_path, "w", encoding="utf-8") as model_file:
        model_file.writelines(s for s in sections if not s.startswith(dropped))

    response = calc_client.post(reverse("main:calculations"), **AJAX)

    _assert_json_error(response, 400, "The model has no subcatchments to compare.")


@pytest.mark.django_db
def test_calculations_ajax_rejects_out_of_bounds_path(client, user):
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "/etc/passwd"
    session.save()

    response = client.post(reverse("main:calculations"), **AJAX)

    _assert_json_error(response, 400, "Invalid file path detected.")
    _assert_no_flash_on_next_page(
        client, reverse("main:calculations"), "Invalid file path detected."
    )


@pytest.mark.django_db
def test_calculations_ajax_oversized_result_returns_413(
    calc_client, fake_calculations, monkeypatch
):
    monkeypatch.setattr("main.views.MAX_RESULT_CACHE_BYTES", 10)

    response = calc_client.post(reverse("main:calculations"), **AJAX)

    _assert_json_error(
        response, 413, "The comparison is too large to keep. Use a model with fewer subcatchments."
    )
    assert CALC_RESULT_TOKEN_SESSION_KEY not in calc_client.session


@pytest.mark.django_db
def test_calculations_ajax_failure_returns_500(calc_client, monkeypatch):
    def failing_simulation(_path):
        raise RuntimeError("swmm crashed")

    monkeypatch.setattr("main.views.Simulation", failing_simulation)
    monkeypatch.setattr("main.views._cleanup_swmm_side_files", lambda _path: None)

    response = calc_client.post(reverse("main:calculations"), **AJAX)

    _assert_json_error(response, 500, "An error occurred while performing calculations.")


@pytest.mark.django_db
def test_calculations_anonymous_ajax_post_returns_401_json(client):
    response = client.post(reverse("main:calculations"), **AJAX)

    body = _assert_json_error(response, 401, "Authentication required.")
    assert body["login_url"] == settings.LOGIN_URL
    assert body["error"] == "Authentication required."


@pytest.mark.django_db
def test_calculations_anonymous_post_redirects_to_login(client):
    calc_url = reverse("main:calculations")

    response = client.post(calc_url)

    assert response.status_code == 302
    assert response.url == f"{settings.LOGIN_URL}?next={calc_url}"


# ---------------------------------------------------------------------------
# Contact and profile
# ---------------------------------------------------------------------------

CONTACT_POST = {
    "email": "sender@example.com",
    "title": "Question",
    "content": "Hello there.",
}


@pytest.mark.django_db
def test_contact_success_sends_mail_and_redirects_with_message(client):
    response = client.post(reverse("main:contact"), data=CONTACT_POST, follow=True)

    assert response.redirect_chain == [(reverse("main:contact"), 302)]
    assert len(mail.outbox) == 1
    assert mail.outbox[0].subject == "Question"
    assert b"alert-success" in response.content
    assert b"Message sent." in response.content


@pytest.mark.django_db
def test_contact_send_failure_rerenders_form_with_error(client, monkeypatch):
    monkeypatch.setattr("main.views.send_message", lambda _message: False)

    response = client.post(reverse("main:contact"), data=CONTACT_POST)

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert b"Your message could not be sent. Please try again later." in response.content
    assert b'value="sender@example.com"' in response.content
    assert b"Hello there." in response.content


@pytest.mark.django_db
def test_contact_form_posts_to_contact_url(client):
    response = client.get(reverse("main:contact"))

    assert f'action="{reverse("main:contact")}"'.encode() in response.content


@pytest.mark.django_db
def test_profile_owner_post_redirects_to_own_profile(client, user):
    client.force_login(user)
    profile_url = reverse("main:userprofile", args=[user.id])

    response = client.post(profile_url, data={"user": user.id, "bio": "Hydrologist."})

    assert response.status_code == 302
    assert response.url == profile_url
    assert UserProfile.objects.get(user=user).bio == "Hydrologist."


@pytest.mark.django_db
def test_profile_save_confirms_with_a_success_message(client, user):
    client.force_login(user)
    profile_url = reverse("main:userprofile", args=[user.id])

    response = client.post(profile_url, data={"user": user.id, "bio": "Hydrologist."}, follow=True)

    assert response.redirect_chain == [(profile_url, 302)]
    assert b"alert-success" in response.content
    assert b"Profile updated." in response.content


@pytest.mark.django_db
def test_profile_form_posts_back_to_profile_url(client, user):
    """Without an action attribute the browser posts to the profile URL itself."""
    client.force_login(user)

    content = client.get(reverse("main:userprofile", args=[user.id])).content.decode()

    form_tag = re.findall(r"<form\b[^>]*>", content[: content.index('id="id_bio"')])[-1]
    assert 'method="post"' in form_tag
    assert "action=" not in form_tag


# ---------------------------------------------------------------------------
# Failed no-JS runs keep the previous results on the page
# ---------------------------------------------------------------------------


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url_name", "post", "failing_post", "heading_id"),
    [
        (
            "main:simulation",
            SIMULATION_POST,
            {**SIMULATION_POST, "start": "10", "stop": "5"},
            "simulation-results-heading",
        ),
        (
            "main:timeseries",
            TIMESERIES_SWEEP_POST,
            {**TIMESERIES_SWEEP_POST, "start": "30"},
            "timeseries-results-heading",
        ),
    ],
)
def test_non_ajax_invalid_run_keeps_previous_results(
    run_client, fake_simulation, url_name, post, failing_post, heading_id
):
    run_client.post(reverse(url_name), data=post, **AJAX)

    response = run_client.post(reverse(url_name), data=failing_post)

    assert response.status_code == 200
    assert "Stop must be" in response.content.decode()
    assert f'id="{heading_id}"' in response.content.decode()


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url_name", "post", "heading_id"),
    [
        ("main:simulation", SIMULATION_POST, "simulation-results-heading"),
        ("main:timeseries", TIMESERIES_SINGLE_POST, "timeseries-results-heading"),
    ],
)
def test_non_ajax_crashed_run_keeps_previous_results(
    run_client, fake_simulation, monkeypatch, url_name, post, heading_id
):
    run_client.post(reverse(url_name), data=post, **AJAX)
    monkeypatch.setattr("main.views.FeaturesSimulation", CrashingFeaturesSimulation)

    response = run_client.post(reverse(url_name), data=post)

    assert response.status_code == 200
    assert "An error occurred" in response.content.decode()
    assert f'id="{heading_id}"' in response.content.decode()


@pytest.mark.django_db
def test_non_ajax_crashed_calculations_keep_previous_comparison(
    calc_client, fake_calculations, monkeypatch
):
    calc_client.post(reverse("main:calculations"), **AJAX)

    def crash(_path):
        raise RuntimeError("boom")

    monkeypatch.setattr("main.views._compare_swmm_and_ann", crash)
    response = calc_client.post(reverse("main:calculations"))

    assert response.status_code == 200
    assert b"An error occurred" in response.content
    assert b'id="calculations-results-heading"' in response.content
    assert b"12340.00" in response.content
