"""Tests for the async (fetch) contract of the run views.

Run views (simulation) answer requests sent with
``X-Requested-With: XMLHttpRequest`` with the results fragment on success and a
``{"message", "field_errors"}`` JSON body on failure, without flash messages.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest
from django.conf import settings
from django.urls import reverse

from main.views import (
    FORM_INVALID_MESSAGE,
    NO_MODEL_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
)

AJAX = {"HTTP_X_REQUESTED_WITH": "XMLHttpRequest"}
SIMULATION_PARTIAL = "main/partials/_simulation_results.html"
SIMULATION_POST = {
    "option": "simulate_percent_slope",
    "start": "1",
    "stop": "5",
    "step": "2",
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
@pytest.mark.parametrize("url_name", ["main:simulation"])
def test_run_view_anonymous_ajax_post_returns_401_json(client, url_name):
    response = client.post(reverse(url_name), data=SIMULATION_POST, **AJAX)

    body = _assert_json_error(response, 401, "Authentication required.")
    assert body["login_url"] == settings.LOGIN_URL


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation"])
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
    [("main:simulation", SIMULATION_POST)],
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
    [("main:simulation", SIMULATION_POST)],
)
def test_run_view_without_model_flashes_message(client, user, url_name, data):
    client.force_login(user)

    response = client.post(reverse(url_name), data=data)

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert NO_MODEL_MESSAGE.encode() in response.content


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
