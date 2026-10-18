"""Unit and integration tests for the JSON REST API endpoints (/api/*)."""

import json
import os

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from catchment_simulation import FeaturesSimulation, build_range
from catchment_simulation.exceptions import SubcatchmentNotFoundError
from main.api import (
    INVALID_JSON_MESSAGE,
    METHOD_NOT_ALLOWED_MESSAGE,
    NO_CALCULATIONS_RESULTS_MESSAGE,
    NO_SIMULATION_RESULTS_MESSAGE,
    NO_TIMESERIES_RESULTS_MESSAGE,
)
from main.views import (
    AUTH_REQUIRED_MESSAGE,
    FORM_INVALID_MESSAGE,
    NO_MODEL_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
    TS_RESULT_TOKEN_SESSION_KEY,
    _user_upload_dir,
)


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
    """Fast deterministic stand-in for FeaturesSimulation."""

    TIMESERIES_KEYS = FeaturesSimulation.TIMESERIES_KEYS

    def __init__(self, subcatchment_id, raw_file):
        self.subcatchment_id = subcatchment_id
        self.raw_file = raw_file

    def simulate_percent_slope(self, start, stop, step):
        values = build_range(start, stop, step)
        frame = pd.DataFrame({"PercSlope": values})
        for offset, key in enumerate(FeaturesSimulation.RESULT_KEYS, start=1):
            frame[key] = [offset * (index + 1.0) for index in range(len(values))]
        return frame

    def calculate_timeseries(self):
        return _timeseries_frame()

    def simulate_subcatchment_timeseries(self, feature, start, stop, step):
        return {value: _timeseries_frame() for value in build_range(start, stop, step)}


@pytest.fixture
def authenticated_client(client, user):
    """Logged-in test client."""
    client.force_login(user)
    return client


@pytest.fixture
def loaded_model_client(authenticated_client):
    """Authenticated client with the bundled sample INP loaded via /api/upload/sample/."""
    res = authenticated_client.post(reverse("main:api_upload_sample"))
    assert res.status_code == 200
    return authenticated_client


@pytest.mark.django_db
class TestApiAuthentication:
    """All /api/* endpoints return 401 JSON when unauthenticated (without XHR headers)."""

    @pytest.mark.parametrize(
        ("url_name", "method"),
        [
            ("main:api_status", "get"),
            ("main:api_subcatchments", "get"),
            ("main:api_upload", "post"),
            ("main:api_upload_sample", "post"),
            ("main:api_upload_clear", "post"),
            ("main:api_simulation", "get"),
            ("main:api_simulation", "post"),
            ("main:api_timeseries", "get"),
            ("main:api_timeseries", "post"),
            ("main:api_calculations", "get"),
            ("main:api_calculations", "post"),
        ],
    )
    def test_unauthenticated_returns_401_json(self, client, url_name, method):
        response = getattr(client, method)(reverse(url_name))
        assert response.status_code == 401
        payload = response.json()
        assert payload["error"] == AUTH_REQUIRED_MESSAGE
        assert payload["message"] == AUTH_REQUIRED_MESSAGE
        assert payload["field_errors"] == {}
        assert "login_url" in payload


@pytest.mark.django_db
class TestApiMethodEnforcement:
    """Endpoints reject unsupported HTTP methods with 405 JSON and an Allow header."""

    @pytest.mark.parametrize(
        ("url_name", "method", "expected_allow"),
        [
            ("main:api_status", "post", "GET"),
            ("main:api_subcatchments", "post", "GET"),
            ("main:api_upload", "get", "POST"),
            ("main:api_upload_sample", "get", "POST"),
            ("main:api_upload_clear", "get", "POST"),
            ("main:api_simulation", "put", "GET, POST"),
            ("main:api_timeseries", "delete", "GET, POST"),
            ("main:api_calculations", "patch", "GET, POST"),
        ],
    )
    def test_unsupported_method_returns_405(
        self, authenticated_client, url_name, method, expected_allow
    ):
        response = getattr(authenticated_client, method)(reverse(url_name))
        assert response.status_code == 405
        assert response["Allow"] == expected_allow
        payload = response.json()
        assert payload["error"] == METHOD_NOT_ALLOWED_MESSAGE
        assert payload["message"] == METHOD_NOT_ALLOWED_MESSAGE


@pytest.mark.django_db
class TestApiModelEndpoints:
    """Tests for /api/status/, /api/upload/, /api/upload/sample/, /api/upload/clear/, /api/subcatchments/."""

    def test_status_without_loaded_model(self, authenticated_client):
        res = authenticated_client.get(reverse("main:api_status"))
        assert res.status_code == 200
        data = res.json()
        assert data["has_file"] is False
        assert data["filename"] is None
        assert data["size"] is None
        assert data["flow_units"] is None
        assert data["subcatchments"] == []
        assert "simulate_area" in data["option_labels"]

    def test_sample_load_status_subcatchments_and_clear(self, authenticated_client):
        load_res = authenticated_client.post(reverse("main:api_upload_sample"))
        assert load_res.status_code == 200
        load_data = load_res.json()
        assert load_data["filename"] == "example.inp"
        assert load_data["size"] > 0

        status_res = authenticated_client.get(reverse("main:api_status"))
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["has_file"] is True
        assert status_data["filename"] == "example.inp"
        assert status_data["flow_units"] == "CMS"
        assert "S1" in status_data["subcatchments"]

        sub_res = authenticated_client.get(reverse("main:api_subcatchments"))
        assert sub_res.status_code == 200
        sub_data = sub_res.json()
        assert "S1" in sub_data["subcatchments"]
        assert sub_data["flow_units"] == "CMS"
        assert "simulate_area" in sub_data["option_labels"]

        clear_res = authenticated_client.post(reverse("main:api_upload_clear"))
        assert clear_res.status_code == 200
        assert clear_res.json()["message"] == "Upload cleared."

        after_clear = authenticated_client.get(reverse("main:api_status")).json()
        assert after_clear["has_file"] is False

    def test_status_cleans_up_invalid_or_missing_session_file(self, authenticated_client, user):
        session = authenticated_client.session
        session["uploaded_file_path"] = "/etc/passwd"
        session.save()

        res = authenticated_client.get(reverse("main:api_status"))
        assert res.status_code == 200
        assert res.json()["has_file"] is False

        missing_path = os.path.join(_user_upload_dir(user.id), "missing.inp")
        session = authenticated_client.session
        session["uploaded_file_path"] = missing_path
        session.save()

        res_missing = authenticated_client.get(reverse("main:api_status"))
        assert res_missing.status_code == 200
        assert res_missing.json()["has_file"] is False

    def test_upload_inp_via_api_without_xhr_header_returns_json(self, authenticated_client):
        valid_inp = (
            b"[TITLE]\nAPI Test\n[OPTIONS]\nFLOW_UNITS CMS\n"
            b"[SUBCATCHMENTS]\nS1 1 2 10 50 100 0.5 0\n"
            b"[SUBAREAS]\nS1 0.01 0.1 0.05 0.05 25 ZERO\n"
        )
        upload_file = SimpleUploadedFile("api_model.inp", valid_inp, content_type="text/plain")
        res = authenticated_client.post(reverse("main:api_upload"), {"file": upload_file})
        assert res.status_code == 200
        assert res.json()["message"] == "File was sent."


@pytest.mark.django_db
class TestApiSimulation:
    """Tests for GET and POST /api/simulation/."""

    def test_get_without_cached_results_returns_404(self, authenticated_client):
        res = authenticated_client.get(reverse("main:api_simulation"))
        assert res.status_code == 404
        assert res.json()["error"] == NO_SIMULATION_RESULTS_MESSAGE

        # Stale token in session is also cleaned up on 404
        session = authenticated_client.session
        session[SIM_RESULT_TOKEN_SESSION_KEY] = "0" * 32
        session.save()
        res_stale = authenticated_client.get(reverse("main:api_simulation"))
        assert res_stale.status_code == 404
        assert SIM_RESULT_TOKEN_SESSION_KEY not in authenticated_client.session

    def test_post_json_and_get_simulation_results(self, loaded_model_client, monkeypatch):
        monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)

        payload = {
            "option": "simulate_percent_slope",
            "start": 1,
            "stop": 5,
            "step": 2,
            "catchment_name": "S1",
        }
        post_res = loaded_model_client.post(
            reverse("main:api_simulation"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert post_res.status_code == 200
        body = post_res.json()
        assert body["feature_name"] == "PercSlope"
        assert body["columns"] == [
            "PercSlope",
            "runoff",
            "peak_runoff_rate",
            "infiltration",
            "evaporation",
        ]
        assert len(body["data"]) == 3
        assert len(body["records"]) == 3
        assert body["download_token"]

        get_res = loaded_model_client.get(reverse("main:api_simulation"))
        assert get_res.status_code == 200
        assert get_res.json()["download_token"] == body["download_token"]

    def test_post_invalid_json_or_boolean_numbers_rejected(self, loaded_model_client):
        bad_json = loaded_model_client.post(
            reverse("main:api_simulation"),
            data="{broken",
            content_type="application/json",
        )
        assert bad_json.status_code == 400
        assert bad_json.json()["error"] == INVALID_JSON_MESSAGE

        list_json = loaded_model_client.post(
            reverse("main:api_simulation"),
            data="[1, 2, 3]",
            content_type="application/json",
        )
        assert list_json.status_code == 400
        assert list_json.json()["error"] == INVALID_JSON_MESSAGE

        bool_json = loaded_model_client.post(
            reverse("main:api_simulation"),
            data=json.dumps(
                {
                    "option": "simulate_percent_slope",
                    "start": True,
                    "stop": 5,
                    "step": 1,
                    "catchment_name": "S1",
                }
            ),
            content_type="application/json",
        )
        assert bool_json.status_code == 400
        assert bool_json.json()["error"] == FORM_INVALID_MESSAGE
        assert "start" in bool_json.json()["field_errors"]

        extra_json = loaded_model_client.post(
            reverse("main:api_simulation"),
            data=json.dumps(
                {
                    "option": "simulate_percent_slope",
                    "start": 1,
                    "stop": 5,
                    "step": 1,
                    "catchment_name": "S1",
                    "unexpected": "value",
                }
            ),
            content_type="application/json",
        )
        assert extra_json.status_code == 400
        assert "unexpected" in extra_json.json()["field_errors"]

        non_str_json = loaded_model_client.post(
            reverse("main:api_simulation"),
            data=json.dumps(
                {
                    "option": "simulate_percent_slope",
                    "start": 1,
                    "stop": 5,
                    "step": 1,
                    "catchment_name": ["S1"],
                }
            ),
            content_type="application/json",
        )
        assert non_str_json.status_code == 400
        assert "catchment_name" in non_str_json.json()["field_errors"]

        bad_utf8 = loaded_model_client.generic(
            "POST",
            reverse("main:api_simulation"),
            b"\xff",
            content_type="application/json",
        )
        assert bad_utf8.status_code == 400
        assert bad_utf8.json()["error"] == INVALID_JSON_MESSAGE

    def test_post_form_validation_error_and_run_failure(
        self, authenticated_client, loaded_model_client, monkeypatch
    ):
        invalid_res = loaded_model_client.post(
            reverse("main:api_simulation"),
            data={"option": "simulate_percent_slope", "start": 10, "stop": 1, "step": 1},
        )
        assert invalid_res.status_code == 400
        assert invalid_res.json()["error"] == FORM_INVALID_MESSAGE
        assert "stop" in invalid_res.json()["field_errors"]

        extra_form_res = loaded_model_client.post(
            reverse("main:api_simulation"),
            data={"option": "simulate_n_imperv", "catchment_name": "S1", "extra": "x"},
        )
        assert extra_form_res.status_code == 400
        assert "extra" in extra_form_res.json()["field_errors"]

        # SubcatchmentNotFoundError mapped to catchment_name field error
        class MissingSubcatchmentSim(DummyFeaturesSimulation):
            def __init__(self, subcatchment_id, raw_file):
                raise SubcatchmentNotFoundError(subcatchment_id, ["S2"])

        monkeypatch.setattr("main.views.FeaturesSimulation", MissingSubcatchmentSim)
        not_found_res = loaded_model_client.post(
            reverse("main:api_simulation"),
            data=json.dumps(
                {
                    "option": "simulate_percent_slope",
                    "start": 1,
                    "stop": 5,
                    "step": 2,
                    "catchment_name": "S1",
                }
            ),
            content_type="application/json",
        )
        assert not_found_res.status_code == 400
        assert "catchment_name" in not_found_res.json()["field_errors"]


@pytest.mark.django_db
class TestApiTimeseries:
    """Tests for GET and POST /api/timeseries/."""

    def test_get_without_cached_results_returns_404(self, authenticated_client):
        res = authenticated_client.get(reverse("main:api_timeseries"))
        assert res.status_code == 404
        assert res.json()["error"] == NO_TIMESERIES_RESULTS_MESSAGE

        session = authenticated_client.session
        session[TS_RESULT_TOKEN_SESSION_KEY] = "0" * 32
        session.save()
        res_stale = authenticated_client.get(reverse("main:api_timeseries"))
        assert res_stale.status_code == 404
        assert TS_RESULT_TOKEN_SESSION_KEY not in authenticated_client.session

    def test_post_single_and_sweep_timeseries(self, loaded_model_client, monkeypatch):
        monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)

        # Omitting "mode" defaults to "single" per TimeseriesRequest schema
        single_res = loaded_model_client.post(
            reverse("main:api_timeseries"),
            data=json.dumps({"catchment_name": "S1"}),
            content_type="application/json",
        )
        assert single_res.status_code == 200
        single_body = single_res.json()
        assert single_body["mode"] == "single"
        assert len(single_body["data"]) == 3
        assert single_body["metrics"] is not None
        assert single_body["download_token"]

        get_res = loaded_model_client.get(reverse("main:api_timeseries"))
        assert get_res.status_code == 200
        assert get_res.json()["mode"] == "single"

        sweep_res = loaded_model_client.post(
            reverse("main:api_timeseries"),
            data=json.dumps(
                {
                    "mode": "sweep",
                    "feature": "PercImperv",
                    "start": 0,
                    "stop": 20,
                    "step": 10,
                    "catchment_name": "S1",
                }
            ),
            content_type="application/json",
        )
        assert sweep_res.status_code == 200
        sweep_body = sweep_res.json()
        assert sweep_body["mode"] == "sweep"
        assert len(sweep_body["sweep_summary"]) == 3

    def test_post_timeseries_validation_and_runtime_errors(self, loaded_model_client, monkeypatch):
        bad_json = loaded_model_client.post(
            reverse("main:api_timeseries"),
            data="{bad",
            content_type="application/json",
        )
        assert bad_json.status_code == 400

        invalid_form = loaded_model_client.post(
            reverse("main:api_timeseries"),
            data=json.dumps({"mode": "sweep", "catchment_name": "S1"}),
            content_type="application/json",
        )
        assert invalid_form.status_code == 400
        assert "start" in invalid_form.json()["field_errors"]

        class BrokenTimeseriesSim(DummyFeaturesSimulation):
            def calculate_timeseries(self):
                raise RuntimeError("boom")

        monkeypatch.setattr("main.views.FeaturesSimulation", BrokenTimeseriesSim)
        err_res = loaded_model_client.post(
            reverse("main:api_timeseries"),
            data=json.dumps({"mode": "single", "catchment_name": "S1"}),
            content_type="application/json",
        )
        assert err_res.status_code == 500


@pytest.mark.django_db
class TestApiCalculations:
    """Tests for GET and POST /api/calculations/."""

    def test_get_without_cached_results_returns_404(self, authenticated_client):
        res = authenticated_client.get(reverse("main:api_calculations"))
        assert res.status_code == 404
        assert res.json()["error"] == NO_CALCULATIONS_RESULTS_MESSAGE

    def test_post_without_model_returns_400(self, authenticated_client):
        res = authenticated_client.post(reverse("main:api_calculations"))
        assert res.status_code == 400
        assert res.json()["error"] == NO_MODEL_MESSAGE

    def test_post_and_get_calculations(self, loaded_model_client, monkeypatch):
        df = pd.DataFrame(
            {
                "Name": ["S1", "S2"],
                "SWMM_Runoff": [10.0, 20.0],
                "ANN_Runoff": [11.0, 19.0],
            }
        )
        monkeypatch.setattr("main.views._compare_swmm_and_ann", lambda path: (df, "10³ m³"))

        post_res = loaded_model_client.post(reverse("main:api_calculations"))
        assert post_res.status_code == 200
        body = post_res.json()
        assert body["unit"] == "10³ m³"
        assert len(body["rows"]) == 2
        assert body["metrics"]["count"] == 2
        assert body["download_token"]

        get_res = loaded_model_client.get(reverse("main:api_calculations"))
        assert get_res.status_code == 200
        assert get_res.json()["rows"] == body["rows"]

    def test_oversized_api_post_returns_413_json(self, authenticated_client, settings):
        settings.DATA_UPLOAD_MAX_MEMORY_SIZE = 32
        res = authenticated_client.post(
            reverse("main:api_simulation"),
            data=json.dumps({"option": "simulate_n_imperv", "catchment_name": "S" * 64}),
            content_type="application/json",
        )
        assert res.status_code == 413
        assert res.json()["error"] == "Request body too large."
