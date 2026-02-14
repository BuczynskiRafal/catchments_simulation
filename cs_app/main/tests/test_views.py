import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.contrib.messages.middleware import MessageMiddleware
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from django.urls import reverse

from main.views import (
    _get_catchment_choices,
    _get_subcatchment_ids,
    calculations,
    clear_session_variables,
    save_output_file,
    simulation_view,
    subcatchments,
    timeseries_view,
    upload,
    upload_clear,
    upload_status,
)


@pytest.mark.django_db
def test_main_view(client):
    """
    Test the main view by sending a GET request to the URL
    and checking that the response status code is 200.
    Also check that the correct template is used to render the response.
    """
    response = client.get(reverse("main:main_view"))
    assert response.status_code == 200
    assert "main/main_view.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_about_view(client):
    """
    Test the about view using Django's test client.
    """
    response = client.get(reverse("main:about"))
    assert response.status_code == 200
    assert "main/about.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_get(client):
    """
    Test the functionality of the contact view by checking the response status
    code and the name of the template used for rendering the response.
    """
    response = client.get(reverse("main:contact"))
    assert response.status_code == 200
    assert "main/contact.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_post_no_data(client):
    """
    Test that the contact view returns a 200 response when
    submitting no data via POST request.
    """
    response = client.post(reverse("main:contact"))
    assert response.status_code == 200


@pytest.mark.django_db
def test_user_profile_view(client, create_user, test_password):
    """
    Test the user profile view by logging in with a client,
    getting the user's profile, and checking that the response
    status code is 200 and the correct template was used.
    """
    client.login(username="testuser", password=test_password)
    response = client.get(reverse("main:userprofile", args=[create_user.id]))
    assert response.status_code == 200
    assert "main/userprofile.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_simulation_view(user):
    """
    Test the simulation view using a Django test client.
    """
    factory = RequestFactory()
    request = factory.post(
        "simulation",
        {
            "option": "simulate_percent_slope",
            "start": "1",
            "stop": "100",
            "step": "1",
            "catchment_name": "S1",
        },
    )
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()

    message_middleware = MessageMiddleware(lambda req: None)
    message_middleware.process_request(request)

    request.user = user
    response = simulation_view(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_simulation_view_get(user):
    """
    Test the simulation view's GET response by creating a new request via
    RequestFactory with middleware to handle sessions.
    The user is set to the provided user fixture.
    The response status code is asserted to be 200.
    """
    factory = RequestFactory()
    request = factory.get("simulation")
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()
    request.user = user

    response = simulation_view(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_simulation_view_prefills_form_state_from_session(client, user):
    """GET /simulation restores the previously selected form state."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1", "S2"]
    session["sim_form_state"] = {
        "option": "simulate_percent_slope",
        "start": 1,
        "stop": 11,
        "step": 2,
        "catchment_name": "S2",
    }
    session.save()

    response = client.get(reverse("main:simulation"))

    assert response.status_code == 200
    form = response.context["form"]
    assert form["option"].value() == "simulate_percent_slope"
    assert int(form["start"].value()) == 1
    assert int(form["stop"].value()) == 11
    assert int(form["step"].value()) == 2
    assert form["catchment_name"].value() == "S2"


@pytest.mark.django_db
def test_simulation_view_clears_unavailable_catchment_in_saved_state(client, user):
    """Saved simulation catchment is reset when not present in current choices."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session["sim_form_state"] = {
        "option": "simulate_percent_slope",
        "start": 1,
        "stop": 11,
        "step": 2,
        "catchment_name": "S2",
    }
    session.save()

    response = client.get(reverse("main:simulation"))

    assert response.status_code == 200
    form = response.context["form"]
    assert not form["catchment_name"].value()

    updated_session = client.session
    assert updated_session["sim_form_state"]["catchment_name"] == ""


@pytest.mark.django_db
def test_simulation_view_sanitizes_invalid_saved_form_values(client, user):
    """Invalid saved values are dropped before being used as form initial data."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session["sim_form_state"] = {
        "option": "invalid-option",
        "start": "<script>alert(1)</script>",
        "stop": "bad",
        "step": "NaN",
        "catchment_name": "S1",
    }
    session.save()

    response = client.get(reverse("main:simulation"))

    assert response.status_code == 200
    form = response.context["form"]
    assert form["catchment_name"].value() == "S1"
    assert int(form["start"].value()) == 1
    assert int(form["stop"].value()) == 10
    assert int(form["step"].value()) == 1

    updated_session = client.session
    assert updated_session["sim_form_state"] == {"catchment_name": "S1"}


@pytest.mark.django_db
def test_simulation_view_ignores_none_numeric_values_in_saved_state(client, user):
    """None numeric values from session should not override form defaults."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session["sim_form_state"] = {
        "option": "simulate_n_imperv",
        "start": None,
        "stop": None,
        "step": None,
        "catchment_name": "S1",
    }
    session.save()

    response = client.get(reverse("main:simulation"))

    assert response.status_code == 200
    form = response.context["form"]
    assert form["option"].value() == "simulate_n_imperv"
    assert int(form["start"].value()) == 1
    assert int(form["stop"].value()) == 10
    assert int(form["step"].value()) == 1
    assert form["catchment_name"].value() == "S1"

    updated_session = client.session
    assert updated_session["sim_form_state"] == {
        "option": "simulate_n_imperv",
        "catchment_name": "S1",
    }


@pytest.mark.django_db
def test_simulation_view_post_range_persists_form_state(client, user, monkeypatch):
    """Successful simulation POST stores selections for subsequent GET requests."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class DummyFeaturesSimulation:
        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def simulate_percent_slope(self, start, stop, step):
            return pd.DataFrame(
                {
                    "PercSlope": [start, stop],
                    "runoff": [1.0, 2.0],
                }
            )

    def fake_save_output_file(request, df, output_file_name, session_prefix=""):
        request.session[f"{session_prefix}output_file_url"] = "/media/fake.xlsx"
        request.session[f"{session_prefix}output_file_name"] = output_file_name

    monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)
    monkeypatch.setattr("main.views.save_output_file", fake_save_output_file)

    response = client.post(
        reverse("main:simulation"),
        data={
            "option": "simulate_percent_slope",
            "start": "1",
            "stop": "5",
            "step": "2",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 302
    assert response.url == reverse("main:simulation")

    updated_session = client.session
    assert updated_session["sim_form_state"] == {
        "option": "simulate_percent_slope",
        "start": 1,
        "stop": 5,
        "step": 2,
        "catchment_name": "S1",
    }

    get_response = client.get(reverse("main:simulation"))
    assert get_response.status_code == 200
    form = get_response.context["form"]
    assert form["option"].value() == "simulate_percent_slope"
    assert int(form["start"].value()) == 1
    assert int(form["stop"].value()) == 5
    assert int(form["step"].value()) == 2
    assert form["catchment_name"].value() == "S1"


@pytest.mark.django_db
def test_simulation_view_post_failure_does_not_persist_form_state(client, user, monkeypatch):
    """Failed simulation run must not persist form state in session."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class FailingFeaturesSimulation:
        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("main.views.FeaturesSimulation", FailingFeaturesSimulation)

    response = client.post(
        reverse("main:simulation"),
        data={
            "option": "simulate_percent_slope",
            "start": "1",
            "stop": "5",
            "step": "2",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 200
    assert "sim_form_state" not in client.session


# @pytest.mark.django_db
# def test_download_result():
#     """
#     Test the download_result view function by simulating a request to the
#     view and checking that the response
#     has status code 200. Uses the Django test client.
#     """
#     factory = RequestFactory()
#     request = factory.get("simulation")

#     response = download_result(request)

#     assert response.status_code == 200


@pytest.mark.django_db
def test_calculations():
    """
    Test calculations function.
    Uses Django test client to create a request to 'calculations' view.
    Asserts that the response status code is 200.
    """
    factory = RequestFactory()
    request = factory.post("calculations")

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    message_middleware = MessageMiddleware(lambda req: None)
    message_middleware.process_request(request)

    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_calculations_get():
    """
    Test calculations get.
    This function tests the calculations view function with a GET request.
    It creates a request factory, sets the request
    method to GET, and sets the user attribute to AnonymousUser. It then calls
    the calculations view function with the
    created request object and asserts that the response status code is 200.

    """
    factory = RequestFactory()
    request = factory.get("calculations")
    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_calculations_uses_model_after_simulation(monkeypatch):
    """
    Ensure calculations rebuilds swmmio.Model after SWMM run so report columns exist.
    """
    state = {"report_ready": False}

    class FakeSimulation:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __iter__(self):
            yield object()

        def __exit__(self, exc_type, exc, tb):
            state["report_ready"] = True

    class FakeModel:
        def __init__(self, _path):
            base = pd.DataFrame(index=["S1"], data={"PercImperv": [10.0]})
            if state["report_ready"]:
                base["TotalRunoffMG"] = [12.34]
            self.subcatchments = SimpleNamespace(dataframe=base)

    monkeypatch.setattr("main.views.Simulation", FakeSimulation)
    monkeypatch.setattr("main.views.swmmio.Model", FakeModel)
    monkeypatch.setattr("main.views.predict_runoff", lambda _model: np.array([4.56]))
    monkeypatch.setattr("main.views._cleanup_swmm_side_files", lambda _path: None)

    factory = RequestFactory()
    request = factory.post("calculations")

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "uploaded_files/test.inp"
    request.session.save()

    message_middleware = MessageMiddleware(lambda req: None)
    message_middleware.process_request(request)

    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200
    assert b"S1" in response.content
    assert b"12.34" in response.content


@pytest.mark.django_db
def test_upload_unauthenticated_ajax_returns_401():
    """
    Test that unauthenticated AJAX requests to upload view return 401 with login URL.
    """
    factory = RequestFactory()
    request = factory.post(
        "/upload/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    request.user = AnonymousUser()

    response = upload(request)

    assert response.status_code == 401
    data = json.loads(response.content)
    assert "error" in data
    assert "login_url" in data
    assert data["login_url"] == settings.LOGIN_URL


@pytest.mark.django_db
def test_upload_unauthenticated_regular_request_redirects():
    """
    Test that unauthenticated regular (non-AJAX) requests redirect to login.
    """
    factory = RequestFactory()
    request = factory.post("/upload/")

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    request.user = AnonymousUser()

    response = upload(request)

    assert response.status_code == 302
    assert settings.LOGIN_URL in response.url


@pytest.mark.django_db
def test_upload_authenticated_user_can_upload(user):
    """
    Test that authenticated users can upload files and the file is saved to disk.
    """
    factory = RequestFactory()

    inp_content = b"[TITLE]\nTest File\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("test_upload.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    request.user = user

    expected_path = os.path.join("uploaded_files", str(user.id), "test_upload.inp")
    try:
        response = upload(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert "message" in data
        assert os.path.exists(expected_path), f"File was not saved at {expected_path}"

        with open(expected_path, "rb") as f:
            saved_content = f.read()
        assert saved_content == inp_content, "Saved file content does not match uploaded content"
        assert request.session.get("uploaded_file_path") == expected_path
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_clears_timeseries_form_state(user):
    """Uploading a file invalidates persisted timeseries form state."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nTest File\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("test_upload.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S1"}
    request.session.save()
    request.user = user

    expected_path = os.path.join("uploaded_files", str(user.id), "test_upload.inp")
    try:
        response = upload(request)

        assert response.status_code == 200
        assert "ts_form_state" not in request.session
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_clears_simulation_form_state(user):
    """Uploading a file invalidates persisted simulation form state."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nTest File\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("test_upload.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["sim_form_state"] = {"option": "simulate_percent_slope", "catchment_name": "S1"}
    request.session.save()
    request.user = user

    expected_path = os.path.join("uploaded_files", str(user.id), "test_upload.inp")
    try:
        response = upload(request)

        assert response.status_code == 200
        assert "sim_form_state" not in request.session
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_failure_preserves_existing_form_state_and_subcatchment_cache(user):
    """Invalid upload should not mutate existing session state."""
    factory = RequestFactory()
    uploaded_file = SimpleUploadedFile(
        "invalid.txt",
        b"not an inp",
        content_type="text/plain",
    )

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "uploaded_files/original.inp"
    request.session["_subcatchment_ids_file"] = "uploaded_files/original.inp"
    request.session["_subcatchment_ids"] = ["S1", "S2"]
    request.session["sim_form_state"] = {"option": "simulate_percent_slope", "catchment_name": "S2"}
    request.session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S2"}
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 400
    assert request.session["uploaded_file_path"] == "uploaded_files/original.inp"
    assert request.session["_subcatchment_ids_file"] == "uploaded_files/original.inp"
    assert request.session["_subcatchment_ids"] == ["S1", "S2"]
    assert request.session["sim_form_state"] == {
        "option": "simulate_percent_slope",
        "catchment_name": "S2",
    }
    assert request.session["ts_form_state"] == {"mode": "sweep", "catchment_name": "S2"}


@pytest.mark.django_db
def test_save_output_file_creates_directory(user):
    """
    Test that save_output_file saves to default_storage, sets session keys,
    and cleans up the temp file.
    """
    from django.core.files.storage import default_storage

    output_dir = "output_files"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    factory = RequestFactory()
    request = factory.get("/")
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()

    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    save_output_file(request, df, "test_result.xlsx")

    assert request.session["output_file_name"] == "test_result.xlsx"
    assert "output_file_url" in request.session
    assert default_storage.exists("test_result.xlsx")

    # Temp file should be cleaned up
    if os.path.isdir(output_dir):
        assert len(os.listdir(output_dir)) == 0

    default_storage.delete("test_result.xlsx")
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.django_db
def test_timeseries_view_get(user):
    """
    Test the timeseries view's GET response by creating a new request via
    RequestFactory with middleware to handle sessions.
    The user is set to the provided user fixture.
    The response status code is asserted to be 200.
    """
    factory = RequestFactory()
    request = factory.get("timeseries")
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()
    request.user = user

    response = timeseries_view(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_timeseries_view_prefills_form_state_from_session(client, user):
    """GET /timeseries restores the previously selected form state."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1", "S2"]
    session["ts_form_state"] = {
        "mode": "sweep",
        "feature": "Area",
        "start": 1.0,
        "stop": 5.0,
        "step": 0.5,
        "catchment_name": "S2",
    }
    session.save()

    response = client.get(reverse("main:timeseries"))

    assert response.status_code == 200
    form = response.context["form"]
    assert form["mode"].value() == "sweep"
    assert form["feature"].value() == "Area"
    assert float(form["start"].value()) == 1.0
    assert float(form["stop"].value()) == 5.0
    assert float(form["step"].value()) == 0.5
    assert form["catchment_name"].value() == "S2"


@pytest.mark.django_db
def test_timeseries_view_clears_unavailable_catchment_in_saved_state(client, user):
    """Saved catchment is reset when it is not present in current choices."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session["ts_form_state"] = {
        "mode": "sweep",
        "feature": "Area",
        "start": 1.0,
        "stop": 5.0,
        "step": 1.0,
        "catchment_name": "S2",
    }
    session.save()

    response = client.get(reverse("main:timeseries"))

    assert response.status_code == 200
    form = response.context["form"]
    assert not form["catchment_name"].value()

    updated_session = client.session
    assert updated_session["ts_form_state"]["catchment_name"] == ""


@pytest.mark.django_db
def test_timeseries_view_post_sweep_persists_form_state(client, user, monkeypatch):
    """Successful sweep POST stores form selections for subsequent GET requests."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class DummyFeaturesSimulation:
        TIMESERIES_KEYS = ("rainfall", "runoff", "infiltration_loss", "evaporation_loss", "runon")

        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def simulate_subcatchment_timeseries(self, feature, start, stop, step):
            idx = pd.date_range("2025-01-01", periods=2, freq="h", name="datetime")
            df = pd.DataFrame(
                {
                    "rainfall": [0.1, 0.0],
                    "runoff": [1.0, 0.5],
                    "infiltration_loss": [0.0, 0.0],
                    "evaporation_loss": [0.0, 0.0],
                    "runon": [0.0, 0.0],
                },
                index=idx,
            )
            return {start: df}

    def fake_save_sweep_output(request, results, output_file_name):
        request.session["ts_output_file_url"] = "/media/fake.xlsx"
        request.session["ts_output_file_name"] = output_file_name

    monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)
    monkeypatch.setattr("main.views._save_sweep_output", fake_save_sweep_output)

    response = client.post(
        reverse("main:timeseries"),
        data={
            "mode": "sweep",
            "feature": "PercSlope",
            "start": "0",
            "stop": "20",
            "step": "5",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 302
    assert response.url == reverse("main:timeseries")

    updated_session = client.session
    assert updated_session["ts_form_state"] == {
        "mode": "sweep",
        "feature": "PercSlope",
        "start": 0.0,
        "stop": 20.0,
        "step": 5.0,
        "catchment_name": "S1",
    }

    get_response = client.get(reverse("main:timeseries"))
    assert get_response.status_code == 200
    form = get_response.context["form"]
    assert form["mode"].value() == "sweep"
    assert form["feature"].value() == "PercSlope"
    assert float(form["start"].value()) == 0.0
    assert float(form["stop"].value()) == 20.0
    assert float(form["step"].value()) == 5.0
    assert form["catchment_name"].value() == "S1"


@pytest.mark.django_db
def test_timeseries_view_post_failure_does_not_persist_form_state(client, user, monkeypatch):
    """Failed timeseries run must not persist form state in session."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class FailingFeaturesSimulation:
        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("main.views.FeaturesSimulation", FailingFeaturesSimulation)

    response = client.post(
        reverse("main:timeseries"),
        data={
            "mode": "sweep",
            "feature": "PercSlope",
            "start": "0",
            "stop": "20",
            "step": "5",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 200
    assert "ts_form_state" not in client.session


@pytest.mark.django_db
def test_simulation_form_predefined_method_valid():
    """
    Test that SimulationForm accepts predefined methods without start/stop/step.
    """
    from main.forms import SimulationForm

    form = SimulationForm(
        data={
            "option": "simulate_n_imperv",
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
    )
    assert form.is_valid(), f"Form errors: {form.errors}"


@pytest.mark.django_db
def test_simulation_form_range_method_requires_params():
    """
    Test that SimulationForm rejects range methods without start/stop/step.
    """
    from main.forms import SimulationForm

    form = SimulationForm(
        data={
            "option": "simulate_percent_slope",
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
    )
    assert not form.is_valid()
    assert "start" in form.errors
    assert "stop" in form.errors
    assert "step" in form.errors


@pytest.mark.django_db
def test_timeseries_form_sweep_start_zero_is_valid():
    """
    Test that TimeseriesForm accepts start=0 in sweep mode (regression for falsy check).
    """
    from main.forms import TimeseriesForm

    form = TimeseriesForm(
        data={
            "mode": "sweep",
            "feature": "PercSlope",
            "start": "0",
            "stop": "100",
            "step": "10",
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
    )
    assert form.is_valid(), f"Form errors: {form.errors}"


@pytest.mark.django_db
def test_timeseries_form_sweep_too_many_steps():
    """
    Test that TimeseriesForm rejects sweeps exceeding MAX_SWEEP_STEPS.
    """
    from main.forms import TimeseriesForm

    form = TimeseriesForm(
        data={
            "mode": "sweep",
            "feature": "PercSlope",
            "start": "0",
            "stop": "1000",
            "step": "1",
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
    )
    assert not form.is_valid()
    assert "step" in form.errors


@pytest.mark.django_db
def test_upload_status_no_file(user):
    """Test that upload_status returns has_file=False when no file in session."""
    factory = RequestFactory()
    request = factory.get(
        "/upload/status/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload_status(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["has_file"] is False


@pytest.mark.django_db
def test_upload_status_with_file(user, tmp_path):
    """Test that upload_status returns file info when a file exists in session."""
    test_file = tmp_path / "test.inp"
    test_file.write_text("[TITLE]\nTest\n")

    factory = RequestFactory()
    request = factory.get(
        "/upload/status/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = str(test_file)
    request.session.save()
    request.user = user

    response = upload_status(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["has_file"] is True
    assert data["filename"] == "test.inp"
    assert data["size"] > 0


@pytest.mark.django_db
def test_upload_status_stale_reference(user):
    """Test that upload_status cleans up session when file no longer exists on disk."""
    factory = RequestFactory()
    request = factory.get(
        "/upload/status/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "/nonexistent/path/file.inp"
    request.session.save()
    request.user = user

    response = upload_status(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["has_file"] is False
    assert "uploaded_file_path" not in request.session


@pytest.mark.django_db
def test_clear_session_preserves_uploaded_file(user):
    """Test that clear_session_variables does not remove uploaded_file_path."""
    factory = RequestFactory()
    request = factory.get("/")

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "uploaded_files/test.inp"
    request.session["show_download_button"] = True
    request.session["chart_config"] = {"data": []}
    request.session["sim_form_state"] = {"option": "simulate_percent_slope", "catchment_name": "S1"}
    request.session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S1"}
    request.session.save()

    clear_session_variables(request)

    assert request.session.get("uploaded_file_path") == "uploaded_files/test.inp"
    assert "show_download_button" not in request.session
    assert "chart_config" not in request.session
    assert "sim_form_state" not in request.session
    assert "ts_form_state" not in request.session


# ── upload_clear tests (#4) ──────────────────────────────────────────────


@pytest.mark.django_db
def test_upload_clear_unauthenticated_ajax_returns_401():
    """upload_clear returns 401 for unauthenticated AJAX requests."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = AnonymousUser()

    response = upload_clear(request)

    assert response.status_code == 401


@pytest.mark.django_db
def test_upload_clear_wrong_method(user):
    """upload_clear rejects GET requests with 405."""
    factory = RequestFactory()
    request = factory.get(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 405


@pytest.mark.django_db
def test_upload_clear_removes_file_and_session(user, tmp_path):
    """upload_clear deletes the file from disk and removes session key."""
    test_file = tmp_path / "to_delete.inp"
    test_file.write_text("[TITLE]\nTest\n")

    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = str(test_file)
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 200
    assert "uploaded_file_path" not in request.session
    assert not test_file.exists()


@pytest.mark.django_db
def test_upload_clear_no_file_in_session(user):
    """upload_clear succeeds even when no file path is stored in session."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["message"] == "Upload cleared."


@pytest.mark.django_db
def test_upload_clear_stale_path(user):
    """upload_clear handles a session path pointing to a non-existent file."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "/nonexistent/file.inp"
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 200
    assert "uploaded_file_path" not in request.session


@pytest.mark.django_db
def test_upload_clear_removes_timeseries_form_state(user):
    """upload_clear removes persisted timeseries form state."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S1"}
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 200
    assert "ts_form_state" not in request.session


@pytest.mark.django_db
def test_upload_clear_removes_simulation_form_state(user):
    """upload_clear removes persisted simulation form state."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/clear/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["sim_form_state"] = {
        "option": "simulate_percent_slope",
        "catchment_name": "S1",
    }
    request.session.save()
    request.user = user

    response = upload_clear(request)

    assert response.status_code == 200
    assert "sim_form_state" not in request.session


# ── upload_status auth test (#5) ─────────────────────────────────────────


@pytest.mark.django_db
def test_upload_status_unauthenticated_ajax_returns_401():
    """upload_status returns 401 for unauthenticated AJAX requests."""
    factory = RequestFactory()
    request = factory.get(
        "/upload/status/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = AnonymousUser()

    response = upload_status(request)

    assert response.status_code == 401


@pytest.mark.django_db
def test_upload_status_unauthenticated_regular_redirects():
    """upload_status redirects non-AJAX unauthenticated requests to login."""
    factory = RequestFactory()
    request = factory.get("/upload/status/")
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = AnonymousUser()

    response = upload_status(request)

    assert response.status_code == 302
    assert settings.LOGIN_URL in response.url


@pytest.mark.django_db
def test_upload_status_rejects_post(user):
    """upload_status rejects POST requests (require_GET)."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/status/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload_status(request)

    assert response.status_code == 405


# ── Core flow test (#8) ──────────────────────────────────────────────────


@pytest.mark.django_db
def test_upload_persists_after_clear_session_and_new_simulation(user):
    """
    Core flow: upload file -> clear_session_variables (as simulation does)
    -> file path remains in session -> upload_status still returns it.
    """
    factory = RequestFactory()

    # 1) Upload a file
    inp_content = b"[TITLE]\nPersistence Test\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("persist_test.inp", inp_content, content_type="text/plain")
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    saved_path = None
    try:
        response = upload(request)
        assert response.status_code == 200
        saved_path = request.session.get("uploaded_file_path")
        assert saved_path is not None

        # 2) Simulate what happens after a simulation run – clear results
        request.session["show_download_button"] = True
        request.session["chart_config"] = {"data": []}
        clear_session_variables(request)

        # 3) Verify file path survived
        assert request.session.get("uploaded_file_path") == saved_path
        assert os.path.exists(saved_path)

        # 4) Verify upload_status reports the file
        status_request = factory.get(
            "/upload/status/",
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        session_middleware.process_request(status_request)
        status_request.session = request.session
        status_request.user = user

        status_response = upload_status(status_request)
        data = json.loads(status_response.content)
        assert data["has_file"] is True
        assert data["filename"] == "persist_test.inp"
    finally:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


@pytest.mark.django_db
def test_upload_replaces_old_file_on_disk(user):
    """Uploading a new file removes the old file from disk (#2, #9)."""
    factory = RequestFactory()

    # Upload first file
    inp_a = b"[TITLE]\nFile A\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    request = factory.post(
        "/upload/",
        {"file": SimpleUploadedFile("file_a.inp", inp_a, content_type="text/plain")},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user
    path_b = None
    try:
        upload(request)
        path_a = request.session["uploaded_file_path"]
        assert os.path.exists(path_a)

        # Upload second file (different name)
        inp_b = b"[TITLE]\nFile B\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
        request2 = factory.post(
            "/upload/",
            {"file": SimpleUploadedFile("file_b.inp", inp_b, content_type="text/plain")},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        session_middleware.process_request(request2)
        request2.session = request.session
        request2.user = user
        upload(request2)
        path_b = request2.session["uploaded_file_path"]

        # Old file should be gone, new file present
        assert not os.path.exists(path_a), "Old file was not cleaned up"
        assert os.path.exists(path_b)
    finally:
        if path_b and os.path.exists(path_b):
            os.remove(path_b)


# ── subcatchments endpoint tests ─────────────────────────────────────────


@pytest.mark.django_db
def test_subcatchments_unauthenticated_ajax_returns_401():
    """subcatchments returns 401 for unauthenticated AJAX requests."""
    factory = RequestFactory()
    request = factory.get(
        "/subcatchments/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = AnonymousUser()

    response = subcatchments(request)

    assert response.status_code == 401


@pytest.mark.django_db
def test_subcatchments_no_file(user):
    """subcatchments returns empty list when no file is uploaded."""
    factory = RequestFactory()
    request = factory.get(
        "/subcatchments/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = subcatchments(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["subcatchments"] == []


@pytest.mark.django_db
def test_subcatchments_stale_path(user):
    """subcatchments returns empty list when session path points to non-existent file."""
    factory = RequestFactory()
    request = factory.get(
        "/subcatchments/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = "/nonexistent/file.inp"
    request.session.save()
    request.user = user

    response = subcatchments(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert data["subcatchments"] == []


@pytest.mark.django_db
def test_get_catchment_choices_no_file():
    """_get_catchment_choices returns placeholder when no file in session."""
    factory = RequestFactory()
    request = factory.get("/")
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    choices = _get_catchment_choices(request)

    assert len(choices) == 1
    assert choices[0][0] == ""
    assert "Upload" in choices[0][1]


@pytest.mark.django_db
def test_subcatchments_with_real_inp_file(user):
    """subcatchments returns correct IDs from a real INP file."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "tests", "fixtures", "example.inp"
    )
    fixture_path = os.path.abspath(fixture_path)
    if not os.path.exists(fixture_path):
        pytest.skip("example.inp fixture not found")

    factory = RequestFactory()
    request = factory.get(
        "/subcatchments/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = fixture_path
    request.session.save()
    request.user = user

    response = subcatchments(request)

    assert response.status_code == 200
    data = json.loads(response.content)
    assert "S1" in data["subcatchments"]
    assert len(data["subcatchments"]) > 0


@pytest.mark.django_db
def test_subcatchment_ids_cached_in_session(user):
    """_get_subcatchment_ids caches results in the session."""
    fixture_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "tests", "fixtures", "example.inp"
    )
    fixture_path = os.path.abspath(fixture_path)
    if not os.path.exists(fixture_path):
        pytest.skip("example.inp fixture not found")

    factory = RequestFactory()
    request = factory.get("/")
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session["uploaded_file_path"] = fixture_path
    request.session.save()

    ids1 = _get_subcatchment_ids(request)
    assert "S1" in ids1
    assert request.session["_subcatchment_ids"] == ids1
    assert request.session["_subcatchment_ids_file"] == fixture_path

    # Second call should use cache (same result)
    ids2 = _get_subcatchment_ids(request)
    assert ids1 == ids2
