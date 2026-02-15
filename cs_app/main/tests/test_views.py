import json
import os
from io import BytesIO, StringIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.contrib.messages.middleware import MessageMiddleware
from django.contrib.sessions.middleware import SessionMiddleware
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from django.urls import reverse

from main.views import (
    SIM_RESULT_TOKEN_SESSION_KEY,
    TS_RESULT_TOKEN_SESSION_KEY,
    _get_catchment_choices,
    _get_subcatchment_ids,
    _result_cache_key,
    _safe_download_filename,
    _validate_inp_file_stream,
    calculations,
    clear_session_variables,
    simulation_view,
    subcatchments,
    timeseries_view,
    upload,
    upload_clear,
    upload_sample,
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

    monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)

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
    token = updated_session[SIM_RESULT_TOKEN_SESSION_KEY]
    cached_payload = cache.get(_result_cache_key("sim", user.id, token))
    assert cached_payload is not None

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


@pytest.mark.django_db
def test_simulation_view_validation_error_shows_human_message(client, user, monkeypatch):
    """Input-related ValueError should be shown as friendly flash message."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class InvalidInputFeaturesSimulation:
        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            raise ValueError("Expected numeric value in slope sheet.")

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("main.views.FeaturesSimulation", InvalidInputFeaturesSimulation)

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
    assert b"Input file contains non-numeric values where numbers are required." in response.content


@pytest.mark.django_db
def test_simulation_view_rejects_oversized_cached_payload(client, user, monkeypatch):
    """Oversized simulation payload should not be stored in cache/session token."""
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = "uploaded_files/test.inp"
    session["_subcatchment_ids_file"] = "uploaded_files/test.inp"
    session["_subcatchment_ids"] = ["S1"]
    session.save()

    class HugeFeaturesSimulation:
        def __init__(self, subcatchment_id, raw_file):
            self.subcatchment_id = subcatchment_id
            self.raw_file = raw_file

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def simulate_percent_slope(self, start, stop, step):
            size = 2000
            return pd.DataFrame({"PercSlope": list(range(size)), "runoff": list(range(size))})

    monkeypatch.setattr("main.views.FeaturesSimulation", HugeFeaturesSimulation)
    monkeypatch.setattr("main.views.MAX_RESULT_CACHE_BYTES", 200)

    response = client.post(
        reverse("main:simulation"),
        data={
            "option": "simulate_percent_slope",
            "start": "1",
            "stop": "2000",
            "step": "1",
            "catchment_name": "S1",
        },
    )

    assert response.status_code == 200
    assert SIM_RESULT_TOKEN_SESSION_KEY not in client.session


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
def test_upload_rejects_get_with_405(client, user):
    """upload rejects GET requests (require_POST)."""
    client.force_login(user)

    response = client.get(
        reverse("main:upload"),
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    assert response.status_code == 405


@pytest.mark.django_db
def test_upload_unauthenticated_ajax_get_returns_405(client):
    """GET is rejected by method guard before auth check."""
    response = client.get(
        reverse("main:upload"),
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    assert response.status_code == 405


@pytest.mark.django_db
def test_upload_returns_413_when_content_length_exceeds_body_limit(user):
    """upload returns 413 before multipart parsing when CONTENT_LENGTH is too large."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    request.META["CONTENT_LENGTH"] = str(settings.INP_UPLOAD_MAX_BODY_BYTES + 1)

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 413
    data = json.loads(response.content)
    assert "Maximum size is" in data["error"]


@pytest.mark.django_db
def test_upload_returns_413_when_uploaded_file_size_exceeds_limit(user):
    """upload returns 413 for files larger than configured size limit."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\n" + (b"x" * settings.INP_UPLOAD_MAX_BYTES)
    uploaded_file = SimpleUploadedFile("too_large.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 413
    data = json.loads(response.content)
    assert "Maximum size is" in data["error"]


@pytest.mark.django_db
def test_upload_returns_413_when_content_length_is_spoofed_low(user, monkeypatch):
    """Streaming body limit still returns 413 when CONTENT_LENGTH is spoofed below real body size."""
    monkeypatch.setattr("main.views.MAX_UPLOAD_BODY_SIZE", 100)
    monkeypatch.setattr("main.views.MAX_UPLOAD_SIZE", 10 * 1024 * 1024)

    factory = RequestFactory()
    inp_content = b"[TITLE]\n" + (b"x" * 2048) + b"\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("spoofed_size.inp", inp_content, content_type="text/plain")
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    request.META["CONTENT_LENGTH"] = "1"

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 413
    data = json.loads(response.content)
    assert "Maximum size is" in data["error"]


@pytest.mark.django_db
@pytest.mark.parametrize("raw_content_length", ["", "not-a-number", "-123"])
def test_upload_handles_invalid_or_negative_content_length(user, raw_content_length):
    """Invalid CONTENT_LENGTH variants should be rejected with 400."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nTest\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile(
        "content_length_variants.inp", inp_content, content_type="text/plain"
    )
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    request.META["CONTENT_LENGTH"] = raw_content_length

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "content_length_variants.inp"
    )
    try:
        response = upload(request)
        assert response.status_code == 400
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_handles_missing_content_length(user):
    """Missing CONTENT_LENGTH is treated as malformed multipart request metadata."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nMissing length\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("missing_length.inp", inp_content, content_type="text/plain")
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    request.META.pop("CONTENT_LENGTH", None)

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "missing_length.inp"
    )
    try:
        response = upload(request)
        assert response.status_code == 400
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


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

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "test_upload.inp"
    )
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
def test_upload_uses_stream_validator_not_bytes_validator(user, monkeypatch):
    """upload should validate INP content from file chunks, not full-file bytes."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nStreamed validation test\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile(
        "streamed_validation.inp", inp_content, content_type="text/plain"
    )

    called_stream_validator = {"called": False}

    def fake_stream_validator(uploaded_stream, chunk_size=8192):
        called_stream_validator["called"] = True
        uploaded_stream.seek(0)
        return True

    def fail_bytes_validator(_file_content):
        raise AssertionError("Legacy bytes validator must not be used in upload().")

    monkeypatch.setattr("main.views._validate_inp_file_stream", fake_stream_validator)
    monkeypatch.setattr("main.views._validate_inp_file_content", fail_bytes_validator)

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "streamed_validation.inp"
    )
    try:
        response = upload(request)

        assert response.status_code == 200
        assert called_stream_validator["called"] is True
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_valid_small_inp_still_succeeds(user):
    """A valid INP under size limit should still upload successfully."""
    factory = RequestFactory()
    inp_content = b"[TITLE]\nValid small file\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("valid_small.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "valid_small.inp"
    )
    try:
        response = upload(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert data["message"] == "File was sent."
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_invalid_content_returns_400(user):
    """upload returns 400 when file extension is valid but SWMM headers are missing."""
    factory = RequestFactory()
    uploaded_file = SimpleUploadedFile(
        "invalid_content.inp",
        b"this file does not contain expected swmm sections",
        content_type="text/plain",
    )

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 400
    data = json.loads(response.content)
    assert "Invalid file content" in data["error"]


@pytest.mark.django_db
def test_upload_binary_blob_with_single_marker_returns_400(user):
    """Single marker embedded in binary-like content should not pass validation."""
    factory = RequestFactory()
    payload = b"\x00\xff\x10garbage[TITLE]\x00\x01still-not-valid"
    uploaded_file = SimpleUploadedFile(
        "binary_single_marker.inp", payload, content_type="text/plain"
    )

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    response = upload(request)

    assert response.status_code == 400
    data = json.loads(response.content)
    assert "Invalid file content" in data["error"]


@pytest.mark.django_db
def test_upload_valid_utf16_bom_file_succeeds(user):
    """upload accepts valid UTF-16 LE files with BOM."""
    factory = RequestFactory()
    inp_text = "[TITLE]\nUTF16 file\n\n[OPTIONS]\nFLOW_UNITS LPS\n"
    inp_content = inp_text.encode("utf-16")
    uploaded_file = SimpleUploadedFile("utf16_valid.inp", inp_content, content_type="text/plain")

    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "utf16_valid.inp"
    )
    try:
        response = upload(request)

        assert response.status_code == 200
        assert request.session.get("uploaded_file_path") == expected_path
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


def test_validate_inp_file_stream_handles_chunk_boundary():
    """Header detection should work when markers are split across chunk boundaries."""

    class ChunkedUpload:
        def __init__(self, chunks):
            self._chunks = chunks
            self.seek_calls = []

        def chunks(self, _chunk_size):
            yield from self._chunks

        def seek(self, offset):
            self.seek_calls.append(offset)

    upload_obj = ChunkedUpload(
        [
            b"[TI",
            b"TLE]\nExample\n",
            b"[OP",
            b"TIONS]\nFLOW_UNITS LPS\n",
        ]
    )

    assert _validate_inp_file_stream(upload_obj, chunk_size=4) is True
    assert upload_obj.seek_calls == [0]


@pytest.mark.django_db
def test_upload_size_equal_limit_is_allowed(user, monkeypatch):
    """File size equal to MAX_UPLOAD_SIZE should be accepted."""
    content = b"[TITLE]\nA\n[OPTIONS]\nFLOW_UNITS LPS\n"
    monkeypatch.setattr("main.views.MAX_UPLOAD_SIZE", len(content))
    monkeypatch.setattr("main.views.MAX_UPLOAD_BODY_SIZE", len(content) + 1024)

    factory = RequestFactory()
    uploaded_file = SimpleUploadedFile("equal_limit.inp", content, content_type="text/plain")
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "equal_limit.inp"
    )
    try:
        response = upload(request)
        assert response.status_code == 200
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_body_length_equal_limit_is_allowed(user, monkeypatch):
    """CONTENT_LENGTH equal to MAX_UPLOAD_BODY_SIZE should not be rejected by pre-check."""
    factory = RequestFactory()
    content = b"[TITLE]\nA\n[OPTIONS]\nFLOW_UNITS LPS\n"
    uploaded_file = SimpleUploadedFile("equal_body_limit.inp", content, content_type="text/plain")
    request = factory.post(
        "/upload/",
        {"file": uploaded_file},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    monkeypatch.setattr("main.views.MAX_UPLOAD_SIZE", len(content) + 1024)
    monkeypatch.setattr(
        "main.views.MAX_UPLOAD_BODY_SIZE", int(request.META.get("CONTENT_LENGTH", "0"))
    )

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = user

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "equal_body_limit.inp"
    )
    try:
        response = upload(request)
        assert response.status_code == 200
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


@pytest.mark.django_db
def test_upload_sample_unauthenticated_ajax_returns_401():
    """Sample upload endpoint returns 401 for unauthenticated AJAX requests."""
    factory = RequestFactory()
    request = factory.post(
        "/upload/sample/",
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()
    request.user = AnonymousUser()

    response = upload_sample(request)

    assert response.status_code == 401
    data = json.loads(response.content)
    assert "login_url" in data


@pytest.mark.django_db
def test_upload_sample_sets_session_and_clears_state(client, user):
    """Sample upload should set uploaded path and invalidate cached form state."""
    client.force_login(user)
    session = client.session
    session["sim_form_state"] = {"option": "simulate_percent_slope", "catchment_name": "S1"}
    session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S1"}
    session["_subcatchment_ids"] = ["S1", "S2"]
    session["_subcatchment_ids_file"] = "uploaded_files/old.inp"
    session.save()

    response = client.post(
        reverse("main:upload_sample"),
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "example.inp"

    updated_session = client.session
    uploaded_path = updated_session.get("uploaded_file_path")
    assert uploaded_path == os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "example.inp"
    )
    assert os.path.exists(uploaded_path)
    assert "sim_form_state" not in updated_session
    assert "ts_form_state" not in updated_session
    assert "_subcatchment_ids" not in updated_session
    assert "_subcatchment_ids_file" not in updated_session

    if uploaded_path and os.path.exists(uploaded_path):
        os.remove(uploaded_path)


@pytest.mark.django_db
def test_upload_sample_returns_500_when_fixture_missing(client, user, monkeypatch):
    """Sample upload returns 500 when bundled example file is unavailable."""
    client.force_login(user)

    original_exists = os.path.exists

    def fake_exists(path):
        if path.endswith(os.path.join("data", "example.inp")):
            return False
        return original_exists(path)

    monkeypatch.setattr("main.views.os.path.exists", fake_exists)

    response = client.post(
        reverse("main:upload_sample"),
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    assert response.status_code == 500
    assert response.json()["error"] == "Sample file is not available."


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

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "test_upload.inp"
    )
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

    expected_path = os.path.join(
        settings.MEDIA_ROOT, "uploaded_files", str(user.id), "test_upload.inp"
    )
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
def test_download_simulation_results_streams_excel(client, user):
    """Simulation download endpoint returns an in-memory Excel attachment."""
    client.force_login(user)
    token = "11111111111111111111111111111111"
    cache.set(
        _result_cache_key("sim", user.id, token),
        json.dumps(
            {
                "output_file_name": "simulation_result.xlsx",
                "results_columns": ["PercSlope", "runoff"],
                "results_data": [[1, 10.5], [2, 20.25]],
            }
        ),
    )

    response = client.post(reverse("main:download_simulation_results"), data={"token": token})

    assert response.status_code == 200
    assert response["Content-Type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "simulation_result.xlsx" in response["Content-Disposition"]

    body = b"".join(response.streaming_content)
    df = pd.read_excel(BytesIO(body), sheet_name="results")
    assert list(df.columns) == ["PercSlope", "runoff"]
    assert df.shape == (2, 2)


@pytest.mark.django_db
def test_download_simulation_results_without_data_redirects(client, user):
    """Simulation download endpoint redirects when session has no results."""
    client.force_login(user)

    response = client.post(reverse("main:download_simulation_results"), data={"token": "missing"})

    assert response.status_code == 302
    assert response.url == reverse("main:simulation")


@pytest.mark.django_db
def test_download_simulation_results_get_not_allowed(client, user):
    """Simulation download endpoint accepts POST only."""
    client.force_login(user)

    response = client.get(reverse("main:download_simulation_results"))

    assert response.status_code == 405


@pytest.mark.django_db
def test_download_simulation_results_isolated_by_token(client, user):
    """Two result tokens should download their own payloads (no cross-tab overwrite)."""
    client.force_login(user)
    cache.set(
        _result_cache_key("sim", user.id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        json.dumps(
            {
                "output_file_name": "a.xlsx",
                "results_columns": ["PercSlope", "runoff"],
                "results_data": [[1, 11.0]],
            }
        ),
    )
    cache.set(
        _result_cache_key("sim", user.id, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        json.dumps(
            {
                "output_file_name": "b.xlsx",
                "results_columns": ["PercSlope", "runoff"],
                "results_data": [[2, 22.0]],
            }
        ),
    )

    response_a = client.post(
        reverse("main:download_simulation_results"),
        data={"token": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    )
    response_b = client.post(
        reverse("main:download_simulation_results"),
        data={"token": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
    )

    df_a = pd.read_excel(BytesIO(b"".join(response_a.streaming_content)), sheet_name="results")
    df_b = pd.read_excel(BytesIO(b"".join(response_b.streaming_content)), sheet_name="results")
    assert df_a.iloc[0]["runoff"] == 11.0
    assert df_b.iloc[0]["runoff"] == 22.0


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

    monkeypatch.setattr("main.views.FeaturesSimulation", DummyFeaturesSimulation)

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
    token = updated_session[TS_RESULT_TOKEN_SESSION_KEY]
    cached_payload = cache.get(_result_cache_key("ts", user.id, token))
    assert cached_payload is not None

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
def test_download_timeseries_results_single_streams_excel(client, user):
    """Timeseries single-mode download returns one-sheet Excel attachment."""
    client.force_login(user)
    token = "22222222222222222222222222222222"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "single",
                "output_file_name": "timeseries_single.xlsx",
                "data": [
                    {"datetime": "2025-01-01 00:00:00", "runoff": 1.1},
                    {"datetime": "2025-01-01 01:00:00", "runoff": 0.9},
                ],
            }
        ),
    )

    response = client.post(reverse("main:download_timeseries_results"), data={"token": token})

    assert response.status_code == 200
    assert response["Content-Type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "timeseries_single.xlsx" in response["Content-Disposition"]

    body = b"".join(response.streaming_content)
    df = pd.read_excel(BytesIO(body), sheet_name="timeseries")
    assert "datetime" in df.columns
    assert "runoff" in df.columns
    assert df.shape[0] == 2


@pytest.mark.django_db
def test_download_timeseries_results_sweep_streams_multisheet_excel(client, user):
    """Timeseries sweep-mode download returns multi-sheet Excel attachment."""
    client.force_login(user)
    token = "33333333333333333333333333333333"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "sweep",
                "output_file_name": "timeseries_sweep.xlsx",
                "data": {
                    "0.0": [{"datetime": "2025-01-01 00:00:00", "runoff": 0.1}],
                    "10.0": [{"datetime": "2025-01-01 00:00:00", "runoff": 0.5}],
                },
            }
        ),
    )

    response = client.post(reverse("main:download_timeseries_results"), data={"token": token})

    assert response.status_code == 200
    assert "timeseries_sweep.xlsx" in response["Content-Disposition"]
    body = b"".join(response.streaming_content)
    workbook = pd.ExcelFile(BytesIO(body))
    assert sorted(workbook.sheet_names) == ["val_0.0", "val_10.0"]


@pytest.mark.django_db
def test_download_timeseries_results_handles_sheet_name_collisions(client, user):
    """Colliding long sheet names should be auto-deduplicated."""
    client.force_login(user)
    token = "44444444444444444444444444444444"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "sweep",
                "output_file_name": "timeseries_collision.xlsx",
                "data": {
                    "123456789012345678901234567890123": [{"runoff": 0.1}],
                    "123456789012345678901234567890124": [{"runoff": 0.2}],
                },
            }
        ),
    )

    response = client.post(reverse("main:download_timeseries_results"), data={"token": token})

    assert response.status_code == 200
    workbook = pd.ExcelFile(BytesIO(b"".join(response.streaming_content)))
    assert len(workbook.sheet_names) == 2
    assert workbook.sheet_names[0] != workbook.sheet_names[1]
    assert all(len(name) <= 31 for name in workbook.sheet_names)


@pytest.mark.django_db
def test_download_timeseries_csv_single_streams_csv(client, user):
    """Timeseries single-mode CSV endpoint returns flat CSV payload."""
    client.force_login(user)
    token = "55555555555555555555555555555555"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "single",
                "output_file_name": "timeseries_single.xlsx",
                "data": [
                    {"datetime": "2025-01-01 00:00:00", "runoff": 1.1},
                    {"datetime": "2025-01-01 01:00:00", "runoff": 0.9},
                ],
            }
        ),
    )

    response = client.post(reverse("main:download_timeseries_csv"), data={"token": token})

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/csv")
    assert "timeseries_single.csv" in response["Content-Disposition"]
    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    assert list(df.columns) == ["datetime", "runoff"]
    assert df.shape == (2, 2)


@pytest.mark.django_db
def test_download_timeseries_csv_sweep_includes_parameter_column(client, user):
    """Sweep CSV export should flatten records and include parameter value column."""
    client.force_login(user)
    token = "66666666666666666666666666666666"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "sweep",
                "output_file_name": "timeseries_sweep.xlsx",
                "data": {
                    "0.0": [{"datetime": "2025-01-01 00:00:00", "runoff": 0.1}],
                    "10.0": [{"datetime": "2025-01-01 00:00:00", "runoff": 0.5}],
                },
            }
        ),
    )

    response = client.post(reverse("main:download_timeseries_csv"), data={"token": token})

    assert response.status_code == 200
    df = pd.read_csv(StringIO(response.content.decode("utf-8")))
    assert "parameter_value" in df.columns
    assert df.shape[0] == 2
    assert sorted(df["parameter_value"].astype(str).tolist()) == ["0.0", "10.0"]


@pytest.mark.django_db
def test_download_timeseries_csv_without_data_redirects(client, user):
    """Timeseries CSV endpoint redirects when result token is missing."""
    client.force_login(user)

    response = client.post(reverse("main:download_timeseries_csv"), data={"token": "missing"})

    assert response.status_code == 302
    assert response.url == reverse("main:timeseries")


@pytest.mark.django_db
def test_download_timeseries_csv_get_not_allowed(client, user):
    """Timeseries CSV endpoint accepts POST only."""
    client.force_login(user)

    response = client.get(reverse("main:download_timeseries_csv"))

    assert response.status_code == 405


def test_safe_download_filename_handles_empty_extension():
    """Empty extension must not produce a trailing dot."""
    filename = _safe_download_filename("results.xlsx", "fallback.xlsx", extension="")
    assert filename == "results"


@pytest.mark.django_db
def test_simulation_template_contains_loading_state(client, user):
    """Simulation page should include loading state container for submit feedback."""
    client.force_login(user)
    response = client.get(reverse("main:simulation"))

    assert response.status_code == 200
    assert b'id="simulation-loading-state"' in response.content


@pytest.mark.django_db
def test_timeseries_template_shows_csv_and_png_buttons_when_results_exist(client, user):
    """Timeseries results view should render CSV and PNG export controls."""
    client.force_login(user)
    token = "77777777777777777777777777777777"
    cache.set(
        _result_cache_key("ts", user.id, token),
        json.dumps(
            {
                "mode": "single",
                "output_file_name": "timeseries_single.xlsx",
                "chart_config": {
                    "mode": "single",
                    "data": [{"datetime": "2025-01-01 00:00:00", "runoff": 1.0}],
                    "columns": ["runoff"],
                    "title": "Timeseries",
                },
                "ts_show_results": True,
            }
        ),
    )
    session = client.session
    session[TS_RESULT_TOKEN_SESSION_KEY] = token
    session.save()

    response = client.get(reverse("main:timeseries"))

    assert response.status_code == 200
    assert b"Export timeseries to CSV" in response.content
    assert b'id="download-timeseries-png-button"' in response.content


@pytest.mark.django_db
def test_base_template_renders_flash_messages(client, user):
    """Base template should render flash alerts from Django messages framework."""
    client.force_login(user)
    response = client.post(
        reverse("main:download_simulation_results"),
        data={"token": "invalid-token"},
        follow=True,
    )

    assert response.status_code == 200
    assert b"alert-danger" in response.content
    assert b"Invalid download token." in response.content


@pytest.mark.django_db
def test_download_timeseries_results_without_data_redirects(client, user):
    """Timeseries download endpoint redirects when session has no results."""
    client.force_login(user)

    response = client.post(reverse("main:download_timeseries_results"), data={"token": "missing"})

    assert response.status_code == 302
    assert response.url == reverse("main:timeseries")


@pytest.mark.django_db
def test_download_timeseries_results_get_not_allowed(client, user):
    """Timeseries download endpoint accepts POST only."""
    client.force_login(user)

    response = client.get(reverse("main:download_timeseries_results"))

    assert response.status_code == 405


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
    request.session[SIM_RESULT_TOKEN_SESSION_KEY] = "cccccccccccccccccccccccccccccccc"
    request.session[TS_RESULT_TOKEN_SESSION_KEY] = "dddddddddddddddddddddddddddddddd"
    request.session["sim_form_state"] = {"option": "simulate_percent_slope", "catchment_name": "S1"}
    request.session["ts_form_state"] = {"mode": "sweep", "catchment_name": "S1"}
    request.session.save()
    request.user = user
    cache.set(
        _result_cache_key("sim", user.id, "cccccccccccccccccccccccccccccccc"),
        json.dumps({"value": 1}),
    )
    cache.set(
        _result_cache_key("ts", user.id, "dddddddddddddddddddddddddddddddd"),
        json.dumps({"value": 1}),
    )

    clear_session_variables(request)

    assert request.session.get("uploaded_file_path") == "uploaded_files/test.inp"
    assert "show_download_button" not in request.session
    assert "chart_config" not in request.session
    assert "sim_form_state" not in request.session
    assert "ts_form_state" not in request.session
    assert cache.get(_result_cache_key("sim", user.id, "cccccccccccccccccccccccccccccccc")) is None
    assert cache.get(_result_cache_key("ts", user.id, "dddddddddddddddddddddddddddddddd")) is None


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
