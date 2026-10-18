"""The upload endpoint through the full middleware stack, with CSRF enforced.

The unit tests in test_views.py call the view directly or use a Client that
skips CSRF, so CsrfViewMiddleware never parses the multipart body before the
view. A browser request does; these tests cover that path.
"""

import os

import pytest
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from main.views import (
    UPLOAD_VALIDATION_MAX_LINE_BUFFER,
    BodySizeLimitUploadHandler,
    _hide_directories,
    _user_upload_dir,
    _validate_inp_file_content,
)

INP_CONTENT = b"[TITLE]\nIntegration\n\n[OPTIONS]\nFLOW_UNITS CMS\n"


@pytest.fixture
def csrf_client(user):
    """Logged-in client that enforces CSRF, holding the token a page load hands out."""
    client = Client(enforce_csrf_checks=True)
    client.force_login(user)
    client.get(reverse("main:simulation"))
    return client


@pytest.fixture
def upload_path(user):
    path = os.path.join(_user_upload_dir(user.id), "integration.inp")
    yield path
    if os.path.exists(path):
        os.remove(path)


def _post_upload(client, content=INP_CONTENT, **headers):
    return client.post(
        reverse("main:upload"),
        {"file": SimpleUploadedFile("integration.inp", content, content_type="text/plain")},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        **headers,
    )


@pytest.mark.django_db
def test_upload_with_csrf_header_stores_the_file(csrf_client, upload_path):
    token = csrf_client.cookies[settings.CSRF_COOKIE_NAME].value

    response = _post_upload(csrf_client, HTTP_X_CSRFTOKEN=token)

    assert response.status_code == 200
    assert response.json() == {"message": "File was sent.", "unchanged": False}
    with open(upload_path, "rb") as saved:
        assert saved.read() == INP_CONTENT
    assert csrf_client.session["uploaded_file_path"] == upload_path


@pytest.mark.django_db
def test_upload_without_csrf_token_is_rejected(csrf_client, upload_path):
    response = _post_upload(csrf_client)

    assert response.status_code == 403
    assert not os.path.exists(upload_path)
    assert "uploaded_file_path" not in csrf_client.session


@pytest.mark.django_db
def test_upload_body_is_parsed_through_the_size_limit_handler(
    csrf_client, upload_path, monkeypatch
):
    received = []

    class SpyHandler(BodySizeLimitUploadHandler):
        def receive_data_chunk(self, raw_data, start):
            received.append(raw_data)
            return super().receive_data_chunk(raw_data, start)

    monkeypatch.setattr("main.views.BodySizeLimitUploadHandler", SpyHandler)
    token = csrf_client.cookies[settings.CSRF_COOKIE_NAME].value

    response = _post_upload(csrf_client, HTTP_X_CSRFTOKEN=token)

    assert response.status_code == 200
    assert b"".join(received) == INP_CONTENT


@pytest.mark.django_db
@pytest.mark.parametrize(
    "malicious_tail",
    [
        b'\n[FILES]\nSAVE OUTFLOWS "/tmp/pwned.txt"\n',
        b'\n[FILE]\nUSE RAINFALL "/etc/passwd"\n',
        b'\n[BACKDROP]\nFILE "/etc/passwd"\n',
        b'\n[TIMESERIES]\nTS1 FILE "/etc/passwd"\n',
        b'\n[RAINGAGES]\n1 INTENSITY 1:00 1.0 FILE "/etc/passwd" STA1 MM\n',
        b'\n[TEMPERATURE]\nFILE "/etc/passwd"\n',
        b"\n[EVAPORATION]\nFILE\n",
        b'\n[LID_USAGE]\nS1 BC 1 10 1 0 0 0 "/tmp/lid_report.txt"\n',
        b"\n[SUBCATCHMENTS]\nS1\x00 1 J1 5 25 100 10 0\n",
        b"\n[SUBCATCHMENTS]\n" + b"A" * (UPLOAD_VALIDATION_MAX_LINE_BUFFER + 10) + b"\n",
    ],
)
def test_upload_rejects_external_file_directives_and_malformed_lines(
    csrf_client, upload_path, malicious_tail
):
    token = csrf_client.cookies[settings.CSRF_COOKIE_NAME].value
    payload = INP_CONTENT + malicious_tail

    response = _post_upload(csrf_client, content=payload, HTTP_X_CSRFTOKEN=token)

    assert response.status_code == 400
    assert "Invalid file content" in response.json()["error"]
    assert not os.path.exists(upload_path)
    assert _validate_inp_file_content(payload) is False


def test_hide_directories_collapses_external_file_oracle_errors():
    missing = _hide_directories("ERROR 361: could not open external file used for Time Series TS1.")
    existing = _hide_directories(
        "ERROR 363: invalid data in external file /etc/passwd used for Time Series TS1."
    )

    assert missing == "ERROR 361: external files are not supported in uploaded models."
    assert existing == missing
