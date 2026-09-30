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

from main.views import BodySizeLimitUploadHandler, _user_upload_dir

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
