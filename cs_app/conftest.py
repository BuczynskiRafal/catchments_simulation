import uuid

import pytest


def _generate_password() -> str:
    """A random password per call, so no password is written in the source (as in e2e)."""
    return f"T!{uuid.uuid4().hex}"


@pytest.fixture(autouse=True)
def isolated_media_root(settings, tmp_path):
    """Give every test its own MEDIA_ROOT.

    Upload directories are named after the user id, and every xdist worker's
    test database hands out the same ids. With the shared MEDIA_ROOT of the test
    settings, one worker's teardown deleted the model that a test on another
    worker was about to run.
    """
    settings.MEDIA_ROOT = str(tmp_path / "media")


@pytest.fixture(autouse=True)
def isolated_cache():
    """Clear Django cache before and after every test to avoid state leakage across tests."""
    from django.core.cache import cache

    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def user_model(user):
    """A copy of the sample model in ``user``'s upload directory, where an upload puts it."""
    import shutil
    from pathlib import Path

    from django.conf import settings
    from main.views import _user_upload_dir

    path = Path(_user_upload_dir(user.id)) / "example.inp"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(settings.BASE_DIR) / "data" / "example.inp", path)
    return str(path)


@pytest.fixture
def db_access():
    from django.db import connection

    with connection.cursor() as cursor:
        yield cursor


@pytest.fixture
def client():
    from django.test import Client

    return Client()


@pytest.fixture
def test_password():
    return _generate_password()


@pytest.fixture
def create_user(db, test_password):
    from django.contrib.auth.models import User

    return User.objects.create_user(username="testuser", password=test_password)


@pytest.fixture
def test_user():
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username="test", password=_generate_password())


@pytest.fixture
def uploaded_file():
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile("example.inp", b"file_content", content_type="text/plain")


@pytest.fixture
def wrong_extension_file():
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile("file.txt", b"file_content", content_type="text/plain")


@pytest.fixture
def user():
    from django.contrib.auth.models import User

    return User.objects.create_user(username="test", password=_generate_password())
