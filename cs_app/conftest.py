import pytest


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
    return "strong-test-pass"


@pytest.fixture
def create_user(db, test_password):
    from django.contrib.auth.models import User

    return User.objects.create_user(username="testuser", password=test_password)


@pytest.fixture
def test_user():
    from django.contrib.auth import get_user_model

    User = get_user_model()
    return User.objects.create_user(username="test", password="testpassword")


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

    return User.objects.create_user(username="test", password="test")
