import pytest
from django.test import Client
from django.contrib.auth import get_user_model
from django.contrib.auth.models import User


@pytest.fixture
def db_access():
    from django.db import connection
    with connection.cursor() as cursor:
        yield cursor


@pytest.fixture
def client():
    return Client()


@pytest.fixture
def test_password():
    return 'strong-test-pass'


@pytest.fixture
def create_user(db, test_password):
    return User.objects.create_user(username='testuser', password=test_password)


@pytest.fixture
def test_user():
    User = get_user_model()
    return User.objects.create_user(username='test', password='testpassword')