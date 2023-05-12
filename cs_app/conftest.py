import pytest
from django.contrib.auth import get_user_model


@pytest.fixture
def db_access():
    from django.db import connection
    with connection.cursor() as cursor:
        yield cursor


@pytest.fixture
def test_user(db):
    User = get_user_model()
    return User.objects.create_user(username='test', password='testpassword')
