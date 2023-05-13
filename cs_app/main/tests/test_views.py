import pytest
from django.urls import reverse

# from cs_app.main.views import *


@pytest.mark.django_db
def test_main_view(client):
    response = client.get(reverse("main:main_view"))
    assert response.status_code == 200
    