import pytest
from django.urls import reverse, resolve


pytestmark = pytest.mark.django_db


def test_main_view_url():
    path = reverse("main:main_view")
    assert resolve(path).view_name == "main:main_view"


def test_upload_url():
    path = reverse("main:upload")
    assert resolve(path).view_name == "main:upload"


def test_user_profile_url():
    path = reverse("main:userprofile", kwargs={"user_id": 1})
    assert resolve(path).view_name == "main:userprofile"


def test_contact_url():
    path = reverse("main:contact")
    assert resolve(path).view_name == "main:contact"


def test_about_url():
    path = reverse("main:about")
    assert resolve(path).view_name == "main:about"


def test_calculations_url():
    path = reverse("main:calculations")
    assert resolve(path).view_name == "main:calculations"


def test_simulation_view_url():
    path = reverse("main:simulation")
    assert resolve(path).view_name == "main:simulation"
