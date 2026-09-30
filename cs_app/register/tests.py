"""Tests for sign-up, log in (by username or email) and log out."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from django.contrib.auth import SESSION_KEY, authenticate
from django.contrib.auth.models import User
from django.urls import reverse

from register.forms import LoginForm, RegisterForm

# Random per run, like the e2e credentials, so no password is written in the source.
PASSWORD = f"T!{uuid.uuid4().hex}"
OTHER_PASSWORD = f"T!{uuid.uuid4().hex}"
WRONG_PASSWORD = f"T!{uuid.uuid4().hex}"


@pytest.fixture
def account(db) -> User:
    return User.objects.create_user(username="test1", email="test1@example.com", password=PASSWORD)


def _logged_in_user_id(client) -> str | None:
    return client.session.get(SESSION_KEY)


# ---------------------------------------------------------------------------
# Authentication backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("identifier", ["test1", "TEST1", " test1 "])
def test_authenticate_by_username_ignoring_case_and_padding(account, identifier):
    assert authenticate(username=identifier, password=PASSWORD) == account


@pytest.mark.parametrize("identifier", ["test1@example.com", "Test1@Example.COM"])
def test_authenticate_by_email_ignoring_case(account, identifier):
    assert authenticate(username=identifier, password=PASSWORD) == account


@pytest.mark.parametrize(
    "identifier", ["test1", "test1@example.com", "nobody", "nobody@example.com"]
)
def test_authenticate_rejects_wrong_password(account, identifier):
    assert authenticate(username=identifier, password=WRONG_PASSWORD) is None


def test_authenticate_rejects_inactive_account(account):
    account.is_active = False
    account.save()

    assert authenticate(username="test1@example.com", password=PASSWORD) is None


def test_authenticate_refuses_email_shared_by_two_accounts(db):
    # Accounts created before emails were unique (or by an admin) may share one.
    User.objects.create_user(username="first", email="shared@example.com", password=PASSWORD)
    second = User.objects.create_user(
        username="second", email="SHARED@example.com", password=PASSWORD
    )

    assert authenticate(username="shared@example.com", password=PASSWORD) is None
    assert authenticate(username="second", password=PASSWORD) == second


def test_exact_username_wins_over_another_accounts_email(account):
    # "@" is a valid username character, so a username can look like an email.
    owner = User.objects.create_user(username="test1@example.com", password=OTHER_PASSWORD)

    assert authenticate(username="test1@example.com", password=OTHER_PASSWORD) == owner
    assert authenticate(username="test1@example.com", password=PASSWORD) is None


def test_usernames_differing_only_in_case_need_the_exact_username(db):
    # Sign-up forbids this since Django 4.2; older databases can still hold both.
    lower = User.objects.create_user(username="anna", password=PASSWORD)
    upper = User.objects.create_user(username="Anna", password=PASSWORD)

    assert authenticate(username="anna", password=PASSWORD) == lower
    assert authenticate(username="Anna", password=PASSWORD) == upper
    assert authenticate(username="ANNA", password=PASSWORD) is None


# ---------------------------------------------------------------------------
# Log in / log out views
# ---------------------------------------------------------------------------


def test_login_form_asks_for_username_or_email(client):
    html = client.get(reverse("login")).content.decode()

    assert "Username or email" in html
    assert 'autocomplete="username"' in html
    assert 'maxlength="254"' in html


@pytest.mark.parametrize("identifier", ["test1", "test1@example.com"])
def test_login_with_username_or_email_redirects_home(client, account, identifier):
    response = client.post(reverse("login"), {"username": identifier, "password": PASSWORD})

    assert response.status_code == 302
    assert response.url == "/"
    assert _logged_in_user_id(client) == str(account.pk)


def test_login_honours_next(client, account):
    response = client.post(
        reverse("login"),
        {"username": "test1@example.com", "password": PASSWORD, "next": "/about"},
    )

    assert response.status_code == 302
    assert response.url == "/about"


def test_login_with_wrong_password_shows_error(client, account):
    response = client.post(
        reverse("login"), {"username": "test1@example.com", "password": WRONG_PASSWORD}
    )

    assert response.status_code == 200
    assert _logged_in_user_id(client) is None
    assert "Please enter a correct username or email and password." in response.content.decode()


def test_login_form_is_the_username_or_email_form(client):
    response = client.get(reverse("login"))

    assert isinstance(response.context["form"], LoginForm)


def test_logout_requires_post(client, account):
    client.force_login(account)

    response = client.post(reverse("logout"))

    assert response.status_code == 302
    assert response.url == "/"
    assert _logged_in_user_id(client) is None


def test_login_enforces_csrf(account):
    from django.test import Client

    csrf_client = Client(enforce_csrf_checks=True)
    response = csrf_client.post(reverse("login"), {"username": "test1", "password": PASSWORD})

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Sign-up
# ---------------------------------------------------------------------------


def _signup_data(**overrides) -> dict:
    data = {
        "username": "test1",
        "email": "test1@example.com",
        "first_name": "Test",
        "last_name": "User",
        "password1": PASSWORD,
        "password2": PASSWORD,
    }
    data.update(overrides)
    return data


def test_register_rejects_email_used_by_another_account_ignoring_case(account):
    form = RegisterForm(data=_signup_data(username="someone-else", email="TEST1@example.com"))

    assert not form.is_valid()
    assert form.errors["email"] == ["An account with this email already exists. Log in instead."]


def test_register_then_logout_then_login_with_email(client, db):
    response = client.post(reverse("register:register"), _signup_data())
    assert response.status_code == 302
    user = User.objects.get(username="test1")
    assert _logged_in_user_id(client) == str(user.pk)

    client.post(reverse("logout"))
    assert _logged_in_user_id(client) is None

    response = client.post(
        reverse("login"), {"username": "test1@example.com", "password": PASSWORD}
    )
    assert response.status_code == 302
    assert _logged_in_user_id(client) == str(user.pk)


# ---------------------------------------------------------------------------
# Production settings: cookie flags
# ---------------------------------------------------------------------------

_READ_COOKIE_FLAGS = """
import json, dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False  # only the env given below counts
from cs_app import settings
print(json.dumps([settings.SESSION_COOKIE_SECURE, settings.CSRF_COOKIE_SECURE]))
"""


def _cookie_flags(**env: str) -> list[bool]:
    """Import cs_app.settings (not the test settings) in a clean process."""
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DEBUG", "SESSION_COOKIE_SECURE", "CSRF_COOKIE_SECURE"}
    }
    clean_env.update(SECRET_KEY="test", DATABASE_URL="sqlite:///:memory:", **env)
    result = subprocess.run(
        [sys.executable, "-c", _READ_COOKIE_FLAGS],
        cwd=Path(__file__).resolve().parent.parent,
        env=clean_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # Plain-HTTP dev server: Secure cookies would be dropped by Safari -> CSRF 403.
        ({"DEBUG": "True"}, [False, False]),
        # Production fails closed.
        ({}, [True, True]),
        ({"DEBUG": "False"}, [True, True]),
        # Explicit settings always win.
        (
            {"DEBUG": "True", "SESSION_COOKIE_SECURE": "True", "CSRF_COOKIE_SECURE": "True"},
            [True, True],
        ),
        (
            {"DEBUG": "False", "SESSION_COOKIE_SECURE": "False", "CSRF_COOKIE_SECURE": "False"},
            [False, False],
        ),
    ],
)
def test_cookies_are_secure_unless_debug(env, expected):
    assert _cookie_flags(**env) == expected
