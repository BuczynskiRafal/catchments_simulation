"""Shared E2E fixtures for Playwright tests.

All fixtures rely on ``pytest-playwright`` providing the ``page`` fixture
and ``pytest-django`` providing ``live_server``.
"""

from __future__ import annotations

import os
import uuid

import pytest
from django.contrib.auth.models import User
from django.test import Client
from playwright.sync_api import Page

# Allow Django ORM access from async Playwright threads.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

# ------------------------------------------------------------------
# User credentials — read from env or generate disposable values at
# import time.  Passwords are random per test-run so no secret ever
# appears in source code.
# ------------------------------------------------------------------


def _generate_password() -> str:
    """Return a strong random password safe for Django validators."""
    return f"T!{uuid.uuid4().hex}"


TEST_USERNAME = os.environ.get("E2E_TEST_USERNAME", "e2etester")
TEST_EMAIL = os.environ.get("E2E_TEST_EMAIL", "e2e@example.com")
TEST_FIRST_NAME = os.environ.get("E2E_TEST_FIRST_NAME", "Test")
TEST_LAST_NAME = os.environ.get("E2E_TEST_LAST_NAME", "User")
TEST_PASSWORD = os.environ.get("E2E_TEST_PASSWORD") or _generate_password()

STRONG_ALT_PASSWORD = os.environ.get("E2E_ALT_PASSWORD") or _generate_password()
OTHER_USER_PASSWORD = os.environ.get("E2E_OTHER_PASSWORD") or _generate_password()


@pytest.fixture()
def test_user(db) -> User:
    """Create and return a standard test user."""
    user = User.objects.create_user(
        username=TEST_USERNAME,
        email=TEST_EMAIL,
        first_name=TEST_FIRST_NAME,
        last_name=TEST_LAST_NAME,
        password=TEST_PASSWORD,
    )
    return user


@pytest.fixture()
def auth_page(page: Page, live_server, test_user) -> Page:
    """Return a Playwright ``Page`` already authenticated via cookie injection.

    Uses Django's ``Client.login()`` to create a server-side session,
    then injects the session cookie into the Playwright browser context.
    This is ~10× faster than filling the login form on every test.
    """
    client = Client()
    client.login(username=TEST_USERNAME, password=TEST_PASSWORD)

    # Extract the session cookie from the Django test client
    session_cookie = client.cookies["sessionid"]

    # Navigate to the site first (required to set cookies for the domain)
    page.goto(f"{live_server.url}/")
    page.context.add_cookies(
        [
            {
                "name": "sessionid",
                "value": session_cookie.value,
                "domain": "localhost",
                "path": "/",
            }
        ]
    )
    # Reload so the server sees the session cookie
    page.reload(wait_until="domcontentloaded")
    return page
