import os
import re
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

pytestmark = pytest.mark.e2e


def _assert_login_redirect(page: Page, expected_next: str) -> None:
    parsed = urlparse(page.url)
    assert parsed.path == "/accounts/login/"
    assert parse_qs(parsed.query).get("next") == [expected_next]
    expect(page.get_by_role("button", name="Login")).to_be_visible()


def test_home_page_smoke(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/")

    expect(page.get_by_role("heading", name="Catchment Simulation")).to_be_visible()
    expect(page.get_by_role("link", name="Simulation")).to_be_visible()


def test_navigation_to_about(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/")

    page.get_by_role("link", name="About").first.click()

    expect(page).to_have_url(re.compile(r".*/about$"))
    expect(page.get_by_role("heading", name="About")).to_be_visible()


def test_simulation_redirects_anonymous_user_to_login(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/simulation")

    _assert_login_redirect(page, "/simulation")


def test_timeseries_redirects_anonymous_user_to_login(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/timeseries")

    _assert_login_redirect(page, "/timeseries")
