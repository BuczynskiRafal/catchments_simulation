import re
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def _assert_login_redirect(page: Page, expected_next: str) -> None:
    parsed = urlparse(page.url)
    assert parsed.path == "/accounts/login/"
    assert parse_qs(parsed.query).get("next") == [expected_next]
    expect(page.get_by_role("button", name="Log in", exact=True)).to_be_visible()


def test_home_page_smoke(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/")

    expect(page.get_by_role("heading", level=1)).to_be_visible()
    header = page.locator("header")
    expect(header.get_by_role("link", name="Catchment Simulation home", exact=True)).to_be_visible()
    expect(header.get_by_role("link", name="Simulation", exact=True)).to_be_visible()


def test_navigation_to_about(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/")

    page.locator("header").get_by_role("link", name="About", exact=True).click()

    expect(page).to_have_url(re.compile(r".*/about$"))
    expect(page.get_by_role("heading", level=1, name="About")).to_be_visible()


def test_simulation_redirects_anonymous_user_to_login(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/simulation")

    _assert_login_redirect(page, "/simulation")


def test_timeseries_redirects_anonymous_user_to_login(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/timeseries")

    _assert_login_redirect(page, "/timeseries")
