"""Navigation tests — header, footer, logo links.

Note: /simulation and /timeseries have @login_required, so clicking those
links as anonymous navigates to /accounts/login/?next=/simulation. The nav
tests for those links therefore verify the redirect target, not the
original URL.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

from .pages.nav_component import NavComponent

pytestmark = pytest.mark.e2e


class TestNavbarLinksAnonymous:
    """Navbar links for an unauthenticated user."""

    def test_home_link_navigates_to_root(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        nav = NavComponent(page)
        nav.click_home()
        expect(page).to_have_url(re.compile(r".*/$"))

    def test_simulation_link_redirects_anonymous_to_login(self, page: Page, live_server) -> None:
        """Clicking Simulation as anonymous → @login_required redirect."""
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_simulation()
        page.wait_for_load_state("domcontentloaded")
        parsed = urlparse(page.url)
        assert parsed.path == "/accounts/login/"
        assert parse_qs(parsed.query).get("next") == ["/simulation"]

    def test_timeseries_link_redirects_anonymous_to_login(self, page: Page, live_server) -> None:
        """Clicking Timeseries as anonymous → @login_required redirect."""
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_timeseries()
        page.wait_for_load_state("domcontentloaded")
        parsed = urlparse(page.url)
        assert parsed.path == "/accounts/login/"
        assert parse_qs(parsed.query).get("next") == ["/timeseries"]

    def test_calculations_link(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_calculations()
        expect(page).to_have_url(re.compile(r".*/calculations$"))

    def test_about_link(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_about()
        expect(page).to_have_url(re.compile(r".*/about$"))

    def test_logo_navigates_to_home(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        nav = NavComponent(page)
        nav.click_logo()
        expect(page).to_have_url(re.compile(r".*/$"))


class TestNavbarLinksAuthenticated:
    """Navbar links for an authenticated user — direct navigation."""

    def test_simulation_link_navigates(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        nav.click_simulation()
        expect(auth_page).to_have_url(re.compile(r".*/simulation$"))

    def test_timeseries_link_navigates(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        nav.click_timeseries()
        expect(auth_page).to_have_url(re.compile(r".*/timeseries$"))


class TestNavbarAuthState:
    """Navbar shows correct auth buttons."""

    def test_anonymous_user_sees_login_and_signup(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        assert nav.is_logged_out()
        expect(page.get_by_role("link", name="Sign-up")).to_be_visible()

    def test_authenticated_user_sees_logout(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        assert nav.is_logged_in()

    def test_login_button_navigates_to_login_page(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_login()
        expect(page).to_have_url(re.compile(r".*/accounts/login/$"))

    def test_signup_button_navigates_to_register_page(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_signup()
        expect(page).to_have_url(re.compile(r".*/register/$"))


class TestFooter:
    """Footer link and content tests."""

    def test_footer_about_link(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.get_footer_about_link().click()
        expect(page).to_have_url(re.compile(r".*/about$"))

    def test_footer_contains_copyright(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        copyright_text = nav.get_footer_copyright()
        assert "Rafał Buczyński" in copyright_text

    def test_footer_logo_navigates_home(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        nav = NavComponent(page)
        nav.click_footer_logo()
        expect(page).to_have_url(re.compile(r".*/$"))
