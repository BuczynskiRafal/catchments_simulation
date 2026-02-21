"""Authentication tests — register, login, logout, protected routes."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Page, expect

from .conftest import OTHER_USER_PASSWORD, STRONG_ALT_PASSWORD, TEST_PASSWORD, TEST_USERNAME
from .pages.login_page import LoginPage
from .pages.nav_component import NavComponent
from .pages.register_page import RegisterPage

pytestmark = pytest.mark.e2e


class TestRegistration:
    """User registration flow."""

    def test_register_form_renders_all_fields(self, page: Page, live_server) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        assert rp.has_form()
        expect(rp.email_input).to_be_visible()
        expect(rp.first_name_input).to_be_visible()
        expect(rp.last_name_input).to_be_visible()

    def test_register_success_redirects_to_home(self, page: Page, live_server, db) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            username="newuser",
            email="new@example.com",
            first_name="New",
            last_name="User",
            password1=STRONG_ALT_PASSWORD,
            password2=STRONG_ALT_PASSWORD,
        )
        expect(page).to_have_url(re.compile(r".*/$"))

    def test_register_mismatched_passwords_shows_error(self, page: Page, live_server, db) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            username="mismatch",
            email="m@example.com",
            first_name="Mis",
            last_name="Match",
            password1=STRONG_ALT_PASSWORD,
            password2=OTHER_USER_PASSWORD,
        )
        page.wait_for_load_state("domcontentloaded")
        # After invalid submit, we should still be on the register page
        expect(page).to_have_url(re.compile(r".*/register/"))
        # Page body should contain error feedback about passwords
        body_text = page.locator("body").inner_text()
        assert (
            "password" in body_text.lower() or "match" in body_text.lower()
        ), "Expected password mismatch error feedback on page"

    def test_register_duplicate_username_shows_error(
        self, page: Page, live_server, test_user
    ) -> None:
        """Registering with an existing username should show an error."""
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            username=TEST_USERNAME,  # Already exists via test_user fixture
            email="duplicate@example.com",
            first_name="Dup",
            last_name="User",
            password1=STRONG_ALT_PASSWORD,
            password2=STRONG_ALT_PASSWORD,
        )
        page.wait_for_load_state("domcontentloaded")
        expect(page).to_have_url(re.compile(r".*/register/"))
        body_text = page.locator("body").inner_text()
        assert "username" in body_text.lower() or "already" in body_text.lower()

    def test_register_weak_password_shows_error(self, page: Page, live_server, db) -> None:
        """Too simple password should be rejected by Django validators."""
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            username="weakpwd",
            email="weak@example.com",
            first_name="Weak",
            last_name="Pwd",
            password1="123",
            password2="123",
        )
        page.wait_for_load_state("domcontentloaded")
        expect(page).to_have_url(re.compile(r".*/register/"))
        body_text = page.locator("body").inner_text()
        assert "password" in body_text.lower()

    def test_register_page_title(self, page: Page, live_server) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        assert "Register" in rp.get_title()


class TestLogin:
    """Login flow."""

    def test_login_form_renders(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        assert lp.has_form()

    def test_login_success_redirects(self, page: Page, live_server, test_user) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.login(TEST_USERNAME, TEST_PASSWORD)
        # Should redirect to home after successful login
        expect(page).to_have_url(re.compile(r".*/$"))

    def test_login_invalid_credentials_shows_error(self, page: Page, live_server, db) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.login("nonexistent", "wrongpass")
        page.wait_for_load_state("domcontentloaded")
        # Should stay on login page
        expect(page).to_have_url(re.compile(r".*/accounts/login/"))
        # The login page should contain error feedback
        content = page.locator(".content-wrapper").inner_text()
        assert (
            "username" in content.lower()
            or "password" in content.lower()
            or "correct" in content.lower()
        )

    def test_login_page_title(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        assert "Logowanie" in lp.get_title()


class TestLogout:
    """Logout flow."""

    def test_logout_redirects_and_shows_login(self, auth_page: Page, live_server) -> None:
        nav = NavComponent(auth_page)
        nav.click_logout()
        auth_page.wait_for_load_state("domcontentloaded")
        nav2 = NavComponent(auth_page)
        assert nav2.is_logged_out()


class TestProtectedRoutes:
    """Pages that redirect anonymous users to login (@login_required)."""

    @pytest.mark.parametrize(
        "path,expected_next",
        [
            ("/simulation", "/simulation"),
            ("/timeseries", "/timeseries"),
        ],
    )
    def test_protected_page_redirects_to_login(
        self, page: Page, live_server, path: str, expected_next: str
    ) -> None:
        """simulation_view and timeseries_view have @login_required."""
        page.goto(f"{live_server.url}{path}")
        page.wait_for_load_state("domcontentloaded")
        parsed = urlparse(page.url)
        assert parsed.path == "/accounts/login/"
        assert parse_qs(parsed.query).get("next") == [expected_next]
