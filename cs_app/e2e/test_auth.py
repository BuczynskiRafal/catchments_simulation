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
        expect(page).to_have_url(f"{live_server.url}/")

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
        expect(page.locator("#error_1_id_password2")).to_contain_text("didn’t match")
        expect(page).to_have_url(re.compile(r".*/register/$"))

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
        expect(page.locator("#error_1_id_username")).to_contain_text("already exists")
        expect(page).to_have_url(re.compile(r".*/register/$"))

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
        expect(page.locator("#error_1_id_password2")).to_contain_text("too short")
        expect(page).to_have_url(re.compile(r".*/register/$"))

    def test_register_page_title(self, page: Page, live_server) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        assert "Create account" in rp.get_title()
        assert rp.get_heading(level=1) == "Create account"

    def test_register_success_logs_the_user_in(self, page: Page, live_server, db) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            "fresh", "fresh@example.com", "Fresh", "User", STRONG_ALT_PASSWORD, STRONG_ALT_PASSWORD
        )

        expect(page.locator(".alert-success")).to_contain_text("Welcome, Fresh.")
        assert NavComponent(page).is_logged_in()

    def test_invalid_fields_are_marked_inline_and_focused(
        self, page: Page, live_server, db
    ) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.register(
            "mismatch2", "m2@example.com", "Mis", "Match", STRONG_ALT_PASSWORD, OTHER_USER_PASSWORD
        )

        feedback = page.locator("#error_1_id_password2")
        expect(feedback).to_be_visible()
        expect(feedback).to_contain_text("didn’t match")
        expect(rp.password2_input).to_have_attribute("aria-invalid", "true")
        expect(rp.password2_input).to_have_attribute(
            "aria-describedby", re.compile(r"\berror_1_id_password2\b")
        )
        expect(rp.password2_input).to_be_focused()
        # Valid fields keep their value and are not flagged.
        expect(rp.email_input).to_have_value("m2@example.com")
        expect(rp.email_input).not_to_have_attribute("aria-invalid", "true")

    def test_links_to_login(self, page: Page, live_server) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        page.locator("main").get_by_role("link", name="Log in", exact=True).click()
        expect(page).to_have_url(re.compile(r".*/accounts/login/$"))


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
        expect(page).to_have_url(f"{live_server.url}/")

    def test_login_invalid_credentials_shows_error(self, page: Page, live_server, db) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.login("nonexistent", "wrongpass")
        expect(page.locator("form .alert-danger")).to_contain_text(
            "Please enter a correct username or email and password"
        )
        expect(page).to_have_url(re.compile(r".*/accounts/login/$"))

    def test_login_page_title(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        assert "Log in" in lp.get_title()
        assert lp.get_heading(level=1) == "Log in"

    def test_wrong_credentials_error_is_tied_to_the_form(self, page: Page, live_server, db) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.login("nobody", "wrong-password")

        assert any("correct username or email and password" in m for m in lp.get_error_messages())
        error_id = page.locator("form .alert-danger").get_attribute("id")
        expect(lp.username_input).to_have_attribute(
            "aria-describedby", re.compile(rf"\b{error_id}\b")
        )
        expect(lp.username_input).to_be_focused()

    def test_login_returns_to_next_page(self, page: Page, live_server, test_user) -> None:
        page.goto(f"{live_server.url}/accounts/login/?next=/about")
        LoginPage(page, live_server.url).login(TEST_USERNAME, TEST_PASSWORD)
        expect(page).to_have_url(re.compile(r".*/about$"))

    def test_links_to_register(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        page.get_by_role("link", name="Create an account", exact=True).click()
        expect(page).to_have_url(re.compile(r".*/register/$"))


class TestPasswordToggle:
    def test_toggle_reveals_and_hides_password(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.password_input.fill(STRONG_ALT_PASSWORD)
        toggle = lp.password_toggle
        expect(toggle).to_have_attribute("aria-pressed", "false")
        expect(toggle).to_have_attribute("aria-controls", "id_password")

        toggle.click()
        expect(lp.password_input).to_have_attribute("type", "text")
        expect(toggle).to_have_attribute("aria-pressed", "true")

        toggle.press("Enter")
        expect(lp.password_input).to_have_attribute("type", "password")
        expect(toggle).to_have_attribute("aria-pressed", "false")
        expect(lp.password_input).to_have_value(STRONG_ALT_PASSWORD)

    def test_each_register_password_gets_its_own_toggle(self, page: Page, live_server) -> None:
        rp = RegisterPage(page, live_server.url)
        rp.navigate_to()
        rp.password_toggle("password confirmation").click()

        expect(rp.password2_input).to_have_attribute("type", "text")
        expect(rp.password1_input).to_have_attribute("type", "password")

    def test_revealed_password_is_hidden_again_on_submit(self, page: Page, live_server) -> None:
        lp = LoginPage(page, live_server.url)
        lp.navigate_to()
        lp.fill_credentials("someone", STRONG_ALT_PASSWORD)
        lp.password_toggle.click()
        # Registered after the page's handler, so it observes the state it leaves behind.
        page.evaluate(
            """() => document.querySelector("form[data-cs-form]").addEventListener("submit", (e) => {
                e.preventDefault();
                window.__typeOnSubmit = document.getElementById("id_password").type;
            })"""
        )

        lp.submit()

        assert page.evaluate("window.__typeOnSubmit") == "password"
        expect(lp.password_toggle).to_have_attribute("aria-pressed", "false")


class TestAuthA11y:
    @pytest.mark.parametrize("scheme", ["light", "dark"])
    @pytest.mark.parametrize("path", ["/accounts/login/", "/register/"])
    def test_no_serious_axe_violations(
        self, page: Page, live_server, db, scheme: str, path: str
    ) -> None:
        page.emulate_media(color_scheme=scheme)
        lp = LoginPage(page, live_server.url)
        lp.navigate(path)
        report = lp.run_axe_audit()
        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]


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
