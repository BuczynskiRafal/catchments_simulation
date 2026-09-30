"""Navigation tests — header, footer, logo links, theme and account menus.

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
        page.wait_for_url(re.compile(r".*/accounts/login/\?"))
        parsed = urlparse(page.url)
        assert parsed.path == "/accounts/login/"
        assert parse_qs(parsed.query).get("next") == ["/simulation"]

    def test_timeseries_link_redirects_anonymous_to_login(self, page: Page, live_server) -> None:
        """Clicking Timeseries as anonymous → @login_required redirect."""
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_timeseries()
        page.wait_for_url(re.compile(r".*/accounts/login/\?"))
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
        expect(page.get_by_role("link", name="Create account")).to_be_visible()

    def test_authenticated_user_sees_logout(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        assert nav.is_logged_in()
        nav.open_account_menu()
        expect(nav.logout_button).to_be_visible()

    def test_account_menu_links_to_own_profile(
        self, auth_page: Page, live_server, test_user
    ) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        expect(nav.account_toggle).to_contain_text(test_user.username)
        nav.click_profile()
        expect(auth_page).to_have_url(re.compile(rf".*/user/{test_user.id}/profile$"))

    def test_account_menu_is_keyboard_operable(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        nav.account_toggle.focus()
        auth_page.keyboard.press("Enter")
        expect(nav.account_toggle).to_have_attribute("aria-expanded", "true")
        expect(nav.link("Profile")).to_be_visible()
        auth_page.keyboard.press("Escape")
        expect(nav.account_toggle).to_have_attribute("aria-expanded", "false")

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


class TestActivePage:
    """The current section is exposed with aria-current, not only colour."""

    @pytest.mark.parametrize(
        ("path", "name"),
        [("/", "Home"), ("/about", "About"), ("/calculations", "Calculations")],
    )
    def test_current_link_is_marked(self, page: Page, live_server, path: str, name: str) -> None:
        page.goto(f"{live_server.url}{path}")
        nav = NavComponent(page)
        expect(nav.current_link()).to_have_count(1)
        expect(nav.current_link()).to_have_text(name)

    def test_authenticated_simulation_is_marked(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/simulation")
        expect(NavComponent(auth_page).current_link()).to_have_text("Simulation")


class TestSkipLink:
    def test_skip_link_is_first_tab_stop_and_targets_main(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        page.keyboard.press("Tab")
        skip = page.get_by_role("link", name="Skip to content")
        expect(skip).to_be_focused()
        expect(skip).to_be_visible()
        page.keyboard.press("Enter")
        expect(page).to_have_url(re.compile(r".*/about#main-content$"))
        expect(page.locator("main#main-content")).to_be_focused()


class TestThemeToggle:
    """Light/dark/auto colour modes, persistence and the cs:themechange contract."""

    def test_dark_mode_applies_persists_and_marks_choice(self, page: Page, live_server) -> None:
        page.emulate_media(color_scheme="light")
        page.goto(f"{live_server.url}/about")
        nav = NavComponent(page)
        assert nav.resolved_theme() == "light"

        nav.set_theme("dark")
        assert nav.resolved_theme() == "dark"
        expect(nav.theme_option("dark")).to_have_attribute("aria-pressed", "true")
        expect(nav.theme_option("light")).to_have_attribute("aria-pressed", "false")

        page.reload()
        assert nav.resolved_theme() == "dark"
        assert page.evaluate("localStorage.getItem('cs-theme')") == "dark"

    def test_keyboard_choice_returns_focus_to_the_toggle(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        toggle = page.locator("#theme-menu")
        toggle.focus()
        page.keyboard.press("Enter")
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        expect(NavComponent(page).theme_option("dark")).to_be_focused()

        page.keyboard.press("Enter")

        assert NavComponent(page).resolved_theme() == "dark"
        expect(toggle).to_be_focused()

    def test_theme_change_event_reports_resolved_theme(self, page: Page, live_server) -> None:
        page.emulate_media(color_scheme="light")
        page.goto(f"{live_server.url}/about")
        page.evaluate(
            """() => {
                window.__themeEvents = [];
                document.addEventListener("cs:themechange", (e) => window.__themeEvents.push(e.detail.theme));
            }"""
        )
        nav = NavComponent(page)
        nav.set_theme("dark")
        nav.set_theme("auto")
        assert page.evaluate("window.__themeEvents") == ["dark", "light"]

    def test_auto_follows_system_preference_live(self, page: Page, live_server) -> None:
        page.emulate_media(color_scheme="light")
        page.goto(f"{live_server.url}/about")
        nav = NavComponent(page)
        nav.set_theme("auto")
        assert page.evaluate("localStorage.getItem('cs-theme')") is None
        page.evaluate(
            """() => {
                window.__themeEvents = [];
                document.addEventListener("cs:themechange", (e) => window.__themeEvents.push(e.detail.theme));
            }"""
        )

        page.emulate_media(color_scheme="dark")
        page.wait_for_function("document.documentElement.dataset.bsTheme === 'dark'")
        assert page.evaluate("window.__themeEvents") == ["dark"]

    def test_stored_theme_applies_before_body_is_parsed(self, page: Page, live_server) -> None:
        """theme-init.js runs in <head>, so the theme is set before any content can paint."""
        page.emulate_media(color_scheme="light")
        page.add_init_script(
            """try { localStorage.setItem('cs-theme', 'dark'); } catch (e) {}
            new MutationObserver((mutations, observer) => {
                if (document.body) {
                    window.__themeAtBody = document.documentElement.dataset.bsTheme;
                    observer.disconnect();
                }
            }).observe(document, { childList: true, subtree: true });"""
        )
        page.goto(f"{live_server.url}/about")
        assert page.evaluate("window.__themeAtBody") == "dark"
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        assert bg == "rgb(11, 22, 32)"


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

    def test_footer_project_links(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        expect(nav.get_footer_link("Contact")).to_have_attribute("href", "/contact")
        expect(nav.get_footer_link("GitHub")).to_have_attribute(
            "href", "https://github.com/BuczynskiRafal/catchments_simulation"
        )
        expect(nav.get_footer_link("PyPI")).to_have_attribute(
            "href", "https://pypi.org/project/catchment-simulation/"
        )
