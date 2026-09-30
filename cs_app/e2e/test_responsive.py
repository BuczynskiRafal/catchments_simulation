"""Responsive layout tests.

Verifies that the app works correctly at different viewport sizes.
The navbar is ``navbar-expand-lg``: below 992px the links collapse behind a
toggler, at and above it they are always visible.

Simulation/Timeseries pages require @login_required, so responsive tests
for those pages use auth_page.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.home_page import HomePage
from .pages.nav_component import NavComponent

pytestmark = pytest.mark.e2e

VIEWPORTS = {
    "mobile": {"width": 375, "height": 667},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1920, "height": 1080},
    "bootstrap_breakpoint": {"width": 992, "height": 768},
}
COLLAPSED = {"mobile", "tablet"}


class TestResponsiveHome:
    """Home page at different viewport sizes."""

    @pytest.mark.parametrize(
        "viewport_name,viewport",
        VIEWPORTS.items(),
        ids=VIEWPORTS.keys(),
    )
    def test_heading_visible_at_all_viewports(
        self, page: Page, live_server, viewport_name: str, viewport: dict
    ) -> None:
        page.set_viewport_size(viewport)
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        assert hp.get_main_heading() is not None

    @pytest.mark.parametrize(
        "viewport_name,viewport",
        VIEWPORTS.items(),
        ids=VIEWPORTS.keys(),
    )
    def test_nav_links_reachable_at_all_viewports(
        self, page: Page, live_server, viewport_name: str, viewport: dict
    ) -> None:
        """Links are visible on lg+, and one toggler click away below it."""
        page.set_viewport_size(viewport)
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)

        if viewport_name in COLLAPSED:
            expect(nav.toggler).to_be_visible()
            expect(nav.toggler).to_have_attribute("aria-expanded", "false")
            expect(nav.link("Home")).to_be_hidden()
            nav.open_menu()
            expect(nav.toggler).to_have_attribute("aria-expanded", "true")
        else:
            expect(nav.toggler).to_be_hidden()

        for name in ("Home", "Simulation", "Timeseries", "Calculations", "About", "Log in"):
            expect(nav.link(name)).to_be_visible()

    def test_no_horizontal_overflow_mobile(self, page: Page, live_server) -> None:
        page.set_viewport_size(VIEWPORTS["mobile"])
        page.goto(f"{live_server.url}/")
        body_width = page.evaluate("document.body.scrollWidth")
        viewport_width = page.evaluate("window.innerWidth")
        # Allow a small tolerance (5px) for scrollbar or rounding
        assert (
            body_width <= viewport_width + 5
        ), f"Horizontal overflow detected: body={body_width}px > viewport={viewport_width}px"

    def test_mobile_menu_navigates_and_closes(self, page: Page, live_server) -> None:
        page.set_viewport_size(VIEWPORTS["mobile"])
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.click_about()
        expect(page).to_have_url(re.compile(r".*/about$"))
        expect(nav.toggler).to_have_attribute("aria-expanded", "false")

    def test_mobile_theme_menu_is_usable(self, page: Page, live_server) -> None:
        page.set_viewport_size(VIEWPORTS["mobile"])
        page.emulate_media(color_scheme="light")
        page.goto(f"{live_server.url}/")
        nav = NavComponent(page)
        nav.set_theme("dark")
        assert nav.resolved_theme() == "dark"

    def test_no_horizontal_overflow_mobile_with_menu_open(
        self, auth_page: Page, live_server
    ) -> None:
        auth_page.set_viewport_size(VIEWPORTS["mobile"])
        auth_page.goto(f"{live_server.url}/")
        nav = NavComponent(auth_page)
        nav.open_account_menu()
        overflow = auth_page.evaluate("document.body.scrollWidth - window.innerWidth")
        assert overflow <= 5, f"Horizontal overflow with menu open: {overflow}px"


class TestResponsiveSimulation:
    """Simulation page form layout at different viewports (requires auth)."""

    @pytest.mark.parametrize(
        "viewport_name,viewport",
        [("mobile", VIEWPORTS["mobile"]), ("tablet", VIEWPORTS["tablet"])],
        ids=["mobile", "tablet"],
    )
    def test_simulation_form_usable(
        self, auth_page: Page, live_server, viewport_name: str, viewport: dict
    ) -> None:
        auth_page.set_viewport_size(viewport)
        auth_page.goto(f"{live_server.url}/simulation")
        # Form and button should be visible
        expect(auth_page.locator("#id_option")).to_be_visible()
        expect(auth_page.locator("#run-simulation-button")).to_be_visible()


class TestWorkbenchPanel:
    """The control panel sticks only while it fits; it never scrolls inside itself."""

    @pytest.mark.parametrize(
        ("height", "sticky"), [(1080, True), (600, False)], ids=["fits", "taller"]
    )
    def test_panel_sticks_only_when_it_fits(
        self, auth_page: Page, live_server, height: int, sticky: bool
    ) -> None:
        auth_page.set_viewport_size({"width": 1366, "height": height})
        auth_page.goto(f"{live_server.url}/simulation")
        panel = auth_page.locator(".cs-panel")

        if sticky:
            expect(panel).to_have_class(re.compile(r"\bis-sticky\b"))
        else:
            expect(panel).not_to_have_class(re.compile(r"\bis-sticky\b"))
        assert panel.evaluate("el => el.scrollHeight <= el.clientHeight")

    def test_panel_is_not_sticky_below_lg(self, auth_page: Page, live_server) -> None:
        auth_page.set_viewport_size({"width": 768, "height": 1024})
        auth_page.goto(f"{live_server.url}/simulation")
        expect(auth_page.locator(".cs-panel")).not_to_have_class(re.compile(r"\bis-sticky\b"))
