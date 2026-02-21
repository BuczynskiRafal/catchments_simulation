"""Responsive layout tests.

Verifies that the app works correctly at different viewport sizes.
The current nav uses ``flex-wrap`` (not a collapsible hamburger menu),
so we test that content wraps properly instead of checking for a toggler.

Simulation/Timeseries pages require @login_required, so responsive tests
for those pages use auth_page.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from .pages.home_page import HomePage

pytestmark = pytest.mark.e2e

VIEWPORTS = {
    "mobile": {"width": 375, "height": 667},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1920, "height": 1080},
    "bootstrap_breakpoint": {"width": 992, "height": 768},
}


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
    def test_nav_links_visible_at_all_viewports(
        self, page: Page, live_server, viewport_name: str, viewport: dict
    ) -> None:
        """Nav uses flex-wrap, so all links should remain visible (wrapped, not collapsed)."""
        page.set_viewport_size(viewport)
        page.goto(f"{live_server.url}/")
        # Use header-scoped selector to avoid ambiguity with footer "About" link
        expect(page.locator("header").get_by_role("link", name="Home")).to_be_visible()
        expect(page.locator("header").get_by_role("link", name="About")).to_be_visible()

    def test_no_horizontal_overflow_mobile(self, page: Page, live_server) -> None:
        page.set_viewport_size(VIEWPORTS["mobile"])
        page.goto(f"{live_server.url}/")
        body_width = page.evaluate("document.body.scrollWidth")
        viewport_width = page.evaluate("window.innerWidth")
        # Allow a small tolerance (5px) for scrollbar or rounding
        assert (
            body_width <= viewport_width + 5
        ), f"Horizontal overflow detected: body={body_width}px > viewport={viewport_width}px"


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
