"""Home page tests."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from .pages.home_page import HomePage

pytestmark = pytest.mark.e2e


class TestHomePage:
    """Home / main_view page content tests."""

    def test_heading_visible(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        assert hp.get_main_heading() == "Catchment Simulation"

    def test_page_title(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        assert "Catchment Simulation" in hp.get_title()

    def test_plotly_chart_containers_present(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        charts = hp.get_plotly_charts()
        # Should have at least 3 chart divs (slope, area, width)
        assert charts.count() >= 3

    def test_code_examples_present(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        examples = hp.get_code_examples()
        assert examples.count() >= 1

    def test_chart_data_script_tag_exists(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        assert hp.has_chart_data_script()

    def test_section_headings_are_present(self, page: Page, live_server) -> None:
        hp = HomePage(page, live_server.url)
        hp.navigate_to()
        # Key section headings
        expect(page.locator("h3", has_text="Simulation Methods")).to_be_visible()
        expect(page.locator("h3", has_text="Analysis Functions")).to_be_visible()
