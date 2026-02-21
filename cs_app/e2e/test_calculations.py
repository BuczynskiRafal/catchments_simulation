"""Calculations page tests."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from .pages.calculations_page import CalculationsPage

pytestmark = pytest.mark.e2e


class TestCalculationsPageRendering:
    """Calculations page renders its content sections."""

    def test_nn_heading_visible(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        heading = cp.get_nn_heading()
        assert heading is not None
        assert "Neural Network" in heading

    def test_results_comparison_heading(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        heading = cp.get_results_heading()
        assert heading is not None
        assert "Results Comparison" in heading

    def test_results_table_has_correct_columns(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        assert cp.has_results_table()
        cols = cp.get_results_columns()
        assert "Name" in cols
        assert "SWMM Runoff [m3]" in cols
        assert "ANN Runoff [m3]" in cols

    def test_nn_architecture_image(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        assert cp.has_nn_architecture_image()

    def test_upload_zone_present(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        assert cp.upload.is_visible()

    def test_run_calculations_button_present(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        expect(cp.run_button).to_be_visible()
        expect(cp.run_button).to_have_text("Run Calculations")

    def test_page_title(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        assert "Calculations" in cp.get_title()

    def test_anonymous_user_can_access_page(self, page: Page, live_server) -> None:
        """Calculations page does NOT require @login_required."""
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        # Page should load OK, not redirect to login
        assert "/calculations" in page.url
        assert cp.has_results_table()

    def test_empty_results_show_placeholder(self, page: Page, live_server) -> None:
        """Without running calculations, the table shows placeholder dashes."""
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        first_cell = page.locator("table.table-striped tbody td").first
        assert first_cell.inner_text().strip() == "-"


class TestCalculationsExecution:
    """Running calculations (requires sample data + SWMM + ANN model)."""

    @pytest.mark.slow
    def test_full_calculation_run(self, auth_page: Page, live_server) -> None:
        cp = CalculationsPage(auth_page, live_server.url)
        cp.navigate_to()

        cp.upload.click_sample_data()

        cp.run_calculations()
        auth_page.wait_for_load_state("domcontentloaded", timeout=60_000)

        assert cp.has_results_table()
        # After running, rows should have actual values, not just dashes
        row_count = cp.get_results_row_count()
        assert row_count > 0
