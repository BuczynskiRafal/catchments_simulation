"""Calculations (SWMM vs ANN comparison) page tests."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.calculations_page import CalculationsPage

pytestmark = pytest.mark.e2e

RESULT_COLUMNS = [
    "Name",
    "SWMM Runoff [m³]",
    "ANN Runoff [m³]",
    "Difference [m³]",
    "Difference [%]",
]


def _open(page: Page, live_server) -> CalculationsPage:
    cp = CalculationsPage(page, live_server.url)
    cp.navigate_to()
    return cp


class TestCalculationsPageRendering:
    """Calculations page renders its content sections."""

    def test_page_heading_and_title(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        assert cp.get_page_heading() == "SWMM vs neural network runoff"
        assert "Calculations" in cp.get_title()

    def test_nn_heading_visible(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        heading = cp.get_nn_heading()
        assert heading is not None
        assert "Neural Network" in heading

    def test_empty_state_heading_before_first_run(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        heading = cp.empty_state.get_by_role("heading", level=2)
        expect(heading).to_have_text("No comparison yet")
        # The results heading (the focus target after a run) comes with the results.
        expect(cp.results_heading).to_have_count(0)

    def test_empty_state_before_first_run(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.empty_state).to_be_visible()
        expect(cp.empty_state).to_contain_text("No comparison yet")
        expect(cp.results_table).to_have_count(0)

    def test_validity_note_visible(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.validity_note).to_be_visible()
        expect(cp.validity_note).to_contain_text("limited precipitation range")

    def test_model_details_expand_with_keyboard(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        architecture = cp.model_details("Architecture")
        assert not cp.has_nn_architecture_image()

        architecture.locator("summary").focus()
        page.keyboard.press("Enter")

        expect(architecture).to_have_attribute("open", "")
        assert cp.has_nn_architecture_image()

    def test_upload_zone_present(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        assert cp.upload.is_visible()
        expect(page.locator(".upload-zone-wrapper")).to_have_class(
            re.compile("upload-zone--compact")
        )

    def test_run_calculations_button_present(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.run_button).to_be_visible()
        expect(cp.run_button).to_have_text("Run Calculations")
        expect(cp.loading_state).to_be_hidden()

    def test_anonymous_user_can_access_page(self, page: Page, live_server) -> None:
        """Calculations page does NOT require @login_required."""
        cp = _open(page, live_server)
        assert "/calculations" in page.url
        expect(cp.empty_state).to_be_visible()

    def test_anonymous_run_redirects_to_login(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        cp.run_calculations()
        page.wait_for_url("**/login/**", timeout=10_000)
        assert "next=%2Fcalculations" in page.url

    def test_no_horizontal_scroll_on_phone(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 360, "height": 780})
        _open(page, live_server)
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0


class TestCalculationsAsyncErrors:
    """Server errors are shown in the control panel without a page reload."""

    def test_run_without_model_shows_inline_error(self, auth_page: Page, live_server) -> None:
        cp = _open(auth_page, live_server)
        auth_page.evaluate("window.__csNoReload = true")

        cp.run_calculations()

        expect(cp.form_error).to_have_text("Please upload a file first.")
        expect(cp.form_error).to_have_attribute("role", "alert")
        expect(cp.run_button).to_be_enabled()
        expect(cp.run_button).to_be_focused()
        expect(cp.loading_state).to_be_hidden()
        expect(cp.empty_state).to_be_visible()
        expect(auth_page.locator(".cs-toast--danger")).to_be_visible()
        assert auth_page.evaluate("window.__csNoReload") is True

    def test_error_clears_on_next_run(self, auth_page: Page, live_server) -> None:
        cp = _open(auth_page, live_server)
        cp.run_calculations()
        expect(cp.form_error).to_have_count(1)

        cp.run_calculations()

        # A repeated failure replaces the alert instead of stacking a second one.
        expect(cp.form_error).to_have_count(1)
        expect(cp.run_button).to_be_enabled()


@pytest.mark.slow
class TestCalculationsExecution:
    """Running calculations (requires sample data + SWMM + ANN model)."""

    @pytest.fixture()
    def results_page(self, auth_page: Page, live_server) -> CalculationsPage:
        cp = _open(auth_page, live_server)
        cp.upload.click_sample_data()
        auth_page.evaluate("window.__csNoReload = true")
        cp.run_and_wait_for_results()
        return cp

    def test_run_swaps_results_without_reload(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        assert page.evaluate("window.__csNoReload") is True
        expect(results_page.results_heading).to_be_focused()
        expect(results_page.run_button).to_be_enabled()
        expect(results_page.loading_state).to_be_hidden()
        expect(results_page.empty_state).to_have_count(0)
        expect(page.locator("#calculations-results")).to_have_count(1)
        expect(page.locator(".cs-toast--success")).to_contain_text("Comparison finished")

    def test_results_table_columns_and_rows(self, results_page: CalculationsPage) -> None:
        assert results_page.get_results_columns() == RESULT_COLUMNS
        assert results_page.get_results_row_count() > 0
        assert results_page.column_values(0)[0] == "S1"
        # SWMM reports 3.36 x 10^6 L for S1 of the sample model: shown in m³, as on the other tools.
        assert results_page.column_values(1)[0] == "3360.00"
        for value in results_page.column_values(1) + results_page.column_values(2):
            assert re.fullmatch(r"-?[\d,]+\.\d{2}", value)

    def test_metric_tiles(self, results_page: CalculationsPage) -> None:
        assert re.fullmatch(r"[\d,]+\.\d{2}m³", results_page.metric_value("Mean absolute error"))
        assert re.fullmatch(r"[\d,]+\.\d{2}m³", results_page.metric_value("Root mean square error"))
        assert results_page.metric_value("Subcatchments") == str(
            results_page.get_results_row_count()
        )

    def test_charts_render_and_follow_theme(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        for chart in (results_page.parity_chart, results_page.bars_chart):
            expect(chart).to_have_attribute("role", "img")
            expect(chart.locator(".main-svg").first).to_be_attached()
        marker_color = (
            "() => document.getElementById('calculations-parity-chart').data[1].marker.color"
        )
        light_color = page.evaluate(marker_color)

        results_page.nav.set_theme("dark")

        page.wait_for_function(f"({marker_color})() !== {light_color!r}")
        expect(results_page.parity_chart.locator(".main-svg").first).to_be_attached()
        expect(results_page.bars_chart.locator(".main-svg").first).to_be_attached()

    def test_table_sorts_by_difference(self, results_page: CalculationsPage) -> None:
        header = results_page.results_table.locator("th", has_text="Difference [m³]")

        results_page.sort_by("Difference [m³]")
        expect(header).to_have_attribute("aria-sort", "ascending")
        ascending = [float(v.replace(",", "")) for v in results_page.column_values(3)]
        assert ascending == sorted(ascending)

        results_page.sort_by("Difference [m³]")
        expect(header).to_have_attribute("aria-sort", "descending")
        descending = [float(v.replace(",", "")) for v in results_page.column_values(3)]
        assert descending == sorted(ascending, reverse=True)

    def test_results_fit_phone_width(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        page.set_viewport_size({"width": 360, "height": 780})
        # Plotly resizes the charts on a debounced window resize.
        page.wait_for_function(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth",
            timeout=5_000,
        )
        # The wide table scrolls inside its keyboard-focusable region instead.
        expect(results_page.page.locator(".cs-table-wrap")).to_have_attribute("tabindex", "0")

    def test_rerun_replaces_results_in_place(self, results_page: CalculationsPage) -> None:
        results_page.page.evaluate(
            "document.getElementById('calculations-table').dataset.firstRun = '1'"
        )
        results_page.run_calculations()
        expect(results_page.results_table).not_to_have_attribute(
            "data-first-run", "1", timeout=60_000
        )
        expect(results_page.results_table).to_have_count(1)
        expect(results_page.results_heading).to_be_focused()
