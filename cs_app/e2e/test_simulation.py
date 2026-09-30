"""Simulation page tests.

Simulation page requires @login_required, so all tests use auth_page.
Tests that run actual SWMM simulations are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, Route, expect

from .pages.nav_component import NavComponent
from .pages.simulation_page import SimulationPage

pytestmark = pytest.mark.e2e

SIMULATION_ROUTE = re.compile(r".*/simulation$")


def _open(page: Page, live_server) -> SimulationPage:
    sp = SimulationPage(page, live_server.url)
    sp.navigate_to()
    return sp


class TestSimulationFormRendering:
    """Form fields render correctly (requires authentication)."""

    def test_form_fields_present(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        assert sp.has_form()
        expect(sp.option_select).to_be_visible()
        expect(sp.start_input).to_be_visible()
        expect(sp.stop_input).to_be_visible()
        expect(sp.step_input).to_be_visible()
        expect(sp.catchment_name_select).to_be_visible()

    def test_page_heading(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        assert sp.get_heading(level=1) == "Parameter sweep"

    def test_default_catchment_placeholder(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.catchment_name_select.locator("option")).to_have_text(
            ["--- Upload a file first ---"]
        )
        assert sp.get_catchment_options() == [""]

    def test_predefined_method_hides_range_fields(self, auth_page: Page, live_server) -> None:
        """Literature-value methods need no range: the fields give way to a note."""
        sp = _open(auth_page, live_server)
        expect(sp.range_fields).to_be_visible()
        expect(sp.literature_note).to_be_hidden()

        sp.set_option("simulate_n_imperv")
        expect(sp.start_input).to_be_hidden()
        expect(sp.range_fields).to_be_hidden()
        expect(sp.literature_note).to_be_visible()

        sp.set_option("simulate_area")
        expect(sp.start_input).to_be_visible()
        expect(sp.literature_note).to_be_hidden()

    def test_run_button_text(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.run_button).to_have_text("Run Simulation")

    def test_empty_state_before_first_run(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.empty_state).to_be_visible()
        expect(sp.empty_state).to_contain_text("No results yet")
        expect(sp.results_heading).to_have_count(0)
        expect(sp.loading_spinner).to_be_hidden()


class TestSimulationRangeHint:
    """Live run-count hint under the range fields."""

    def test_hint_counts_runs(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.set_range(0, 100, 10)
        expect(sp.range_hint).to_have_text("11 runs · 0 → 100, step 10")
        expect(sp.range_hint).not_to_have_class(re.compile(r"\bis-warning\b"))
        # The package sweeps np.arange(start, stop + step / 2, step): 0, 4, 8 and 12.
        sp.set_range(0, 11, 4)
        expect(sp.range_hint).to_have_text("4 runs · 0 → 11, step 4")

    def test_hint_warns_over_server_limit(self, auth_page: Page, live_server) -> None:
        """Warns (without blocking) once (stop - start) / step reaches the form's limit of 100."""
        sp = _open(auth_page, live_server)
        sp.set_range(0, 99, 1)
        expect(sp.range_hint).to_have_text("100 runs · 0 → 99, step 1")

        sp.set_stop(100)
        expect(sp.range_hint).to_contain_text("over the limit of 100 runs")
        expect(sp.range_hint).to_have_class(re.compile(r"\bis-warning\b"))
        expect(sp.run_button).to_be_enabled()

    def test_hint_flags_reversed_range(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.set_range(10, 1, 1)
        expect(sp.range_hint).to_have_text("Stop must be greater than or equal to start.")

    def test_hidden_invalid_range_does_not_block_a_literature_run(
        self, auth_page: Page, live_server
    ) -> None:
        """Leftovers in the hidden range are neither checked by the browser nor sent."""
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(-5, 1, 1)
        sp.set_option("simulate_n_imperv")
        auth_page.route(
            "**/simulation",
            lambda route: route.fulfill(
                status=400, json={"message": "Stubbed.", "field_errors": {}}
            ),
        )

        with auth_page.expect_request(lambda request: request.method == "POST") as posted:
            sp.run_simulation()

        assert 'name="start"' not in (posted.value.post_data or "")
        expect(sp.error_summary).to_contain_text("Stubbed.")


class TestSimulationModelHint:
    """Step 1 explains what to do while no model is loaded."""

    def test_hint_follows_model_state(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.model_hint).to_be_visible()

        sp.load_sample_and_select_catchment()
        expect(sp.model_hint).to_be_hidden()

        sp.upload.clear_upload()
        expect(sp.model_hint).to_be_visible()

    def test_hint_hidden_when_model_restored(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.navigate_to()
        expect(sp.model_hint).to_be_hidden()

    def test_sample_button_keeps_keyboard_focus(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sample = auth_page.get_by_role("button", name="Try sample data")
        sample.focus()

        auth_page.keyboard.press("Enter")

        sp.wait_for_catchment_options()
        expect(sample).to_be_enabled()
        expect(sample).to_be_focused()


class TestSimulationModelChange:
    """Results belong to the model they were computed from."""

    @pytest.mark.slow
    def test_new_model_marks_results_stale_and_drops_them(
        self, auth_page: Page, live_server
    ) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 2, 1)
        sp.run_simulation()
        sp.wait_for_results()
        stale = sp.canvas.locator(".cs-stale-notice")

        # The model restored on page load is the one the results came from.
        sp.navigate_to()
        sp.wait_for_catchment_options()
        expect(sp.results_heading).to_be_visible()
        expect(stale).to_have_count(0)

        auth_page.get_by_role("button", name="Try sample data").click()
        expect(stale).to_contain_text("previously loaded model")

        sp.navigate_to()
        expect(sp.empty_state).to_be_visible()


class TestSimulationValidation:
    """Server-side validation errors are shown inline without reloading the page."""

    def test_start_greater_than_stop_shows_field_error(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(10, 1, 1)
        sp.mark_window()

        sp.run_simulation()

        expect(sp.stop_input).to_have_class(re.compile(r"\bis-invalid\b"))
        expect(sp.page.locator("#id_stop-async-error")).to_have_text(
            "Stop must be greater than or equal to start."
        )
        expect(sp.error_summary).to_contain_text("Please correct the highlighted fields.")
        expect(sp.stop_input).to_be_focused()
        expect(sp.run_button).to_be_enabled()
        expect(sp.loading_spinner).to_be_hidden()
        expect(sp.empty_state).to_be_visible()
        assert sp.window_is_marked()

    def test_field_error_clears_on_input(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(10, 1, 1)
        sp.run_simulation()
        expect(sp.stop_input).to_have_class(re.compile(r"\bis-invalid\b"))

        sp.set_stop(20)
        expect(sp.stop_input).not_to_have_class(re.compile(r"\bis-invalid\b"))


class TestSimulationProgress:
    """Loading state while the server works (response held back by a route)."""

    def test_loading_state_while_running(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        held: list[Route] = []
        auth_page.route(
            SIMULATION_ROUTE,
            lambda route: held.append(route)
            if route.request.method == "POST"
            else route.continue_(),
        )

        sp.run_simulation()

        expect(sp.loading_spinner).to_be_visible()
        expect(sp.elapsed_counter).to_have_text(re.compile(r"^\d+ s$"))
        expect(sp.run_button).to_be_disabled()
        expect(sp.run_button).to_have_attribute("aria-busy", "true")
        expect(sp.canvas).to_have_attribute("aria-busy", "true")
        assert len(held) == 1
        assert held[0].request.headers.get("x-requested-with") == "XMLHttpRequest"

        held[0].fulfill(
            status=500,
            content_type="application/json",
            body='{"message": "An error occurred while running the simulation.", "field_errors": {}}',
        )
        expect(sp.loading_spinner).to_be_hidden()
        expect(sp.run_button).to_be_enabled()
        expect(sp.error_summary).to_contain_text("An error occurred while running the simulation.")


class TestSimulationExecution:
    """Full simulation execution tests (require sample data + SWMM)."""

    @pytest.mark.slow
    def test_full_simulation_run(self, auth_page: Page, live_server) -> None:
        """Load sample → select catchment → run → results swap in without a page load."""
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_option("simulate_percent_slope")
        sp.set_range(1, 5, 1)
        sp.mark_window()

        sp.run_simulation()
        sp.wait_for_results()

        assert sp.window_is_marked(), "the run reloaded the page"
        expect(sp.results_heading).to_be_focused()
        expect(auth_page.locator(".toast", has_text="Simulation finished")).to_be_visible()
        expect(sp.empty_state).to_have_count(0)
        expect(sp.loading_spinner).to_be_hidden()
        assert sp.has_results_table()
        assert sp.get_results_row_count() == 5
        assert sp.has_chart()
        expect(sp.chart).to_have_attribute("role", "img")
        expect(sp.metric_tiles).to_have_count(4)
        expect(sp.metric_tiles.first).to_contain_text("Runoff at lowest value")
        assert sp.has_download_button()
        expect(sp.copy_table_button).to_be_visible()
        expect(sp.download_png_button).to_be_visible()

    @pytest.mark.slow
    def test_results_table_sorts_and_copies(self, auth_page: Page, live_server) -> None:
        auth_page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        sp.run_simulation()
        sp.wait_for_results()

        swept_header = sp.results_table.locator("thead th").first
        swept_header.locator(".cs-sort").click()
        expect(swept_header).to_have_attribute("aria-sort", "ascending")
        swept_header.locator(".cs-sort").click()
        expect(swept_header).to_have_attribute("aria-sort", "descending")
        assert sp.column_values(0) == ["3", "2", "1"]

        sp.copy_table_button.click()
        expect(auth_page.locator(".toast", has_text="Table copied as CSV.")).to_be_visible()
        csv = auth_page.evaluate("navigator.clipboard.readText()")
        lines = csv.split("\r\n")
        assert lines[0].startswith("Percent Slope [%],Total Runoff Volume")
        assert [line.split(",")[0] for line in lines[1:]] == ["3", "2", "1"]

    @pytest.mark.slow
    def test_download_does_not_start_a_run(self, auth_page: Page, live_server) -> None:
        """Audit S2: the download submits its own form and leaves the run form idle."""
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        sp.run_simulation()
        sp.wait_for_results()

        with auth_page.expect_download() as download_info:
            sp.download_button.click()

        assert download_info.value.suggested_filename.endswith(".xlsx")
        expect(sp.loading_spinner).to_be_hidden()
        expect(sp.run_button).to_be_enabled()
        expect(sp.results_heading).to_be_visible()

    @pytest.mark.slow
    def test_second_run_replaces_results(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        sp.run_simulation()
        sp.wait_for_results()
        sp.mark_window()

        sp.set_option("simulate_area")
        sp.set_range(1, 4, 1)
        sp.run_simulation()
        expect(sp.results_rows).to_have_count(4, timeout=SimulationPage.RUN_TIMEOUT_MS)

        assert sp.window_is_marked()
        expect(sp.results_heading).to_have_count(1)
        expect(sp.results_table.locator("thead th").first).to_contain_text("Area")
        expect(sp.chart.locator(".main-svg").first).to_be_attached()

    @pytest.mark.slow
    def test_theme_switch_keeps_chart(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        NavComponent(auth_page).set_theme("light")
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        sp.run_simulation()
        sp.wait_for_results()
        chart_font = "document.getElementById('simulation-chart').layout.font.color"
        light_color = auth_page.evaluate(chart_font)

        NavComponent(auth_page).set_theme("dark")

        auth_page.wait_for_function(f"{chart_font} !== {light_color!r}")
        expect(sp.chart.locator(".main-svg").first).to_be_attached()
        expect(sp.results_rows).to_have_count(3)

    @pytest.mark.slow
    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_results_have_no_serious_a11y_violations(
        self, auth_page: Page, live_server, theme: str
    ) -> None:
        # No reveal animation: axe would measure contrast mid-fade.
        auth_page.emulate_media(reduced_motion="reduce")
        sp = _open(auth_page, live_server)
        NavComponent(auth_page).set_theme(theme)
        sp.load_sample_and_select_catchment()
        sp.set_range(1, 3, 1)
        sp.run_simulation()
        sp.wait_for_results()

        report = sp.run_axe_audit()

        assert not report.critical + report.serious, report.critical + report.serious
