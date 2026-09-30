"""Timeseries analysis page tests.

Timeseries page requires @login_required, so all tests use auth_page.
Tests that run actual SWMM simulations are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Browser, Page, expect

from .conftest import new_context_without_js, wait_for_styles
from .pages.timeseries_page import TimeseriesPage

pytestmark = pytest.mark.e2e

# Set on window before a run; a full page load would wipe it.
MARK_PAGE = "() => { window.__tsNoReload = true; }"
PAGE_NOT_RELOADED = "() => window.__tsNoReload === true"


def _open(auth_page: Page, live_server) -> TimeseriesPage:
    tp = TimeseriesPage(auth_page, live_server.url)
    tp.navigate_to()
    return tp


class TestTimeseriesFormRendering:
    """Form fields render correctly (requires authentication)."""

    def test_form_fields_present(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        assert tp.has_form()
        expect(tp.catchment_name_select).to_be_visible()
        expect(auth_page.get_by_role("radio", name="Single run")).to_be_visible()
        expect(auth_page.get_by_role("radio", name="Parameter sweep")).to_be_visible()

    def test_page_heading_is_level_one(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        assert tp.get_heading(level=1) == "Timeseries analysis"

    def test_run_button_text(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        expect(tp.run_button).to_have_text("Run Timeseries Analysis")

    def test_default_mode_is_single_and_hides_sweep_range(
        self, auth_page: Page, live_server
    ) -> None:
        tp = _open(auth_page, live_server)
        assert tp.selected_mode() == "single"
        expect(tp.sweep_fields).to_be_hidden()
        # Disabled so hidden inputs are neither validated nor submitted.
        expect(tp.start_input).to_be_disabled()

    def test_sweep_mode_shows_feature_field(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.set_mode("sweep")
        expect(auth_page.locator("#feature-wrapper")).to_be_visible()
        expect(tp.feature_select).to_be_enabled()

    def test_sweep_mode_shows_range_fields(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.set_mode("sweep")
        expect(auth_page.locator("#start-wrapper")).to_be_visible()
        expect(auth_page.locator("#stop-wrapper")).to_be_visible()
        expect(auth_page.locator("#step-wrapper")).to_be_visible()

    def test_mode_is_keyboard_operable(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.mode_radio("single").focus()
        auth_page.keyboard.press("ArrowRight")
        assert tp.selected_mode() == "sweep"
        expect(tp.sweep_fields).to_be_visible()

    def test_empty_state_before_first_run(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        expect(tp.empty_state).to_contain_text("No results yet")
        expect(tp.loading_spinner).to_be_hidden()
        expect(tp.results_heading).to_have_count(0)


class TestTimeseriesRunsHint:
    """Live sweep hint mirrors the package's run count and the form's step limit."""

    def test_hint_counts_runs(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.set_mode("sweep")
        tp.set_start(0)
        tp.set_stop(100)
        tp.set_step(10)
        expect(tp.runs_hint).to_have_text("11 runs · 0 → 100, step 10")
        tp.set_stop(0)
        expect(tp.runs_hint).to_have_text("1 run · 0 → 0, step 10")

    def test_hint_warns_about_limit_and_reversed_range(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.set_mode("sweep")
        tp.set_start(0)
        tp.set_stop(100)
        tp.set_step(0.5)
        expect(tp.runs_hint).to_contain_text("over the limit of 100 runs")
        expect(tp.runs_hint).to_have_class(re.compile(r"\bis-warning\b"))
        tp.set_step(10)
        tp.set_start(200)
        expect(tp.runs_hint).to_contain_text("Stop must be greater than or equal to start")


class TestTimeseriesValidation:
    """Server-side validation errors come back inline without a page load."""

    def test_sweep_start_greater_than_stop(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.set_mode("sweep")
        tp.set_feature("PercSlope")
        tp.set_start(100)
        tp.set_stop(10)
        tp.set_step(10)
        auth_page.evaluate(MARK_PAGE)

        tp.run_analysis()

        error = auth_page.locator("#id_stop-async-error")
        expect(error).to_have_text("Stop must be greater than or equal to start.")
        expect(tp.stop_input).to_have_class(re.compile(r"\bis-invalid\b"))
        expect(tp.stop_input).to_be_focused()
        expect(auth_page.locator("#timeseries-form .alert-danger")).to_be_visible()
        expect(tp.run_button).to_be_enabled()
        expect(tp.loading_spinner).to_be_hidden()
        expect(tp.empty_state).to_be_visible()
        assert auth_page.evaluate(PAGE_NOT_RELOADED)


class TestTimeseriesExecution:
    """Full timeseries execution tests."""

    @pytest.mark.slow
    def test_single_timeseries_run(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        catchment = tp.load_sample_and_pick_catchment()
        tp.set_mode("single")
        auth_page.evaluate(MARK_PAGE)

        tp.run_analysis()
        tp.wait_for_results()

        assert auth_page.evaluate(PAGE_NOT_RELOADED), "the run reloaded the page"
        expect(tp.results_heading).to_be_focused()
        expect(tp.results_heading).to_contain_text(catchment)
        expect(auth_page.locator(".toast", has_text="Timeseries analysis finished")).to_be_visible()
        expect(tp.run_button).to_be_enabled()
        expect(tp.loading_spinner).to_be_hidden()

        assert tp.has_chart()
        expect(tp.chart).to_have_attribute("role", "img")
        # One unit per axis: infiltration shares the rainfall rate (mm/h), evaporation (mm/day)
        # gets a panel of its own, hidden until its legend entry is selected.
        axes = tp.chart.evaluate(
            "el => Object.fromEntries(el.data.filter(t => t.uid).map(t => [t.uid, t.yaxis]))"
        )
        assert axes["runoff"] == axes["runon"] == "y"
        assert axes["rainfall"] == axes["infiltration_loss"] == "y2"
        assert axes["evaporation_loss"] not in ("y", "y2")
        evap_axis = "el.layout.yaxis" + axes["evaporation_loss"][1:]  # "y3" -> el.layout.yaxis3
        assert "loss rates [mm/h]" in tp.chart.evaluate("el => el.layout.yaxis2.title.text")
        assert tp.chart.evaluate(f"el => {evap_axis}.visible") is False
        assert tp.chart.evaluate("el => el.layout.yaxis.domain") == [0, 1]
        height = tp.chart.evaluate("el => el.layout.height")

        legend_item = tp.chart.locator(".legend .traces", has_text="Evaporation Loss [mm/day]")
        legend_item.locator(".legendtoggle").click()
        expect(
            tp.chart.locator(".annotation-text", has_text="Evaporation Loss [mm/day]")
        ).to_be_visible()
        assert tp.chart.evaluate(f"el => {evap_axis}.visible") is True
        assert tp.chart.evaluate(f"el => {evap_axis}.title.text") == "mm/day"
        # The panel sits below the main plot, without overlapping it, and carries the time axis.
        assert tp.chart.evaluate(f"el => {evap_axis}.domain[1] <= el.layout.yaxis.domain[0]")
        assert tp.chart.evaluate("el => el.layout.xaxis.anchor") == axes["evaporation_loss"]
        assert tp.chart.evaluate("el => el.layout.height") > height

        legend_item.locator(".legendtoggle").click()
        expect(
            tp.chart.locator(".annotation-text", has_text="Evaporation Loss [mm/day]")
        ).to_be_hidden()
        assert tp.chart.evaluate(f"el => {evap_axis}.visible") is False
        assert tp.chart.evaluate("el => el.layout.yaxis.domain") == [0, 1]
        assert tp.chart.evaluate("el => el.layout.xaxis.anchor") == "y"
        assert tp.chart.evaluate("el => el.layout.height") == height
        # The time axis spans exactly the data (the peak label must not widen it).
        assert tp.chart.evaluate(
            """el => { const t = el.data[0].x; return el.layout.xaxis.range[0] === t[0]
                && el.layout.xaxis.range[1] === t[t.length - 1]; }"""
        )
        assert re.fullmatch(r"[\d.,]+\s*CMS", tp.get_peak_runoff() or "")
        assert re.fullmatch(r"[\d.,]+\s*m³", tp.get_runoff_volume() or "")
        assert re.search(r"\d+ (d|h|min|s)\b", tp.get_time_to_peak() or "")

        tp.timestep_toggle.click()
        expect(tp.timestep_table.locator("tbody tr").first).to_be_visible()

        assert tp.has_download_results_button()
        assert tp.has_csv_download_button()
        assert tp.has_png_download_button()

    @pytest.mark.slow
    def test_exports_do_not_trigger_the_run_state(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.run_analysis()
        tp.wait_for_results()

        with auth_page.expect_download() as xlsx_download:
            auth_page.get_by_role("button", name="Download Results (.xlsx)").click()
        assert xlsx_download.value.suggested_filename.endswith(".xlsx")
        expect(tp.loading_spinner).to_be_hidden()

        with auth_page.expect_download() as csv_download:
            auth_page.get_by_role("button", name="Download Results (.csv)").click()
        assert csv_download.value.suggested_filename.endswith(".csv")
        expect(tp.loading_spinner).to_be_hidden()
        expect(tp.run_button).to_be_enabled()

        with auth_page.expect_download() as png_download:
            auth_page.get_by_role("button", name="Download PNG").click()
        assert png_download.value.suggested_filename.endswith(".png")
        expect(auth_page.locator("#timeseries-export-feedback")).to_be_hidden()

    @pytest.mark.slow
    def test_theme_switch_keeps_chart(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.run_analysis()
        tp.wait_for_results()
        rain_fill = "() => document.querySelector('#timeseries-chart .bars .point path').style.fill"
        light_fill = auth_page.evaluate(rain_fill)

        tp.nav.set_theme("dark")

        auth_page.wait_for_function(f"({rain_fill})() !== {light_fill!r}")
        assert tp.has_chart()

    @pytest.mark.slow
    def test_second_run_replaces_results(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.run_analysis()
        tp.wait_for_results()
        auth_page.evaluate(
            "() => { document.getElementById('timeseries-results').dataset.stale = '1'; }"
        )

        tp.run_analysis()

        expect(auth_page.locator("#timeseries-results[data-stale]")).to_have_count(
            0, timeout=60_000
        )
        tp.wait_for_results()
        expect(auth_page.locator("#timeseries-results")).to_have_count(1)
        expect(tp.results_heading).to_have_count(1)

    @pytest.mark.slow
    def test_sweep_timeseries_run(self, auth_page: Page, live_server) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.set_mode("sweep")
        tp.set_feature("PercImperv")
        tp.set_start(0)
        tp.set_stop(50)
        tp.set_step(25)
        expect(tp.runs_hint).to_have_text("3 runs · 0 → 50, step 25")
        auth_page.evaluate(MARK_PAGE)

        tp.run_analysis()
        tp.wait_for_results()

        assert auth_page.evaluate(PAGE_NOT_RELOADED)
        assert tp.has_chart()
        rows = tp.sweep_table.locator("tbody tr")
        expect(rows).to_have_count(3)
        expect(rows.first.locator("th")).to_have_text("0")
        expect(tp.sweep_table.locator("thead")).to_contain_text("Peak runoff [CMS]")
        expect(tp.sweep_table.locator("th button.cs-sort").first).to_be_visible()
        assert tp.get_time_to_peak() is None, "sweep shows a summary table instead of tiles"
        # Values only: the legend title names the parameter.
        assert tp.sweep_legend_names() == ["0", "25", "50"]
        assert tp.chart.evaluate("el => el.layout.legend.title.text") == "Impervious Area [%]"

        before = auth_page.evaluate(
            "() => document.getElementById('timeseries-chart').layout.yaxis.title.text"
        )
        tp.sweep_series_select.select_option("infiltration_loss")
        auth_page.wait_for_function(
            "(before) => document.getElementById('timeseries-chart').layout.yaxis.title.text !== before",
            arg=before,
        )

    @pytest.mark.slow
    def test_leftover_invalid_sweep_values_do_not_block_a_single_run(
        self, auth_page: Page, live_server
    ) -> None:
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.set_mode("sweep")
        tp.set_start(-5)
        tp.set_mode("single")

        tp.run_analysis()

        tp.wait_for_results()
        expect(auth_page.locator("#timeseries-form .alert-danger")).to_have_count(0)


class TestTimeseriesWithoutJavaScript:
    """Without JS the form posts natively (PRG) and the server renders every number."""

    @pytest.mark.slow
    def test_single_run_renders_metrics_after_redirect(
        self, auth_page: Page, live_server, browser: Browser
    ) -> None:
        # Loading the model needs JS (the upload zone); the session then carries it over.
        _open(auth_page, live_server).load_sample_and_pick_catchment()
        context = new_context_without_js(browser)
        context.add_cookies(auth_page.context.cookies())
        try:
            tp = TimeseriesPage(context.new_page(), live_server.url)
            tp.navigate_to()
            wait_for_styles(tp.page)
            tp.set_catchment_name(tp.get_catchment_options()[-1])

            tp.run_analysis()

            expect(tp.results_heading).to_be_visible(timeout=120_000)
            assert re.fullmatch(r"[\d.,]+\s*CMS", tp.get_peak_runoff() or "")
            assert re.fullmatch(r"\d+ h( \d+ min)?", tp.get_time_to_peak() or "")
            assert re.fullmatch(r"[\d.,]+\s*m³", tp.get_runoff_volume() or "")
        finally:
            context.close()


class TestTimeseriesResultsAccessibility:
    """axe finds no critical or serious issues in the results, in either theme."""

    @pytest.mark.slow
    @pytest.mark.parametrize("mode", ["single", "sweep"])
    def test_results_have_no_serious_a11y_violations(
        self, auth_page: Page, live_server, mode: str
    ) -> None:
        # axe would otherwise sample colours mid-way through the reveal and theme transitions.
        auth_page.emulate_media(reduced_motion="reduce")
        tp = _open(auth_page, live_server)
        tp.load_sample_and_pick_catchment()
        tp.set_mode(mode)
        if mode == "sweep":
            tp.set_start(0)
            tp.set_stop(50)
            tp.set_step(25)
        tp.run_analysis()
        tp.wait_for_results()
        if mode == "single":
            tp.timestep_toggle.click()

        for theme in ("light", "dark"):
            tp.nav.set_theme(theme)
            report = tp.run_axe_audit()
            issues = [f"{v.id} ({v.nodes_count})" for v in report.critical + report.serious]
            assert not issues, f"{theme}: {issues}"
