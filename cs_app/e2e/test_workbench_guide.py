"""The workbench leads from model to insight: step states, the run checklist, the
parameter card, the suggested range, the run status panel and results that explain
themselves (reading line, model value marker, water balance, links).

SWMM runs are marked ``slow``; everything else stubs or holds the run request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, Route, expect

from .pages.calculations_page import CalculationsPage
from .pages.nav_component import NavComponent
from .pages.simulation_page import SimulationPage
from .pages.timeseries_page import TimeseriesPage

pytestmark = pytest.mark.e2e

SAMPLE_INP = Path(__file__).resolve().parent.parent / "data" / "example.inp"
SIMULATION_ROUTE = re.compile(r".*/simulation$")
SUBCATCHMENTS_ROUTE = re.compile(r".*/subcatchments/$")
DISCARDED_TEXT = "its results were discarded"


@pytest.fixture
def console_errors(auth_page: Page) -> list[str]:
    errors: list[str] = []
    auth_page.on("console", lambda message: message.type == "error" and errors.append(message.text))
    auth_page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def _open(page: Page, live_server) -> SimulationPage:
    sp = SimulationPage(page, live_server.url)
    sp.navigate_to()
    return sp


def _two_subcatchment_model(tmp_path) -> str:
    """The sample with a copy of S1 named S2 in every subcatchment section."""
    lines = []
    section = ""
    for line in SAMPLE_INP.read_text(encoding="utf-8").splitlines():
        lines.append(line)
        if line.startswith("["):
            section = line.strip().upper()
        if section in ("[SUBCATCHMENTS]", "[SUBAREAS]", "[INFILTRATION]") and line.startswith(
            "S1 "
        ):
            lines.append("S2 " + line[3:])
    path = tmp_path / "two_subcatchments.inp"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _wait_for_model_value(sp: SimulationPage) -> None:
    expect(sp.insight).to_contain_text(re.compile(r"In this model: \d"))


class TestSteps:
    def test_steps_and_checklist_follow_the_model(
        self, auth_page: Page, live_server, console_errors
    ) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.step("model")).to_have_attribute("data-state", "active")
        expect(sp.step("run")).to_have_attribute("data-state", "todo")
        expect(sp.step("run").locator("h2")).to_contain_text("(load a model first)")
        expect(sp.checklist_item("model")).to_have_attribute("data-state", "current")
        expect(sp.next_step).to_contain_text("Next: load a model")

        sp.upload.click_sample_data()

        expect(sp.step("model")).to_have_attribute("data-state", "done")
        expect(sp.step("model").locator("h2")).to_have_text("Step 1: Model (completed)")
        expect(sp.step("run")).to_have_attribute("data-state", "active")
        expect(sp.checklist_item("model")).to_have_attribute("data-state", "done")
        expect(sp.checklist_item("catchment")).to_have_attribute("data-state", "done")
        expect(sp.checklist_item("run")).to_have_attribute("data-state", "current")
        expect(sp.next_step).to_contain_text("select Run Simulation")

        sp.upload.clear_upload()

        expect(sp.step("model")).to_have_attribute("data-state", "active")
        expect(sp.checklist_item("catchment")).to_have_attribute("data-state", "todo")
        assert console_errors == []

    def test_several_subcatchments_keep_the_placeholder_and_ask_for_a_choice(
        self, auth_page: Page, live_server, tmp_path
    ) -> None:
        sp = _open(auth_page, live_server)
        sp.upload.upload_file(_two_subcatchment_model(tmp_path))
        sp.wait_for_catchment_options()

        expect(sp.catchment_name_select).to_have_value("")
        expect(sp.catchment_hint).to_have_text("Choose one of the 2 subcatchments in this model.")
        expect(sp.next_step).to_contain_text("choose the subcatchment")

        sp.run_simulation()

        expect(sp.attention).to_have_text("Choose the subcatchment to run.")
        expect(sp.catchment_name_select).to_be_focused()
        expect(sp.catchment_name_select).to_have_attribute("aria-invalid", "true")

        sp.set_catchment_name("S2")
        expect(sp.attention).to_have_count(0)
        expect(sp.checklist_item("catchment")).to_have_attribute("data-state", "done")

    def test_run_without_a_model_points_at_step_one(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)

        sp.run_simulation()

        expect(sp.attention).to_contain_text("Load a model first")
        expect(sp.upload.trigger).to_be_focused()
        expect(sp.results_heading).to_have_count(0)

    def test_calculations_steps_follow_the_model(self, auth_page: Page, live_server) -> None:
        auth_page.goto(f"{live_server.url}/calculations")
        step = auth_page.locator(".cs-step[data-cs-step='model']")
        expect(step).to_have_attribute("data-state", "active")
        auth_page.locator("#load-sample-data-button").click()
        expect(step).to_have_attribute("data-state", "done")


class TestSampleButton:
    def test_hidden_while_the_sample_is_loaded_and_offered_instead_of_another_model(
        self, auth_page: Page, live_server, tmp_path
    ) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.upload.sample_button).to_have_text("Try sample data")

        sp.upload.click_sample_data()
        expect(sp.upload.sample_button).to_be_hidden()

        # Still hidden when the page restores the sample from the session.
        sp.navigate_to()
        expect(sp.upload.upload_status_text).to_have_text("example.inp")
        expect(sp.upload.sample_button).to_be_hidden()

        sp.upload.upload_file(_two_subcatchment_model(tmp_path))
        expect(sp.upload.sample_button).to_be_visible()
        expect(sp.upload.sample_button).to_have_text("Use sample data instead")

        sp.upload.clear_upload()
        expect(sp.upload.sample_button).to_have_text("Try sample data")


class TestParameterInsight:
    def test_one_values_request_when_the_only_subcatchment_is_selected(
        self, auth_page: Page, live_server
    ) -> None:
        """cs:model-changed and the auto-selection's `change` share one request."""
        sp = _open(auth_page, live_server)
        requests: list[str] = []
        auth_page.on(
            "request",
            lambda request: (
                "/subcatchments/values/" in request.url and requests.append(request.url)
            ),
        )

        sp.upload.click_sample_data()
        _wait_for_model_value(sp)
        auth_page.wait_for_load_state("networkidle")

        assert len(requests) == 1

    def test_card_explains_the_parameter_and_shows_the_models_value(
        self, auth_page: Page, live_server
    ) -> None:
        sp = _open(auth_page, live_server)
        expect(sp.insight).to_contain_text("Slope")
        expect(sp.insight.locator(".cs-insight__model")).to_be_hidden()

        sp.load_sample_and_select_catchment()

        _wait_for_model_value(sp)
        expect(sp.insight).to_contain_text("In this model: 10\u00a0%")
        expect(sp.insight).to_contain_text("Steeper surfaces drain faster")
        expect(sp.insight.locator(".cs-insight__where")).to_contain_text("W·√S")
        link = sp.insight.get_by_role("link", name="How it enters the model")
        expect(link).to_have_attribute("href", "/how-it-works#parameter-PercSlope")

        sp.set_option("simulate_curb_length")
        expect(sp.insight).to_contain_text("does not enter the runoff equations")
        expect(sp.insight).to_contain_text("In this model: 0\u00a0m")

        sp.set_option("simulate_n_perv")
        expect(sp.literature_note).to_be_visible()
        expect(sp.insight).to_contain_text("Manning's n, pervious")
        expect(sp.insight).to_contain_text("In this model: 1")
        expect(link).to_have_attribute("href", "/how-it-works#parameter-N-Perv")

    def test_timeseries_sweep_gets_the_card_and_the_suggestion(
        self, auth_page: Page, live_server
    ) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        tp.load_sample_and_pick_catchment()
        tp.set_mode("sweep")
        insight = auth_page.locator("#ts-insight")
        expect(insight).to_contain_text("In this model: 25\u00a0%")
        expect(auth_page.locator("#ts-range-suggestion")).to_have_text(
            "Range covers every possible value."
        )

        tp.set_feature("Width")

        expect(insight).to_contain_text("In this model: 100\u00a0m")
        expect(tp.start_input).to_have_value("50")
        expect(tp.stop_input).to_have_value("150")
        expect(tp.step_input).to_have_value("10")
        expect(tp.runs_hint).to_have_text("11 runs · 50 → 150, step 10")


class TestRangeSuggestion:
    def test_applied_on_load_and_parameter_change_kept_after_manual_edits(
        self, auth_page: Page, live_server
    ) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        _wait_for_model_value(sp)

        assert sp.range_values() == ["5", "15", "1"]
        expect(sp.suggestion).to_have_text("Range centred on the model's value.")
        expect(sp.range_hint).to_have_text("11 runs · 5 → 15, step 1")

        sp.set_option("simulate_width")
        assert sp.range_values() == ["50", "150", "10"]

        sp.set_start(70)
        expect(sp.suggestion).to_contain_text("Suggested for this model: 50 → 150, step 10")
        # A new look at the same subcatchment keeps what the user typed.
        sp.set_catchment_name("S1")
        auth_page.wait_for_load_state("networkidle")
        assert sp.range_values() == ["70", "150", "10"]

        sp.use_suggestion_button.click()
        assert sp.range_values() == ["50", "150", "10"]
        expect(sp.range_hint).to_have_text("11 runs · 50 → 150, step 10")
        expect(sp.start_input).to_be_focused()

        # The old range belonged to another parameter: changing it applies the new suggestion.
        sp.set_start(70)
        sp.set_option("simulate_area")
        assert sp.range_values() == ["2.5", "7.5", "0.5"]

        # The sample's curb length is 0: there is nothing to centre a range on.
        sp.set_option("simulate_curb_length")
        assert sp.range_values() == ["0", "100", "10"]
        expect(sp.suggestion).to_have_text("Default range: the model's value is 0.")


class TestRunRightAfterLoading:
    """
    The upload zone announces a model (cs:model-changed) only once its subcatchments are
    listed, after the server has taken it. A run started in between belongs to the new
    model and must be kept; a run started before a different model is loaded must not.
    """

    @staticmethod
    def _hold_subcatchments(page: Page) -> list[Route]:
        held: list[Route] = []
        page.route(SUBCATCHMENTS_ROUTE, lambda route: held.append(route))
        return held

    @staticmethod
    def _release(page: Page, held: list[Route]) -> None:
        for route in held:
            route.continue_()
        page.unroute(SUBCATCHMENTS_ROUTE)

    @pytest.mark.slow
    def test_calculations_run_started_before_the_announcement_is_kept(
        self, auth_page: Page, live_server
    ) -> None:
        cp = CalculationsPage(auth_page, live_server.url)
        cp.navigate_to()
        held = self._hold_subcatchments(auth_page)

        cp.upload.click_sample_data()
        cp.run_calculations()
        expect(cp.loading_state).to_be_visible()
        self._release(auth_page, held)

        expect(cp.results_table.locator("tbody tr")).to_have_count(1, timeout=cp.RUN_TIMEOUT_MS)
        expect(auth_page.locator(".toast", has_text=DISCARDED_TEXT)).to_have_count(0)
        expect(auth_page.locator(".cs-stale-notice")).to_have_count(0)

    @pytest.mark.slow
    def test_simulation_run_waits_for_the_subcatchments_then_starts(
        self, auth_page: Page, live_server
    ) -> None:
        sp = _open(auth_page, live_server)
        held = self._hold_subcatchments(auth_page)

        sp.upload.click_sample_data()
        sp.run_simulation()

        expect(auth_page.locator("#cs-run-waiting")).to_contain_text("the run starts as soon")
        expect(sp.run_status).to_have_count(0)
        self._release(auth_page, held)

        sp.wait_for_results()
        expect(sp.catchment_name_select).to_have_value("S1")
        expect(auth_page.locator(".toast", has_text=DISCARDED_TEXT)).to_have_count(0)
        expect(auth_page.locator("#cs-run-waiting")).to_have_count(0)

    def test_run_for_a_model_replaced_mid_run_is_discarded(
        self, auth_page: Page, live_server, tmp_path
    ) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        held: list[Route] = []
        auth_page.route(
            SIMULATION_ROUTE,
            lambda route: (
                held.append(route) if route.request.method == "POST" else route.continue_()
            ),
        )
        sp.run_simulation()
        expect(sp.run_status).to_be_visible()

        sp.upload.upload_file(_two_subcatchment_model(tmp_path))
        sp.wait_for_catchment_options()
        held[0].fulfill(
            status=200,
            content_type="text/html",
            body='<div id="simulation-results"><h2 id="simulation-results-heading">Simulation results</h2></div>',
        )

        expect(auth_page.locator(".toast", has_text=DISCARDED_TEXT)).to_be_visible()
        expect(sp.results_heading).to_have_count(0)
        expect(sp.empty_state).to_be_visible()


class TestRunStatus:
    def test_panel_says_what_runs_and_goes_away(self, auth_page: Page, live_server) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        _wait_for_model_value(sp)
        held: list[Route] = []
        auth_page.route(
            SIMULATION_ROUTE,
            lambda route: (
                held.append(route) if route.request.method == "POST" else route.continue_()
            ),
        )

        sp.run_simulation()

        expect(sp.run_status).to_be_visible()
        expect(sp.run_status).to_have_attribute("aria-hidden", "true")
        expect(sp.run_status.locator(".cs-run-status__title")).to_have_text(
            "Running SWMM — 11 runs for S1, Slope 5 → 15 %"
        )
        expect(sp.run_status.locator(".cs-run-status__stage")).to_have_text(
            ["Copy the model", "Write the value", "Run SWMM", "Collect results"]
        )
        expect(sp.run_status.locator(".cs-run-status__elapsed")).to_have_text(
            re.compile(r"^\d+ s$")
        )
        # The existing polite status stays the one announcement.
        expect(sp.loading_spinner).to_be_visible()
        assert "%" not in sp.run_status.locator(".cs-run-status__track").inner_text()

        # The empty state's next-step line would still ask for the run.
        expect(sp.next_step).to_be_hidden()

        held[0].fulfill(status=500, json={"message": "Stubbed failure.", "field_errors": {}})

        expect(sp.run_status).to_have_count(0)
        expect(sp.error_summary).to_contain_text("Stubbed failure.")
        expect(sp.next_step).to_be_visible()

    def test_literature_run_and_reduced_motion(self, auth_page: Page, live_server) -> None:
        auth_page.emulate_media(reduced_motion="reduce")
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_option("simulate_n_imperv")
        auth_page.route(
            SIMULATION_ROUTE,
            lambda route: None if route.request.method == "POST" else route.continue_(),
        )

        sp.run_simulation()

        expect(sp.run_status).to_contain_text(
            "one run per literature value of Manning's n, impervious for S1"
        )
        bar = sp.run_status.locator(".cs-run-status__track > span")
        assert bar.evaluate("el => getComputedStyle(el).animationName") == "none"


# Rows of a #chart-config for the reading line: (x, runoff, peak, infiltration).
def _config(rows, x_label="Percent Slope [%]"):
    return {
        "x": "X",
        "xLabel": x_label,
        "y": ["runoff", "peak_runoff_rate", "infiltration"],
        "yLabels": {
            "runoff": "Total Runoff Volume [m³]",
            "peak_runoff_rate": "Peak Runoff Rate [CMS]",
            "infiltration": "Total Infiltration Volume [m³]",
        },
        "data": [
            {"X": x, "runoff": r, "peak_runoff_rate": p, "infiltration": i} for x, r, p, i in rows
        ],
    }


READINGS = [
    (
        "increase",
        _config([(10, 3360.4, 0.19, 406.7), (1, 2434.6, 0.13, 442.6)]),
        "Slope",
        "Raising Slope from 1 to 10\u00a0% increases the runoff volume by 38.0\u00a0% "
        "(2,434.6 → 3,360.4\u00a0m³) and the peak rate by 46.2\u00a0%; infiltration falls by 8.1\u00a0%.",
    ),
    (
        "decrease",
        _config([(0.011, 4507.9, 0.27, 292), (0.8, 3516.3, 0.2, 400)], "Manning n (Pervious) [-]"),
        "",
        "Raising Manning n (Pervious) from 0.011 to 0.8 decreases the runoff volume by 22.0\u00a0% "
        "(4,507.9 → 3,516.3\u00a0m³) and the peak rate by 25.9\u00a0%; infiltration rises by 37.0\u00a0%.",
    ),
    (
        "no change",
        _config([(0, 3360.4, 0.1933, 406.7), (100, 3360.4, 0.1933, 406.7)], "Curb Length [m]"),
        "Curb length",
        "Changing Curb length from 0 to 100\u00a0m does not change the runoff: every run gives "
        "3,360.4\u00a0m³ at a peak rate of 0.1933\u00a0CMS.",
    ),
    (
        "non-monotonic",
        _config([(1, 10, 1, 5), (2, 30, 2, 5), (3, 20, 3, 5)]),
        "Slope",
        "The runoff volume does not move steadily with Slope: it is highest at 2\u00a0% "
        "(30.00\u00a0m³) and lowest at 1\u00a0% (10.00\u00a0m³).",
    ),
    (
        "single run",
        _config([(10, 3360.4, 0.1933, 406.7)]),
        "Slope",
        "A single run at Slope = 10\u00a0%: the runoff volume is 3,360.4\u00a0m³ and the peak rate "
        "0.1933\u00a0CMS.",
    ),
    (
        "from zero, opposite peak, missing infiltration",
        _config([(1, 0, 0.3, None), (2, 12.5, 0.2, None)], "Area [ha]"),
        "Area",
        "Raising Area from 1 to 2\u00a0ha increases the runoff volume from 0 to 12.50\u00a0m³, "
        "while the peak rate falls by 33.3\u00a0%.",
    ),
    (
        "no values",
        _config([(1, None, None, None)]),
        "Slope",
        "SWMM returned no runoff volume for these runs.",
    ),
]


class TestReading:
    @pytest.mark.parametrize(
        ("config", "name", "expected"),
        [case[1:] for case in READINGS],
        ids=[case[0] for case in READINGS],
    )
    def test_reading_sentence(self, auth_page: Page, live_server, config, name, expected) -> None:
        _open(auth_page, live_server)
        sentence = auth_page.evaluate(
            "([config, name]) => CS.simulation.reading(config, name)", [config, name]
        )
        assert sentence == expected


class TestResults:
    @pytest.mark.slow
    def test_sample_run_explains_itself(self, auth_page: Page, live_server, console_errors) -> None:
        """Try sample data → Run at once: reading line, marker, balance and links."""
        sp = _open(auth_page, live_server)
        sp.upload.click_sample_data()
        expect(sp.catchment_name_select).to_have_value("S1")
        _wait_for_model_value(sp)

        sp.run_simulation()
        sp.wait_for_results()

        expect(sp.reading).to_have_text(
            re.compile(
                r"^Raising Slope from 5 to 15\u00a0% increases the runoff volume by [\d.]+\u00a0%"
            )
        )
        expect(sp.guide_link).to_have_text("How Slope enters the model")
        expect(sp.guide_link).to_have_attribute("href", "/how-it-works#parameter-PercSlope")
        expect(sp.metric_tiles).to_have_count(5)
        expect(sp.metric_tiles.last).to_contain_text("Model value")
        shapes = sp.chart.evaluate("el => el.layout.shapes.map(s => [s.x0, s.line.dash])")
        assert shapes == [[10, "dash"]] * 4
        labels = sp.chart.evaluate("el => el.layout.annotations.map(a => a.text)")
        assert labels == ["Model value"]

        sp.balance_chart.locator(".main-svg").first.wait_for(state="attached")
        names = sp.balance_chart.evaluate("el => el.data.map(t => t.name)")
        assert names == ["Runoff", "Infiltration", "Evaporation", "Still on the surface"]
        totals = sp.balance_chart.evaluate(
            "el => el.data[0].y.map((_, i) => el.data.reduce((s, t) => s + t.y[i], 0))"
        )
        assert all(abs(total - 100) < 0.01 for total in totals)
        sp.balance_mode("volume").click()
        expect(sp.balance_chart).to_have_attribute("aria-label", re.compile("as volumes in m³"))
        title = sp.balance_chart.evaluate("el => el.layout.yaxis.title.text")
        assert title == "Volume [m³]"

        href = sp.lab_link.get_attribute("href")
        assert href.startswith("/how-it-works?")
        assert href.endswith("#lab")
        assert "vary=PercSlope" in href
        assert "slope=10" in href
        assert console_errors == []

    @pytest.mark.slow
    def test_curb_length_reads_as_no_change_and_restored_range_is_kept(
        self, auth_page: Page, live_server
    ) -> None:
        sp = _open(auth_page, live_server)
        sp.load_sample_and_select_catchment()
        sp.set_option("simulate_curb_length")
        sp.set_range(0, 20, 10)
        sp.run_simulation()
        sp.wait_for_results()

        expect(sp.reading).to_contain_text("does not change the runoff")

        # The restored form keeps the user's range instead of the suggestion.
        sp.navigate_to()
        sp.wait_for_catchment_options()
        _wait_for_model_value(sp)
        assert [float(value) for value in sp.range_values()] == [0, 20, 10]

    @pytest.mark.slow
    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_phone_width_has_no_overflow_and_no_serious_a11y_issues(
        self, auth_page: Page, live_server, theme: str
    ) -> None:
        auth_page.set_viewport_size({"width": 390, "height": 844})
        auth_page.emulate_media(reduced_motion="reduce")
        sp = _open(auth_page, live_server)
        NavComponent(auth_page).set_theme(theme)
        sp.load_sample_and_select_catchment()
        _wait_for_model_value(sp)
        sp.set_range(5, 7, 1)
        sp.run_simulation()
        sp.wait_for_results()
        sp.balance_chart.locator(".main-svg").first.wait_for(state="attached")

        overflow = auth_page.evaluate(
            "document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0
        report = sp.run_axe_audit()
        assert not report.critical + report.serious, report.critical + report.serious
