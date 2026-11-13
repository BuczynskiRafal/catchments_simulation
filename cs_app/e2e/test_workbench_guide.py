"""The workbench leads from model to the first run: step states, the run checklist,
the parameter card and the suggested range.

SWMM runs are marked ``slow``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from playwright.sync_api import Page, Route, expect

from .pages.calculations_page import CalculationsPage
from .pages.simulation_page import SimulationPage
from .pages.timeseries_page import TimeseriesPage

pytestmark = pytest.mark.e2e

SAMPLE_INP = Path(__file__).resolve().parent.parent / "data" / "example.inp"
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
        self._release(auth_page, held)

        sp.wait_for_results()
        expect(sp.catchment_name_select).to_have_value("S1")
        expect(auth_page.locator(".toast", has_text=DISCARDED_TEXT)).to_have_count(0)
        expect(auth_page.locator("#cs-run-waiting")).to_have_count(0)
