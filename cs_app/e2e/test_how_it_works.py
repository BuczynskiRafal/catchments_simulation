"""How it works page: rendered equations, parameter guide and the interactive model."""

from __future__ import annotations

import math
import re

import pytest
from playwright.sync_api import Browser, Page, expect

from .pages.how_it_works_page import HowItWorksPage

pytestmark = pytest.mark.e2e

PHONE = {"width": 390, "height": 844}
# Running animations and transitions inside the interactive model that last long enough to see
# (under reduced motion app.css turns every change into a 0.01 ms transition).
LAB_ANIMATIONS = """() => document.getAnimations().filter((a) =>
    a.playState === "running" && a.effect.getTiming().duration > 1 &&
    document.getElementById("hiw-lab").contains(a.effect.target)
).length"""


def _number(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-"))


@pytest.fixture
def hiw(page: Page, live_server) -> HowItWorksPage:
    hp = HowItWorksPage(page, live_server.url)
    hp.navigate_to()
    return hp


class TestContent:
    def test_heading(self, hiw: HowItWorksPage) -> None:
        assert hiw.get_heading(level=1) == "How the simulation works"

    def test_equations_are_typeset_and_numbered_in_order(self, hiw: HowItWorksPage) -> None:
        hiw.page.wait_for_function(
            "document.querySelectorAll('.hiw-equation .katex-display').length === 18"
        )
        assert hiw.rendered_equation_numbers() == [f"({n})" for n in range(1, 19)]
        # Inline symbols are typeset too; no TeX source is left showing.
        assert hiw.page.locator("[data-cs-math]:not(.is-rendered)").count() == 0

    def test_try_it_selects_the_parameter_and_focuses_its_slider(self, hiw: HowItWorksPage) -> None:
        hiw.try_button("Width").click()
        expect(hiw.sweep_parameter).to_have_value("Width")
        expect(hiw.slider("width")).to_be_focused()
        expect(hiw.page).to_have_url(re.compile(r"vary=Width"))

    def test_parameter_without_a_model_key_has_no_button(self, hiw: HowItWorksPage) -> None:
        expect(hiw.try_button("CurbLength")).to_have_count(0)
        expect(hiw.try_button("PercSlope")).to_be_visible()


class TestInteractiveModel:
    def test_initial_numbers_come_from_the_solver(self, hiw: HowItWorksPage) -> None:
        solved = hiw.solve()
        assert _number(hiw.metric_value("peak")) == pytest.approx(
            solved["peak"]["runoff"], rel=5e-3
        )
        assert _number(hiw.metric_value("volume")) == pytest.approx(
            solved["totals"]["runoff"], abs=1
        )
        coefficient = solved["totals"]["runoff"] / solved["totals"]["rainfall"] * 100
        assert _number(hiw.metric_value("coefficient")) == pytest.approx(coefficient, abs=0.05)
        assert hiw.metric_value("timeToPeak") == "9 h"
        # The tools' 5-minute step gives the engine's peak of the example model.
        assert hiw.solve(time_step=300)["peak"]["runoff"] == pytest.approx(0.193, abs=5e-4)
        expect(hiw.metric_delta("peak")).to_be_hidden()
        assert hiw.path("hiw-chart__ghost") == ""

    def test_slider_changes_numbers_ghost_and_deltas(self, hiw: HowItWorksPage) -> None:
        before = hiw.metric_value("peak")
        hiw.nudge("imperviousness", 25)
        expect(hiw.slider_readout("imperviousness")).to_have_text("50\u00a0%")
        expect(hiw.page.locator("[data-hiw-metric='peak'] [data-hiw-value]")).not_to_have_text(
            before
        )
        solved = hiw.solve("{ imperviousness: 50 }")
        assert _number(hiw.metric_value("volume")) == pytest.approx(
            solved["totals"]["runoff"], abs=1
        )
        expect(hiw.metric_delta("peak")).to_contain_text(re.compile(r"^\+[\d.]+\s%\svs.\sstart$"))
        assert hiw.path("hiw-chart__ghost").startswith("M")
        expect(hiw.page.locator("[data-hiw-legend='ghost']")).to_be_visible()

    def test_unchanged_number_says_no_change(self, hiw: HowItWorksPage) -> None:
        hiw.nudge("width", 1)
        # A slightly wider surface moves the volume but not the time to peak (whole minutes).
        expect(hiw.metric_delta("volume")).to_contain_text("% vs.\u00a0start")
        expect(hiw.metric_delta("timeToPeak")).to_have_text("no change vs.\u00a0start")

    def test_reset_restores_the_starting_values(self, hiw: HowItWorksPage) -> None:
        start = hiw.metric_value("volume")
        hiw.nudge("slope", 30)
        expect(hiw.page.locator("[data-hiw-metric='volume'] [data-hiw-value]")).not_to_have_text(
            start
        )
        hiw.reset_button.click()
        expect(hiw.slider_readout("slope")).to_have_text("10\u00a0%")
        expect(hiw.page.locator("[data-hiw-metric='volume'] [data-hiw-value]")).to_have_text(start)
        expect(hiw.metric_delta("volume")).to_be_hidden()
        assert hiw.path("hiw-chart__ghost") == ""

    def test_storm_switch_redraws_the_rainfall(self, hiw: HowItWorksPage) -> None:
        bars = hiw.path("hiw-chart__rain")
        hiw.choose_storm("cloudburst")
        expect(hiw.page.locator("#lab-storm-description")).to_contain_text("45 minutes")
        assert hiw.path("hiw-chart__rain") != bars
        solved = hiw.solve(storm="cloudburst")
        assert _number(hiw.metric_value("volume")) == pytest.approx(
            solved["totals"]["runoff"], abs=1
        )

    def test_play_advances_time_and_pause_stops_it(self, hiw: HowItWorksPage) -> None:
        # The label says what the button does next; it is not a toggle (no aria-pressed).
        expect(hiw.page.get_by_role("button", name="Play", exact=True)).to_be_visible()
        hiw.play_button.click()
        expect(hiw.play_button).to_have_accessible_name("Pause")
        expect(hiw.play_button).not_to_have_attribute("aria-pressed", re.compile(".*"))
        hiw.page.wait_for_timeout(1500)
        hiw.play_button.click()
        expect(hiw.play_button).to_have_accessible_name("Play")
        paused_at = hiw.time_value.inner_text()
        # Play starts the storm over: about 1.5 h of the 12 h storm have passed.
        hours, minutes = map(int, paused_at.split(":"))
        assert 0 < hours * 60 + minutes < 9 * 60
        hiw.page.wait_for_timeout(600)
        expect(hiw.time_value).to_have_text(paused_at)

    def test_scrubber_moves_the_cursor(self, hiw: HowItWorksPage) -> None:
        cursor = hiw.path("hiw-chart__cursor")
        hiw.scrubber.focus()
        hiw.page.keyboard.press("Home")
        expect(hiw.time_value).to_have_text("0:00")
        assert hiw.path("hiw-chart__cursor") != cursor
        hiw.page.keyboard.press("End")
        expect(hiw.time_value).to_have_text("12:00")

    def test_chart_is_an_image_that_pointer_drags_move(self, hiw: HowItWorksPage) -> None:
        # An image, not a control: the time slider gives the keyboard the same access.
        expect(hiw.hydrograph).to_have_attribute("role", "img")
        expect(hiw.hydrograph).not_to_have_attribute("tabindex", re.compile(".*"))
        expect(hiw.hydrograph).to_have_attribute("aria-label", re.compile(r"Peak runoff 0\.193"))
        hiw.scroll_into_view(hiw.hydrograph)
        box = hiw.hydrograph.bounding_box()
        # The plot spans the chart between its axis margins; its left edge is the start of the storm.
        hiw.page.mouse.move(box["x"] + 56, box["y"] + box["height"] / 2)
        hiw.page.mouse.down()
        expect(hiw.time_value).to_have_text(re.compile(r"^0:\d\d$"))
        hiw.page.mouse.move(box["x"] + box["width"] - 50, box["y"] + box["height"] / 2)
        hiw.page.mouse.up()
        expect(hiw.time_value).to_have_text(re.compile(r"^1[12]:\d\d$"))

    def test_sweep_draws_one_line_per_value_and_the_response(self, hiw: HowItWorksPage) -> None:
        hiw.sweep_parameter.select_option("PercSlope")
        hiw.sweep_button.click()
        expect(hiw.sweep_status).to_contain_text("This is a parameter sweep", timeout=10_000)
        assert hiw.drawn_sweep_lines() == 9
        expect(hiw.response_points()).to_have_count(9)
        expect(hiw.page.locator("#lab-response-table tbody tr")).to_have_count(9)
        expect(hiw.sweep_status.get_by_role("link", name="Simulation tool")).to_have_attribute(
            "href", "/simulation"
        )
        # Any slider change clears the sweep.
        hiw.nudge("area", 1)
        expect(hiw.response_points()).to_have_count(0)
        assert hiw.drawn_sweep_lines() == 0
        expect(hiw.sweep_button).to_have_text("Run sweep")

    def test_query_string_sets_starting_values_and_vary(self, page: Page, live_server) -> None:
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to("?area=20&slope=50&imperviousness=abc&vary=PercSlope#lab")
        expect(hp.slider_readout("area")).to_have_text("20\u00a0ha")
        # A valid value outside the slider widens it instead of being clamped.
        expect(hp.slider_readout("slope")).to_have_text("50\u00a0%")
        expect(hp.slider_max("slope")).to_have_text("50\u00a0%")
        expect(hp.slider_readout("imperviousness")).to_have_text("25\u00a0%")
        expect(hp.sweep_parameter).to_have_value("PercSlope")
        solved = hp.solve("{ area: 20, slope: 50 }")
        assert _number(hp.metric_value("volume")) == pytest.approx(
            solved["totals"]["runoff"], abs=1
        )
        expect(hp.metric_delta("volume")).to_be_hidden()


class TestRobustness:
    """Starting values come from the URL: anything there must leave a working lab."""

    @pytest.mark.parametrize(
        "query",
        [
            "?width=1e308&area=1e-12&nImperv=1e-12&vary=a'b",
            "?area=abc&slope=-5&imperviousness=250",
            "?area=10000&width=100000&slope=100&nPerv=5",
        ],
        ids=["extreme", "malformed", "largest"],
    )
    def test_any_query_string_leaves_a_working_lab(
        self, page: Page, live_server, query: str
    ) -> None:
        messages: list[str] = []
        page.on(
            "console", lambda m: messages.append(m.text) if m.type in ("error", "warning") else None
        )
        page.on("pageerror", lambda e: messages.append(str(e)))
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to(query + "#lab")
        expect(page.locator("#hiw-lab")).to_be_visible()
        for name in ("peak", "volume", "coefficient"):
            assert math.isfinite(_number(hp.metric_value(name))), name
        # Readouts stay short however large or small the value.
        for key in ("area", "width", "slope", "nImperv", "nPerv"):
            assert len(hp.slider_readout(key).inner_text()) <= 12, key
        # Still responsive: a slider change updates the numbers within a second (the chart's
        # label carries them at three significant digits, which a tiny area still moves).
        before = hp.hydrograph.get_attribute("aria-label")
        hp.nudge("imperviousness", 10, "ArrowLeft")
        expect(hp.hydrograph).not_to_have_attribute("aria-label", before, timeout=1000)
        # The rest of the page still started: equations typeset, try buttons shown.
        page.wait_for_function(
            "document.querySelectorAll('.hiw-equation .katex-display').length === 18"
        )
        expect(hp.try_button("Width")).to_be_visible()
        assert messages == []

    def test_values_are_clamped_to_physical_bounds(self, page: Page, live_server) -> None:
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to("?width=1e308&area=1e-12&slope=-5&imperviousness=250&vary=a'b")
        expect(hp.slider_readout("width")).to_have_text("100000\u00a0m")
        expect(hp.slider_readout("area")).to_have_text("0.001\u00a0ha")
        expect(hp.slider_readout("slope")).to_have_text("0.001\u00a0%")
        expect(hp.slider_readout("imperviousness")).to_have_text("100\u00a0%")
        # An unknown vary keeps the default parameter.
        expect(hp.sweep_parameter).to_have_value("PercImperv")

    def test_a_lab_that_cannot_start_leaves_a_note(self, page: Page, live_server) -> None:
        # Without the solver the lab cannot run; the page says so and the rest still works.
        page.route(re.compile(r"runoff-model\.js"), lambda route: route.abort())
        page.goto(f"{live_server.url}{HowItWorksPage.PATH}", wait_until="domcontentloaded")
        expect(page.get_by_text("The interactive model could not be started")).to_be_visible()
        expect(page.locator("#hiw-lab")).to_be_hidden()
        expect(page.locator("[data-hiw-try='Width']")).to_be_hidden()
        page.wait_for_function(
            "document.querySelectorAll('.hiw-equation .katex-display').length === 18"
        )


class TestLayout:
    """The sliders stay next to what they move: sticky beside the stage, or a docked readout."""

    def test_controls_stick_beside_the_stage(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 1440, "height": 900})
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        for target in (hp.hydrograph_card, hp.page.locator(".hiw-balance")):
            hp.scroll_into_view(target)
            assert hp.in_viewport(hp.controls_column)
            assert hp.in_viewport(hp.slider("conductivity"))
        # The divider runs the full height of the lab.
        heights = page.evaluate(
            """() => [document.getElementById('lab-controls').offsetHeight,
                      document.getElementById('hiw-lab').clientHeight]"""
        )
        assert heights[0] == heights[1]

    def test_short_viewport_scrolls_the_controls_inside(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 1440, "height": 700})
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        hp.scroll_into_view(hp.hydrograph_card)
        column = hp.controls_column
        assert column.evaluate("el => el.scrollHeight > el.clientHeight")
        assert hp.in_viewport(column)
        hp.slider("conductivity").focus()
        page.keyboard.press("ArrowRight")
        assert hp.in_viewport(hp.slider("conductivity"))
        expect(hp.dock).to_be_hidden()

    def test_phone_docks_the_numbers_while_sliding(self, page: Page, live_server) -> None:
        page.set_viewport_size(PHONE)
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        expect(hp.dock).not_to_have_class(re.compile(r"\bis-shown\b"))
        hp.nudge("storagePerv", 20)
        hp.settle_scroll()
        expect(hp.dock).to_have_class(re.compile(r"\bis-shown\b"))
        assert hp.dock_value("peak") == hp.metric_value("peak")
        assert hp.dock_value("volume") == hp.metric_value("volume")
        assert (
            hp.dock_value("volumeChange") == hp.metric_delta("volume").inner_text().split(" vs.")[0]
        )
        # The focused slider is not under the dock.
        slider_bottom = hp.slider("storagePerv").evaluate("el => el.getBoundingClientRect().bottom")
        dock_top = hp.dock.evaluate("el => el.getBoundingClientRect().top")
        assert slider_bottom <= dock_top
        hp.scroll_into_view(hp.hydrograph_card)
        expect(hp.dock).not_to_have_class(re.compile(r"\bis-shown\b"))
        hp.scroll_into_view(hp.page.locator("#references"), "start")
        expect(hp.dock).not_to_have_class(re.compile(r"\bis-shown\b"))


class TestQuality:
    def test_no_console_errors(self, page: Page, live_server) -> None:
        messages: list[str] = []
        page.on(
            "console", lambda m: messages.append(m.text) if m.type in ("error", "warning") else None
        )
        page.on("pageerror", lambda e: messages.append(str(e)))
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        hp.nudge("width", 5)
        hp.sweep_button.click()
        expect(hp.sweep_status).to_contain_text("This is a parameter sweep", timeout=10_000)
        assert messages == []

    def test_reduced_motion_does_not_animate(self, browser: Browser, live_server) -> None:
        context = browser.new_context(reduced_motion="reduce")
        try:
            page = context.new_page()
            hp = HowItWorksPage(page, live_server.url)
            hp.navigate_to()
            time = hp.time_value.inner_text()
            page.wait_for_timeout(500)
            expect(hp.time_value).to_have_text(time)
            assert page.evaluate(LAB_ANIMATIONS) == 0
            # The sweep is drawn at once.
            hp.sweep_button.click()
            assert hp.drawn_sweep_lines() == 9
            expect(hp.sweep_status).to_contain_text("This is a parameter sweep")
            # Playing is the reader's choice; the rain then stands still.
            hp.play_button.click()
            expect(hp.play_button).to_have_accessible_name("Pause")
            page.wait_for_timeout(300)
            assert page.evaluate(LAB_ANIMATIONS) == 0
        finally:
            context.close()

    def test_no_horizontal_overflow_on_a_phone(self, page: Page, live_server) -> None:
        page.set_viewport_size(PHONE)
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0, f"page scrolls sideways by {overflow}px"

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(self, page: Page, live_server, scheme: str) -> None:
        page.emulate_media(color_scheme=scheme)
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        page.wait_for_function("document.querySelectorAll('.katex').length > 0")
        report = hp.run_axe_audit()
        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]
