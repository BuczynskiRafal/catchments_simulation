"""Home page tests: hero hydrograph, tool cards, documentation and example charts."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.home_page import HomePage
from .pages.nav_component import NavComponent

pytestmark = pytest.mark.e2e

HERO_HEADING = "See how a subcatchment turns rain into runoff."
PHONE = {"width": 360, "height": 780}


@pytest.fixture()
def home(page: Page, live_server) -> HomePage:
    hp = HomePage(page, live_server.url)
    hp.navigate_to()
    return hp


class TestHomeContent:
    def test_heading_visible(self, home: HomePage) -> None:
        assert home.get_main_heading() == HERO_HEADING

    def test_page_title(self, home: HomePage) -> None:
        assert "Catchment Simulation" in home.get_title()

    def test_plotly_chart_containers_present(self, home: HomePage) -> None:
        for chart_id in ("plot-hydrograph", "plot-slope", "plot-area", "plot-width"):
            expect(home.page.locator(f"#{chart_id}")).to_have_count(1)

    def test_code_examples_present(self, home: HomePage) -> None:
        expect(home.get_code_examples()).to_have_count(7)

    def test_chart_data_script_tag_exists(self, home: HomePage) -> None:
        assert home.has_chart_data_script()

    def test_section_headings_are_present(self, home: HomePage) -> None:
        expect(home.page.locator("h3", has_text="Simulation methods")).to_be_visible()
        expect(home.page.locator("h3", has_text="Analysis functions")).to_be_visible()

    @pytest.mark.parametrize(
        ("name", "path"),
        [
            ("Parameter sweep", "/simulation"),
            ("Hydrograph", "/timeseries"),
            ("SWMM vs. neural network", "/calculations"),
        ],
    )
    def test_tool_cards_link_to_tools(self, home: HomePage, name: str, path: str) -> None:
        expect(home.tool_link(name)).to_have_attribute("href", path)


class TestHeroChart:
    def test_hydrograph_draws_rainfall_and_runoff(self, home: HomePage) -> None:
        home.wait_for_chart(home.hydrograph)

        traces = home.page.evaluate(
            "() => document.getElementById('plot-hydrograph').data.map(t => t.name)"
        )
        assert "Rainfall Intensity [mm/h]" in traces
        assert "Runoff Rate [CMS]" in traces
        expect(home.hydrograph).to_have_attribute("role", "img")
        expect(home.hydrograph).to_have_attribute("aria-describedby", "hero-metrics")

    def test_metrics_state_the_numbers_with_units(self, home: HomePage) -> None:
        metrics = home.hero_metrics
        expect(metrics).to_have_count(4)
        expect(metrics.nth(0)).to_contain_text("Peak rainfall")
        expect(metrics.nth(0)).to_contain_text("mm/h")
        expect(metrics.nth(1)).to_contain_text("CMS")
        expect(metrics.nth(2)).to_contain_text(re.compile(r"\d+\s*h\s*\d+\s*min"))
        expect(metrics.nth(3)).to_contain_text("m³")

    def test_theme_switch_redraws_chart_in_place(self, page: Page, live_server) -> None:
        page.emulate_media(color_scheme="light")
        home = HomePage(page, live_server.url)
        home.navigate_to()
        home.wait_for_chart(home.hydrograph)
        font_color = "() => document.getElementById('plot-hydrograph').layout.font.color"
        light_ink = page.evaluate(font_color)

        NavComponent(page).set_theme("dark")

        expect(page.locator("html")).to_have_attribute("data-bs-theme", "dark")
        page.wait_for_function(f"({font_color})() !== {light_ink!r}")
        expect(home.hydrograph.locator(".main-svg").first).to_be_attached()
        expect(home.hydrograph.locator(".cs-empty-state")).to_have_count(0)


class TestExampleCharts:
    @pytest.mark.parametrize(
        ("key", "x_label"),
        [("slope", "Percent Slope [%]"), ("area", "Area [ha]"), ("width", "Width [m]")],
    )
    def test_sweep_chart_uses_server_axis_labels(
        self, home: HomePage, key: str, x_label: str
    ) -> None:
        chart = home.page.locator(f"#plot-{key}")
        home.wait_for_chart(chart)

        expect(chart.locator(".xtitle")).to_have_text(x_label)
        expect(chart.locator(".ytitle")).to_have_text("Total Runoff Volume [m³]")

    def test_each_chart_has_a_numeric_summary(self, home: HomePage) -> None:
        summary = home.page.locator("#plot-slope-summary")
        expect(home.page.locator("#plot-slope")).to_have_attribute(
            "aria-describedby", "plot-slope-summary"
        )
        expect(summary).to_contain_text("Percent Slope [%]")
        expect(summary).to_contain_text("1 → 100")


class TestCodeCopy:
    def test_copy_button_copies_snippet(self, page: Page, live_server) -> None:
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        home = HomePage(page, live_server.url)
        home.navigate_to()

        home.copy_button("example-slope").click()

        expect(page.locator(".toast-container .cs-toast--success")).to_contain_text("copied")
        copied = page.evaluate("navigator.clipboard.readText()")
        assert copied.startswith("from catchment_simulation import FeaturesSimulation")
        assert "simulate_percent_slope(start=1, stop=100, step=1)" in copied

    def test_copy_buttons_have_distinct_names(self, home: HomePage) -> None:
        expect(
            home.page.get_by_role("button", name="Copy the install command", exact=True)
        ).to_be_visible()
        expect(
            home.page.get_by_role("button", name="Copy the width example", exact=True)
        ).to_be_visible()


class TestTableOfContents:
    def test_scrollspy_marks_the_visible_section(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 1366, "height": 900})
        home = HomePage(page, live_server.url)
        home.navigate_to()
        page.wait_for_load_state("load")

        page.locator("#analysis-functions").scroll_into_view_if_needed()
        page.mouse.wheel(0, 200)

        expect(home.toc_link("Analysis functions")).to_have_class(re.compile(r"\bactive\b"))
        expect(home.toc_link("The Python package")).not_to_have_class(re.compile(r"\bactive\b"))

    def test_toc_link_jumps_to_section(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 1366, "height": 900})
        home = HomePage(page, live_server.url)
        home.navigate_to()

        home.toc_link("Width sweep").click()

        expect(page).to_have_url(re.compile(r"#example-width$"))
        expect(page.locator("#example-width-title")).to_be_in_viewport()

    def test_toc_is_hidden_on_phones(self, page: Page, live_server) -> None:
        page.set_viewport_size(PHONE)
        home = HomePage(page, live_server.url)
        home.navigate_to()
        expect(page.locator("#docs-toc")).to_be_hidden()


class TestHomeLayoutAndA11y:
    def test_no_horizontal_overflow_at_360px(self, page: Page, live_server) -> None:
        page.set_viewport_size(PHONE)
        home = HomePage(page, live_server.url)
        home.navigate_to()
        home.wait_for_chart(home.hydrograph)
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        ), "home page scrolls horizontally at 360px"

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(self, page: Page, live_server, scheme: str) -> None:
        page.emulate_media(color_scheme=scheme)
        home = HomePage(page, live_server.url)
        home.navigate_to()
        home.wait_for_chart(home.hydrograph)

        report = home.run_axe_audit()

        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]
