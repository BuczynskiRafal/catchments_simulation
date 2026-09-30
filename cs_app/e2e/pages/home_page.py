"""Home / main_view page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class HomePage(BasePage):
    """POM for the home page (``/``): hero chart, tool cards and documentation."""

    PATH = "/"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Hero
    # ------------------------------------------------------------------

    def get_main_heading(self) -> str | None:
        return self.get_heading(level=1)

    @property
    def hydrograph(self) -> Locator:
        return self.page.locator("#plot-hydrograph")

    @property
    def hero_metrics(self) -> Locator:
        return self.page.locator("#hero-metrics .cs-metric")

    def wait_for_chart(self, chart: Locator) -> None:
        """Wait until Plotly has drawn into *chart*."""
        chart.locator(".main-svg").first.wait_for(state="attached")

    # ------------------------------------------------------------------
    # Tools and documentation
    # ------------------------------------------------------------------

    def tool_link(self, name: str) -> Locator:
        return self.page.locator(".home-tools").get_by_role("link", name=name, exact=True)

    def toc_link(self, name: str) -> Locator:
        return self.page.locator("#docs-toc").get_by_role("link", name=name, exact=True)

    def get_plotly_charts(self) -> Locator:
        """Return locator for all Plotly chart containers."""
        return self.page.locator("[id^='plot-']")

    def get_code_examples(self) -> Locator:
        """Return locator for code snippet blocks."""
        return self.page.locator("pre.code-snippet")

    def copy_button(self, example_id: str) -> Locator:
        """Copy button of the snippet inside the ``#<example_id>`` section."""
        return self.page.locator(f"#{example_id} .cs-code__copy")

    def has_chart_data_script(self) -> bool:
        """Check whether the embedded chart-data JSON script tag exists."""
        return self.page.locator("#chart-data").count() > 0
