"""Home / main_view page object."""

from __future__ import annotations

from .base_page import BasePage


class HomePage(BasePage):
    """POM for the home page (``/``)."""

    PATH = "/"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def get_main_heading(self) -> str | None:
        return self.get_heading(level=2)

    def get_plotly_charts(self):
        """Return locator for all Plotly chart containers."""
        return self.page.locator("[id^='plot-']")

    def get_code_examples(self):
        """Return locator for code snippet blocks."""
        return self.page.locator("pre.code-snippet")

    def has_chart_data_script(self) -> bool:
        """Check whether the embedded chart-data JSON script tag exists."""
        return self.page.locator("#chart-data").count() > 0
