"""Timeseries page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage
from .upload_component import UploadComponent


class TimeseriesPage(BasePage):
    """POM for the timeseries analysis page (``/timeseries``)."""

    PATH = "/timeseries"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload = UploadComponent(self.page)

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form fields (public — tests assert on these)
    # ------------------------------------------------------------------

    @property
    def mode_select(self) -> Locator:
        return self.page.locator("#id_mode")

    @property
    def feature_select(self) -> Locator:
        return self.page.locator("#id_feature")

    @property
    def start_input(self) -> Locator:
        return self.page.locator("#id_start")

    @property
    def stop_input(self) -> Locator:
        return self.page.locator("#id_stop")

    @property
    def step_input(self) -> Locator:
        return self.page.locator("#id_step")

    @property
    def catchment_name_select(self) -> Locator:
        return self.page.locator("#id_catchment_name")

    @property
    def run_button(self) -> Locator:
        return self.page.locator("#run-timeseries-button")

    @property
    def loading_spinner(self) -> Locator:
        return self.page.locator("#timeseries-loading-state")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def set_mode(self, value: str) -> None:
        """Set analysis mode: 'single' or 'sweep'."""
        self.mode_select.select_option(value)

    def set_feature(self, value: str) -> None:
        self.feature_select.select_option(value)

    def set_start(self, value: float | str) -> None:
        self.start_input.fill(str(value))

    def set_stop(self, value: float | str) -> None:
        self.stop_input.fill(str(value))

    def set_step(self, value: float | str) -> None:
        self.step_input.fill(str(value))

    def set_catchment_name(self, value: str) -> None:
        self.catchment_name_select.select_option(value)

    def run_analysis(self) -> None:
        """Click 'Run Timeseries Analysis'."""
        self.run_button.click()

    def wait_for_catchment_options(self) -> None:
        """Wait until the catchment dropdown has real options."""
        self.page.wait_for_function(
            "document.querySelectorAll('#id_catchment_name option').length > 1",
            timeout=10_000,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.mode_select.is_visible() and self.run_button.is_visible()

    def is_loading(self) -> bool:
        return self.loading_spinner.is_visible()

    def has_chart(self) -> bool:
        return self.page.locator("#timeseries-chart").is_visible()

    def get_time_to_peak(self) -> str | None:
        card = self.page.locator(".metric-card", has_text="Time to Peak")
        if card.count() > 0:
            return card.locator(".card-title").inner_text()
        return None

    def get_runoff_volume(self) -> str | None:
        card = self.page.locator(".metric-card", has_text="Runoff Volume")
        if card.count() > 0:
            return card.locator(".card-title").inner_text()
        return None

    def has_download_results_button(self) -> bool:
        return self.page.get_by_role("button", name="Download Results").is_visible()

    def has_csv_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Export timeseries to CSV").is_visible()

    def has_png_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Download chart as PNG").is_visible()

    def get_catchment_options(self) -> list[str]:
        options = self.catchment_name_select.locator("option")
        return [options.nth(i).get_attribute("value") or "" for i in range(options.count())]
