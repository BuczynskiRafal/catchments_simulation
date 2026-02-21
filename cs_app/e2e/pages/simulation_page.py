"""Simulation page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage
from .upload_component import UploadComponent


class SimulationPage(BasePage):
    """POM for the simulation page (``/simulation``)."""

    PATH = "/simulation"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload = UploadComponent(self.page)

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form fields (public — tests assert on these)
    # ------------------------------------------------------------------

    @property
    def option_select(self) -> Locator:
        return self.page.locator("#id_option")

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
        return self.page.locator("#run-simulation-button")

    @property
    def loading_spinner(self) -> Locator:
        return self.page.locator("#simulation-loading-state")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def set_option(self, value: str) -> None:
        self.option_select.select_option(value)

    def set_start(self, value: int | str) -> None:
        self.start_input.fill(str(value))

    def set_stop(self, value: int | str) -> None:
        self.stop_input.fill(str(value))

    def set_step(self, value: int | str) -> None:
        self.step_input.fill(str(value))

    def set_catchment_name(self, value: str) -> None:
        self.catchment_name_select.select_option(value)

    def run_simulation(self) -> None:
        self.run_button.click()

    def wait_for_catchment_options(self) -> None:
        """Wait until the catchment dropdown has real options (not just placeholder)."""
        self.page.wait_for_function(
            "document.querySelectorAll('#id_catchment_name option').length > 1",
            timeout=10_000,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.option_select.is_visible() and self.run_button.is_visible()

    def is_loading(self) -> bool:
        return self.loading_spinner.is_visible()

    def has_results_table(self) -> bool:
        return self.page.locator("table.table-bordered").is_visible()

    def get_results_row_count(self) -> int:
        return self.page.locator("table.table-bordered tbody tr").count()

    def has_chart(self) -> bool:
        return self.page.locator("#simulation-chart").is_visible()

    def has_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Download Results").is_visible()

    def get_catchment_options(self) -> list[str]:
        """Return all option values currently available in the catchment select."""
        options = self.catchment_name_select.locator("option")
        return [options.nth(i).get_attribute("value") or "" for i in range(options.count())]

    def is_run_button_enabled(self) -> bool:
        return self.run_button.is_enabled()
