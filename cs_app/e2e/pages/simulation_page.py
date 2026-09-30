"""Simulation page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage
from .upload_component import UploadComponent

WINDOW_MARKER = "__simulationPageMarker"


class SimulationPage(BasePage):
    """POM for the simulation page (``/simulation``)."""

    PATH = "/simulation"
    RUN_TIMEOUT_MS = 120_000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload = UploadComponent(self.page)

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form fields (public — tests assert on these)
    # ------------------------------------------------------------------

    @property
    def form(self) -> Locator:
        return self.page.locator("#simulation-form")

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
    def range_fields(self) -> Locator:
        return self.page.locator("#simulation-range")

    @property
    def range_hint(self) -> Locator:
        return self.page.locator("#simulation-range-hint")

    @property
    def literature_note(self) -> Locator:
        return self.page.locator("#simulation-literature-note")

    @property
    def model_hint(self) -> Locator:
        return self.page.locator("#simulation-model-hint")

    @property
    def run_button(self) -> Locator:
        return self.page.locator("#run-simulation-button")

    @property
    def loading_spinner(self) -> Locator:
        return self.page.locator("#simulation-loading-state")

    @property
    def elapsed_counter(self) -> Locator:
        return self.loading_spinner.locator("[data-cs-elapsed]")

    @property
    def error_summary(self) -> Locator:
        return self.form.locator(".alert-danger[role='alert']")

    # ------------------------------------------------------------------
    # Results canvas
    # ------------------------------------------------------------------

    @property
    def canvas(self) -> Locator:
        return self.page.locator("#simulation-canvas")

    @property
    def empty_state(self) -> Locator:
        return self.canvas.locator(".cs-empty-state")

    @property
    def results_heading(self) -> Locator:
        return self.page.locator("#simulation-results-heading")

    @property
    def metric_tiles(self) -> Locator:
        return self.page.locator("#simulation-metrics .cs-metric")

    @property
    def chart(self) -> Locator:
        return self.page.locator("#simulation-chart")

    @property
    def results_table(self) -> Locator:
        return self.page.locator("#simulation-table")

    @property
    def results_rows(self) -> Locator:
        return self.results_table.locator("tbody tr")

    @property
    def download_button(self) -> Locator:
        return self.page.locator("#download-simulation-results-button")

    @property
    def download_png_button(self) -> Locator:
        return self.page.locator("[data-sim-action='download-png']")

    @property
    def copy_table_button(self) -> Locator:
        return self.page.locator("[data-sim-action='copy-table']")

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

    def set_range(self, start: int | str, stop: int | str, step: int | str) -> None:
        self.set_start(start)
        self.set_stop(stop)
        self.set_step(step)

    def set_catchment_name(self, value: str) -> None:
        self.catchment_name_select.select_option(value)

    def run_simulation(self) -> None:
        self.run_button.click()

    def wait_for_results(self) -> None:
        """Wait until an async run has swapped in the results fragment and drawn the chart."""
        self.results_heading.wait_for(timeout=self.RUN_TIMEOUT_MS)
        self.chart.locator(".main-svg").first.wait_for(state="attached")

    def wait_for_catchment_options(self) -> None:
        """Wait until the catchment dropdown has real options (not just placeholder)."""
        self.page.wait_for_function(
            "document.querySelectorAll('#id_catchment_name option').length > 1",
            timeout=10_000,
        )

    def load_sample_and_select_catchment(self) -> str:
        """Load the bundled sample model and select its first subcatchment; return its id."""
        self.upload.click_sample_data()
        self.wait_for_catchment_options()
        catchment = next(option for option in self.get_catchment_options() if option)
        self.set_catchment_name(catchment)
        return catchment

    def mark_window(self) -> None:
        """Tag the current document; the tag survives only if no full page load happens."""
        self.page.evaluate(f"window.{WINDOW_MARKER} = true")

    def window_is_marked(self) -> bool:
        return self.page.evaluate(f"window.{WINDOW_MARKER} === true")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.option_select.is_visible() and self.run_button.is_visible()

    def is_loading(self) -> bool:
        return self.loading_spinner.is_visible()

    def has_results_table(self) -> bool:
        return self.results_table.is_visible()

    def get_results_row_count(self) -> int:
        return self.results_rows.count()

    def has_chart(self) -> bool:
        return self.chart.is_visible()

    def has_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Download Results").is_visible()

    def get_catchment_options(self) -> list[str]:
        """Return all option values currently available in the catchment select."""
        options = self.catchment_name_select.locator("option")
        return [options.nth(i).get_attribute("value") or "" for i in range(options.count())]

    def is_run_button_enabled(self) -> bool:
        return self.run_button.is_enabled()

    def column_values(self, index: int) -> list[str]:
        """Text of the given column (0 = swept parameter) in current row order."""
        return self.results_rows.locator(f":scope > :nth-child({index + 1})").all_inner_texts()
