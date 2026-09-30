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

    def mode_radio(self, value: str) -> Locator:
        """Radio of the 'single' / 'sweep' segmented control."""
        return self.page.locator(f"input[name='mode'][value='{value}']")

    @property
    def sweep_fields(self) -> Locator:
        return self.page.locator("#ts-sweep-fields")

    @property
    def runs_hint(self) -> Locator:
        return self.page.locator("#ts-runs-hint")

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

    @property
    def canvas(self) -> Locator:
        """Results region whose content the async run replaces."""
        return self.page.locator("#timeseries-canvas")

    @property
    def results_heading(self) -> Locator:
        return self.page.locator("#timeseries-results-heading")

    @property
    def chart(self) -> Locator:
        return self.page.locator("#timeseries-chart")

    @property
    def empty_state(self) -> Locator:
        return self.canvas.locator(".cs-empty-state")

    @property
    def sweep_series_select(self) -> Locator:
        return self.page.locator("#ts-sweep-series")

    @property
    def sweep_table(self) -> Locator:
        return self.page.locator("table[data-ts-table='sweep']")

    @property
    def timestep_table(self) -> Locator:
        return self.page.locator("table[data-ts-table='single']")

    @property
    def timestep_toggle(self) -> Locator:
        return self.page.locator("details[data-ts-data] > summary")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def set_mode(self, value: str) -> None:
        """Set analysis mode: 'single' or 'sweep' (clicks the segmented control)."""
        radio_id = self.mode_radio(value).get_attribute("id")
        self.page.locator(f"label[for='{radio_id}']").click()

    def selected_mode(self) -> str | None:
        return self.page.locator("input[name='mode']:checked").get_attribute("value")

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

    def wait_for_results(self, timeout: float = 120_000) -> None:
        """Wait until an async run has swapped in results and drawn the chart."""
        self.results_heading.wait_for(state="visible", timeout=timeout)
        self.chart.locator(".main-svg").first.wait_for(state="attached", timeout=timeout)

    def load_sample_and_pick_catchment(self) -> str:
        """Load the bundled sample model and select its first subcatchment."""
        self.upload.click_sample_data()
        self.wait_for_catchment_options()
        options = [o for o in self.get_catchment_options() if o]
        assert options, "sample model has no subcatchments"
        self.set_catchment_name(options[0])
        return options[0]

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
        return self.mode_radio("single").is_enabled() and self.run_button.is_visible()

    def is_loading(self) -> bool:
        return self.loading_spinner.is_visible()

    def has_chart(self) -> bool:
        """True once Plotly has drawn into the chart container."""
        return self.chart.locator(".main-svg").count() > 0

    def get_metric(self, label: str) -> str | None:
        """Value text (with unit) of the metric tile labelled *label*."""
        tile = self.page.locator(
            ".cs-metric", has=self.page.locator(".cs-metric__label", has_text=label)
        )
        if tile.count() > 0:
            return tile.locator(".cs-metric__value").inner_text()
        return None

    def get_time_to_peak(self) -> str | None:
        return self.get_metric("Time to peak")

    def get_runoff_volume(self) -> str | None:
        return self.get_metric("Runoff volume")

    def get_peak_runoff(self) -> str | None:
        return self.get_metric("Peak runoff")

    def sweep_legend_names(self) -> list[str]:
        """Legend entries of the sweep chart, read from the Plotly traces ([] with a colour bar)."""
        return self.chart.evaluate(
            "el => el.layout.showlegend === false ? []"
            " : el.data.filter(t => t.showlegend !== false).map(t => t.name)"
        )

    def has_download_results_button(self) -> bool:
        return self.page.get_by_role("button", name="Download Results (.xlsx)").is_visible()

    def has_csv_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Download Results (.csv)").is_visible()

    def has_png_download_button(self) -> bool:
        return self.page.get_by_role("button", name="Download PNG").is_visible()

    def get_catchment_options(self) -> list[str]:
        options = self.catchment_name_select.locator("option")
        return [options.nth(i).get_attribute("value") or "" for i in range(options.count())]
