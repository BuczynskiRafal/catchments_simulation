"""Calculations page object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_page import BasePage
from .upload_component import UploadComponent

if TYPE_CHECKING:
    from playwright.sync_api import Locator


class CalculationsPage(BasePage):
    """POM for the SWMM vs ANN comparison page (``/calculations``).

    GET is public; running the comparison requires login. For anonymous users
    the "Run Calculations" button redirects to login via JS
    (``data-authenticated``). With JS the run is submitted by ``CS.asyncForm``
    and the results fragment replaces the contents of ``#calculations-output``.
    """

    PATH = "/calculations"
    RUN_TIMEOUT_MS = 60_000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload = UploadComponent(self.page)

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def get_page_heading(self) -> str | None:
        return self.get_heading(level=1)

    def get_nn_heading(self) -> str | None:
        """Return the ANN model description heading text."""
        loc = self.page.locator("h2", has_text="Catchment Area Neural Network Model")
        if loc.count() > 0:
            return loc.inner_text()
        return None

    @property
    def results_heading(self) -> Locator:
        return self.page.locator("#calculations-results-heading")

    @property
    def validity_note(self) -> Locator:
        return self.page.locator(".calc-note")

    @property
    def empty_state(self) -> Locator:
        return self.page.locator("#calculations-results .cs-empty-state")

    def model_details(self, title: str) -> Locator:
        """The ``<details>`` section of the model description with *title*."""
        return self.page.locator(
            "details.calc-details__item", has=self.page.locator("summary", has_text=title)
        )

    # ------------------------------------------------------------------
    # Inside the network
    # ------------------------------------------------------------------

    @property
    def network_heading(self) -> Locator:
        return self.page.locator("#calc-network-title")

    @property
    def network_stage(self) -> Locator:
        """The keyboard-focusable wrapper of the diagram (arrow keys read the neurons)."""
        return self.page.locator("#calc-network-stage")

    @property
    def network_diagram(self) -> Locator:
        """The ``role="img"`` SVG drawn from ``#calculations-network``."""
        return self.page.locator("#calc-network-diagram")

    @property
    def network_caption(self) -> Locator:
        return self.page.locator("#calc-network-caption")

    @property
    def network_readout(self) -> Locator:
        """The line that reads out the neuron under the pointer or the keyboard."""
        return self.page.locator("#calc-network-readout")

    @property
    def network_output(self) -> Locator:
        """ "ANN … · SWMM … (…)" of the traced subcatchment; hidden without a trace."""
        return self.page.locator("#calc-network-output")

    @property
    def trace_select(self) -> Locator:
        return self.page.get_by_label("Trace subcatchment")

    def trace_button(self, name: str) -> Locator:
        """The Trace button in the results table row of subcatchment *name*."""
        return self.results_table.get_by_role("button", name=f"Trace {name} through the network")

    def traced_ann_value(self) -> float:
        """The ANN value of the network output readout, in the page's volume unit."""
        text = self.network_output.inner_text()
        return float(text.split("ANN", 1)[1].split()[0].replace(",", ""))

    def forward(self, inputs: list[float]) -> dict:
        """``CS.network.forward`` of the embedded network on *inputs*, run in the page."""
        return self.page.evaluate(
            """(inputs) => CS.network.forward(
                JSON.parse(document.getElementById('calculations-network').textContent), inputs
            )""",
            inputs,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @property
    def run_button(self) -> Locator:
        return self.page.locator("#run-calculations-button")

    @property
    def loading_state(self) -> Locator:
        return self.page.locator("#calculations-loading-state")

    @property
    def form_error(self) -> Locator:
        return self.page.locator("#calculations-form .alert-danger")

    def run_calculations(self) -> None:
        self.run_button.click()

    def run_and_wait_for_results(self) -> None:
        """Run the comparison and wait until the swapped-in results table is shown."""
        self.run_calculations()
        self.results_table.wait_for(state="visible", timeout=self.RUN_TIMEOUT_MS)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def results_table(self) -> Locator:
        return self.page.locator("#calculations-table")

    def has_results_table(self) -> bool:
        return self.results_table.is_visible()

    def get_results_columns(self) -> list[str]:
        """Return the data column headers of the results table (not the Trace column)."""
        headers = self.results_table.locator("thead th:not([data-calc-trace-col])")
        return [text.strip() for text in headers.all_inner_texts()]

    def get_results_row_count(self) -> int:
        return self.results_table.locator("tbody tr").count()

    def column_values(self, index: int) -> list[str]:
        """Text of every body cell in column *index* (0 = Name)."""
        return [
            text.strip()
            for text in self.results_table.locator(
                f"tbody tr > :nth-child({index + 1})"
            ).all_inner_texts()
        ]

    def sort_by(self, column: str) -> None:
        self.results_table.get_by_role("button", name=column, exact=True).click()

    def metric_value(self, label: str) -> str:
        """Value text of the metric tile whose label is *label*."""
        tile = self.page.locator(".cs-metric", has=self.page.locator("dt", has_text=label))
        return tile.locator(".cs-metric__value").inner_text()

    @property
    def parity_chart(self) -> Locator:
        return self.page.locator("#calculations-parity-chart")

    @property
    def bars_chart(self) -> Locator:
        return self.page.locator("#calculations-bars-chart")

    @property
    def copy_table_button(self) -> Locator:
        return self.page.locator("#copy-calculations-table-button")
