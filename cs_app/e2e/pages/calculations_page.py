"""Calculations page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage
from .upload_component import UploadComponent


class CalculationsPage(BasePage):
    """POM for the calculations page (``/calculations``).

    Note: This page does NOT require authentication — anonymous users
    can access it, but the "Run Calculations" button redirects
    unauthenticated users to login via JS (``data-authenticated``).
    """

    PATH = "/calculations"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upload = UploadComponent(self.page)

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def get_nn_heading(self) -> str | None:
        """Return the ANN model heading text."""
        loc = self.page.locator("h3", has_text="Catchment Area Neural Network Model")
        if loc.count() > 0:
            return loc.inner_text()
        return None

    def get_results_heading(self) -> str | None:
        loc = self.page.locator("h4", has_text="Results Comparison")
        if loc.count() > 0:
            return loc.inner_text()
        return None

    @property
    def run_button(self) -> Locator:
        return self.page.locator("#run-calculations-button")

    def run_calculations(self) -> None:
        self.run_button.click()

    def has_results_table(self) -> bool:
        return self.page.locator("table.table-striped").is_visible()

    def get_results_columns(self) -> list[str]:
        """Return column headers of the results table."""
        return self.page.locator("table.table-striped thead th").all_inner_texts()

    def get_results_row_count(self) -> int:
        return self.page.locator("table.table-striped tbody tr").count()

    def has_nn_architecture_image(self) -> bool:
        return self.page.locator("img[alt='Opis obrazu']").is_visible()
