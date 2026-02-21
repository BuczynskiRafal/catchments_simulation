"""Reusable Upload Zone (Dropzone.js) component POM.

Used by Simulation, Timeseries, and Calculations pages.  Upload clearing is
done via the Dropzone's ×  remove-link (``dz-remove``), not a standalone
"Clear" button.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from playwright.sync_api import Locator, expect

if TYPE_CHECKING:
    from playwright.sync_api import Page


class UploadComponent:
    """POM for the Dropzone.js upload zone shared across pages."""

    def __init__(self, page: Page) -> None:
        self.page = page

    # ------------------------------------------------------------------
    # Selectors (public — tests may assert on these)
    # ------------------------------------------------------------------

    @property
    def dropzone(self) -> Locator:
        return self.page.locator("#my-dropzone")

    @property
    def sample_button(self) -> Locator:
        return self.page.locator("#load-sample-data-button")

    @property
    def upload_status(self) -> Locator:
        return self.page.locator("#upload-status")

    @property
    def upload_status_text(self) -> Locator:
        return self.page.locator("#upload-status-text")

    @property
    def dropzone_remove_link(self) -> Locator:
        """The × remove link added by Dropzone ``addRemoveLinks: true``."""
        return self.page.locator(".dz-remove")

    @property
    def file_input(self) -> Locator:
        """The hidden file input created by Dropzone.js (class ``dz-hidden-input``).

        Dropzone dynamically creates an ``<input type='file'>`` with class
        ``dz-hidden-input`` outside of regular DOM flow.
        """
        return self.page.locator("input.dz-hidden-input")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def is_visible(self) -> bool:
        return self.dropzone.is_visible()

    def click_sample_data(self) -> None:
        """Click 'Try sample data' and wait for upload status to appear."""
        self.sample_button.click()
        # Wait for the status bar to become visible (sample loaded)
        self.upload_status.wait_for(state="visible", timeout=15_000)

    def upload_file(self, path: str) -> None:
        """Upload a file by setting it on the hidden input.

        Waits for the Dropzone success event (upload status becomes visible)
        or for a Dropzone error element to appear.
        """
        self.file_input.set_input_files(path)
        # Wait for async Dropzone processing to complete
        self.page.wait_for_function(
            """() => {
                const status = document.getElementById('upload-status');
                const error = document.querySelector('.dz-error');
                return (status && status.style.display !== 'none') || !!error;
            }""",
            timeout=15_000,
        )

    def get_upload_status_text_value(self) -> str:
        """Return the text shown in the upload status bar."""
        return self.upload_status_text.inner_text()

    def has_upload_status(self) -> bool:
        return self.upload_status.is_visible()

    def clear_upload(self) -> None:
        """Remove the file via Dropzone's × remove link."""
        self.dropzone_remove_link.click()

    def wait_for_sample_button_ready(self) -> None:
        """Wait until the sample button is enabled (not loading)."""
        expect(self.sample_button).to_be_enabled(timeout=10_000)

    def has_dropzone_error(self) -> bool:
        """Check whether a Dropzone error element is present."""
        return self.page.locator(".dz-error").count() > 0

    def get_dropzone_error_message(self) -> str | None:
        """Return the Dropzone error message text, if any."""
        error_el = self.page.locator("[data-dz-errormessage]")
        if error_el.count() > 0:
            return error_el.first.inner_text()
        return None
