"""Reusable Upload Zone (Dropzone.js) component POM.

Used by Simulation, Timeseries, and Calculations pages.  The loaded model is
shown as a chip (``#upload-status``); clearing it is done via the chip's ×
button (``.dz-remove``), not a standalone "Clear" button.
"""

from __future__ import annotations

import os
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
    def upload_status_meta(self) -> Locator:
        """Size and subcatchment count shown in the model chip."""
        return self.page.locator("#upload-status-meta")

    @property
    def dropzone_remove_link(self) -> Locator:
        """The × button of the loaded-model chip (class ``dz-remove``)."""
        return self.page.locator(".dz-remove")

    @property
    def trigger(self) -> Locator:
        """Keyboard-focusable button inside the drop target that opens the file dialog."""
        return self.page.locator("#my-dropzone .upload-trigger")

    @property
    def error_preview_dismiss(self) -> Locator:
        """Dismiss button of the failed upload preview."""
        return self.page.locator("#my-dropzone .dz-error .upload-preview-dismiss")

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

        Waits until the model chip shows this file (success) or a Dropzone
        error element appears; matching the name keeps it correct when a
        model is already loaded.
        """
        self.file_input.set_input_files(path)
        self.page.wait_for_function(
            """(name) => {
                const status = document.getElementById('upload-status');
                const text = document.getElementById('upload-status-text');
                const loaded = status && status.style.display !== 'none'
                    && text && text.textContent.includes(name);
                return loaded || !!document.querySelector('.dz-error');
            }""",
            arg=os.path.basename(path),
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

    def record_model_events(self) -> None:
        """Store ``cs:model-changed`` / ``cs:model-cleared`` details on ``window`` for assertions."""
        self.page.evaluate(
            """() => {
                window.__csModelChanged = null;
                window.__csModelCleared = false;
                document.addEventListener('cs:model-changed', (e) => { window.__csModelChanged = e.detail; });
                document.addEventListener('cs:model-cleared', () => { window.__csModelCleared = true; });
            }"""
        )

    def wait_for_model_changed(self) -> dict:
        """Wait for the next recorded ``cs:model-changed`` event and return its detail."""
        handle = self.page.wait_for_function("() => window.__csModelChanged", timeout=15_000)
        return handle.json_value()

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
