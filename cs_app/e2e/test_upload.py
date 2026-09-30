"""Upload zone tests — Dropzone.js interactions.

Simulation and Timeseries require @login_required, so those tests use auth_page.
Calculations does NOT require auth.
"""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import Browser, Page, expect

from .conftest import new_context_without_js, wait_for_styles
from .pages.calculations_page import CalculationsPage
from .pages.simulation_page import SimulationPage
from .pages.timeseries_page import TimeseriesPage

pytestmark = pytest.mark.e2e

SAMPLE_INP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "example.inp")


class TestUploadZoneVisibility:
    """Upload zone is present on pages that include it."""

    def test_simulation_has_upload_zone(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        assert sp.upload.is_visible()

    def test_timeseries_has_upload_zone(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        assert tp.upload.is_visible()

    def test_calculations_has_upload_zone(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()
        assert cp.upload.is_visible()

    def test_sample_button_needs_javascript(self, browser: Browser, live_server) -> None:
        context = new_context_without_js(browser)
        try:
            cp = CalculationsPage(context.new_page(), live_server.url)
            cp.navigate_to()
            wait_for_styles(cp.page)
            expect(cp.upload.sample_button).to_be_hidden()
            expect(cp.page.locator("#upload-fallback-file")).to_be_visible()
        finally:
            context.close()


class TestSampleDataUpload:
    """'Try sample data' button loads sample file and updates UI."""

    def test_sample_data_loads_on_simulation(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        assert sp.upload.has_upload_status()
        status_text = sp.upload.get_upload_status_text_value()
        assert "example.inp" in status_text

    def test_sample_data_populates_catchment_dropdown(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        sp.wait_for_catchment_options()
        options = sp.get_catchment_options()
        assert len(options) > 1


class TestUploadClear:
    """Removing file via Dropzone × remove link."""

    def test_remove_link_clears_upload_status(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        assert sp.upload.has_upload_status()
        sp.upload.clear_upload()
        expect(auth_page.locator("#upload-status")).to_be_hidden()


class TestRealFileUpload:
    """Upload using an actual .inp file via the hidden file input."""

    @pytest.mark.skipif(
        not os.path.isfile(SAMPLE_INP_PATH),
        reason="example.inp not found",
    )
    def test_upload_real_inp_file(self, auth_page: Page, live_server) -> None:
        """Upload the sample .inp file and verify status updates."""
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.upload_file(SAMPLE_INP_PATH)
        assert sp.upload.has_upload_status()
        status_text = sp.upload.get_upload_status_text_value()
        assert "example.inp" in status_text


class TestUploadEdgeCases:
    """Upload validation edge cases."""

    def test_wrong_extension_rejected(self, auth_page: Page, live_server, tmp_path) -> None:
        """Uploading a non-.inp file should show a Dropzone error."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("not an inp file")

        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()

        # Dropzone client-side validation rejects non-.inp files
        sp.upload.file_input.set_input_files(str(wrong_file))
        expect(auth_page.locator(".dz-error")).to_be_visible()
        assert sp.upload.get_dropzone_error_message()
        assert not sp.upload.has_upload_status()


class TestModelEvents:
    """Pages learn about the loaded model through document events."""

    def test_sample_data_dispatches_model_changed(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.record_model_events()

        sp.upload.click_sample_data()
        detail = sp.upload.wait_for_model_changed()

        assert detail["filename"] == "example.inp"
        assert detail["size"] > 0
        assert len(detail["subcatchments"]) > 0
        count = len(detail["subcatchments"])
        expect(sp.upload.upload_status_meta).to_contain_text(f"{count} subcatchment")

    def test_restored_model_dispatches_model_changed_once(
        self, auth_page: Page, live_server
    ) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()

        auth_page.add_init_script(
            """window.__csModelEvents = 0;
            document.addEventListener('cs:model-changed', () => { window.__csModelEvents += 1; });"""
        )
        sp.navigate_to()
        auth_page.wait_for_function("() => window.__csModelEvents > 0", timeout=15_000)
        expect(sp.catchment_name_select).to_be_enabled()

        assert auth_page.evaluate("window.__csModelEvents") == 1
        assert "example.inp" in sp.upload.get_upload_status_text_value()

    def test_remove_dispatches_model_cleared(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        sp.upload.record_model_events()

        sp.upload.clear_upload()

        auth_page.wait_for_function("() => window.__csModelCleared === true", timeout=5_000)
        expect(sp.upload.trigger).to_be_focused()
        assert sp.get_catchment_options() == [""]


class TestUploadZoneKeyboard:
    """The drop target opens the file dialog from the keyboard."""

    @pytest.mark.parametrize("key", ["Enter", " "])
    def test_trigger_opens_file_dialog(self, auth_page: Page, live_server, key: str) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        # Listening for file choosers turns their interception on without waiting for it,
        # so a key pressed right after expect_file_chooser() can reach the page first and
        # open an unseen native dialog. Listen before the navigation instead.
        auth_page.on("filechooser", lambda _chooser: None)
        sp.navigate_to()
        expect(sp.upload.trigger).to_have_attribute("aria-describedby", "upload-zone-hint")

        sp.upload.trigger.focus()
        with auth_page.expect_file_chooser(timeout=5_000) as chooser_info:
            auth_page.keyboard.press(key)

        assert not chooser_info.value.is_multiple()


class TestUploadErrorPreview:
    """A rejected file never discards the model that is already loaded."""

    def test_dismissing_rejected_file_keeps_loaded_model(
        self, auth_page: Page, live_server, tmp_path
    ) -> None:
        wrong_file = tmp_path / "notes.txt"
        wrong_file.write_text("not an inp file")
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()

        sp.upload.file_input.set_input_files(str(wrong_file))
        expect(auth_page.locator(".dz-error [data-dz-errormessage]")).not_to_be_empty()
        sp.upload.error_preview_dismiss.focus()
        auth_page.keyboard.press("Enter")

        expect(auth_page.locator(".dz-error")).to_have_count(0)
        expect(sp.upload.trigger).to_be_focused()
        assert sp.upload.has_upload_status()
        sp.navigate_to()
        expect(sp.upload.upload_status_text).to_contain_text("example.inp")


class TestUploadRequiresLogin:
    """Anonymous sample-data requests are redirected to the login page."""

    def test_anonymous_sample_click_redirects_to_login(self, page: Page, live_server) -> None:
        cp = CalculationsPage(page, live_server.url)
        cp.navigate_to()

        cp.upload.sample_button.click()

        page.wait_for_url("**/login/**", timeout=10_000)
        assert "next=" in page.url
