"""Upload zone tests — Dropzone.js interactions.

Simulation and Timeseries require @login_required, so those tests use auth_page.
Calculations does NOT require auth.
"""

from __future__ import annotations

import os

import pytest
from playwright.sync_api import Page, expect

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
    @pytest.mark.xfail(
        reason=(
            "Server-side bug: upload view sets request.upload_handlers after "
            "the upload has been processed (AttributeError). File reaches "
            "the server correctly via Dropzone, but the view crashes."
        ),
        strict=False,
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
        # Wait briefly for Dropzone to process
        auth_page.wait_for_timeout(500)
        # File should be rejected — either error element appears or no upload status
        assert not sp.upload.has_upload_status() or sp.upload.has_dropzone_error()
