"""Timeseries analysis page tests.

Timeseries page requires @login_required, so all tests use auth_page.
Tests that run actual SWMM simulations are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from .pages.timeseries_page import TimeseriesPage

pytestmark = pytest.mark.e2e


class TestTimeseriesFormRendering:
    """Form fields render correctly (requires authentication)."""

    def test_form_fields_present(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        assert tp.has_form()
        expect(tp.mode_select).to_be_visible()
        expect(tp.catchment_name_select).to_be_visible()

    def test_run_button_text(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        expect(tp.run_button).to_have_text("Run Timeseries Analysis")

    def test_default_mode_is_single(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        expect(tp.mode_select).to_have_value("single")

    def test_sweep_mode_shows_feature_field(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        tp.set_mode("sweep")
        expect(auth_page.locator("#feature-wrapper")).to_be_visible()

    def test_sweep_mode_shows_range_fields(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        tp.set_mode("sweep")
        expect(auth_page.locator("#start-wrapper")).to_be_visible()
        expect(auth_page.locator("#stop-wrapper")).to_be_visible()
        expect(auth_page.locator("#step-wrapper")).to_be_visible()


class TestTimeseriesExecution:
    """Full timeseries execution tests."""

    @pytest.mark.slow
    def test_single_timeseries_run(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()

        tp.upload.click_sample_data()
        tp.wait_for_catchment_options()

        options = tp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        assert len(real_options) > 0
        tp.set_catchment_name(real_options[0])
        tp.set_mode("single")

        tp.run_analysis()
        auth_page.wait_for_load_state("domcontentloaded", timeout=60_000)

        assert tp.has_chart()
        ttp = tp.get_time_to_peak()
        vol = tp.get_runoff_volume()
        assert ttp is not None or vol is not None, "Expected at least one metric card"

        assert tp.has_download_results_button()
        assert tp.has_csv_download_button()
        assert tp.has_png_download_button()

    @pytest.mark.slow
    def test_sweep_timeseries_run(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()

        tp.upload.click_sample_data()
        tp.wait_for_catchment_options()

        options = tp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        assert len(real_options) > 0
        tp.set_catchment_name(real_options[0])

        tp.set_mode("sweep")
        tp.set_feature("PercImperv")
        tp.set_start(0)
        tp.set_stop(50)
        tp.set_step(25)

        tp.run_analysis()
        auth_page.wait_for_load_state("domcontentloaded", timeout=120_000)

        assert tp.has_chart()


class TestTimeseriesValidation:
    """Timeseries form validation (requires auth + sample data)."""

    @pytest.mark.slow
    def test_sweep_start_greater_than_stop(self, auth_page: Page, live_server) -> None:
        tp = TimeseriesPage(auth_page, live_server.url)
        tp.navigate_to()
        tp.upload.click_sample_data()
        tp.wait_for_catchment_options()

        options = tp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        if not real_options:
            pytest.skip("No catchments available")
        tp.set_catchment_name(real_options[0])
        tp.set_mode("sweep")
        tp.set_feature("PercSlope")
        tp.set_start(100)
        tp.set_stop(10)
        tp.set_step(10)

        tp.run_analysis()
        auth_page.wait_for_load_state("domcontentloaded")

        errors = auth_page.locator(".invalid-feedback")
        expect(errors.first).to_be_visible()
