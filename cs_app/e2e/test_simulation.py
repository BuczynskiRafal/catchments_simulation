"""Simulation page tests.

Simulation page requires @login_required, so all tests use auth_page.
Tests that run actual SWMM simulations are marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from .pages.simulation_page import SimulationPage

pytestmark = pytest.mark.e2e


class TestSimulationFormRendering:
    """Form fields render correctly (requires authentication)."""

    def test_form_fields_present(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        assert sp.has_form()
        expect(sp.option_select).to_be_visible()
        expect(sp.start_input).to_be_visible()
        expect(sp.stop_input).to_be_visible()
        expect(sp.step_input).to_be_visible()
        expect(sp.catchment_name_select).to_be_visible()

    def test_default_catchment_placeholder(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        options = sp.get_catchment_options()
        assert any("Upload a file first" in opt or opt == "" for opt in options)

    def test_predefined_method_hides_range_fields(self, auth_page: Page, live_server) -> None:
        """When a predefined method is selected, start/stop/step should be hidden."""
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.set_option("simulate_n_imperv")
        col = auth_page.locator("#id_start").locator(
            "xpath=ancestor::div[contains(@class,'col-md-2')]"
        )
        expect(col).to_be_hidden()

    def test_run_button_text(self, auth_page: Page, live_server) -> None:
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        expect(sp.run_button).to_have_text("Run Simulation")


class TestSimulationExecution:
    """Full simulation execution tests (require sample data + SWMM)."""

    @pytest.mark.slow
    def test_full_simulation_run(self, auth_page: Page, live_server) -> None:
        """Load sample → select catchment → run → verify results table and chart."""
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()

        sp.upload.click_sample_data()
        sp.wait_for_catchment_options()

        options = sp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        assert len(real_options) > 0, "No catchments available after sample upload"
        sp.set_catchment_name(real_options[0])

        sp.set_option("simulate_percent_slope")
        sp.set_start(1)
        sp.set_stop(5)
        sp.set_step(1)

        sp.run_simulation()
        auth_page.wait_for_load_state("domcontentloaded", timeout=60_000)

        assert sp.has_results_table()
        assert sp.get_results_row_count() > 0
        assert sp.has_chart()
        assert sp.has_download_button()

    @pytest.mark.slow
    def test_loading_spinner_appears(self, auth_page: Page, live_server) -> None:
        """Spinner should appear immediately after clicking Run Simulation."""
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        sp.wait_for_catchment_options()

        options = sp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        if not real_options:
            pytest.skip("No catchments available")
        sp.set_catchment_name(real_options[0])
        sp.set_option("simulate_percent_slope")
        sp.set_start(1)
        sp.set_stop(3)
        sp.set_step(1)

        # Prevent actual form submission so the page stays and the spinner
        # (shown by the existing simulation.js submit handler) remains visible.
        # Note: the page has two <form> elements (upload zone + simulation),
        # so we target the simulation form via the run button's parent form.
        auth_page.evaluate(
            "document.getElementById('run-simulation-button').closest('form')"
            ".addEventListener('submit', e => e.preventDefault())"
        )
        sp.run_simulation()
        expect(sp.loading_spinner).to_be_visible(timeout=2000)


class TestSimulationValidation:
    """Form validation edge cases (require auth + sample data)."""

    @pytest.mark.slow
    def test_start_greater_than_stop_shows_error(self, auth_page: Page, live_server) -> None:
        """start > stop should show a validation error."""
        sp = SimulationPage(auth_page, live_server.url)
        sp.navigate_to()
        sp.upload.click_sample_data()
        sp.wait_for_catchment_options()

        options = sp.get_catchment_options()
        real_options = [o for o in options if o and "Upload" not in o and "Select" not in o]
        if not real_options:
            pytest.skip("No catchments available")
        sp.set_catchment_name(real_options[0])
        sp.set_option("simulate_percent_slope")
        sp.set_start(10)
        sp.set_stop(1)
        sp.set_step(1)
        sp.run_simulation()
        auth_page.wait_for_load_state("domcontentloaded")

        errors = auth_page.locator(".invalid-feedback")
        expect(errors.first).to_be_visible()
