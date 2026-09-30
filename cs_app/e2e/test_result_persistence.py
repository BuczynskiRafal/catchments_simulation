"""Results of every tool stay on its page while the user moves between tools.

Each run's result is kept server-side for the session, so a tool's page renders its
latest results again when the user comes back to it. Loading the same model again
(the sample data, from another tool's page) keeps them too.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.calculations_page import CalculationsPage
from .pages.nav_component import NavComponent
from .pages.simulation_page import SimulationPage
from .pages.timeseries_page import TimeseriesPage

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

STALE_NOTICE = ".cs-stale-notice"


def _expect_simulation_results(page: Page, sp: SimulationPage) -> None:
    expect(page).to_have_url(re.compile(r".*/simulation$"))
    expect(sp.results_heading).to_be_visible()
    sp.chart.locator(".main-svg").first.wait_for(state="attached")
    assert sp.get_results_row_count() > 0
    expect(page.locator(STALE_NOTICE)).to_have_count(0)


def _expect_timeseries_results(page: Page, tp: TimeseriesPage) -> None:
    expect(page).to_have_url(re.compile(r".*/timeseries$"))
    expect(tp.results_heading).to_be_visible()
    tp.chart.locator(".main-svg").first.wait_for(state="attached")
    expect(page.locator(STALE_NOTICE)).to_have_count(0)


def test_results_survive_moving_between_tools(auth_page: Page, live_server) -> None:
    nav = NavComponent(auth_page)
    sp = SimulationPage(auth_page, live_server.url)
    tp = TimeseriesPage(auth_page, live_server.url)
    cp = CalculationsPage(auth_page, live_server.url)

    sp.navigate_to()
    sp.load_sample_and_select_catchment()
    sp.set_option("simulate_n_imperv")
    sp.run_simulation()
    sp.wait_for_results()

    nav.click_timeseries()
    expect(auth_page).to_have_url(re.compile(r".*/timeseries$"))
    tp.wait_for_catchment_options()
    tp.set_catchment_name(next(o for o in tp.get_catchment_options() if o))
    tp.set_mode("single")
    tp.run_analysis()
    tp.wait_for_results()

    nav.click_calculations()
    expect(auth_page).to_have_url(re.compile(r".*/calculations$"))
    cp.run_and_wait_for_results()

    nav.click_simulation()
    _expect_simulation_results(auth_page, sp)
    nav.click_timeseries()
    _expect_timeseries_results(auth_page, tp)
    nav.click_calculations()
    expect(auth_page).to_have_url(re.compile(r".*/calculations$"))
    expect(cp.results_heading).to_be_visible()
    assert cp.get_results_row_count() > 0

    # The browser's back button renders the same results.
    auth_page.go_back()
    _expect_timeseries_results(auth_page, tp)


def test_loading_the_same_model_again_keeps_results(auth_page: Page, live_server) -> None:
    sp = SimulationPage(auth_page, live_server.url)
    sp.navigate_to()
    sp.load_sample_and_select_catchment()
    sp.set_option("simulate_n_imperv")
    sp.run_simulation()
    sp.wait_for_results()

    NavComponent(auth_page).click_timeseries()
    expect(auth_page).to_have_url(re.compile(r".*/timeseries$"))
    tp = TimeseriesPage(auth_page, live_server.url)
    tp.load_sample_and_pick_catchment()
    NavComponent(auth_page).click_simulation()

    _expect_simulation_results(auth_page, sp)

    # On the page itself, reloading the model it shows does not mark its results stale.
    sp.upload.record_model_events()
    sp.upload.sample_button.click()
    assert sp.upload.wait_for_model_changed()["restored"] is True
    expect(auth_page.locator(STALE_NOTICE)).to_have_count(0)
