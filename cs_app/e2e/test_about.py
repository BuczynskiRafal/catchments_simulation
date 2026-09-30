"""About page tests."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.about_page import AboutPage

pytestmark = pytest.mark.e2e


@pytest.fixture()
def about(page: Page, live_server) -> AboutPage:
    ap = AboutPage(page, live_server.url)
    ap.navigate_to()
    return ap


def test_heading_and_sections(about: AboutPage) -> None:
    assert about.get_main_heading() == "About"
    for heading in ("What it is", "Who it is for", "How it works", "Open source"):
        expect(about.page.get_by_role("heading", level=2, name=heading, exact=True)).to_be_visible()


def test_open_source_links(about: AboutPage) -> None:
    main = about.page.locator("main")
    expected = {
        "GitHub": "https://github.com/BuczynskiRafal/catchments_simulation",
        "PyPI": "https://pypi.org/project/catchment-simulation/",
        "Documentation": "https://catchments-simulation.readthedocs.io",
    }
    for name, href in expected.items():
        expect(main.get_by_role("link", name=re.compile(f"^{name}"))).to_have_attribute(
            "href", href
        )


def test_contact_call_to_action(about: AboutPage) -> None:
    about.page.locator("main").get_by_role("link", name="Get in touch", exact=True).click()
    expect(about.page).to_have_url(re.compile(r".*/contact$"))


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_no_serious_axe_violations(page: Page, live_server, scheme: str) -> None:
    page.emulate_media(color_scheme=scheme)
    ap = AboutPage(page, live_server.url)
    ap.navigate_to()
    report = ap.run_axe_audit()
    assert not report.critical + report.serious, [v.id for v in report.critical + report.serious]
