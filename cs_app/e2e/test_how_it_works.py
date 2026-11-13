"""How it works page: rendered equations and parameter guide."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from .pages.how_it_works_page import HowItWorksPage

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.e2e

PHONE = {"width": 390, "height": 844}


@pytest.fixture
def hiw(page: Page, live_server) -> HowItWorksPage:
    hp = HowItWorksPage(page, live_server.url)
    hp.navigate_to()
    return hp


class TestContent:
    def test_heading(self, hiw: HowItWorksPage) -> None:
        assert hiw.get_heading(level=1) == "How the simulation works"

    def test_equations_are_typeset_and_numbered_in_order(self, hiw: HowItWorksPage) -> None:
        hiw.page.wait_for_function(
            "document.querySelectorAll('.hiw-equation .katex-display').length === 18"
        )
        assert hiw.rendered_equation_numbers() == [f"({n})" for n in range(1, 19)]
        # Inline symbols are typeset too; no TeX source is left showing.
        assert hiw.page.locator("[data-cs-math]:not(.is-rendered)").count() == 0


class TestQuality:
    def test_no_console_errors(self, page: Page, live_server) -> None:
        messages: list[str] = []
        page.on(
            "console", lambda m: messages.append(m.text) if m.type in ("error", "warning") else None
        )
        page.on("pageerror", lambda e: messages.append(str(e)))
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        assert messages == []

    def test_no_horizontal_overflow_on_a_phone(self, page: Page, live_server) -> None:
        page.set_viewport_size(PHONE)
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
        assert overflow <= 0, f"page scrolls sideways by {overflow}px"

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(self, page: Page, live_server, scheme: str) -> None:
        page.emulate_media(color_scheme=scheme)
        hp = HowItWorksPage(page, live_server.url)
        hp.navigate_to()
        page.wait_for_function("document.querySelectorAll('.katex').length > 0")
        report = hp.run_axe_audit()
        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]
