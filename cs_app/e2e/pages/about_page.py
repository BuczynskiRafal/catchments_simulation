"""About page object."""

from __future__ import annotations

from .base_page import BasePage


class AboutPage(BasePage):
    """POM for the about page (``/about``)."""

    PATH = "/about"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    def get_main_heading(self) -> str | None:
        return self.get_heading(level=2)

    def get_body_text(self) -> str:
        return self.page.locator(".content-wrapper").inner_text()
