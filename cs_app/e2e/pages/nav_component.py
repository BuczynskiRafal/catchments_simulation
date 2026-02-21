"""Navigation component POM — header navbar and footer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from playwright.sync_api import Locator

if TYPE_CHECKING:
    from playwright.sync_api import Page


class NavComponent:
    """Header and footer navigation shared across all pages."""

    def __init__(self, page: Page) -> None:
        self.page = page

    # ------------------------------------------------------------------
    # Header nav links
    # ------------------------------------------------------------------

    @property
    def header(self) -> Locator:
        return self.page.locator("header")

    def click_home(self) -> None:
        self.header.get_by_role("link", name="Home").click()

    def click_simulation(self) -> None:
        self.header.get_by_role("link", name="Simulation").click()

    def click_timeseries(self) -> None:
        self.header.get_by_role("link", name="Timeseries").click()

    def click_calculations(self) -> None:
        self.header.get_by_role("link", name="Calculations").click()

    def click_about(self) -> None:
        self.header.get_by_role("link", name="About").click()

    def click_logo(self) -> None:
        self.header.locator("a img[alt='Logo']").first.click()

    # ------------------------------------------------------------------
    # Auth buttons
    # ------------------------------------------------------------------

    def click_login(self) -> None:
        self.header.get_by_role("link", name="Login").click()

    def click_signup(self) -> None:
        self.header.get_by_role("link", name="Sign-up").click()

    def click_logout(self) -> None:
        self.header.get_by_role("link", name="Logout").click()

    def is_logged_in(self) -> bool:
        return self.header.get_by_role("link", name="Logout").is_visible()

    def is_logged_out(self) -> bool:
        return self.header.get_by_role("link", name="Login").is_visible()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    @property
    def footer(self) -> Locator:
        return self.page.locator("footer")

    def get_footer_about_link(self) -> Locator:
        return self.footer.get_by_role("link", name="About")

    def get_footer_copyright(self) -> str:
        return self.footer.locator("p.text-muted").inner_text()

    def click_footer_logo(self) -> None:
        self.footer.locator("a img[alt='Logo']").click()
