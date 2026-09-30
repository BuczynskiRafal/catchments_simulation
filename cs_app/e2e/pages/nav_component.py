"""Navigation component POM — header navbar, theme menu, account menu and footer.

Below the ``lg`` breakpoint the navbar collapses behind a toggler; every
action opens it first, so callers do not need to care about the viewport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from playwright.sync_api import Locator

if TYPE_CHECKING:
    from playwright.sync_api import Page

THEME_MODES = ("light", "dark", "auto")


class NavComponent:
    """Header and footer navigation shared across all pages."""

    def __init__(self, page: Page) -> None:
        self.page = page

    # ------------------------------------------------------------------
    # Header structure
    # ------------------------------------------------------------------

    @property
    def header(self) -> Locator:
        return self.page.locator("header")

    @property
    def toggler(self) -> Locator:
        return self.header.locator(".navbar-toggler")

    @property
    def menu(self) -> Locator:
        return self.header.locator("#main-nav")

    def is_collapsed(self) -> bool:
        """True when the hamburger is shown and the menu is closed."""
        return self.toggler.is_visible() and self.toggler.get_attribute("aria-expanded") != "true"

    def open_menu(self) -> None:
        """Expand the collapsed navbar (no-op on wide viewports)."""
        if self.is_collapsed():
            self.toggler.click()
            self.header.locator("#main-nav.show").wait_for(state="visible")

    def link(self, name: str) -> Locator:
        return self.header.get_by_role("link", name=name, exact=True)

    def current_link(self) -> Locator:
        return self.header.locator("a[aria-current='page']")

    # ------------------------------------------------------------------
    # Header nav links
    # ------------------------------------------------------------------

    def _click_link(self, name: str) -> None:
        self.open_menu()
        self.link(name).click()

    def click_home(self) -> None:
        self._click_link("Home")

    def click_simulation(self) -> None:
        self._click_link("Simulation")

    def click_timeseries(self) -> None:
        self._click_link("Timeseries")

    def click_calculations(self) -> None:
        self._click_link("Calculations")

    def click_about(self) -> None:
        self._click_link("About")

    def click_logo(self) -> None:
        self.header.locator("a img[alt='Logo']").first.click()

    # ------------------------------------------------------------------
    # Auth / account
    # ------------------------------------------------------------------

    @property
    def account_toggle(self) -> Locator:
        return self.header.locator("#account-menu")

    @property
    def logout_button(self) -> Locator:
        """Logout submits a POST form, so it is a button rather than a link."""
        return self.header.get_by_role("button", name="Logout", exact=True)

    def open_account_menu(self) -> None:
        self.open_menu()
        if self.account_toggle.get_attribute("aria-expanded") != "true":
            self.account_toggle.click()
        self.logout_button.wait_for(state="visible")

    def click_login(self) -> None:
        self._click_link("Log in")

    def click_signup(self) -> None:
        self._click_link("Create account")

    def click_profile(self) -> None:
        self.open_account_menu()
        self.link("Profile").click()

    def click_logout(self) -> None:
        self.open_account_menu()
        self.logout_button.click()

    def is_logged_in(self) -> bool:
        """Server rendered the account menu (independent of viewport/menu state)."""
        return self.account_toggle.count() == 1

    def is_logged_out(self) -> bool:
        return self.link("Log in").count() == 1 and self.account_toggle.count() == 0

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    @property
    def theme_toggle(self) -> Locator:
        return self.header.locator("#theme-menu")

    def theme_option(self, mode: str) -> Locator:
        return self.header.locator(f"[data-cs-theme-value='{mode}']")

    def set_theme(self, mode: str) -> None:
        """Pick light/dark/auto from the theme menu."""
        if mode not in THEME_MODES:
            raise ValueError(f"Unknown theme mode: {mode}")
        self.open_menu()
        self.theme_toggle.click()
        self.theme_option(mode).click()

    def resolved_theme(self) -> str | None:
        return self.page.locator("html").get_attribute("data-bs-theme")

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    @property
    def footer(self) -> Locator:
        return self.page.locator("footer")

    def get_footer_link(self, name: str) -> Locator:
        return self.footer.get_by_role("link", name=name, exact=True)

    def get_footer_about_link(self) -> Locator:
        return self.get_footer_link("About")

    def get_footer_copyright(self) -> str:
        return self.footer.locator("p.text-muted").inner_text()

    def click_footer_logo(self) -> None:
        self.footer.locator("a img[alt='Logo']").click()
