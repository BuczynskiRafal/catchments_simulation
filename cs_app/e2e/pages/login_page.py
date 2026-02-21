"""Login page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class LoginPage(BasePage):
    """POM for the Django login page (``/accounts/login/``)."""

    PATH = "/accounts/login/"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form interaction
    # ------------------------------------------------------------------

    @property
    def username_input(self) -> Locator:
        return self.page.locator("#id_username")

    @property
    def password_input(self) -> Locator:
        return self.page.locator("#id_password")

    @property
    def submit_button(self) -> Locator:
        return self.page.get_by_role("button", name="Login")

    def fill_credentials(self, username: str, password: str) -> None:
        self.username_input.fill(username)
        self.password_input.fill(password)

    def submit(self) -> None:
        self.submit_button.click()

    def login(self, username: str, password: str) -> None:
        """Convenience: fill + submit."""
        self.fill_credentials(username, password)
        self.submit()

    # ------------------------------------------------------------------
    # Assertions / queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.username_input.is_visible() and self.password_input.is_visible()

    def get_error_messages(self) -> list[str]:
        """Return non-field error messages shown by crispy forms.

        Uses ``all_inner_texts()`` to avoid race conditions.
        """
        return self.page.locator(".errorlist li").all_inner_texts()
