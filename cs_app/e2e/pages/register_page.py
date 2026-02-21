"""Registration page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class RegisterPage(BasePage):
    """POM for the registration page (``/register/``)."""

    PATH = "/register/"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form fields  (6 fields: username, email, first_name, last_name,
    #               password1, password2)
    # ------------------------------------------------------------------

    @property
    def username_input(self) -> Locator:
        return self.page.locator("#id_username")

    @property
    def email_input(self) -> Locator:
        return self.page.locator("#id_email")

    @property
    def first_name_input(self) -> Locator:
        return self.page.locator("#id_first_name")

    @property
    def last_name_input(self) -> Locator:
        return self.page.locator("#id_last_name")

    @property
    def password1_input(self) -> Locator:
        return self.page.locator("#id_password1")

    @property
    def password2_input(self) -> Locator:
        return self.page.locator("#id_password2")

    @property
    def submit_button(self) -> Locator:
        return self.page.get_by_role("button", name="Register")

    def fill_form(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        password1: str,
        password2: str,
    ) -> None:
        self.username_input.fill(username)
        self.email_input.fill(email)
        self.first_name_input.fill(first_name)
        self.last_name_input.fill(last_name)
        self.password1_input.fill(password1)
        self.password2_input.fill(password2)

    def submit(self) -> None:
        self.submit_button.click()

    def register(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        password1: str,
        password2: str,
    ) -> None:
        """Convenience: fill all fields + submit."""
        self.fill_form(username, email, first_name, last_name, password1, password2)
        self.submit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.username_input.is_visible() and self.password1_input.is_visible()

    def get_error_messages(self) -> list[str]:
        return self.page.locator(".errorlist li").all_inner_texts()
