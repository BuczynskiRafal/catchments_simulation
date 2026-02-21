"""Contact page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class ContactPage(BasePage):
    """POM for the contact form page (``/contact``).

    Form labels are in Polish:
      - email  → "Adres email"
      - title  → "Tytuł"
      - content → "Treść"
      - submit → "Wyślij"
    """

    PATH = "/contact"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Form fields
    # ------------------------------------------------------------------

    @property
    def email_input(self) -> Locator:
        return self.page.locator("#id_email")

    @property
    def title_input(self) -> Locator:
        return self.page.locator("#id_title")

    @property
    def content_input(self) -> Locator:
        return self.page.locator("#id_content")

    @property
    def send_to_me_checkbox(self) -> Locator:
        return self.page.locator("#id_send_to_me")

    @property
    def submit_button(self) -> Locator:
        return self.page.get_by_role("button", name="Wyślij")

    def fill_form(self, email: str, title: str, content: str, *, send_to_me: bool = False) -> None:
        self.email_input.fill(email)
        self.title_input.fill(title)
        self.content_input.fill(content)
        if send_to_me:
            self.send_to_me_checkbox.check()

    def submit(self) -> None:
        self.submit_button.click()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.email_input.is_visible() and self.title_input.is_visible()

    def get_error_messages(self) -> list[str]:
        return self.page.locator(".errorlist li").all_inner_texts()
