"""User profile page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class UserProfilePage(BasePage):
    """POM for the user profile page (``/user/<id>/profile``).

    The profile view is public for GET, but the form is read-only when
    viewed by a user other than the owner (fields disabled, submit hidden).
    Only the owner can POST updates.
    """

    def navigate_to(self, user_id: int) -> None:
        self.navigate(f"/user/{user_id}/profile")

    # ------------------------------------------------------------------
    # Form fields
    # ------------------------------------------------------------------

    @property
    def user_field(self) -> Locator:
        return self.page.locator("#id_user")

    @property
    def bio_field(self) -> Locator:
        return self.page.locator("#id_bio")

    @property
    def submit_button(self) -> Locator:
        return self.page.get_by_role("button", name="Wyślij")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def fill_bio(self, bio: str) -> None:
        self.bio_field.fill(bio)

    def submit(self) -> None:
        self.submit_button.click()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_form(self) -> bool:
        return self.bio_field.is_visible()

    def is_read_only(self) -> bool:
        """Check if the form is in read-only mode (fields disabled)."""
        return self.bio_field.is_disabled()

    def has_submit_button(self) -> bool:
        return self.submit_button.count() > 0 and self.submit_button.is_visible()
