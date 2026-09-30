"""User profile page object."""

from __future__ import annotations

from playwright.sync_api import Locator

from .base_page import BasePage


class UserProfilePage(BasePage):
    """POM for the user profile page (``/user/<id>/profile``).

    The profile view is public for GET. The owner gets an editable bio form;
    everyone else sees the bio as text, without a form or submit button.
    """

    def navigate_to(self, user_id: int) -> None:
        self.navigate(f"/user/{user_id}/profile")

    # ------------------------------------------------------------------
    # Elements
    # ------------------------------------------------------------------

    @property
    def profile_name(self) -> Locator:
        return self.page.locator("#profile-name")

    @property
    def profile_meta(self) -> Locator:
        return self.page.locator(".profile-card__meta")

    @property
    def bio_field(self) -> Locator:
        return self.page.locator("#id_bio")

    @property
    def bio_text(self) -> Locator:
        return self.page.locator(".profile-card__bio")

    @property
    def submit_button(self) -> Locator:
        return self.page.get_by_role("button", name="Save profile", exact=True)

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
        """The bio is shown as text and there is no form to edit it."""
        return self.bio_field.count() == 0 and self.bio_text.is_visible()

    def has_submit_button(self) -> bool:
        return self.submit_button.count() > 0 and self.submit_button.is_visible()
