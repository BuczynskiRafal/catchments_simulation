"""User profile page tests.

Profile page (``/user/<id>/profile``) is public for GET, but only the
owner can edit and submit the form.  Other users see a read-only view.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page

from .conftest import OTHER_USER_PASSWORD
from .pages.user_profile_page import UserProfilePage

pytestmark = pytest.mark.e2e


class TestProfileRendering:
    """Profile page renders for the owner."""

    def test_own_profile_shows_form(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)
        assert pp.has_form()

    def test_own_profile_has_submit_button(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)
        assert pp.has_submit_button()

    def test_own_profile_fields_are_editable(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)
        assert not pp.is_read_only()


class TestProfileReadOnly:
    """Other users see the profile in read-only mode."""

    def test_other_user_profile_is_read_only(
        self, auth_page: Page, live_server, test_user, db
    ) -> None:
        """Create a second user and view their profile — should be read-only."""
        from django.contrib.auth.models import User

        other = User.objects.create_user(
            username="other_user",
            email="other@example.com",
            password=OTHER_USER_PASSWORD,
        )
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(other.id)
        assert pp.has_form()
        assert pp.is_read_only()

    def test_other_user_profile_hides_submit(
        self, auth_page: Page, live_server, test_user, db
    ) -> None:
        from django.contrib.auth.models import User

        other = User.objects.create_user(
            username="other_user2",
            email="other2@example.com",
            password=OTHER_USER_PASSWORD,
        )
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(other.id)
        assert not pp.has_submit_button()


class TestProfileAnonymous:
    """Anonymous users can view profiles (no @login_required)."""

    def test_anonymous_can_view_profile(self, page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(page, live_server.url)
        pp.navigate_to(test_user.id)
        assert pp.has_form()
        assert pp.is_read_only()
