"""User profile page tests.

Profile page (``/user/<id>/profile``) is public for GET, but only the
owner can edit the bio. Everyone else sees it as text.
"""

from __future__ import annotations

import re

import pytest
from django.contrib.auth.models import User
from playwright.sync_api import Page, expect

from .conftest import OTHER_USER_PASSWORD, TEST_EMAIL, TEST_FIRST_NAME, TEST_LAST_NAME
from .pages.user_profile_page import UserProfilePage

pytestmark = pytest.mark.e2e


@pytest.fixture()
def other_user(db) -> User:
    return User.objects.create_user(
        username="other_user", email="other@example.com", password=OTHER_USER_PASSWORD
    )


class TestOwnProfile:
    def test_shows_identity_and_editable_bio(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)

        assert pp.get_heading(level=1) == "Profile"
        expect(pp.profile_name).to_have_text(f"{TEST_FIRST_NAME} {TEST_LAST_NAME}")
        expect(pp.profile_meta).to_contain_text(TEST_EMAIL)
        assert pp.has_form()
        assert not pp.is_read_only()
        assert pp.has_submit_button()
        expect(auth_page.locator("#id_user")).to_have_count(0)

    def test_save_bio_round_trip(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)

        pp.fill_bio("Urban drainage engineer.")
        pp.submit()

        expect(auth_page).to_have_url(re.compile(rf".*/user/{test_user.id}/profile$"))
        expect(pp.bio_field).to_have_value("Urban drainage engineer.")

    def test_page_title(self, auth_page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(test_user.id)
        assert "Profile" in pp.get_title()


class TestOtherProfile:
    def test_other_user_profile_is_read_only(
        self, auth_page: Page, live_server, test_user, other_user
    ) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(other_user.id)
        expect(pp.profile_name).to_have_text("other_user")
        assert pp.is_read_only()
        expect(pp.bio_text).to_have_text("No bio yet.")

    def test_other_user_profile_hides_submit_and_email(
        self, auth_page: Page, live_server, test_user, other_user
    ) -> None:
        pp = UserProfilePage(auth_page, live_server.url)
        pp.navigate_to(other_user.id)
        assert not pp.has_submit_button()
        expect(pp.profile_meta).not_to_contain_text("other@example.com")


class TestProfileAnonymous:
    """Anonymous users can view profiles (no @login_required)."""

    def test_anonymous_can_view_profile(self, page: Page, live_server, test_user) -> None:
        pp = UserProfilePage(page, live_server.url)
        pp.navigate_to(test_user.id)
        expect(pp.profile_name).to_be_visible()
        assert pp.is_read_only()

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(
        self, page: Page, live_server, test_user, scheme: str
    ) -> None:
        page.emulate_media(color_scheme=scheme)
        pp = UserProfilePage(page, live_server.url)
        pp.navigate_to(test_user.id)
        report = pp.run_axe_audit()
        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]
