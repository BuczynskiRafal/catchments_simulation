"""Contact page tests."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from .pages.contact_page import ContactPage

pytestmark = pytest.mark.e2e


@pytest.fixture()
def contact(page: Page, live_server) -> ContactPage:
    cp = ContactPage(page, live_server.url)
    cp.navigate_to()
    return cp


class TestContactForm:
    """Contact form rendering and submission."""

    def test_form_renders_with_english_labels(self, contact: ContactPage) -> None:
        assert contact.has_form()
        page = contact.page
        expect(page.get_by_label("Email")).to_be_visible()
        expect(page.get_by_label("Subject")).to_be_visible()
        expect(page.get_by_label("Message")).to_be_visible()
        expect(page.get_by_label("Send me a copy")).to_be_visible()

    def test_submit_button_text(self, contact: ContactPage) -> None:
        expect(contact.submit_button).to_be_visible()

    def test_valid_submission_redirects_with_success_message(
        self, contact: ContactPage, db
    ) -> None:
        contact.fill_form(
            email="test@example.com", title="Test Subject", content="Test message body."
        )
        contact.submit()

        expect(contact.page).to_have_url(re.compile(r".*/contact$"))
        expect(contact.page.locator(".alert-success")).to_contain_text("Message sent.")
        expect(contact.email_input).to_have_value("")

    def test_empty_submission_is_blocked_by_the_browser(self, contact: ContactPage) -> None:
        contact.submit()
        expect(contact.page).to_have_url(re.compile(r".*/contact$"))
        expect(contact.page.locator(".alert-success")).to_have_count(0)

    def test_server_errors_are_inline_and_focused(self, contact: ContactPage, db) -> None:
        """Server-side validation (the browser's is bypassed) marks and focuses the field."""
        page = contact.page
        page.evaluate("() => document.querySelector('form[data-cs-form]').noValidate = true")
        contact.fill_form(email="not-an-email", title="Subject", content="Body")
        contact.submit()

        expect(page.locator("#error_1_id_email")).to_contain_text("Enter a valid email address.")
        expect(contact.email_input).to_have_attribute("aria-invalid", "true")
        expect(contact.email_input).to_be_focused()
        expect(contact.content_input).to_have_value("Body")

    def test_page_title_and_heading(self, contact: ContactPage) -> None:
        assert "Contact" in contact.get_title()
        assert contact.get_heading(level=1) == "Contact"

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(self, page: Page, live_server, scheme: str) -> None:
        page.emulate_media(color_scheme=scheme)
        cp = ContactPage(page, live_server.url)
        cp.navigate_to()
        report = cp.run_axe_audit()
        assert not report.critical + report.serious, [
            v.id for v in report.critical + report.serious
        ]
