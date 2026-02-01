"""Tests for Pydantic schemas in the Django web application."""

import pytest
from pydantic import ValidationError

from main.schemas import ContactMessage


class TestContactMessage:
    """Tests for ContactMessage schema."""

    def test_valid_message(self):
        msg = ContactMessage(
            email="user@example.com",
            title="Test subject",
            content="Hello, this is a test.",
            send_to_me=True,
        )
        assert msg.email == "user@example.com"
        assert msg.title == "Test subject"
        assert msg.send_to_me is True

    def test_default_send_to_me(self):
        msg = ContactMessage(
            email="user@example.com",
            title="Subject",
            content="Body",
        )
        assert msg.send_to_me is False

    def test_invalid_email_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage(
                email="not-an-email",
                title="Subject",
                content="Body",
            )

    def test_empty_title_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage(
                email="user@example.com",
                title="",
                content="Body",
            )

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage(
                email="user@example.com",
                title="Subject",
                content="",
            )

    def test_title_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage(
                email="user@example.com",
                title="x" * 201,
                content="Body",
            )

    def test_content_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage(
                email="user@example.com",
                title="Subject",
                content="x" * 5001,
            )

    def test_whitespace_stripping(self):
        msg = ContactMessage(
            email="user@example.com",
            title="  Subject  ",
            content="  Body  ",
        )
        assert msg.title == "Subject"
        assert msg.content == "Body"

    def test_from_dict(self):
        data = {
            "email": "user@example.com",
            "title": "Subject",
            "content": "Body",
            "send_to_me": True,
        }
        msg = ContactMessage.model_validate(data)
        assert msg.email == "user@example.com"
        assert msg.send_to_me is True

    def test_from_dict_missing_required_field(self):
        with pytest.raises(ValidationError):
            ContactMessage.model_validate({"email": "user@example.com", "title": "Subject"})
