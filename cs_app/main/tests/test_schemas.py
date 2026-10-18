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

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ContactMessage.model_validate(
                {
                    "email": "user@example.com",
                    "title": "Subject",
                    "content": "Body",
                    "unexpected": "field",
                }
            )

    def test_instance_is_frozen(self):
        msg = ContactMessage(email="user@example.com", title="Subject", content="Body")
        with pytest.raises(ValidationError):
            msg.title = "Mutated"


class TestSendMessageService:
    """Tests for main.services.send_message email delivery and relay protection."""

    def test_send_message_uses_default_from_and_reply_to(self, mailoutbox, settings):
        from main.services import send_message

        settings.DEFAULT_FROM_EMAIL = "noreply@example.org"
        settings.CONTACT_RECIPIENT_EMAIL = "owner@example.org"
        settings.CONTACT_ALLOW_SENDER_COPY = False

        ok = send_message(
            {
                "email": "visitor@example.com",
                "title": "Inquiry",
                "content": "Hello!",
                "send_to_me": True,
            }
        )

        assert ok is True
        assert len(mailoutbox) == 1
        sent = mailoutbox[0]
        assert sent.from_email == "noreply@example.org"
        assert sent.to == ["owner@example.org"]
        assert sent.reply_to == ["visitor@example.com"]
        assert "From: visitor@example.com" in sent.body

    def test_send_message_allows_sender_copy_when_explicitly_enabled(self, mailoutbox, settings):
        from main.services import send_message

        settings.DEFAULT_FROM_EMAIL = "noreply@example.org"
        settings.CONTACT_RECIPIENT_EMAIL = "owner@example.org"
        settings.CONTACT_ALLOW_SENDER_COPY = True

        ok = send_message(
            {
                "email": "visitor@example.com",
                "title": "Inquiry",
                "content": "Copy me please.",
                "send_to_me": True,
            }
        )

        assert ok is True
        assert len(mailoutbox) == 1
        assert mailoutbox[0].to == ["owner@example.org", "visitor@example.com"]

    def test_send_message_returns_false_when_no_recipient_configured(self, settings, caplog):
        from main.services import send_message

        settings.CONTACT_RECIPIENT_EMAIL = ""
        settings.DEFAULT_FROM_EMAIL = ""

        with caplog.at_level("WARNING"):
            ok = send_message(
                {"email": "visitor@example.com", "title": "Subject", "content": "Body"}
            )

        assert ok is False
        assert "No recipients configured" in caplog.text

    def test_send_message_logs_exception_on_smtp_failure(self, monkeypatch, caplog):
        from smtplib import SMTPException

        from main import services

        def broken_send(self, fail_silently=False):
            raise SMTPException("SMTP down")

        monkeypatch.setattr(services.EmailMessage, "send", broken_send)

        with caplog.at_level("ERROR"):
            ok = services.send_message(
                {"email": "visitor@example.com", "title": "Subject", "content": "Body"}
            )

        assert ok is False
        assert "Failed to send contact form message" in caplog.text


class TestApiSchemas:
    """Tests for SimulationRequest and TimeseriesRequest Pydantic schemas."""

    def test_simulation_request_valid(self):
        from main.schemas import SimulationRequest

        req = SimulationRequest(
            option="simulate_percent_slope",
            catchment_name="S1",
            start=1.0,
            stop=5.0,
            step=1.0,
        )
        assert req.option == "simulate_percent_slope"
        assert req.catchment_name == "S1"
        assert req.start == 1.0

    def test_simulation_request_rejects_extra_fields(self):
        from main.schemas import SimulationRequest

        with pytest.raises(ValidationError):
            SimulationRequest(
                option="simulate_n_imperv",
                catchment_name="S1",
                extra_field="nope",
            )

    def test_timeseries_request_valid_modes(self):
        from main.schemas import TimeseriesRequest

        single = TimeseriesRequest(mode="single", catchment_name="S1")
        assert single.mode == "single"
        assert single.feature is None

        sweep = TimeseriesRequest(
            mode="sweep",
            catchment_name="S1",
            feature="PercImperv",
            start=0.0,
            stop=20.0,
            step=10.0,
        )
        assert sweep.mode == "sweep"
        assert sweep.feature == "PercImperv"

    def test_timeseries_request_rejects_invalid_mode(self):
        from main.schemas import TimeseriesRequest

        with pytest.raises(ValidationError):
            TimeseriesRequest(mode="invalid", catchment_name="S1")

    def test_simulation_request_enforces_range_and_rejects_bool(self):
        from main.schemas import SimulationRequest

        with pytest.raises(ValidationError):
            SimulationRequest(option="simulate_percent_slope", catchment_name="S1")

        with pytest.raises(ValidationError):
            SimulationRequest(
                option="simulate_percent_slope",
                catchment_name="S1",
                start=10.0,
                stop=1.0,
                step=1.0,
            )

        with pytest.raises(ValidationError):
            SimulationRequest(
                option="simulate_percent_slope",
                catchment_name="S1",
                start=True,
                stop=5.0,
                step=1.0,
            )

    def test_timeseries_request_enforces_sweep_fields_and_rejects_bool(self):
        from main.schemas import TimeseriesRequest

        with pytest.raises(ValidationError):
            TimeseriesRequest(mode="sweep", catchment_name="S1")

        with pytest.raises(ValidationError):
            TimeseriesRequest(
                mode="sweep",
                catchment_name="S1",
                feature="PercImperv",
                start=20.0,
                stop=5.0,
                step=1.0,
            )

        with pytest.raises(ValidationError):
            TimeseriesRequest(
                mode="sweep",
                catchment_name="S1",
                feature="PercImperv",
                start=False,
                stop=20.0,
                step=10.0,
            )
