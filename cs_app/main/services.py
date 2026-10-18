import logging
from smtplib import SMTPException
from typing import Any

from django.conf import settings
from django.core.mail import EmailMessage

from main.schemas import ContactMessage

logger = logging.getLogger(__name__)


def send_message(data: ContactMessage | dict[str, Any]) -> bool:
    """
    Send a contact form message via email.

    Parameters
    ----------
    data : ContactMessage | dict
        Validated ``ContactMessage`` instance **or** a raw dict (which will
        be validated on the fly).  Passing a ``ContactMessage`` is preferred
        because validation happens once, at the boundary.

    Returns
    -------
    bool
        True if message was sent successfully, False otherwise.

    Raises
    ------
    pydantic.ValidationError
        If *data* is a dict that fails schema validation.
    """
    if not isinstance(data, ContactMessage):
        data = ContactMessage.model_validate(data)

    recipient_email = getattr(settings, "CONTACT_RECIPIENT_EMAIL", None) or getattr(
        settings, "DEFAULT_FROM_EMAIL", None
    )
    recipients = [recipient_email] if recipient_email else []
    if (
        data.send_to_me
        and data.email
        and getattr(settings, "CONTACT_ALLOW_SENDER_COPY", False)
        and data.email not in recipients
    ):
        recipients.append(data.email)

    if not recipients:
        logger.warning(
            "No recipients configured for contact form. Set CONTACT_RECIPIENT_EMAIL in settings."
        )
        return False

    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", None) or None
    message = f"From: {data.email}\n\n{data.content}"

    try:
        email = EmailMessage(
            subject=data.title,
            body=message,
            from_email=from_email,
            to=recipients,
            reply_to=[data.email] if data.email else None,
        )
        email.send(fail_silently=False)
        logger.info("Contact form message sent: %s", data.title)
        return True
    except (SMTPException, OSError, Exception):
        logger.exception("Failed to send contact form message from %s", data.email)
        return False
