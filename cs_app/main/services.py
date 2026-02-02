import logging
from typing import Any, Union

from django.conf import settings
from django.core.mail import send_mail

from main.schemas import ContactMessage

logger = logging.getLogger(__name__)


def send_message(data: Union[ContactMessage, dict[str, Any]]) -> bool:
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

    # Build recipients list
    recipients = [settings.DEFAULT_FROM_EMAIL] if hasattr(settings, "DEFAULT_FROM_EMAIL") else []
    if data.send_to_me and data.email:
        recipients.append(data.email)

    if not recipients:
        logger.warning(
            "No recipients configured for contact form. Set DEFAULT_FROM_EMAIL in settings."
        )
        return False

    message = f"From: {data.email}\n\n{data.content}"

    try:
        send_mail(
            subject=data.title,
            message=message,
            from_email=data.email or None,
            recipient_list=recipients,
            fail_silently=False,
        )
        logger.info(f"Contact form message sent: {data.title}")
        return True
    except Exception as e:
        logger.error(f"Failed to send contact form message: {e}")
        return False
