import logging
from typing import Any

from django.conf import settings
from django.core.mail import send_mail

logger = logging.getLogger(__name__)


def send_message(data: dict[str, Any]) -> bool:
    """
    Send a contact form message via email.

    Parameters
    ----------
    data : dict
        Form data containing 'email', 'title', 'content', and optionally 'send_to_me'.

    Returns
    -------
    bool
        True if message was sent successfully, False otherwise.
    """
    email_from = data.get("email", "")
    title = data.get("title", "Contact Form Message")
    content = data.get("content", "")
    send_to_me = data.get("send_to_me", False)

    # Build recipients list
    recipients = [settings.DEFAULT_FROM_EMAIL] if hasattr(settings, "DEFAULT_FROM_EMAIL") else []
    if send_to_me and email_from:
        recipients.append(email_from)

    if not recipients:
        logger.warning(
            "No recipients configured for contact form. Set DEFAULT_FROM_EMAIL in settings."
        )
        return False

    message = f"From: {email_from}\n\n{content}"

    try:
        send_mail(
            subject=title,
            message=message,
            from_email=email_from or None,
            recipient_list=recipients,
            fail_silently=False,
        )
        logger.info(f"Contact form message sent: {title}")
        return True
    except Exception as e:
        logger.error(f"Failed to send contact form message: {e}")
        return False
