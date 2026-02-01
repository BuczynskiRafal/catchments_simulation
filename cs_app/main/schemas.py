"""Pydantic schemas for the Django web application.

These schemas provide runtime validation for data flowing between
the views, services, and external systems (email, file uploads).
"""

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ContactMessage(BaseModel):
    """Validated payload for the contact form email service.

    Replaces the untyped ``dict[str, Any]`` that was passed to ``send_message()``.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    email: EmailStr = Field(..., description="Sender email address")
    title: str = Field(..., min_length=1, max_length=200, description="Message subject")
    content: str = Field(..., min_length=1, max_length=5000, description="Message body")
    send_to_me: bool = Field(default=False, description="CC a copy to the sender")
