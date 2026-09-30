"""The account pages."""

import uuid

import pytest
from django.urls import reverse

# Random per run, like the e2e credentials, so no password is written in the source.
SIGNUP_PASSWORD = f"T!{uuid.uuid4().hex}"


@pytest.mark.django_db
def test_register_success_logs_in_and_welcomes(client):
    response = client.post(
        reverse("register:register"),
        data={
            "username": "newbie",
            "email": "newbie@example.com",
            "first_name": "New",
            "last_name": "Bie",
            "password1": SIGNUP_PASSWORD,
            "password2": SIGNUP_PASSWORD,
        },
        follow=True,
    )

    assert response.redirect_chain == [(reverse("main:main_view"), 302)]
    assert response.context["user"].username == "newbie"
    assert b"Welcome, New. Your account is ready." in response.content
