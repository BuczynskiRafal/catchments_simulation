"""
This module contains the view for user registration in the web application.

It provides a view function, register, that handles the registration process
using a custom registration form (RegisterForm).
"""

import logging

from django.contrib import messages
from django.contrib.auth import login
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from .forms import RegisterForm

logger = logging.getLogger(__name__)


def register(request: HttpRequest) -> HttpResponse:
    """
    Render the registration form and handle form submission.

    A valid submission creates the user, logs them in and redirects to the
    home page. An invalid one re-renders the form with inline field errors.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered registration template.
    """
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            name = user.get_short_name() or user.get_username()
            messages.success(request, f"Welcome, {name}. Your account is ready.")
            return redirect("main:main_view")
        # Field names only: the submitted values (and passwords) never reach the log.
        logger.info("Registration rejected: invalid fields %s", sorted(form.errors))
    else:
        form = RegisterForm()
    return render(request, "account/register.html", {"form": form})
