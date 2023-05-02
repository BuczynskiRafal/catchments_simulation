"""
This module contains the view for user registration in the web application.

It provides a view function, register, that handles the registration process
using a custom registration form (RegisterForm).
"""

from django.shortcuts import render
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib.auth.models import Permission
from django.http import HttpRequest, HttpResponse
from .forms import RegisterForm


def register(response: HttpRequest) -> HttpResponse:
    """
    Render the registration form and handle form submission.

    The registration form is validated upon submission. If the form is valid,
    the user is saved and redirected to the main view. If the form is invalid,
    any errors are printed and the user remains on the registration page.

    Parameters
    ----------
    response : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered registration template.
    """
    if response.method == "POST":
        form = RegisterForm(response.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse("main:main_view"))
        else:
            print("Form errors:", form.errors)
    else:
        form = RegisterForm()
    return render(response, "account/register.html", {"form": form})
