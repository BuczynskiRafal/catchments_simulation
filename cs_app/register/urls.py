"""
This module contains the URL configuration for the registration app.

It defines a single URL pattern for the user registration view, using the
register view function from the views module.
"""
from django.urls import path

from .views import register

app_name = "register"


urlpatterns = [
    path("register/", register, name="register"),
]
