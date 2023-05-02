"""
This module defines a custom registration form for user sign-up, extending
the built-in Django UserCreationForm.

The custom form adds additional fields (email, first_name, and last_name) to
the standard registration form.
"""
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField()  # Additional email field
    first_name = forms.CharField()  # Additional first_name field
    last_name = forms.CharField()  # Additional last_name field

    class Meta:
        model = User  # The User model from Django's authentication framework
        fields = [
            "username",
            "email",
            "first_name",
            "last_name",
            "password1",  # Password field, required by UserCreationForm
            "password2",  # Password confirmation field, required by UserCreationForm
        ]
