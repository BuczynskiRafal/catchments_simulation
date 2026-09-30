"""
This module defines the account forms: sign-up and log in.

RegisterForm extends the built-in Django UserCreationForm with email,
first_name and last_name fields; the email must not belong to another account,
because it can be used to log in. LoginForm extends AuthenticationForm so the
login field accepts a username or an email address (see register.backends).
"""

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(widget=forms.EmailInput(attrs={"autocomplete": "email"}))
    first_name = forms.CharField(widget=forms.TextInput(attrs={"autocomplete": "given-name"}))
    last_name = forms.CharField(widget=forms.TextInput(attrs={"autocomplete": "family-name"}))

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

    def clean_email(self) -> str:
        """Reject an email address already used by another account, ignoring case."""
        email = self.cleaned_data["email"]
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError(
                "An account with this email already exists. Log in instead.",
                code="duplicate_email",
            )
        return email


class LoginForm(AuthenticationForm):
    """Log in with a username or an email address and a password."""

    error_messages = {
        **AuthenticationForm.error_messages,
        "invalid_login": (
            "Please enter a correct username or email and password. "
            "Note that the password is case-sensitive."
        ),
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        username = self.fields["username"]
        username.label = "Username or email"
        # Room for a full email address; the stock limit is the username column length.
        username.max_length = User._meta.get_field("email").max_length
        username.widget.attrs["maxlength"] = username.max_length
        username.widget.attrs["autocomplete"] = "username"
