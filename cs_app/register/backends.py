"""
Authentication backend that accepts either a username or an email address.

People sign up with both and routinely log in with whichever they remember,
so the login identifier is resolved to a username before the standard
ModelBackend password check runs.
"""

from django.contrib.auth import get_user_model
from django.contrib.auth.backends import ModelBackend
from django.http import HttpRequest

UserModel = get_user_model()


def resolve_username(identifier: str) -> str:
    """
    Map a login identifier to the username of the account it names.

    Tried in order: the exact username, then the username and the email
    address compared case-insensitively. A case-insensitive match counts only
    when exactly one account has it; an ambiguous identifier is returned
    unchanged, so it matches no account rather than an arbitrary one.

    Parameters
    ----------
    identifier : str
        What the user typed into the login field.

    Returns
    -------
    str
        The matching username, or ``identifier`` when no single account matches.
    """
    identifier = identifier.strip()
    users = UserModel._default_manager
    username_field = UserModel.USERNAME_FIELD
    if users.filter(**{username_field: identifier}).exists():
        return identifier

    lookups = [f"{username_field}__iexact"]
    if "@" in identifier:
        lookups.append(f"{UserModel.get_email_field_name()}__iexact")
    for lookup in lookups:
        matches = list(users.filter(**{lookup: identifier})[:2])
        if len(matches) == 1:
            return matches[0].get_username()
    return identifier


class UsernameOrEmailBackend(ModelBackend):
    """ModelBackend whose login identifier may be a username or an email address."""

    def authenticate(
        self,
        request: HttpRequest | None,
        username: str | None = None,
        password: str | None = None,
        **kwargs,
    ):
        if username is None:
            username = kwargs.get(UserModel.USERNAME_FIELD)
        if username:
            username = resolve_username(username)
        # ModelBackend still runs the password hasher when no account matches,
        # so a miss takes as long as a wrong password.
        return super().authenticate(request, username=username, password=password, **kwargs)
