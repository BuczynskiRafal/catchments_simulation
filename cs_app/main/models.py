"""
This module contains the UserProfile model for the web application.

The UserProfile model extends the default Django User model with additional
fields, such as a biography.
"""
from django.db import models


class UserProfile(models.Model):
    """
    A model representing a user's profile in the web application.

    The UserProfile model is linked to the default Django User model through
    a OneToOneField relationship.

    Attributes:
        user: The related User object.
        bio: A text field for the user's biography.
    """

    user = models.OneToOneField("auth.User", on_delete=models.CASCADE)
    bio = models.TextField()
