"""
This module contains form classes for the web application. It includes:
1. ContactForm - for sending messages through the contact page.
2. UserProfileForm - for updating user profile information.
3. SimulationForm - for selecting simulation parameters.

Each form class is built with the help of the Django Forms library and the
Crispy Forms package.
"""
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from django import forms
from django.forms import ModelForm
from .models import UserProfile


class ContactForm(forms.Form):
    """
    A form for sending messages through the contact page.

    Attributes:
        email: A field for the sender's email address.
        title: A field for the message's title.
        content: A field for the message's content.
        send_to_me: A boolean field to indicate whether to send the message to the sender.
    """

    email = forms.EmailField(label="Adres email")
    title = forms.CharField(label="Tytuł")
    content = forms.CharField(widget=forms.Textarea, label="Treść")
    send_to_me = forms.BooleanField(required=False, label="Prześlij")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_action = "contact"
        self.helper.add_input(Submit("submit", "Wyślij"))


class UserProfileForm(forms.ModelForm):
    """
    A form for updating user profile information.

    Attributes:
        user: A field for the related user object.
        bio: A field for the user's biography.
    """

    class Meta:
        model = UserProfile
        fields = ["user", "bio"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_action = "contact"
        self.helper.add_input(Submit("submit", "Wyślij"))


class SimulationForm(forms.Form):
    """
    A form for selecting simulation parameters.

    Attributes:
        option: A choice field for selecting the parameter to simulate.
        start: An integer field for the starting value of the parameter range.
        stop: An integer field for the ending value of the parameter range.
        step: An integer field for the step size in the parameter range.
        catchment_name: A field for the name of the catchment.
    """

    OPTIONS = (
        ("simulate_percent_slope", "Slope"),
        ("simulate_area", "Area"),
        ("simulate_width", "Width"),
        ("simulate_percent_impervious", "Impervious"),
        ("simulate_percent_zero_imperv", "Zero-Imperv"),
    )
    option = forms.ChoiceField(
        choices=OPTIONS, widget=forms.Select(attrs={"class": "form-select"})
    )
    start = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=1,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    stop = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=10,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    step = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=1,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    catchment_name = forms.CharField(
        max_length=100, widget=forms.TextInput(attrs={"class": "form-control"})
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_action = "simulation_view"
        self.helper.add_input(Submit("submit", "Run Simulation"))
