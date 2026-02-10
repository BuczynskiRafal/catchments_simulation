"""
This module contains form classes for the web application. It includes:
1. ContactForm - for sending messages through the contact page.
2. UserProfileForm - for updating user profile information.
3. SimulationForm - for selecting simulation parameters.
4. TimeseriesForm - for selecting timeseries analysis parameters.

Each form class is built with the help of the Django Forms library and the
Crispy Forms package.
"""

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from django import forms

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
        self.helper.form_action = "userprofile"
        self.helper.add_input(Submit("submit", "Wyślij"))


class SimulationForm(forms.Form):
    """
    A form for selecting simulation parameters.

    Supports both range-based methods (requiring start/stop/step) and
    predefined methods (using literature values, no range parameters).

    Attributes:
        option: A choice field for selecting the parameter to simulate.
        start: An integer field for the starting value of the parameter range.
        stop: An integer field for the ending value of the parameter range.
        step: An integer field for the step size in the parameter range.
        catchment_name: A field for the name of the catchment.
    """

    OPTIONS = (
        (
            "Range-based Parameters",
            (
                ("simulate_percent_slope", "Slope (%)"),
                ("simulate_area", "Area (ha)"),
                ("simulate_width", "Width (m)"),
                ("simulate_percent_impervious", "Impervious (%)"),
                ("simulate_percent_zero_imperv", "Zero-Imperv (%)"),
                ("simulate_curb_length", "Curb Length (m)"),
            ),
        ),
        (
            "Predefined Literature Values",
            (
                ("simulate_n_imperv", "Manning's n - Impervious"),
                ("simulate_n_perv", "Manning's n - Pervious"),
                ("simulate_s_imperv", "Depression Storage - Impervious"),
                ("simulate_s_perv", "Depression Storage - Pervious"),
            ),
        ),
    )

    PREDEFINED_METHODS = frozenset(
        {
            "simulate_n_imperv",
            "simulate_n_perv",
            "simulate_s_imperv",
            "simulate_s_perv",
        }
    )

    MAX_SWEEP_STEPS = 100

    option = forms.ChoiceField(choices=OPTIONS, widget=forms.Select(attrs={"class": "form-select"}))
    start = forms.IntegerField(
        min_value=0,
        max_value=10000,
        initial=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    stop = forms.IntegerField(
        min_value=0,
        max_value=10000,
        initial=10,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    step = forms.IntegerField(
        min_value=1,
        max_value=10000,
        initial=1,
        required=False,
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

    def clean(self):
        cleaned_data = super().clean()
        option = cleaned_data.get("option")
        if option not in self.PREDEFINED_METHODS:
            for field in ("start", "stop", "step"):
                if cleaned_data.get(field) is None:
                    self.add_error(field, "This field is required for the selected method.")
            start = cleaned_data.get("start")
            stop = cleaned_data.get("stop")
            step = cleaned_data.get("step")
            if start is not None and stop is not None and start > stop:
                self.add_error("stop", "Stop must be >= start.")
            if start is not None and stop is not None and step and step > 0:
                if (stop - start) / step >= self.MAX_SWEEP_STEPS:
                    self.add_error(
                        "step",
                        f"Too many steps (max {self.MAX_SWEEP_STEPS}). "
                        "Increase step size or reduce range.",
                    )
        return cleaned_data


class TimeseriesForm(forms.Form):
    """
    A form for timeseries analysis. Supports single-run and parameter sweep modes.

    Attributes:
        mode: Choice between single timeseries or parameter sweep.
        feature: The subcatchment feature to vary (sweep mode only).
        start: Start value for parameter sweep.
        stop: Stop value for parameter sweep.
        step: Step size for parameter sweep.
        catchment_name: The subcatchment identifier.
    """

    MODE_CHOICES = (
        ("single", "Single Timeseries"),
        ("sweep", "Parameter Sweep Timeseries"),
    )

    FEATURE_CHOICES = (
        ("PercSlope", "Slope (%)"),
        ("Area", "Area (ha)"),
        ("Width", "Width (m)"),
        ("PercImperv", "Impervious (%)"),
        ("CurbLength", "Curb Length (m)"),
    )

    mode = forms.ChoiceField(
        choices=MODE_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    feature = forms.ChoiceField(
        choices=FEATURE_CHOICES,
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    start = forms.FloatField(
        min_value=0,
        max_value=10000,
        initial=0,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    stop = forms.FloatField(
        min_value=0,
        max_value=10000,
        initial=100,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    step = forms.FloatField(
        min_value=0.1,
        initial=10,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    catchment_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={"class": "form-control"}),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_action = "timeseries"
        self.helper.add_input(Submit("submit", "Run Analysis"))

    MAX_SWEEP_STEPS = 100

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("mode") == "sweep":
            if not cleaned_data.get("feature"):
                self.add_error("feature", "Required for parameter sweep mode.")
            for field in ("start", "stop", "step"):
                if cleaned_data.get(field) is None:
                    self.add_error(field, "Required for parameter sweep mode.")
            start = cleaned_data.get("start")
            stop = cleaned_data.get("stop")
            step = cleaned_data.get("step")
            if start is not None and stop is not None and start > stop:
                self.add_error("stop", "Stop must be >= start.")
            if start is not None and stop is not None and step and step > 0:
                if (stop - start) / step >= self.MAX_SWEEP_STEPS:
                    self.add_error(
                        "step",
                        f"Too many steps (max {self.MAX_SWEEP_STEPS}). "
                        "Increase step size or reduce range.",
                    )
        return cleaned_data
