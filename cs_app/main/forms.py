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
from crispy_forms.layout import Layout, Submit
from django import forms

from .models import UserProfile

# Same wording as the live range hint on the simulation and timeseries pages.
STOP_BEFORE_START_MESSAGE = "Stop must be greater than or equal to start."


class ContactForm(forms.Form):
    """
    A form for sending messages through the contact page.

    Attributes:
        email: A field for the sender's email address.
        title: A field for the message's title.
        content: A field for the message's content.
        send_to_me: A boolean field to indicate whether to send the message to the sender.
    """

    # Length limits mirror main.schemas.ContactMessage.
    email = forms.EmailField(
        label="Email", widget=forms.EmailInput(attrs={"autocomplete": "email"})
    )
    title = forms.CharField(label="Subject", max_length=200)
    content = forms.CharField(
        label="Message", max_length=5000, widget=forms.Textarea(attrs={"rows": 6})
    )
    send_to_me = forms.BooleanField(required=False, label="Send me a copy")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = "post"
        self.helper.form_action = "main:contact"
        self.helper.attrs = {"data-cs-form": ""}  # progressive enhancement hook (pages/forms.js)
        self.helper.add_input(Submit("submit", "Send message"))


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
        labels = {"bio": "Bio"}
        help_texts = {
            "bio": "A few words about you and your work. Anyone with the link can read it."
        }
        widgets = {"bio": forms.Textarea(attrs={"rows": 5})}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The profile's account comes from the URL (via instance or initial), never
        # from the request: the field is locked to that one account and not rendered.
        initial_user = self.initial.get("user")
        owner_pk = self.instance.user_id or getattr(initial_user, "pk", initial_user)
        user_field = self.fields["user"]
        user_field.disabled = True
        user_field.queryset = user_field.queryset.filter(pk=owner_pk)
        self.owner = user_field.queryset.first()

        self.helper = FormHelper()
        # No form_action: the form posts back to the profile URL it was rendered on.
        self.helper.form_method = "post"
        self.helper.attrs = {"data-cs-form": ""}  # progressive enhancement hook (pages/forms.js)
        self.helper.layout = Layout("bio")
        self.helper.add_input(Submit("submit", "Save profile"))


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

    option = forms.ChoiceField(
        label="Parameter to vary",
        choices=OPTIONS,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    start = forms.IntegerField(
        label="Start",
        min_value=0,
        max_value=10000,
        initial=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    stop = forms.IntegerField(
        label="Stop",
        min_value=0,
        max_value=10000,
        initial=10,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    step = forms.IntegerField(
        label="Step",
        min_value=1,
        max_value=10000,
        initial=1,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    catchment_name = forms.CharField(
        label="Subcatchment",
        widget=forms.Select(
            choices=[("", "--- Upload a file first ---")],
            attrs={"class": "form-select"},
        ),
    )

    RANGE_FIELDS = ("start", "stop", "step")

    def __init__(self, *args, catchment_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if catchment_choices is not None:
            self.fields["catchment_name"].widget.choices = catchment_choices

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("option") in self.PREDEFINED_METHODS:
            # Literature-value methods take no range, so values the user left in the
            # range fields (even invalid ones) must not block the run.
            for field in self.RANGE_FIELDS:
                self.errors.pop(field, None)
            return cleaned_data

        for field in self.RANGE_FIELDS:
            if cleaned_data.get(field) is None and field not in self.errors:
                self.add_error(field, "This field is required for the selected method.")
        start = cleaned_data.get("start")
        stop = cleaned_data.get("stop")
        step = cleaned_data.get("step")
        if start is not None and stop is not None and start > stop:
            self.add_error("stop", STOP_BEFORE_START_MESSAGE)
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
        ("single", "Single run"),
        ("sweep", "Parameter sweep"),
    )

    FEATURE_CHOICES = (
        ("PercSlope", "Slope (%)"),
        ("Area", "Area (ha)"),
        ("Width", "Width (m)"),
        ("PercImperv", "Impervious (%)"),
        ("CurbLength", "Curb Length (m)"),
    )

    # Rendered as a segmented control (radios + button labels) by timeseries.html.
    mode = forms.ChoiceField(
        label="Analysis mode",
        choices=MODE_CHOICES,
        initial="single",
        widget=forms.RadioSelect(attrs={"class": "btn-check", "autocomplete": "off"}),
    )
    feature = forms.ChoiceField(
        label="Parameter to vary",
        choices=FEATURE_CHOICES,
        required=False,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    start = forms.FloatField(
        label="Start",
        min_value=0,
        max_value=10000,
        initial=0,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    stop = forms.FloatField(
        label="Stop",
        min_value=0,
        max_value=10000,
        initial=100,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    step = forms.FloatField(
        label="Step",
        min_value=0.1,
        initial=10,
        required=False,
        widget=forms.NumberInput(attrs={"class": "form-control"}),
    )
    catchment_name = forms.CharField(
        label="Subcatchment",
        widget=forms.Select(
            choices=[("", "--- Upload a file first ---")],
            attrs={"class": "form-select"},
        ),
    )

    def __init__(self, *args, catchment_choices=None, **kwargs):
        super().__init__(*args, **kwargs)
        if catchment_choices is not None:
            self.fields["catchment_name"].widget.choices = catchment_choices

    MAX_SWEEP_STEPS = 100
    SWEEP_FIELDS = ("feature", "start", "stop", "step")

    def clean(self):
        cleaned_data = super().clean()
        if cleaned_data.get("mode") != "sweep":
            # The sweep range is not used outside sweep mode, so values the user left
            # there (even invalid ones) must not block the run.
            for field in self.SWEEP_FIELDS:
                self.errors.pop(field, None)
            return cleaned_data

        for field in self.SWEEP_FIELDS:
            if cleaned_data.get(field) in (None, "") and field not in self.errors:
                self.add_error(field, "Required for parameter sweep mode.")
        start = cleaned_data.get("start")
        stop = cleaned_data.get("stop")
        step = cleaned_data.get("step")
        if start is not None and stop is not None and start > stop:
            self.add_error("stop", STOP_BEFORE_START_MESSAGE)
        if start is not None and stop is not None and step and step > 0:
            if (stop - start) / step >= self.MAX_SWEEP_STEPS:
                self.add_error(
                    "step",
                    f"Too many steps (max {self.MAX_SWEEP_STEPS}). "
                    "Increase step size or reduce range.",
                )
        return cleaned_data
