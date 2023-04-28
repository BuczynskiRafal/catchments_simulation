from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from django import forms
from django.forms import ModelForm
from .models import UserProfile


class ContactForm(forms.Form):
    email = forms.EmailField(label="Adres email")
    title = forms.CharField(label='Tytuł')
    content = forms.CharField(widget=forms.Textarea, label='Treść')
    send_to_me = forms.BooleanField(required=False, label='Prześlij')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_action = 'contact'
        self.helper.add_input(Submit('submit', 'Wyślij'))


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['user', 'bio']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_action = 'contact'
        self.helper.add_input(Submit('submit', 'Wyślij'))


class SimulationForm(forms.Form):
    OPTIONS = (
        ('simulate_percent_slope', 'Slope'),
        ('simulate_area', 'Area'),
        ('simulate_width', 'Width'),
        ('simulate_percent_impervious', 'Impervious'),
        ('simulate_percent_zero_imperv', 'Zero-Imperv'),
    )
    option = forms.ChoiceField(choices=OPTIONS, widget=forms.Select(attrs={'class': 'form-select'}))
    start = forms.IntegerField(min_value=1, max_value=100, initial=1, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    stop = forms.IntegerField(min_value=1, max_value=100, initial=10, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    step = forms.IntegerField(min_value=1, max_value=100, initial=1, widget=forms.NumberInput(attrs={'class': 'form-control'}))
    catchment_name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_action = 'simulation_view'
        self.helper.add_input(Submit('submit', 'Run Simulation'))