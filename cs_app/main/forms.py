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