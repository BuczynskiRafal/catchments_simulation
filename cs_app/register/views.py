from django.shortcuts import render
from django.shortcuts import redirect
from django.urls import reverse
from django.contrib.auth.models import Permission
from .forms import RegisterForm


def register(response):
    if response.method == 'POST':
        form = RegisterForm(response.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse('main:main_view'))
        else:
            print("Form errors:", form.errors)
    else:
        form = RegisterForm()
    return render(response, "account/register.html", {'form': form})
